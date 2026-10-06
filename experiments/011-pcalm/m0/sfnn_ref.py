"""experiment-011 M0: SFNN (HalfKA2 1024-7-64 k3k3) の float 参照実装 (PyTorch) と PC-ALM の勾配。

- 順伝播は BulletOu `2a8e5ed` の CUDA カーネル (`sfnn_sparse_l0_pairwise_concat_kernel` ほか) と
  experiment-009 の `loss_eval` の float forward と同じ式。crelu の勾配は BulletOu と同じく
  「出力値が開区間 (0, 1) にあるときだけ 1」
- パラメータは state.bin の畳み込み前の値 (FT の仮想特徴量の行、L1/L2/L3 の shared factorizer を含む) をそのまま持ち、
  順伝播の中で実効重みを作る。したがって自動微分の勾配は BulletOu が Ranger に渡す勾配と同じ並び・同じ意味になる
- loss は BulletOu の sigmoid-MSE (`loss_sigmoid_mse_reduce_kernel`, output_inv_scale = 1):
  ``mean_s w_s (sigmoid(out_s) - t_s)^2``, ``t_s = sigmoid(score_s / 290)``, ``w_s = [|score_s| < 32000]``
- PC-ALM は参照実装 (SakanaAI/pc-alm `pcalm/inference.py`) と同じ手順。状態は ``h1 = c`` (pairwise 後, 1024)、
  ``h2 = z1`` (L1 出力, 8)、``h3 = z2`` (L2 出力, 64)。エネルギーはサンプルごとの和 (参照実装はバッチ平均に
  ``state_lr * batch`` を掛けており、同じ更新になる)。重みの勾配はバッチ平均 (BP の勾配と同じ単位)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

N_BASE = 131_949
N_VIRT = 1_629
FT = 1024
HALF = FT // 2
L1_OUT = 8
L1_H = 7
L2_IN = 14
L2 = 64
STACKS = 9
PAIR_SCALE = 127.0 / 128.0
PAD = 0xFFFF_FFFF

SHAPES: dict[str, tuple[int, ...]] = {
    "l0w": (N_BASE + N_VIRT, FT),
    "l0b": (FT,),
    "l1w": (STACKS, L1_OUT, FT),
    "l1b": (STACKS, L1_OUT),
    "l1fw": (FT, L1_OUT),
    "l1fb": (L1_OUT,),
    "l2w": (STACKS, L2, L2_IN),
    "l2b": (STACKS, L2),
    "l2fw": (L2, L2_IN),
    "l2fb": (L2,),
    "l3w": (STACKS, L2),
    "l3b": (STACKS,),
    "l3fw": (L2,),
    "l3fb": (1,),
}
# 勾配を比べる層のまとまり (FT / L1 / L2 / L3)
GROUPS: dict[str, tuple[str, ...]] = {
    "FT": ("l0w", "l0b"),
    "L1": ("l1w", "l1b", "l1fw", "l1fb"),
    "L2": ("l2w", "l2b", "l2fw", "l2fb"),
    "L3": ("l3w", "l3b", "l3fw", "l3fb"),
}


# ---------------------------------------------------------------------------- 入出力


def read_state(path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """state.bin から重み (nnue/weights/*) だけを読む。レコード = id + '\\n' + u64 個数 + f32 × 個数。"""
    out: dict[str, torch.Tensor] = {}
    size = path.stat().st_size
    with path.open("rb") as fh:
        off = 0
        while off < size:
            fh.seek(off)
            head = fh.read(512)
            nl = head.index(b"\n")
            rid = head[:nl].decode()
            n = int.from_bytes(head[nl + 1 : nl + 9], "little")
            val_off = off + nl + 9
            if rid.startswith("nnue/weights/"):
                key = rid.removeprefix("nnue/weights/")
                if key not in SHAPES:
                    raise ValueError(f"未対応の重み {rid} (axis factorizer など)")
                arr = np.fromfile(path, dtype="<f4", count=n, offset=val_off)
                if arr.size != int(np.prod(SHAPES[key])):
                    raise ValueError(f"{rid}: 長さ {arr.size} != {SHAPES[key]}")
                out[key] = torch.from_numpy(arr.reshape(SHAPES[key]).copy()).to(device)
            off = val_off + 4 * n
    missing = set(SHAPES) - set(out)
    if missing:
        raise ValueError(f"state.bin に無い重み: {sorted(missing)}")
    return out


@dataclass
class Batch:
    idx_stm: torch.Tensor  # long [B, 80] 基本 40 + 仮想 40 (不足は 0 で重み 0)
    idx_nstm: torch.Tensor
    w_stm: torch.Tensor  # float [B, 80] (有効 1 / 不足 0)
    w_nstm: torch.Tensor
    bucket: torch.Tensor  # long [B]
    target: torch.Tensor  # float [B]
    weight: torch.Tensor  # float [B]

    @property
    def size(self) -> int:
        return int(self.bucket.shape[0])


def _with_virtual(feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = feat != PAD
    base = np.where(valid, feat, 0).astype(np.int64)
    virt = np.where(valid & (base < N_BASE), N_BASE + base % N_VIRT, 0)
    vvalid = valid & (base < N_BASE)
    idx = np.concatenate([base, virt], axis=1)
    w = np.concatenate([valid, vvalid], axis=1).astype(np.float32)
    return idx, w


def load_batch(prefix: Path, start: int, size: int, device: str, scale: float = 290.0) -> Batch:
    """make_batch.py の出力 (<prefix>.feat.npy / .bucket.npy / .meta.npz) から [start, start+size) を取る。"""
    feat = np.load(f"{prefix}.feat.npy", mmap_mode="r")[start : start + size]
    bucket = np.load(f"{prefix}.bucket.npy", mmap_mode="r")[start : start + size]
    score = np.load(f"{prefix}.meta.npz")["score"][start : start + size].astype(np.float32)
    is_, ws = _with_virtual(np.asarray(feat[:, :40]))
    in_, wn = _with_virtual(np.asarray(feat[:, 40:]))
    t = 1.0 / (1.0 + np.exp(-score / scale))
    w = (np.abs(score) < 32000).astype(np.float32)
    dev = torch.device(device)
    return Batch(
        idx_stm=torch.from_numpy(is_).to(dev),
        idx_nstm=torch.from_numpy(in_).to(dev),
        w_stm=torch.from_numpy(ws).to(dev),
        w_nstm=torch.from_numpy(wn).to(dev),
        bucket=torch.from_numpy(np.asarray(bucket).astype(np.int64)).to(dev),
        target=torch.from_numpy(t.astype(np.float32)).to(dev),
        weight=torch.from_numpy(w).to(dev),
    )


# ---------------------------------------------------------------------------- 順伝播


class _CRelu(torch.autograd.Function):
    """clamp(x, 0, 1)。勾配は BulletOu と同じく出力値が開区間 (0, 1) のときだけ通す (2 階微分も可)。"""

    @staticmethod
    def forward(ctx, x):  # noqa: ANN001, ANN205
        y = x.clamp(0.0, 1.0)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):  # noqa: ANN001, ANN205
        (y,) = ctx.saved_tensors
        return g * ((y > 0.0) & (y < 1.0)).to(g.dtype)


def crelu(x: torch.Tensor) -> torch.Tensor:
    return _CRelu.apply(x)


def ft_combined(
    P: dict[str, torch.Tensor], b: Batch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FT (両視点, 仮想特徴量を含む) → crelu → pairwise。戻り値 (a_stm, a_nstm, c)。"""
    a_s = crelu(
        F.embedding_bag(b.idx_stm, P["l0w"], per_sample_weights=b.w_stm, mode="sum") + P["l0b"]
    )
    a_n = crelu(
        F.embedding_bag(b.idx_nstm, P["l0w"], per_sample_weights=b.w_nstm, mode="sum") + P["l0b"]
    )
    c = (
        torch.cat([a_s[:, :HALF] * a_s[:, HALF:], a_n[:, :HALF] * a_n[:, HALF:]], dim=1)
        * PAIR_SCALE
    )
    return a_s, a_n, c


def _gather(all_stacks: torch.Tensor, bucket: torch.Tensor) -> torch.Tensor:
    """[B, STACKS, k] から各サンプルの stack の [B, k] を取る。"""
    return all_stacks[torch.arange(all_stacks.shape[0], device=bucket.device), bucket]


def l1(P: dict[str, torch.Tensor], c: torch.Tensor, bucket: torch.Tensor) -> torch.Tensor:
    w = P["l1w"] + P["l1fw"].t().unsqueeze(0)  # [S, 8, 1024]
    bias = P["l1b"] + P["l1fb"]
    z = (c @ w.reshape(STACKS * L1_OUT, FT).t()).reshape(-1, STACKS, L1_OUT) + bias
    return _gather(z, bucket)


def l2_input(z1: torch.Tensor) -> torch.Tensor:
    h = z1[:, :L1_H]
    return torch.cat([crelu(h * h * PAIR_SCALE), crelu(h)], dim=1)


def l2(P: dict[str, torch.Tensor], u: torch.Tensor, bucket: torch.Tensor) -> torch.Tensor:
    w = P["l2w"] + P["l2fw"].unsqueeze(0)  # [S, 64, 14]
    bias = P["l2b"] + P["l2fb"]
    z = (u @ w.reshape(STACKS * L2, L2_IN).t()).reshape(-1, STACKS, L2) + bias
    return crelu(_gather(z, bucket))


def l3(
    P: dict[str, torch.Tensor], z2: torch.Tensor, z1: torch.Tensor, bucket: torch.Tensor
) -> torch.Tensor:
    w = P["l3w"] + P["l3fw"].unsqueeze(0)  # [S, 64]
    bias = P["l3b"] + P["l3fb"]
    z = z2 @ w.t() + bias  # [B, S]
    return _gather(z.unsqueeze(-1), bucket).squeeze(-1) + z1[:, L1_H]


def per_sample_loss(out: torch.Tensor, b: Batch) -> torch.Tensor:
    return b.weight * (torch.sigmoid(out) - b.target) ** 2


def forward(P: dict[str, torch.Tensor], b: Batch) -> dict[str, torch.Tensor]:
    _, _, c = ft_combined(P, b)
    z1 = l1(P, c, b.bucket)
    z2 = l2(P, l2_input(z1), b.bucket)
    out = l3(P, z2, z1, b.bucket)
    return {"c": c, "z1": z1, "z2": z2, "out": out}


# ---------------------------------------------------------------------------- 勾配


def bp_grads(P: dict[str, torch.Tensor], b: Batch) -> tuple[dict[str, torch.Tensor], float]:
    """BP の勾配 (バッチ平均の loss)。"""
    Pg = {k: v.detach().requires_grad_(True) for k, v in P.items()}
    loss = per_sample_loss(forward(Pg, b)["out"], b).mean()
    grads = torch.autograd.grad(loss, list(Pg.values()))
    return dict(zip(Pg.keys(), grads, strict=True)), float(loss)


def energy(
    P: dict[str, torch.Tensor],
    b: Batch,
    pred1: torch.Tensor,
    h: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    lam: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rho: float,
) -> torch.Tensor:
    """拡張 Lagrangian のサンプルごとの和 (``pred1`` は FT + pairwise の順伝播値。P の勾配は呼び出し側で決める)。"""
    h1, h2, h3 = h
    r1 = h1 - pred1
    r2 = h2 - l1(P, h1, b.bucket)
    r3 = h3 - l2(P, l2_input(h2), b.bucket)
    sup = per_sample_loss(l3(P, h3, h2, b.bucket), b).sum()
    total = sup
    for r, lm in zip((r1, r2, r3), lam, strict=True):
        total = total + (lm * r).sum() + 0.5 * rho * (r * r).sum()
    return total


def residuals(P, b, pred1, h):  # noqa: ANN001, ANN201
    h1, h2, h3 = h
    return (h1 - pred1, h2 - l1(P, h1, b.bucket), h3 - l2(P, l2_input(h2), b.bucket))


def pcalm_grads(
    P: dict[str, torch.Tensor],
    b: Batch,
    *,
    steps: int,
    alpha: float,
    rho: float,
    eta: float,
    timing: str = "pre_dual_energy",
    trace: list | None = None,
) -> dict[str, torch.Tensor]:
    """PC-ALM の重み勾配 (参照実装 ``run_pcalm`` と同じ手順。``alpha = 0`` で PC)。"""
    Pd = {k: v.detach() for k, v in P.items()}
    with torch.no_grad():
        fw = forward(Pd, b)
    pred1 = fw["c"]
    h = [fw["c"].clone(), fw["z1"].clone(), fw["z2"].clone()]
    lam = [torch.zeros_like(x) for x in h]

    def primal(h, lam):  # noqa: ANN001, ANN202
        hv = [x.detach().requires_grad_(True) for x in h]
        e = energy(Pd, b, pred1, tuple(hv), tuple(lam), rho)
        g = torch.autograd.grad(e, hv)
        return [x.detach() - eta * gx for x, gx in zip(hv, g, strict=True)]

    for t in range(steps - 1):
        h = primal(h, lam)
        with torch.no_grad():
            r = residuals(Pd, b, pred1, h)
            lam = [lm + alpha * rr for lm, rr in zip(lam, r, strict=True)]
        if trace is not None:
            trace.append((t, [float(x.norm()) for x in r], [float(x.norm()) for x in lam]))
    lam_before = lam
    h = primal(h, lam)
    if timing == "post_dual_energy":
        with torch.no_grad():
            r = residuals(Pd, b, pred1, h)
            lam_w = [lm + alpha * rr for lm, rr in zip(lam_before, r, strict=True)]
    else:
        lam_w = lam_before

    Pg = {k: v.detach().requires_grad_(True) for k, v in P.items()}
    # pred1 は重みの関数として作り直す (FT の勾配のため)。h と λ は定数
    _, _, pred1_g = ft_combined(Pg, b)
    e = energy(Pg, b, pred1_g, tuple(x.detach() for x in h), tuple(lam_w), rho) / b.size
    grads = torch.autograd.grad(e, list(Pg.values()))
    return dict(zip(Pg.keys(), grads, strict=True))


def group_compare(
    ga: dict[str, torch.Tensor], gb: dict[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    """層のまとまりごとの cosine とノルム比 (a / b)。"""
    out = {}
    for name, keys in GROUPS.items():
        a = torch.cat([ga[k].reshape(-1) for k in keys])
        c = torch.cat([gb[k].reshape(-1) for k in keys])
        na, nc = float(a.norm()), float(c.norm())
        cos = float(a @ c) / (na * nc) if na > 0 and nc > 0 else float("nan")
        out[name] = {"cos": cos, "norm_ratio": na / nc if nc > 0 else float("nan")}
    return out


def hessian_lambda_max(
    P: dict[str, torch.Tensor], b: Batch, rho: float, iters: int = 50, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """推論の作用素 (エネルギーの状態に関する Hessian, λ = 0, 順伝播の点) の最大固有値をサンプルごとに求める (べき乗法)。

    サンプルは互いに独立なので、サンプルごとに正規化したべき乗法でサンプルごとの λ_max が出る。戻り値 (λ_max [B], 最後の残差)。
    """
    Pd = {k: v.detach() for k, v in P.items()}
    with torch.no_grad():
        fw = forward(Pd, b)
    pred1 = fw["c"]
    h0 = [fw["c"].clone(), fw["z1"].clone(), fw["z2"].clone()]
    lam = tuple(torch.zeros_like(x) for x in h0)
    hv = [x.requires_grad_(True) for x in h0]
    e = energy(Pd, b, pred1, tuple(hv), lam, rho)
    g = torch.autograd.grad(e, hv, create_graph=True)
    gen = torch.Generator(device=pred1.device).manual_seed(seed)
    v = [torch.randn(x.shape, generator=gen, device=x.device) for x in h0]

    def norm(vs):  # noqa: ANN001, ANN202
        return torch.sqrt(sum((x * x).sum(dim=1) for x in vs))

    n = norm(v)
    v = [x / n[:, None] for x in v]
    lam_max = torch.zeros(pred1.shape[0], device=pred1.device)
    for _ in range(iters):
        hvp = torch.autograd.grad(g, hv, grad_outputs=v, retain_graph=True)
        lam_max = sum((a * c).sum(dim=1) for a, c in zip(hvp, v, strict=True))
        n = norm(hvp)
        v = [x / n[:, None].clamp_min(1e-30) for x in hvp]
    resid = n - lam_max.abs()
    return lam_max.detach(), resid.detach()
