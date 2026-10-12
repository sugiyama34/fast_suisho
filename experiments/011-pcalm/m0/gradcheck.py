"""experiment-011 M1 の検証: 改造した BulletOu (CUDA) の勾配を Python の参照実装 (sfnn_ref.py) と突き合わせる。

1. 重み (state.bin か合成) と M0 の標本の先頭 B 局面を生の f32 / i32 ファイルに書く
2. ``pcalm_gradcheck`` (BulletOu-pcalm の cuda_cpp crate の bin) で BP と PC-ALM の勾配を出す
3. 同じ重み・同じ局面で sfnn_ref.py の BP / PC-ALM の勾配を計算し、テンソルごとの相対誤差と cosine を表示する

使い方 (GPU は MIG の UUID):
    CUDA_VISIBLE_DEVICES=<MIG UUID> data/pcalmenv/bin/python experiments/011-pcalm/m0/gradcheck.py \
        --weights synthetic --batch 4096 --work /mnt/D/sugiyama/011/gradcheck/synth
    ... --weights /mnt/D/sugiyama/checkpoints/011-s-bp-lr1/0001/state.bin --work .../e1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from sfnn_ref import PAD, SHAPES, bp_grads, load_batch, pcalm_grads, read_state  # noqa: E402

TOOL = REPO / "data" / "bulletou" / "BulletOu-pcalm" / "target" / "release" / "pcalm_gradcheck"
NAMES = list(SHAPES)


def synthetic_weights(seed: int) -> dict[str, torch.Tensor]:
    """全分岐 (crelu の 0 < x < 1、二乗の分岐、shortcut) を通る大きさの合成重み。"""
    g = torch.Generator().manual_seed(seed)

    def n(shape, std, mean=0.0):  # noqa: ANN001, ANN202
        return torch.randn(shape, generator=g) * std + mean

    P = {
        "l0w": n(SHAPES["l0w"], 0.06),
        "l0b": n(SHAPES["l0b"], 0.1, 0.35),
        "l1w": n(SHAPES["l1w"], 0.06),
        "l1b": n(SHAPES["l1b"], 0.1),
        "l1fw": n(SHAPES["l1fw"], 0.03),
        "l1fb": n(SHAPES["l1fb"], 0.05),
        "l2w": n(SHAPES["l2w"], 0.4),
        "l2b": n(SHAPES["l2b"], 0.1, 0.2),
        "l2fw": n(SHAPES["l2fw"], 0.2),
        "l2fb": n(SHAPES["l2fb"], 0.05),
        "l3w": n(SHAPES["l3w"], 0.5),
        "l3b": n(SHAPES["l3b"], 0.1),
        "l3fw": n(SHAPES["l3fw"], 0.2),
        "l3fb": n(SHAPES["l3fb"], 0.05),
    }
    return P


def write_inputs(P: dict[str, torch.Tensor], prefix: Path, batch: int, work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for k in NAMES:
        P[k].detach().cpu().numpy().astype("<f4").tofile(work / f"{k}.f32")
    feat = np.load(f"{prefix}.feat.npy", mmap_mode="r")[:batch]
    bucket = np.load(f"{prefix}.bucket.npy", mmap_mode="r")[:batch]
    score = np.load(f"{prefix}.meta.npz")["score"][:batch].astype(np.float32)
    f = np.asarray(feat).astype(np.int64)
    f[f == PAD] = -1
    f[:, :40].astype("<i4").tofile(work / "stm.i32")
    f[:, 40:].astype("<i4").tofile(work / "nstm.i32")
    np.asarray(bucket).astype("<i4").tofile(work / "buckets.i32")
    (1.0 / (1.0 + np.exp(-score / 290.0))).astype("<f4").tofile(work / "targets.f32")
    (np.abs(score) < 32000).astype("<f4").tofile(work / "weights.f32")


def read_grads(out: Path) -> dict[str, torch.Tensor]:
    return {
        k: torch.from_numpy(np.fromfile(out / f"{k}.f32", dtype="<f4").reshape(SHAPES[k]))
        for k in NAMES
    }


def compare(
    cuda: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    res = {}
    for k in NAMES:
        a = cuda[k].double().reshape(-1)
        b = ref[k].detach().cpu().double().reshape(-1)
        nb = float(b.norm())
        err = float((a - b).norm()) / nb if nb > 0 else float((a - b).norm())
        cos = (
            float(a @ b) / (float(a.norm()) * nb)
            if nb > 0 and float(a.norm()) > 0
            else float("nan")
        )
        res[k] = {"rel_err": err, "cos": cos, "ref_norm": nb}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--weights", required=True, help="state.bin のパス、または synthetic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--batch-prefix", type=Path, default=Path("/mnt/D/sugiyama/011/m0/sample"))
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument(
        "--cases",
        default="1:1:1:0.3,2:1:1:0.3,4:1:1:0.3,8:1:1:0.3,8:0:1:0.3,4:1.5:2:0.2,300:1:1:0.3",
        help="PC-ALM の (steps:alpha:rho:eta) をカンマ区切り",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.weights == "synthetic":
        P = synthetic_weights(args.seed)
    else:
        P = read_state(Path(args.weights))
    write_inputs(P, args.batch_prefix, args.batch, args.work)
    Pd = {k: v.to(args.device) for k, v in P.items()}
    b = load_batch(args.batch_prefix, 0, args.batch, args.device)

    report: dict = {"weights": args.weights, "batch": args.batch, "cases": {}}
    out = args.work / "out-bp"
    subprocess.run([str(TOOL), str(args.work), str(out), "bp"], check=True)
    g_ref, loss = bp_grads(Pd, b)
    report["cases"]["bp"] = compare(read_grads(out), g_ref)
    report["loss_ref"] = loss
    report["loss_cuda"] = float((out / "loss.txt").read_text())
    print(f"loss ref={loss:.8f} cuda={report['loss_cuda']:.8f}")
    g_bp = g_ref
    for case in args.cases.split(","):
        T, alpha, rho, eta = case.split(":")
        out = args.work / f"out-pcalm-{case.replace(':', '_')}"
        subprocess.run(
            [str(TOOL), str(args.work), str(out), "pcalm", T, alpha, rho, eta], check=True
        )
        g_ref = pcalm_grads(Pd, b, steps=int(T), alpha=float(alpha), rho=float(rho), eta=float(eta))
        cuda = read_grads(out)
        report["cases"][f"pcalm {case}"] = compare(cuda, g_ref)
        report["cases"][f"pcalm {case} vs bp (cuda)"] = compare(cuda, g_bp)
    for name, res in report["cases"].items():
        worst = max(res.items(), key=lambda kv: kv[1]["rel_err"] if kv[1]["ref_norm"] > 0 else 0)
        row = " ".join(
            f"{k}:{v['rel_err']:.1e}"
            for k, v in res.items()
            if k in ("l0w", "l1w", "l2w", "l3w", "l1fw")
        )
        print(f"{name:<34} worst {worst[0]} rel_err={worst[1]['rel_err']:.2e}  | {row}")
    (args.work / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
