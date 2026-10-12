"""experiment-011 M0: PC-ALM / PC の重み勾配と BP の勾配の向き (cosine) と大きさ (ノルム比) を層ごとに測る。

- 重み: state.bin (BulletOu の checkpoint)。初期値に近いもの・学習済みのものを並べて比べる
- 推論の刻み η_h: 推論の作用素 (状態に関する Hessian) のサンプルごとの λ_max を測り、既定は 1 / (λ_max の最大値)。
  参照実装 (SakanaAI/pc-alm) の ``eta_best = 1 / lambda_max`` に合わせる (サンプルごとに刻みを変えないので、
  全サンプルで安定になるよう最大値を使う)
- 出力: JSON (--out) と標準出力の表

使い方:
    data/pcalmenv/bin/python experiments/011-pcalm/m0/grad_alignment.py \
        --state /mnt/D/sugiyama/checkpoints/011-init/0001/state.bin --label init \
        --batch-prefix /mnt/D/sugiyama/011/m0/sample --out /mnt/D/sugiyama/011/m0/align-init.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sfnn_ref import (
    bp_grads,
    group_compare,
    hessian_lambda_max,
    load_batch,
    pcalm_grads,
    read_state,
)  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--state", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--batch-prefix", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--steps", default="1,2,3,4,8,16,32,64")
    ap.add_argument("--alphas", default="0,0.5,1,1.5")
    ap.add_argument("--rhos", default="1")
    ap.add_argument(
        "--eta",
        type=float,
        default=None,
        help="省略時は 1 / max(λ_max) (全サンプルで安定な最大の刻み)",
    )
    ap.add_argument("--eta-scales", default="1", help="η_h に掛ける倍率の一覧 (感度を見る)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()
    P = read_state(args.state, args.device)
    b = load_batch(args.batch_prefix, args.start, args.batch, args.device)
    g_bp, loss = bp_grads(P, b)
    res: dict = {
        "label": args.label,
        "state": str(args.state),
        "batch": args.batch,
        "loss": loss,
        "bp_norm": {k: float(v.norm()) for k, v in g_bp.items()},
        "lambda_max": {},
        "runs": [],
    }
    print(f"[{args.label}] loss={loss:.6f} device={args.device}", flush=True)
    for rho in [float(x) for x in args.rhos.split(",")]:
        lm, resid = hessian_lambda_max(P, b, rho)
        q = torch.quantile(lm, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0], device=lm.device)).tolist()
        per_stack = {int(s): float(lm[b.bucket == s].median()) for s in b.bucket.unique()}
        res["lambda_max"][str(rho)] = {
            "quantiles_0_50_90_99_100": q,
            "median_by_stack": per_stack,
            "power_iter_resid_median": float(resid.median()),
        }
        eta0 = args.eta if args.eta is not None else 1.0 / q[4]  # 全サンプルで安定 (最大の λ_max)
        print(
            f"  rho={rho}: lambda_max q0/50/90/99/100 = {[round(x, 3) for x in q]} -> eta0={eta0:.4f}",
            flush=True,
        )
        for es in [float(x) for x in args.eta_scales.split(",")]:
            eta = eta0 * es
            for alpha in [float(x) for x in args.alphas.split(",")]:
                for T in [int(x) for x in args.steps.split(",")]:
                    g = pcalm_grads(P, b, steps=T, alpha=alpha, rho=rho, eta=eta)
                    cmp = group_compare(g, g_bp)
                    finite = all(torch.isfinite(v).all() for v in g.values())
                    res["runs"].append(
                        {
                            "rho": rho,
                            "eta": eta,
                            "alpha": alpha,
                            "T": T,
                            "finite": bool(finite),
                            **cmp,
                        }
                    )
                    row = "  ".join(
                        f"{k} {v['cos']:+.4f}/{v['norm_ratio']:.3f}" for k, v in cmp.items()
                    )
                    print(f"  rho={rho} eta={eta:.4f} alpha={alpha:<4} T={T:<3} {row}", flush=True)
    res["seconds"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out} ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
