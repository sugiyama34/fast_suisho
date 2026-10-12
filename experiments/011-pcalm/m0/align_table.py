"""grad_alignment.py の JSON を markdown の表にする (notes/m0.md 用)。

python3 experiments/011-pcalm/m0/align_table.py /mnt/D/sugiyama/011/m0/align-init.json [--rho 1] [--eta-index 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=Path)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument(
        "--eta-index", type=int, default=0, help="η の候補のうち何番目 (0 = 1/max λ_max)"
    )
    args = ap.parse_args()
    r = json.loads(args.json.read_text())
    runs = [x for x in r["runs"] if x["rho"] == args.rho]
    etas = sorted({x["eta"] for x in runs})
    eta = etas[args.eta_index] if args.eta_index < len(etas) else etas[0]
    lm = r["lambda_max"][str(args.rho)]["quantiles_0_50_90_99_100"]
    print(
        f"{r['label']}: loss {r['loss']:.4f}, ρ = {args.rho:g}, λ_max (最小/中央/90%/99%/最大) = "
        + " / ".join(f"{x:.2f}" for x in lm)
        + f", η_h = {eta:.4f}\n"
    )
    print("| α | T | FT cos (比) | L1 cos (比) | L2 cos (比) | L3 cos (比) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for x in runs:
        if x["eta"] != eta:
            continue
        cells = []
        for g in ("FT", "L1", "L2", "L3"):
            c, n = x[g]["cos"], x[g]["norm_ratio"]
            cells.append("–" if c != c else f"{c:+.4f} ({n:.2f})")
        print(f"| {x['alpha']:g} | {x['T']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
