"""experiment-011: 書き出した nn.bin (量子化) と state.bin (float) の関数がどれだけ違うかを測る。

loss_eval (experiment-009) で同じ 10 万局面の出力を両方計算し、量子化の生出力を float の logit に線形で合わせたときの
倍率 (出力側を 1/k に縮めた nn.bin なら BP の k 倍になる) と、|差| の平均・99 パーセンタイル (cp, scale 290)、符号の一致率を表示する。
PC-ALM で縮めない層 (L1 の 0〜6 行目, L2) の int8 の切り詰めが関数を変えていないかの確認用 (notes/m0.md)。

使い方: .venv/bin/python experiments/011-pcalm/quant_check.py --net <dir with nn.bin> --ckpt <checkpoint dir>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOSS_EVAL = (
    REPO / "experiments" / "009-data-scaling" / "loss_eval" / "target" / "release" / "loss_eval"
)
SAMPLE = Path("/mnt/D/sugiyama/011/m0/sample.psv")
RAW_TO_LOGIT = 1.0 / (127 * 64)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--net", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=100000)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [str(LOSS_EVAL), "eval", "--net", str(args.net / "nn.bin"), "--state", str(args.ckpt / "state.bin"),
             "--psv", str(SAMPLE), "--limit", str(args.limit), "--threads", "8", "--out", f"{td}/x"],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        q = np.load(f"{td}/x.q.npy").astype(np.float64)
        f = np.load(f"{td}/x.f.npy").astype(np.float64)
    acc = dict(
        re.findall(r"^(quantized|float) \(\S+\): accuracy=([0-9.]+)", r.stdout + r.stderr, re.M)
    )
    c = float(np.dot(q, f) / np.dot(q, q))
    d = np.abs(q * c - f) * 290
    out = {
        "net": str(args.net),
        "k_fit": round(c / RAW_TO_LOGIT, 3),
        "mean_abs_diff_cp": round(float(d.mean()), 2),
        "p99_abs_diff_cp": round(float(np.percentile(d, 99)), 2),
        "corr": round(float(np.corrcoef(q, f)[0, 1]), 6),
        "acc_quantized": float(acc.get("quantized", "nan")),
        "acc_float": float(acc.get("float", "nan")),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
