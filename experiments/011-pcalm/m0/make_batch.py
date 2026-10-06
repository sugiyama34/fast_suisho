"""M0 用の固定標本: 教師 .psv から一様に N 局面を選び、.psv と特徴量 (loss_eval features) を書く。

出力 (--out-dir):
    sample.psv          PackedSfenValue 40 B × N (元ファイルでの位置の昇順)
    sample.index.npy    u64 [N] 元ファイルでの record 番号
    sample.feat.npy     u32 [N, 80] HalfKA2 特徴 (手番側 40 + 非手番側 40, 不足は u32::MAX)。loss_eval features の出力
    sample.bucket.npy   u8 [N] layer stack 番号 (k3k3)
    sample.meta.npz     score (i16), result (i8), ply (u16)

使い方:
    python3 experiments/011-pcalm/m0/make_batch.py --psv data/teacher/sojo/train/dlsuisho_unique_001.psv \
        --n 262144 --seed 20261006 --out-dir /mnt/D/sugiyama/011/m0
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
LOSS_EVAL = (
    REPO / "experiments" / "009-data-scaling" / "loss_eval" / "target" / "release" / "loss_eval"
)
REC = 40
DT = np.dtype(
    [
        ("sfen", "u1", 32),
        ("score", "<i2"),
        ("move", "<u2"),
        ("ply", "<u2"),
        ("result", "i1"),
        ("pad", "u1"),
    ]
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--psv", required=True, type=Path)
    ap.add_argument("--n", type=int, default=262_144)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    total = args.psv.stat().st_size // REC
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(total, size=args.n, replace=False)).astype(np.uint64)
    mm = np.memmap(args.psv, dtype=DT, mode="r")
    recs = np.asarray(mm[idx.astype(np.int64)])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "sample"
    recs.tofile(f"{out}.psv")
    np.save(f"{out}.index.npy", idx)
    np.savez(f"{out}.meta.npz", score=recs["score"], result=recs["result"], ply=recs["ply"])
    subprocess.run(
        [
            str(LOSS_EVAL),
            "features",
            "--psv",
            f"{out}.psv",
            "--out",
            f"{out}.feat.npy",
            "--bucket-out",
            f"{out}.bucket.npy",
        ],
        check=True,
    )
    print(f"wrote {args.n} records from {args.psv.name} (of {total}) -> {out}.*")


if __name__ == "__main__":
    main()
