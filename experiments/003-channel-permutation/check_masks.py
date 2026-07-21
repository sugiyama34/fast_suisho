"""マスクファイルのチャンク非ゼロ率を報告する (experiment-003 機構確認用)。

permuted ネットで再収集したマスクは既に並べ替え済みの座標なので identity で評価し、
optimize_permutation.py が予測した値と突き合わせる。

usage: check_masks.py --masks FILE
"""

from __future__ import annotations

import argparse

import numpy as np

from optimize_permutation import N_PAIRS, chunk_nonzero_rate, load_half_masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    args = ap.parse_args()

    nz = load_half_masks(args.masks)
    identity = np.arange(N_PAIRS, dtype=np.int32)
    print(f"samples (half-vectors): {nz.shape[0]}")
    print(f"channel nonzero rate mean: {nz.mean():.4f}")
    print(f"chunk nonzero rate: {chunk_nonzero_rate(nz, identity):.4f}")


if __name__ == "__main__":
    main()
