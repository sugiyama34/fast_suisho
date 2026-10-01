"""探索中の発火回数 (match_featcount.py の search.npy / played.npy) と教師の発火回数を比べる。

使い方:
    data/matchenv/bin/python experiments/009-data-scaling/search_featcount/compare.py \
        experiments/009-data-scaling/games/featcount/<name> \
        [--teacher '/mnt/nvme1/sugiyama/feature_counts/dlsuisho_unique_0[0-3][0-9].both.npy']
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "feature_count"))
from analyze import decode, structural_impossible  # noqa: E402


def norm(h: np.ndarray) -> np.ndarray:
    return h.astype(np.float64) / float(h.sum())


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a: np.ndarray) -> float:
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / m[nz])).sum())

    return 0.5 * kl(p) + 0.5 * kl(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument(
        "--teacher",
        default="/mnt/nvme1/sugiyama/feature_counts/dlsuisho_unique_[0-9][0-9][0-9].both.npy",
    )
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(args.teacher) if re.search(r"_\d{3}\.both\.npy$", f))
    teacher = np.zeros(131949, dtype=np.uint64)
    for f in files:
        teacher += np.load(f)
    run = Path(args.run_dir)
    search, played = np.load(run / "search.npy"), np.load(run / "played.npy")
    imp = structural_impossible()
    print(f"teacher files={len(files)} sum={int(teacher.sum()):,}")
    print(f"search sum={int(search.sum()):,} (evaluations={int(search.sum()) // 80:,})")
    print(f"played sum={int(played.sum()):,} (positions={int(played.sum()) // 80:,})")
    print(
        f"structurally-impossible features with count>0: search={int((search[imp] > 0).sum())} "
        f"played={int((played[imp] > 0).sum())}"
    )

    ps, pp, pt = norm(search), norm(played), norm(teacher)
    print(f"TV(search, teacher)={tv(ps, pt):.4f}  JS={js(ps, pt):.4f} bit")
    print(f"TV(played, teacher)={tv(pp, pt):.4f}  JS={js(pp, pt):.4f} bit")
    print(f"TV(search, played) ={tv(ps, pp):.4f}  JS={js(ps, pp):.4f} bit")

    seen = search > 0
    print(f"features seen in search: {int(seen.sum()):,}; in played: {int((played > 0).sum()):,}")
    for thr in (1, 10, 100, 1000, 10000):
        m = seen & (teacher < thr)
        print(
            f"  seen in search & teacher < {thr:>5}: {int(m.sum()):>6,} features, "
            f"search mass {ps[m].sum():.2e}"
        )

    print(f"top {args.top} in search (share search / teacher / played):")
    for i in np.argsort(search, kind="stable")[::-1][: args.top]:
        print(f"  {i:6d} {decode(int(i))}: {ps[i]:.4%} / {pt[i]:.4%} / {pp[i]:.4%}")
    ratio = np.where(teacher > 0, ps / np.maximum(pt, 1e-300), np.inf)
    print(
        f"top {args.top} over-represented in search vs teacher (search share / teacher share, teacher count):"
    )
    cand = np.where(search >= 1000)[0]
    for i in cand[np.argsort(-ratio[cand])][: args.top]:
        print(
            f"  {i:6d} {decode(int(i))}: x{ratio[i]:.1f}  ({ps[i]:.2e} / {pt[i]:.2e}, n_teacher={int(teacher[i]):,})"
        )


if __name__ == "__main__":
    main()
