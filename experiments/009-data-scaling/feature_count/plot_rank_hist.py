"""教師データでの出現順位 (多い順) ごとに、特徴量の出現が全体の何 % を占めるかのヒストグラム。

横軸はどちらも「教師データでの出現回数の順位」(1 = 最多、現れ得る 123,053 個)。順位を 1% ずつ
(約 1,231 個ずつ) の区間に分け、縦軸はその区間の特徴量の出現が全出現の何 % か。
上段 = 教師データ (30 ファイル)、下段 = 水匠 11 の対局 (探索中の評価, 2,000 ペア 300k ノード)。
同数のタイはアブレーション (`ablate/select_features.py`) と同じシード 0 の乱数順で切る。

    .venv/bin/python experiments/009-data-scaling/feature_count/plot_rank_hist.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from analyze import structural_impossible  # noqa: E402
from plot_results import FIG, GRID, INK2, SETTINGS, SURFACE, plt, style  # noqa: E402

COUNT_DIR = Path("/mnt/nvme1/sugiyama/feature_counts")
N_BINS = 100
ABL = (5, 10, 20, 30, 50, 75)  # 最終アブレーションの下位 X%
OUT = FIG / "feature_rank_hist.png"


def main() -> None:
    teach = sum(
        np.load(COUNT_DIR / f"dlsuisho_unique_{n:03d}.both.npy").astype(np.uint64)
        for n in range(1, 31)
    )
    search = np.load(COUNT_DIR / "search_s11_2000_300k.npy").astype(np.uint64)
    possible = np.flatnonzero(~structural_impossible())
    n = possible.size
    tb = np.random.default_rng(0).permutation(n)
    order = possible[np.lexsort((tb, teach[possible]))[::-1]]  # 多い順 (順位 1 = 最多)
    bins = np.array_split(np.arange(n), N_BINS)
    left = np.array([b[0] for b in bins])
    width = np.array([b.size for b in bins])

    blue = SETTINGS[0][3]  # 教師順位の色 (アブレーションの図と同じ)
    green = SETTINGS[2][3]  # 対局の色
    panels = [
        (teach, "Training data (teacher positions, 30 files)", blue),
        (search, "Suisho 11 matches (evaluations during search)", green),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, sharey=True, facecolor=SURFACE)
    ylo, yhi = np.inf, 0.0
    for ax, (c, title, color) in zip(axes, panels):
        cs = c[order].astype(np.float64)
        share = np.array([cs[b].sum() for b in bins]) / cs.sum() * 100
        style(ax, title, "", "% of all occurrences (per 1% of features)")
        ax.bar(
            left, share, width=width, align="edge", color=color, edgecolor=SURFACE, linewidth=0.6
        )
        ax.set_yscale("log")
        ax.grid(axis="y", which="major", color=GRID, linewidth=0.8)
        lo = share[share > 0].min()
        ylo, yhi = min(ylo, lo / 3), max(yhi, share.max() * 3)
        for x in ABL:
            r = n * (1 - x / 100)
            ax.axvline(r, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
            ax.text(r - n * 0.004, 0.97, f"rarest {x}%", transform=ax.get_xaxis_transform(),
                    rotation=90, color=INK2, fontsize=7, ha="right", va="top")  # fmt: skip
        tail = cs[n - int(round(n * 0.75)) :].sum() / cs.sum() * 100
        ax.set_title(f"{title}  —  most frequent 1% of features: {share[0]:.1f}% of occurrences, "
                     f"rarest 75%: {tail:.1f}%", color=INK2, fontsize=10, loc="left")  # fmt: skip
        print(f"{title}: top1%={share[0]:.2f}% top10%={share[:10].sum():.2f}% "
              f"rarest75%={tail:.2f}% min-bin={lo:.2e}% last-bin={share[-1]:.2e}%")  # fmt: skip
    axes[1].set_xlabel(
        f"Rank of feature by occurrence in training data (1 = most frequent; {n:,} possible features)",
        color=INK2, fontsize=9,
    )  # fmt: skip
    axes[1].set_xlim(0, n)
    axes[1].set_ylim(ylo, yhi)
    fig.text(0.01, 0.005, "Bars: 1% of features each (rank bins). Dashed lines: start of the rarest X% "
             "(the ablated sets of the final grid). Ties broken by the same seeded order as the ablation.",
             color=INK2, fontsize=7.5)  # fmt: skip
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT, dpi=110, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
