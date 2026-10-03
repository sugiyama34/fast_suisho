"""教師データでの出現順位 (多い順) ごとに、特徴量の出現が全体の何 % を占めるかの図 (2 枚)。

横軸はどちらも「教師データでの出現回数の順位」(1 = 最多、現れ得る 123,053 個)。
上段 = 教師データ (30 ファイル)、下段 = 水匠 11 の対局 (探索中の評価, 2,000 ペア 300k ノード)。
同数のタイはアブレーション (`ablate/select_features.py`) と同じシード 0 の乱数順で切る。

- `feature_rank_hist.png`: 順位を 1% ずつ (約 1,231 個ずつ) の区間に分けたヒストグラム。縦軸 = 区間の出現の割合
- `feature_rank_line.png`: 1 特徴量 = 1 点の折れ線。縦軸 = その特徴量 1 個の出現の割合
- `feature_rank_line_match.png`: 上の図の下段 (対局) だけ

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
from plot_results import FIG, GRID, INK, INK2, SETTINGS, SURFACE, plt, style  # noqa: E402

COUNT_DIR = Path("/mnt/nvme1/sugiyama/feature_counts")
N_BINS = 100
ABL = (5, 10, 20, 30, 50, 75)  # 最終アブレーションの下位 X%
XLAB = "Rank of feature by occurrence in training data (1 = most frequent; {n:,} possible features)"
TIES = "Ties broken by the same seeded order as the ablation."


def cutoffs(ax, n: int) -> None:
    """最終アブレーションの「下位 X%」の境界を点線で描く。"""
    for x in ABL:
        r = n * (1 - x / 100)
        ax.axvline(r, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(r - n * 0.004, 0.97, f"rarest {x}%", transform=ax.get_xaxis_transform(),
                rotation=90, color=INK2, fontsize=7, ha="right", va="top")  # fmt: skip


def finish(fig, axes, n: int, ylim: tuple[float, float], note: str, out: Path) -> None:
    axes[-1].set_xlabel(XLAB.format(n=n), color=INK2, fontsize=9)
    axes[-1].set_xlim(0, n)
    axes[-1].set_ylim(*ylim)
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    fig.text(0.01, 0.005, note, color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(f"wrote {out}")


def plot_bins(panels: list, n: int) -> None:
    bins = np.array_split(np.arange(n), N_BINS)
    left = np.array([b[0] for b in bins])
    width = np.array([b.size for b in bins])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, sharey=True, facecolor=SURFACE)
    ylo, yhi = np.inf, 0.0
    for ax, (cs, title, color) in zip(axes, panels):
        share = np.array([cs[b].sum() for b in bins]) / cs.sum() * 100
        style(ax, title, "", "% of all occurrences (per 1% of features)")
        ax.bar(
            left, share, width=width, align="edge", color=color, edgecolor=SURFACE, linewidth=0.6
        )
        ax.set_yscale("log")
        lo = share[share > 0].min()
        ylo, yhi = min(ylo, lo / 3), max(yhi, share.max() * 3)
        cutoffs(ax, n)
        tail = cs[n - int(round(n * 0.75)) :].sum() / cs.sum() * 100
        ax.set_title(f"{title}  —  most frequent 1% of features: {share[0]:.1f}% of occurrences, "
                     f"rarest 75%: {tail:.1f}%", color=INK2, fontsize=10, loc="left")  # fmt: skip
        print(f"{title}: top1%={share[0]:.2f}% top10%={share[:10].sum():.2f}% "
              f"rarest75%={tail:.2f}% min-bin={lo:.2e}% last-bin={share[-1]:.2e}%")  # fmt: skip
    note = "Bars: 1% of features each (rank bins). Dashed lines: start of the rarest X% (the ablated sets "
    finish(
        fig,
        axes,
        n,
        (ylo, yhi),
        note + "of the final grid). " + TIES,
        FIG / "feature_rank_hist.png",
    )


def plot_per_feature(panels: list, n: int, idx: tuple[int, ...], out: Path, height: float) -> None:
    """idx = 描く段 (0 = 教師, 1 = 対局)。対局の段には 1% 区間ごとの中央値と教師の曲線を重ねる。"""
    rank = np.arange(1, n + 1)
    fig, axes = plt.subplots(len(idx), 1, figsize=(10, height), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)  # fmt: skip
    axes = axes[:, 0]
    ylo, yhi = np.inf, 0.0
    bins = np.array_split(np.arange(n), N_BINS)
    mid = np.array([b.mean() + 1 for b in bins])
    ref = panels[0][0] / panels[0][0].sum() * 100  # 教師の曲線 (下段に重ねる参照線)
    ref_label = "training data (top panel)" if len(idx) > 1 else "training data (same ranking)"
    for ax, i in zip(axes, idx):
        cs, title, color = panels[i]
        share = cs / cs.sum() * 100
        pos = share > 0
        style(ax, title, "", "% of all occurrences (one feature)")
        if i == 0:
            ax.plot(rank[pos], share[pos], color=color, linewidth=1.2)
        else:
            # 教師の順位で並べると対局の値はばらつくので、1% 区間ごとの中央値と教師の曲線を重ねる
            ax.plot(rank[pos], share[pos], color=color, linewidth=0.4, alpha=0.6,
                    label="each feature")  # fmt: skip
            med = np.array([np.median(share[b]) for b in bins])
            ax.plot(mid[med > 0], med[med > 0], color=INK, linewidth=1.6,
                    label="median of each 1% of features")  # fmt: skip
            ax.plot(rank[ref > 0], ref[ref > 0], color=panels[0][2], linewidth=1.2,
                    label=ref_label)  # fmt: skip
            ax.legend(loc="lower left", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID)
        ax.set_yscale("log")
        ylo, yhi = min(ylo, share[pos].min() / 3), max(yhi, share.max() * 3)
        cutoffs(ax, n)
        zero = f", {int((~pos).sum()):,} features never occur (not drawn)" if (~pos).any() else ""
        ax.set_title(f"{title}  —  max {share.max():.2f}% (rank {int(share.argmax()) + 1}){zero}",
                     color=INK2, fontsize=10, loc="left")  # fmt: skip
        print(
            f"{title}: max={share.max():.3f}% min>0={share[pos].min():.2e}% zeros={int((~pos).sum())}"
        )
    note = "One point per feature, joined by a line. Dashed lines: start of the rarest X% (the ablated sets "
    finish(
        fig,
        axes,
        n,
        (ylo, yhi),
        note + "of the final grid). " + TIES,
        out,
    )


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
    blue = SETTINGS[0][3]  # 教師順位の色 (アブレーションの図と同じ)
    green = SETTINGS[2][3]  # 対局の色
    panels = [
        (teach[order].astype(np.float64), "Training data (teacher positions, 30 files)", blue),
        (search[order].astype(np.float64), "Suisho 11 matches (evaluations during search)", green),
    ]
    plot_bins(panels, n)
    plot_per_feature(panels, n, (0, 1), FIG / "feature_rank_line.png", 7.5)
    plot_per_feature(panels, n, (1,), FIG / "feature_rank_line_match.png", 4.6)


if __name__ == "__main__":
    main()
