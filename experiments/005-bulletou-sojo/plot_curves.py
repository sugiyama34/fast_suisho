"""学習曲線の図を figures/ に生成する (report.md 埋め込み用)。

3 アーム (main / var-b / var-c) の floodgate test accuracy / test loss / train loss を
positions 軸で重ね描きする。配色は dataviz skill の検証済みリファレンスパレット
(slot 1-3 固定順)。ラベルは CJK フォント非依存のため英語表記。

uv run python experiments/005-bulletou-sojo/plot_curves.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIGS = HERE / "figures"

ARMS = [
    ("main", "#2a78d6"),
    ("var-b", "#eb6834"),
    ("var-c", "#1baf7a"),
]

TEXT = "#1a1a19"
MUTED = "#6f6e66"
GRID = "#e5e4dd"


def read_summary(arm: str) -> dict[str, list[float]]:
    path = REPO / "data" / "bulletou" / "checkpoints" / f"005-{arm}" / "summary-learn.log"
    cols: dict[str, list[float]] = {
        "positions": [],
        "test_value_accuracy": [],
        "test_value_loss": [],
        "train_value_loss": [],
    }
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                vals = {k: float(row[k]) for k in cols}
            except (TypeError, ValueError):
                continue  # '-' 行 (test 列が無い行) は飛ばす
            for k, v in vals.items():
                cols[k].append(v)
    return cols


def style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=10)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    data = {arm: read_summary(arm) for arm, _ in ARMS}

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True, facecolor="white")
    panels = [
        ("test_value_accuracy", "floodgate test accuracy"),
        ("test_value_loss", "floodgate test loss"),
        ("train_value_loss", "train loss (32-batch avg)"),
    ]
    for ax, (key, ylabel) in zip(axes, panels):
        for arm, color in ARMS:
            xs = [p / 1e9 for p in data[arm]["positions"]]
            ax.plot(xs, data[arm][key], color=color, linewidth=2, label=arm)
        style_axis(ax, ylabel)
        # 直接ラベル (最終値の右側)
        for arm, color in ARMS:
            xs = data[arm]["positions"]
            if xs:
                ax.annotate(
                    f" {arm}",
                    (xs[-1] / 1e9, data[arm][key][-1]),
                    color=color,
                    fontsize=9,
                    va="center",
                )
    axes[0].legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower right")
    axes[-1].set_xlabel("teacher positions consumed (billions)", color=TEXT, fontsize=10)
    axes[0].set_title(
        "BulletOu SFNN_halfka2_1024_7_64_k3k3 on Sojo data — 3 arms (teacher file order rotated)",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    fig.tight_layout()
    out = FIGS / "training_curves.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")

    # 拡大図: 最後の 4 epoch の test accuracy (アーム間差が見える倍率)
    fig2, ax = plt.subplots(figsize=(8.5, 3.5), facecolor="white")
    for arm, color in ARMS:
        xs = [p / 1e9 for p in data[arm]["positions"]]
        ys = data[arm]["test_value_accuracy"]
        tail = [(x, y) for x, y in zip(xs, ys) if x >= xs[-1] * 0.75]
        ax.plot(*zip(*tail), color=color, linewidth=2, label=arm)
        ax.annotate(f" {arm}", tail[-1], color=color, fontsize=9, va="center")
    style_axis(ax, "floodgate test accuracy")
    ax.set_xlabel("teacher positions consumed (billions)", color=TEXT, fontsize=10)
    ax.set_title("Final quarter of training (zoom)", color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower right")
    fig2.tight_layout()
    out2 = FIGS / "test_accuracy_zoom.png"
    fig2.savefig(out2, dpi=150, facecolor="white")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
