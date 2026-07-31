"""学習指標 (train/test loss, test accuracy) は Elo の代理指標としてどの程度使えるか。

Elo 実測済みの 17 checkpoint について、対応する epoch 末の学習指標を
summary-learn.log から取り出し、散布図行列と相関係数 (Pearson r / Spearman ρ) を出す。
注意: BulletOu は train accuracy を記録しないため対象は 3 指標 + Elo。

uv run python experiments/006-standard-epoch/analyze_metric_proxy.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIGS = HERE / "figures"
GAMES = HERE / "games"
CKPT = REPO / "data" / "bulletou" / "checkpoints"

SERIES = [
    ("sb=12", "bu-ep", CKPT / "005-main", [1, 2, 4, 8, 16], "#2a78d6"),
    ("sb=108", "yane-ep", CKPT / "006-yane20", [1, 2, 4, 8, 16, 20, 32], "#eb6834"),
    ("sb=367", "std-ep", CKPT / "006-std", [1, 2, 4, 8, 16], "#1baf7a"),
]

METRICS = ["train_value_loss", "test_value_loss", "test_value_accuracy", "elo"]
LABELS = {
    "train_value_loss": "train loss",
    "test_value_loss": "test loss (floodgate)",
    "test_value_accuracy": "test accuracy (floodgate)",
    "elo": "Elo vs Suisho 11",
}

TEXT = "#1a1a19"
MUTED = "#6f6e66"
GRID = "#e5e4dd"


def epoch_end_metrics(log_path: Path, epoch: int) -> dict[str, float] | None:
    """summary-learn.log から epoch 末 (その epoch の最終検証行) の指標を取る。"""
    last = None
    with log_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                if int(row["epoch"]) != epoch:
                    continue
            except (TypeError, ValueError):
                continue
            if row.get("test_value_accuracy") in (None, "", "-"):
                continue
            last = row
    if last is None:
        return None
    return {
        "train_value_loss": float(last["train_value_loss"]),
        "test_value_loss": float(last["test_value_loss"]),
        "test_value_accuracy": float(last["test_value_accuracy"]),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    points: list[dict] = []
    for label, prefix, ckpt_dir, epochs, color in SERIES:
        for n in epochs:
            sfile = GAMES / f"{prefix}{n}-vs-suisho11" / "summary.json"
            if not sfile.exists():
                continue
            elo = json.loads(sfile.read_text())["elo"]
            m = epoch_end_metrics(ckpt_dir / "summary-learn.log", n)
            if m is None:
                print(f"warn: no metrics for {label} ep{n}")
                continue
            points.append({**m, "elo": elo, "series": label, "color": color, "ep": n})

    print(f"{len(points)} points assembled")
    data = {k: np.array([p[k] for p in points]) for k in METRICS}

    print("\n相関 (指標 vs Elo, 全 17 点):")
    for k in METRICS[:-1]:
        r = float(np.corrcoef(data[k], data["elo"])[0, 1])
        rho = spearman(data[k], data["elo"])
        print(f"  {LABELS[k]:28s} Pearson r={r:+.3f}  Spearman ρ={rho:+.3f}")

    # 高精度域 (accuracy >= 0.70 ≒ 強い checkpoint 群) に絞ると proxy はどうなるか
    mask = data["test_value_accuracy"] >= 0.70
    print(f"\n相関 (accuracy >= 0.70 の {int(mask.sum())} 点に限定):")
    print(f"  Elo 範囲: {data['elo'][mask].min():.0f} .. {data['elo'][mask].max():.0f}")
    for k in METRICS[:-1]:
        r = float(np.corrcoef(data[k][mask], data["elo"][mask])[0, 1])
        rho = spearman(data[k][mask], data["elo"][mask])
        print(f"  {LABELS[k]:28s} Pearson r={r:+.3f}  Spearman ρ={rho:+.3f}")

    n = len(METRICS)
    fig, axes = plt.subplots(n, n, figsize=(11, 11), facecolor="white")
    for i, ky in enumerate(METRICS):
        for j, kx in enumerate(METRICS):
            ax = axes[i][j]
            if i == j:
                ax.annotate(
                    LABELS[ky],
                    (0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=TEXT,
                    wrap=True,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color(GRID)
                continue
            for p in points:
                ax.plot(p[kx], p[ky], "o", color=p["color"], markersize=5, alpha=0.85)
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(MUTED)
            if i == n - 1:
                ax.set_xlabel(LABELS[kx], color=TEXT, fontsize=8)
            if j == 0:
                ax.set_ylabel(LABELS[ky], color=TEXT, fontsize=8)

    handles = [
        plt.Line2D([], [], color=c, marker="o", linestyle="", label=lab)
        for lab, _p, _d, _e, c in SERIES
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=10,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        labelcolor=TEXT,
    )
    fig.suptitle(
        "Training metrics vs measured Elo — 17 checkpoints (scatter matrix)",
        color=TEXT,
        fontsize=13,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGS / "metric_proxy_scatter.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
