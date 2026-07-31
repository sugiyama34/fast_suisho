"""学習量 → Elo カーブ (sb=12 / 108 / 367 の 3 系列) を figures/ に描く。

各点は games/<name>/summary.json の固定ペア測定 (Elo ± 95% CI)。
x 軸は総学習局面数 (log)。dataviz 検証済みパレット (slot 1-3 固定順)。

uv run python experiments/006-standard-epoch/plot_elo_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
GAMES = HERE / "games"

SB_POS = 39_976_960  # 実効 superbatch 局面数

SERIES = [
    ("sb=12 (005 template)", "bu-ep", 12, [1, 2, 4, 8, 16], "#2a78d6"),
    ("sb=108 (yaneurao actual config)", "yane-ep", 108, [1, 2, 4, 8, 16, 20, 32], "#eb6834"),
    ("sb=367 (~ teacher 1 pass/epoch)", "std-ep", 367, [1, 2, 4, 8, 16], "#1baf7a"),
]

TEXT = "#1a1a19"
MUTED = "#6f6e66"
GRID = "#e5e4dd"
TEACHER_POSITIONS = 14_668_949_437  # 奏乗全 30 ファイル


def load_point(prefix: str, n: int) -> dict | None:
    path = GAMES / f"{prefix}{n}-vs-suisho11" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")

    for label, prefix, sb, epochs, color in SERIES:
        xs, ys, lo, hi = [], [], [], []
        for n in epochs:
            s = load_point(prefix, n)
            if s is None:
                continue
            xs.append(n * sb * SB_POS / 1e9)
            ys.append(s["elo"])
            lo.append(s["elo"] - s["elo_ci95"][0])
            hi.append(s["elo_ci95"][1] - s["elo"])
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            capsize=3,
            label=label,
        )
        if xs:
            ax.annotate(
                f" {label.split()[0]}", (xs[-1], ys[-1]), color=color, fontsize=9, va="center"
            )

    ax.set_xscale("log")
    ax.set_ylim(-700, 40)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(" Suisho 11 (=WCSC36)", (ax.get_xlim()[0], 0), color=MUTED, fontsize=8, va="bottom")
    ax.axhline(-100, color=MUTED, linewidth=0.8, linestyle=":")
    ax.annotate(
        " yaneurao corrected claim (-100)",
        (ax.get_xlim()[0], -100),
        color=MUTED,
        fontsize=8,
        va="bottom",
    )
    ax.axvline(TEACHER_POSITIONS / 1e9, color=MUTED, linewidth=0.8, linestyle=":")
    ax.annotate(
        " 1 standard-epoch (teacher 1 pass)",
        (TEACHER_POSITIONS / 1e9, ax.get_ylim()[0]),
        color=MUTED,
        fontsize=8,
        rotation=90,
        va="bottom",
    )

    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("training positions (billions, log scale)", color=TEXT, fontsize=10)
    ax.set_ylabel("Elo vs Suisho 11 (fixed 50-pair matches, 95% CI)", color=TEXT, fontsize=10)
    ax.set_title(
        "BulletOu + Sojo data: training amount vs strength (3 LR-cycle lengths)",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    out = FIGS / "elo_curves.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")

    # 表も出力 (report 用)
    for label, prefix, sb, epochs, _c in SERIES:
        print(f"\n== {label}")
        for n in epochs:
            s = load_point(prefix, n)
            if s:
                print(
                    f"ep{n:>2} {n * sb * SB_POS / 1e9:8.1f}B pos: elo={s['elo']:+8.1f} "
                    f"ci=[{s['elo_ci95'][0]:.1f}, {s['elo_ci95'][1]:.1f}] pairs={s['pairs']}"
                )


if __name__ == "__main__":
    main()
