"""007 の Elo 測定結果 (5 arm × ep16/ep20 相当 + 006 anchor) を figures/ に描く。

各点は games/<name>/summary.json (Elo ± 95% CI, vs Suisho 11)。wrm の 2 点は
200 ペア精密測定、他は 50 ペア。dataviz 検証済みパレット (slot 1-5 固定順)。

uv run python experiments/007-wrm-factorizer/plot_results.py
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

# (表示名, games dir 名の {ep} テンプレート, palette slot 固定順)
ARMS = [
    ("ctrl (006 recipe @2a8e5ed)", "007-ctrl-ep{ep}-vs-suisho11", "#2a78d6"),
    ("wrm (win-rate-model pow2.5)", "007-wrm-ep{ep}-vs-suisho11", "#eb6834"),
    ("nofact (factorizer none)", "007-nofact-ep{ep}-vs-suisho11", "#1baf7a"),
    ("both (wrm+nofact)", "007-both-ep{ep}-vs-suisho11", "#eda100"),
    ("lr1shot (single anneal)", "007-lr1shot-ep{ep}-vs-suisho11", "#e87ba4"),
]
ANCHOR = ("006 net re-anchor", "006-yane-ep16-reanchor-vs-suisho11", "#6f6e66")

TEXT = "#1a1a19"
MUTED = "#6f6e66"
GRID = "#e5e4dd"


def load(name: str) -> dict | None:
    p = GAMES / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")

    xs_group = {16: 0.0, 20: 1.0}
    n_arms = len(ARMS)
    for i, (label, tmpl, color) in enumerate(ARMS):
        offs = (i - (n_arms - 1) / 2) * 0.09
        xs, ys, lo, hi, ns = [], [], [], [], []
        for ep in (16, 20):
            s = load(tmpl.format(ep=ep))
            if s is None:
                continue
            xs.append(xs_group[ep] + offs)
            ys.append(s["elo"])
            lo.append(s["elo"] - s["elo_ci95"][0])
            hi.append(s["elo_ci95"][1] - s["elo"])
            ns.append(s["pairs"])
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            color=color,
            linewidth=0,
            elinewidth=2,
            marker="o",
            markersize=7,
            capsize=3,
            label=label,
        )
        if xs:
            ax.annotate(
                f" {label.split()[0]}",
                (xs[-1], ys[-1]),
                color=color,
                fontsize=9,
                va="center",
            )

    s = load(ANCHOR[1])
    if s:
        ax.errorbar(
            [xs_group[16] - 0.33],
            [s["elo"]],
            yerr=[[s["elo"] - s["elo_ci95"][0]], [s["elo_ci95"][1] - s["elo"]]],
            color=ANCHOR[2],
            linewidth=0,
            elinewidth=2,
            marker="s",
            markersize=6,
            capsize=3,
            label=ANCHOR[0],
        )

    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(" Suisho 11 (baseline)", (-0.45, 0), color=MUTED, fontsize=8, va="bottom")

    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["ep16-equiv (69.1B pos)", "ep20-equiv (86.4B pos)"], color=TEXT, fontsize=10
    )
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel("Elo vs Suisho 11 (95% CI)", color=TEXT, fontsize=10)
    ax.set_title(
        "007: yaneurao's suggestions vs Elo — WRM loss is decisive (+~280 over ctrl)",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT, loc="lower left", ncol=2)
    fig.tight_layout()
    out = FIGS / "elo_by_arm.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")

    # report 用テーブル
    for label, tmpl, _c in ARMS + [(ANCHOR[0], ANCHOR[1].replace("007-", "{ep}"), ANCHOR[2])]:
        pass
    for label, tmpl, _c in ARMS:
        row = [label]
        for ep in (16, 20):
            s = load(tmpl.format(ep=ep))
            row.append(
                f"{s['elo']:+.0f} [{s['elo_ci95'][0]:+.0f},{s['elo_ci95'][1]:+.0f}] n={s['pairs']}"
                if s
                else "-"
            )
        print(" | ".join(row))


if __name__ == "__main__":
    main()
