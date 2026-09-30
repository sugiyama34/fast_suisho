"""experiment-009 の結果の図と表を作る (対局結果が増えるたびに何度でも実行してよい)。

アブレーション (ユーザー指定, 2026-09-30):
  x = アブレーションした特徴量の割合 (5/10/20/30%, 0% = 親 = full)、y = 水匠 11 (FV_SCALE 32) に対する Elo
  設定 4 通り = 順位 {教師 (training), 対局 (match)} × {zero, random}
  図 5 枚 = 設定ごとの 4 枚 + 4 設定を重ねた 1 枚
  対象: 最終 (pipeline の state.json にある full の最良 epoch / FV_SCALE) があればそれ、無ければ暫定 (full-e12)

データ削減との比較: 基準からの Elo の低下 (アブレーション: ablated − full 最良, データ削減: arm-e10 − full-e10) を
同じ軸に並べる。親と子は同じ開始局面で対局し、対局は決定的なので、局面ごとの対応のある差 (paired) も出す。

出力: experiments/009-data-scaling/figures/*.png と notes/results.md
使い方: .venv/bin/python experiments/009-data-scaling/plot_results.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
GAMES = HERE / "games"
FIG = HERE / "figures"
NOTE = HERE / "notes" / "results.md"
STATE = Path("/mnt/nvme1/sugiyama/pipeline/state.json")
FV_FLAG = Path("/mnt/nvme1/sugiyama/fv/full.decided")
PCTS = (5, 10, 20, 30)
ARMS = ("p90", "p80", "p70", "p60", "p50", "p10")

# 設定ごとの見た目。色はデータ可視化ガイドの検証済みカテゴリ順 (スロット 1〜4) をそのまま使い、
# 色だけに頼らないよう、順位の種類を marker、zero/random を線種でも区別する
SETTINGS = [
    ("teach", "zs", "Training-dist. rank, zero", "#2a78d6", "o", "-"),
    ("teach", "rs", "Training-dist. rank, random", "#eb6834", "o", "--"),
    ("match", "zs", "Match-dist. rank, zero", "#1baf7a", "s", "-"),
    ("match", "rs", "Match-dist. rank, random", "#eda100", "s", "--"),
]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ---------------------------------------------------------------- data


def summary(name: str) -> dict | None:
    """完了した対局の要約 (未完了なら None)。"""
    d = GAMES / name
    if not (d / "queue_done.json").exists():
        return None
    return json.loads((d / "summary.json").read_text())


def pair_scores(name: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for line in (GAMES / name / "games.jsonl").read_text().splitlines():
        r = json.loads(line)
        out[r["pair"]] = out.get(r["pair"], 0.0) + r["cand_score"]
    return out


def elo(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def paired_delta(child: str, parent: str) -> tuple[float, float, float, int] | None:
    """同じ開始局面のペアごとの得点差から Elo 差 (child − parent) と 95% CI、棋譜が変わったペア数。"""
    if summary(child) is None or summary(parent) is None:
        return None
    c, p = pair_scores(child), pair_scores(parent)
    keys = sorted(set(c) & set(p))
    if not keys:
        return None
    d = [(c[k] - p[k]) / 2 for k in keys]  # ペア得点 (0〜2) → 1 局あたりの得点
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / max(n - 1, 1)
    se = math.sqrt(var / n)
    base = sum(p[k] for k in keys) / (2 * n)
    delta = elo(base + mean) - elo(base)
    lo, hi = elo(base + mean - 1.96 * se) - elo(base), elo(base + mean + 1.96 * se) - elo(base)
    changed = sum(1 for x in d if x != 0)
    return delta, lo, hi, changed


def ablation_names() -> tuple[str, str, dict, str]:
    """(見出し, 親の対局名, {(rank, mode, pct): 対局名}, 親の表記)。最終があれば最終、無ければ暫定 e12。"""
    if STATE.exists():
        st = json.loads(STATE.read_text())
        if "best" in st:
            ep, fv = st["best"]["epoch"], st["best"]["fv"]
            names = {
                (r, m, p): f"final-full-e{ep}-{m}-{r}-f{p}@{fv}-vs-s11@32-300k"
                for r in ("teach", "match")
                for m in ("zs", "rs")
                for p in PCTS
            }
            return (
                f"final: full-e{ep} @ FV_SCALE {fv}, 1,000 pairs",
                f"final-full-e{ep}@{fv}-vs-s11@32-300k",
                names,
                f"full-e{ep}",
            )
    fv = int(FV_FLAG.read_text().strip()) if FV_FLAG.exists() else 0
    names = {
        (r, "zs", p): f"full-e12-zs-{r}-f{p}-vs-s11@32-300k"
        for r in ("teach", "match")
        for p in PCTS
    }
    return (
        f"provisional: full-e12 @ FV_SCALE {fv}, 400 pairs",
        f"fv-full-e12@{fv}-vs-s11@32-300k",
        names,
        "full-e12",
    )


# ---------------------------------------------------------------- plots


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(
        "Ablated features (% of structurally possible, lowest count first)", color=INK2, fontsize=9
    )
    ax.set_ylabel("Elo vs Suisho 11 (FV_SCALE 32)", color=INK2, fontsize=9)
    ax.set_xticks([0, *PCTS])
    ax.set_xticklabels(["0\n(parent)", *[f"{p}" for p in PCTS]])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def series(parent: dict, names: dict, rank: str, mode: str) -> tuple[list, list, list, list]:
    xs, ys, lo, hi = [0], [parent["elo"]], [parent["elo_ci95"][0]], [parent["elo_ci95"][1]]
    for p in PCTS:
        s = summary(names.get((rank, mode, p), ""))
        if s is None:
            continue
        xs.append(p)
        ys.append(s["elo"])
        lo.append(s["elo_ci95"][0])
        hi.append(s["elo_ci95"][1])
    return xs, ys, lo, hi


def draw(ax, xs, ys, lo, hi, color, marker, ls, label, offset=0.0, end_label=False) -> None:
    x = [v + offset for v in xs]
    yerr = [[y - a for y, a in zip(ys, lo)], [b - y for y, b in zip(ys, hi)]]
    ax.errorbar(
        x, ys, yerr=yerr, color=color, marker=marker, markersize=6, linestyle=ls, linewidth=2,
        capsize=3, elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label,
    )  # fmt: skip
    if end_label and len(xs) > 1:
        ax.annotate(label, (x[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=7,
                    color=INK2, va="center")  # fmt: skip


def plot_ablation() -> list[dict]:
    head, parent_name, names, parent_label = ablation_names()
    parent = summary(parent_name)
    FIG.mkdir(exist_ok=True)
    if parent is None:
        print(f"parent {parent_name} not finished yet; no ablation plots")
        return []
    rows = []
    fig_all, ax_all = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    for i, (rank, mode, label, color, marker, ls) in enumerate(SETTINGS):
        xs, ys, lo, hi = series(parent, names, rank, mode)
        for p, y, a, b in zip(xs[1:], ys[1:], lo[1:], hi[1:]):
            pd = paired_delta(names[(rank, mode, p)], parent_name)
            rows.append(
                {"rank": rank, "mode": mode, "pct": p, "elo": y, "ci": (a, b), "paired": pd}
            )
        if len(xs) > 1:
            draw(ax_all, xs, ys, lo, hi, color, marker, ls, label, offset=(i - 1.5) * 0.35)
        fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor=SURFACE)
        ax.axhline(parent["elo"], color=INK2, linewidth=1, linestyle=":", zorder=0)
        draw(ax, xs, ys, lo, hi, color, marker, ls, label)
        style(ax, f"{label}\n{head}")
        fig.tight_layout()
        fig.savefig(FIG / f"ablation_{rank}_{mode}.png", dpi=160)
        plt.close(fig)
    ax_all.axhline(parent["elo"], color=INK2, linewidth=1, linestyle=":", zorder=0)
    ax_all.annotate(
        f"{parent_label} (parent)",
        (30.8, parent["elo"]),
        fontsize=7,
        color=INK2,
        va="bottom",
        ha="right",
    )
    style(ax_all, f"Rare-feature ablation, all settings\n{head}")
    ax_all.legend(fontsize=8, frameon=False, labelcolor=INK)
    fig_all.tight_layout()
    fig_all.savefig(FIG / "ablation_merged.png", dpi=160)
    plt.close(fig_all)
    return rows


def plot_data_vs_ablation(abl_rows: list[dict]) -> list[dict]:
    """データ削減 arm (e10) の Elo 低下と、アブレーションの Elo 低下を同じ軸に並べる。"""
    fv = None
    if STATE.exists():
        st = json.loads(STATE.read_text())
        fv = st.get("best", {}).get("fv")
    if fv is None:
        return []
    ref = f"full-e10@{fv}-vs-s11@32-300k-1000p"
    rows = []
    for arm in ARMS:
        name = f"arm-{arm}-e10@{fv}-vs-s11@32-300k-1000p"
        s, pd = summary(name), paired_delta(name, ref)
        if s is None or pd is None:
            continue
        removed = 100 - int(arm[1:])
        rows.append(
            {
                "arm": arm,
                "removed": removed,
                "elo": s["elo"],
                "ci": tuple(s["elo_ci95"]),
                "paired": pd,
            }
        )
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    ax.axhline(0, color=INK2, linewidth=1, linestyle=":", zorder=0)
    xs = [r["removed"] for r in rows]
    ys = [r["paired"][0] for r in rows]
    lo = [r["paired"][1] for r in rows]
    hi = [r["paired"][2] for r in rows]
    draw(ax, xs, ys, lo, hi, "#4a3aa7", "D", "-", "Training data removed (arm-e10 − full-e10)")
    for i, (rank, mode, label, color, marker, ls) in enumerate(SETTINGS):
        pts = [r for r in abl_rows if r["rank"] == rank and r["mode"] == mode and r["paired"]]
        if pts:
            draw(ax, [r["pct"] for r in pts], [r["paired"][0] for r in pts], [r["paired"][1] for r in pts],
                 [r["paired"][2] for r in pts], color, marker, ls, f"Features ablated: {label}",
                 offset=(i - 1.5) * 0.35)  # fmt: skip
    ax.set_facecolor(SURFACE)
    ax.set_title("Elo lost vs own reference (paired, 95% CI)", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Removed (% of training data, or % of features ablated)", color=INK2, fontsize=9)
    ax.set_ylabel("Δ Elo vs reference (vs Suisho 11)", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_vs_data.png", dpi=160)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------- note


def fmt_ci(a: float, b: float) -> str:
    return f"[{a:+.0f}, {b:+.0f}]"


def write_note(abl_rows: list[dict], data_rows: list[dict]) -> None:
    head, parent_name, _, parent_label = ablation_names()
    parent = summary(parent_name)
    lines = [
        "# experiment-009 の結果 (自動生成)\n",
        "`plot_results.py` が対局結果から作る (手で編集しない)。Elo は水匠 11 (FV_SCALE 32) 相手、300k ノード。",
        "「対応のある差」は親と同じ開始局面のペアごとの得点差から計算した Elo 差 (対局は決定的なので、",
        "アブレーションで指し手が変わらない局は親と同じ棋譜になる)。\n",
        f"## 稀な特徴量のアブレーション ({head})\n",
    ]
    if parent:
        lo, hi = parent["elo_ci95"]
        lines.append(
            f"- 親 {parent_label}: **{parent['elo']:+.1f} Elo** {fmt_ci(lo, hi)} ({parent['pairs']} ペア)\n"
        )
    lines += [
        "![all settings](../figures/ablation_merged.png)\n",
        "| 順位 | モード | 下位 % | Elo vs 水匠 11 | 95% CI | 親との対応のある差 | 95% CI | 棋譜が変わったペア |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rank_ja = {"teach": "教師", "match": "対局"}
    mode_ja = {"zs": "zero", "rs": "random"}
    for r in abl_rows:
        pd = r["paired"]
        pdtxt = (f"{pd[0]:+.1f}", fmt_ci(pd[1], pd[2]), str(pd[3])) if pd else ("–", "–", "–")
        lines.append(
            f"| {rank_ja[r['rank']]} | {mode_ja[r['mode']]} | {r['pct']} | {r['elo']:+.1f} | "
            f"{fmt_ci(*r['ci'])} | {pdtxt[0]} | {pdtxt[1]} | {pdtxt[2]} |"
        )
    lines.append("")
    for rank, mode, label, *_ in SETTINGS:
        if (FIG / f"ablation_{rank}_{mode}.png").exists():
            lines.append(f"![{label}](../figures/ablation_{rank}_{mode}.png)")
    if data_rows:
        lines += [
            "\n## データ削減との比較 (e10, 同じ 1,000 ペア)\n",
            "![ablation vs data](../figures/ablation_vs_data.png)\n",
            "| arm | 削ったデータ | Elo vs 水匠 11 | 95% CI | full-e10 との対応のある差 | 95% CI |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in data_rows:
            pd = r["paired"]
            lines.append(
                f"| {r['arm']} | {r['removed']}% | {r['elo']:+.1f} | {fmt_ci(*r['ci'])} | "
                f"{pd[0]:+.1f} | {fmt_ci(pd[1], pd[2])} |"
            )
    NOTE.write_text("\n".join(lines) + "\n")


def main() -> None:
    abl = plot_ablation()
    data = plot_data_vs_ablation(abl)
    write_note(abl, data)
    print(f"ablation rows {len(abl)}, data rows {len(data)} -> {NOTE}")


if __name__ == "__main__":
    main()
