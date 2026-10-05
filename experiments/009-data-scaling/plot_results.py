"""experiment-009 の結果の図と表を作る (対局結果が増えるたびに何度でも実行してよい)。

基準は常に full (最良 epoch・最良 FV_SCALE, 2026-09-30 ユーザー決定)。すべてのネットは水匠 11 (FV_SCALE 32) と
対局し、Elo を full と比べる。

アブレーション (図 5 枚):
  x = アブレーションした特徴量の割合 (5/10/20/30/50/75%, 0% = full)、y = 水匠 11 に対する Elo
  設定 4 通り = 順位 {教師 (training), 対局 (match)} × {zero, random}。設定ごとの 4 枚 + 重ねた 1 枚
  最終 (pipeline の state.json の full 最良) があればそれ、無ければ暫定 (full-e12, zero のみ)

「アブレーション X% ≈ データ削減 Y%」:
  アブレーション: ablated − full、データ削減: arm (その arm の最良 epoch) − full (arm = p90〜p10)。
  対局は決定的で、開始局面は共通 (アブレーションの 1,000 ペアは full の 2,000 ペアの部分集合、arm は同じ 2,000 ペア)
  なので、開始局面ごとの対応のある差を取る。データ側の曲線 (削減 0〜90%) に単調減少の当てはめ (PAV) を行い、
  アブレーションの低下と等しくなる削減率 Y を逆算する。区間は開始局面のブートストラップ (全対局で同じ再標本)。
  補助: full-e10 − full (epoch の効果)、full-rot-e10 − full-e10 (学習の揺らぎ)

出力: experiments/009-data-scaling/figures/*.png と notes/results.md
使い方: .venv/bin/python experiments/009-data-scaling/plot_results.py
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from subsets import ARMS, N_FILES  # noqa: E402

HERE = Path(__file__).resolve().parent
GAMES = HERE / "games"
FIG = HERE / "figures"
NOTE = HERE / "notes" / "results.md"
STATE = Path("/mnt/nvme1/sugiyama/pipeline/state.json")
FV_FLAG = Path("/mnt/nvme1/sugiyama/fv/full.decided")
PCTS = (5, 10, 20, 30, 50, 75)
# 削った割合はファイル数から (removed_pct)
DATA_ARMS = ("p90", "p80", "p70", "p60", "p50", "p30", "p10", "p3")
ARM_PAIRS = 2000
N_BOOT = 2000

# 色はデータ可視化ガイドの検証済みカテゴリ順 (スロット 1〜4) をそのまま使い、色だけに頼らないよう
# 順位の種類を marker、zero/random を線種でも区別する
SETTINGS = [
    ("teach", "zs", "Training-dist. rank, zero", "#2a78d6", "o", "-"),
    ("teach", "rs", "Training-dist. rank, random", "#eb6834", "o", "--"),
    ("match", "zs", "Match-dist. rank, zero", "#1baf7a", "s", "-"),
    ("match", "rs", "Match-dist. rank, random", "#eda100", "s", "--"),
]
DATA_COLOR = "#4a3aa7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RANK_JA = {"teach": "教師", "match": "対局"}
MODE_JA = {"zs": "zero", "rs": "random"}
XLAB = "Ablated features (% of possible, rarest first)"
YLAB = "Elo vs Suisho 11 (FV_SCALE 32)"


# ---------------------------------------------------------------- data


def summary(name: str) -> dict | None:
    """完了した対局の要約 (未完了なら None)。"""
    d = GAMES / name
    if not (d / "queue_done.json").exists():
        return None
    return json.loads((d / "summary.json").read_text())


def pair_scores(name: str) -> dict[str, float]:
    """開始局面 (sfen) → ペア得点 (0〜2, 先後 2 局の合計)。2 局そろったペアだけ。

    ペア数が違う対局どうしでも開始局面で対応が取れるよう、キーは pair 番号ではなく sfen。"""
    tot: dict[str, float] = {}
    cnt: dict[str, int] = {}
    for line in (GAMES / name / "games.jsonl").read_text().splitlines():
        r = json.loads(line)
        tot[r["sfen"]] = tot.get(r["sfen"], 0.0) + r["cand_score"]
        cnt[r["sfen"]] = cnt.get(r["sfen"], 0) + 1
    return {k: v for k, v in tot.items() if cnt[k] == 2}


def elo(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def elo_on(scores: dict[str, float], keys: list[str]) -> tuple[float, float, float]:
    """開始局面の集合 keys の上での Elo と 95% CI (ペア得点の正規近似)。"""
    v = [scores[k] / 2 for k in keys]
    n = len(v)
    m = sum(v) / n
    se = math.sqrt(sum((x - m) ** 2 for x in v) / max(n - 1, 1) / n)
    return elo(m), elo(m - 1.96 * se), elo(m + 1.96 * se)


def pct_ci(vals: list) -> tuple[float, float]:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return (math.nan, math.nan)
    return v[int(0.025 * (len(v) - 1))], v[int(0.975 * (len(v) - 1))]


def best() -> tuple[int, int] | None:
    if STATE.exists():
        b = json.loads(STATE.read_text()).get("best")
        if b:
            return b["epoch"], b["fv"]
    return None


def ablation_names() -> tuple[str, str, dict, str]:
    """(見出し, 基準 full の対局名, {(rank, mode, pct): 対局名}, 基準の表記)。"""
    b = best()
    if b:
        ep, fv = b
        names = {
            (r, m, p): f"final-full-e{ep}-{m}-{r}-f{p}@{fv}-vs-s11@32-300k"
            for r in ("teach", "match")
            for m in ("zs", "rs")
            for p in PCTS
        }
        return (
            f"final: full-e{ep} @ FV_SCALE {fv}",
            f"final-full-e{ep}@{fv}-vs-s11@32-300k-{ARM_PAIRS}p",
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
        f"provisional: full-e12 @ FV_SCALE {fv}",
        f"fv-full-e12@{fv}-vs-s11@32-300k",
        names,
        "full-e12",
    )


def removed_pct(arm: str) -> float:
    """arm で削った教師データの割合 (%)。p3 は 1/30 ファイルなので 96.7%。"""
    return round(100 * (1 - len(ARMS[arm]) / N_FILES), 1)


def arm_best_job(arm: str, fv: int) -> tuple[str, int] | None:
    """arm の最良 epoch の本命対局 (pipeline が追加する arm-<arm>-best-e<N>@...-2000p) の名前と epoch。"""
    for d in sorted(GAMES.glob(f"arm-{arm}-best-e*@{fv}-vs-s11@32-300k-{ARM_PAIRS}p")):
        m = re.match(rf"arm-{re.escape(arm)}-best-e(\d+)@", d.name)
        if m and summary(d.name) is not None:
            return d.name, int(m.group(1))
    return None


# ---------------------------------------------------------------- plots


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def draw(ax, xs, ys, lo, hi, color, marker, ls, label, offset=0.0) -> None:
    x = [v + offset for v in xs]
    yerr = [[y - a for y, a in zip(ys, lo)], [b - y for y, b in zip(ys, hi)]]
    ax.errorbar(
        x, ys, yerr=yerr, color=color, marker=marker, markersize=6, linestyle=ls, linewidth=2,
        capsize=3, elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label,
    )  # fmt: skip


def plot_ablation() -> list[dict]:
    """アブレーションの図 5 枚。各点は基準 full と同じ開始局面 (アブレーション側の局面) の上の Elo。"""
    head, parent_name, names, parent_label = ablation_names()
    FIG.mkdir(exist_ok=True)
    if summary(parent_name) is None:
        print(f"parent {parent_name} not finished yet; no ablation plots")
        return []
    P = pair_scores(parent_name)
    rows = []
    fig_all, ax_all = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    parent_y = None
    ticks = [0, *PCTS]
    tick_labels = [f"0\n({parent_label})", *[str(p) for p in PCTS]]
    for i, (rank, mode, label, color, marker, ls) in enumerate(SETTINGS):
        pts = []
        for p in PCTS:
            n = names.get((rank, mode, p))
            if n is None or summary(n) is None:
                continue
            C = pair_scores(n)
            keys = sorted(set(C) & set(P))
            y, lo, hi = elo_on(C, keys)
            parent_y = elo_on(P, keys)
            changed = sum(1 for k in keys if C[k] != P[k])
            rows.append({"rank": rank, "mode": mode, "pct": p, "elo": y, "ci": (lo, hi),
                         "parent": parent_y[0], "n": len(keys), "changed": changed})  # fmt: skip
            pts.append((p, y, lo, hi))
        if not pts:
            continue
        xs = [0] + [q[0] for q in pts]
        ys = [parent_y[0]] + [q[1] for q in pts]
        lo = [parent_y[1]] + [q[2] for q in pts]
        hi = [parent_y[2]] + [q[3] for q in pts]
        draw(ax_all, xs, ys, lo, hi, color, marker, ls, label, offset=(i - 1.5) * 0.35)
        fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor=SURFACE)
        ax.axhline(parent_y[0], color=INK2, linewidth=1, linestyle=":", zorder=0)
        draw(ax, xs, ys, lo, hi, color, marker, ls, label)
        style(ax, f"{label}\n{head}", XLAB, YLAB)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        fig.tight_layout()
        fig.savefig(FIG / f"ablation_{rank}_{mode}.png", dpi=160)
        plt.close(fig)
    if parent_y is None:
        plt.close(fig_all)
        return rows
    ax_all.axhline(parent_y[0], color=INK2, linewidth=1, linestyle=":", zorder=0)
    style(ax_all, f"Rare-feature ablation, all settings\n{head}", XLAB, YLAB)
    ax_all.set_xticks(ticks)
    ax_all.set_xticklabels(tick_labels)
    ax_all.legend(fontsize=8, frameon=False, labelcolor=INK)
    fig_all.tight_layout()
    fig_all.savefig(FIG / "ablation_merged.png", dpi=160)
    plt.close(fig_all)
    return rows


# ---------------------------------------------------------------- equivalence


def pav_decreasing(ys: list[float]) -> list[float]:
    """単調非増加の最小二乗当てはめ (pool-adjacent-violators, 等重み)。"""
    blocks = [[y, 1] for y in ys]  # [平均, 個数]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out: list[float] = []
    for m, n in blocks:
        out += [m] * n
    return out


def invert(xs: list[float], fs: list[float], target: float) -> float:
    """単調非増加の折れ線 f で f(Y) = target となる最小の Y (0 以下なら 0、末端より下なら inf)。"""
    if target >= fs[0]:
        return 0.0
    for (x0, f0), (x1, f1) in zip(zip(xs, fs), zip(xs[1:], fs[1:])):
        if f1 <= target <= f0:
            return x0 if f0 == f1 else x0 + (x1 - x0) * (f0 - target) / (f0 - f1)
    return math.inf


def equivalence() -> dict | None:
    b = best()
    if not b:
        return None
    ep, fv = b
    ref = f"final-full-e{ep}@{fv}-vs-s11@32-300k-{ARM_PAIRS}p"
    if summary(ref) is None:
        return None
    R = pair_scores(ref)
    keys = sorted(R)
    data = {}
    for arm in DATA_ARMS:
        found = arm_best_job(arm, fv)
        if found:
            data[removed_pct(arm)] = (arm, found[1], pair_scores(found[0]))
    abl = {}
    for rank in ("teach", "match"):
        for m in ("zs", "rs"):
            for pct in PCTS:
                n = f"final-full-e{ep}-{m}-{rank}-f{pct}@{fv}-vs-s11@32-300k"
                if summary(n) is not None:
                    abl[(rank, m, pct)] = pair_scores(n)
    aux = {}
    for key, n in (("e10", f"full-e10@{fv}-vs-s11@32-300k-{ARM_PAIRS}p"),
                   ("rot", f"arm-full-rot-e10@{fv}-vs-s11@32-300k-{ARM_PAIRS}p")):  # fmt: skip
        if summary(n) is not None:
            aux[key] = pair_scores(n)

    def delta(child: dict[str, float], idx: list[int]) -> float | None:
        """基準 full に対する Elo 差 (child − full)。idx は keys の添字の再標本。"""
        ks = [keys[i] for i in idx if keys[i] in child]
        if not ks:
            return None
        base = sum(R[k] for k in ks) / (2 * len(ks))
        diff = sum(child[k] - R[k] for k in ks) / (2 * len(ks))
        return elo(base + diff) - elo(base)

    removed = sorted(data)

    def curve(idx: list[int]) -> tuple[list[float], list[float]]:
        return [0.0, *map(float, removed)], pav_decreasing(
            [0.0, *(delta(data[y][2], idx) for y in removed)]
        )

    allidx = list(range(len(keys)))
    rng = random.Random(20260930)
    boots = [[rng.randrange(len(keys)) for _ in keys] for _ in range(N_BOOT)]
    out = {"ref_label": f"full-e{ep}", "data": [], "abl": [], "aux": {}}
    for y in removed:
        arm, e, c = data[y]
        out["data"].append({"arm": arm, "epoch": e, "removed": y, "delta": delta(c, allidx),
                            "ci": pct_ci([delta(c, bb) for bb in boots])})  # fmt: skip
    if "e10" in aux:
        out["aux"]["epoch"] = (
            delta(aux["e10"], allidx),
            pct_ci([delta(aux["e10"], bb) for bb in boots]),
        )
        if "rot" in aux:

            def dd(idx: list[int]) -> float | None:
                a, b_ = delta(aux["rot"], idx), delta(aux["e10"], idx)
                return None if a is None or b_ is None else a - b_

            out["aux"]["noise"] = (dd(allidx), pct_ci([dd(bb) for bb in boots]))
    xs, fs = curve(allidx) if removed else ([], [])
    curves = [curve(bb) for bb in boots] if removed else []
    out["top"] = removed[-1] if removed else None

    def add_y(row: dict, bd: list) -> None:
        if removed:
            row["y"] = invert(xs, fs, row["delta"])
            row["y_ci"] = pct_ci([invert(*cv, d) for cv, d in zip(curves, bd)])

    bds = {}
    for (rank, m, pct), c in abl.items():
        bds[(rank, m, pct)] = bd = [delta(c, bb) for bb in boots]
        row = {"rank": rank, "mode": m, "pct": pct, "delta": delta(c, allidx), "ci": pct_ci(bd)}
        add_y(row, bd)
        out["abl"].append(row)
    # まとめ: 下位 X% ごとに 4 設定 (順位 2 × モード 2) の差を平均した 1 行。区間は同じ再標本で平均してから取る
    out["pooled"] = []
    for pct in PCTS:
        ks = [k for k in bds if k[2] == pct]
        if len(ks) < 4:
            continue
        bd = [sum(bds[k][i] for k in ks) / len(ks) for i in range(N_BOOT)]
        row = {"pct": pct, "delta": sum(r["delta"] for r in out["abl"] if r["pct"] == pct) / len(ks),
               "ci": pct_ci(bd)}  # fmt: skip
        add_y(row, bd)
        out["pooled"].append(row)
    return out


def plot_equivalence(eq: dict) -> None:
    """補助の図: 基準 full に対する Elo の差 (データ削減と、アブレーションの 4 設定)。"""
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    ax.axhline(0, color=INK2, linewidth=1, linestyle=":", zorder=0)
    if "noise" in eq["aux"]:
        lo, hi = eq["aux"]["noise"][1]
        ax.axhspan(
            lo,
            hi,
            color=GRID,
            alpha=0.7,
            zorder=0,
            label="Run-to-run noise (full-rot − full at e10, 95%)",
        )
    r = eq["data"]
    if r:
        draw(ax, [0] + [x["removed"] for x in r], [0] + [x["delta"] for x in r], [0] + [x["ci"][0] for x in r],
             [0] + [x["ci"][1] for x in r], DATA_COLOR, "D", "-", "Training data removed (arm at its best epoch)")  # fmt: skip
    for i, (rank, mode, label, color, marker, ls) in enumerate(SETTINGS):
        a = [x for x in eq["abl"] if x["rank"] == rank and x["mode"] == mode]
        if a:
            draw(ax, [0] + [x["pct"] for x in a], [0] + [x["delta"] for x in a], [0] + [x["ci"][0] for x in a],
                 [0] + [x["ci"][1] for x in a], color, marker, ls, f"Features ablated: {label}",
                 offset=(i - 1.5) * 0.35)  # fmt: skip
    style(ax, f"Elo change vs {eq['ref_label']} (same openings, bootstrap 95% CI)",
          "% of training data removed  /  % of features ablated", f"Δ Elo vs {eq['ref_label']} (all vs Suisho 11)")  # fmt: skip
    ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_vs_data.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- note


def fmt_ci(a: float, b: float) -> str:
    return f"[{a:+.0f}, {b:+.0f}]"


def fy(v: float, top: float) -> str:
    """Y の表示。データ側の曲線の末端 (top %) より大きい低下は「>top%」。"""
    return "0%" if v == 0 else (f">{top:g}%" if math.isinf(v) else f"{v:.0f}%")


def write_note(abl_rows: list[dict], eq: dict | None) -> None:
    head, _, _, parent_label = ablation_names()
    lines = [
        "# experiment-009 の結果 (自動生成)\n",
        "`plot_results.py` が対局結果から作る (手で編集しない)。すべて水匠 11 (FV_SCALE 32) との対局、300k ノード。",
        "**基準は常に full** (最良 epoch・最良 FV_SCALE)。対局は決定的なので、アブレーションで指し手が変わらない局は",
        "full と同じ棋譜になり、同じ開始局面どうしの差は対応のある比較になる。\n",
        f"## 稀な特徴量のアブレーション ({head})\n",
    ]
    if abl_rows:
        lines += [
            "![all settings](../figures/ablation_merged.png)\n",
            f"| 順位 | モード | 下位 % | Elo vs 水匠 11 | 95% CI | 同じ局面での {parent_label} | 局面数 | 棋譜が変わったペア |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in abl_rows:
            lines.append(
                f"| {RANK_JA[r['rank']]} | {MODE_JA[r['mode']]} | {r['pct']} | {r['elo']:+.1f} | "
                f"{fmt_ci(*r['ci'])} | {r['parent']:+.1f} | {r['n']} | {r['changed']} |"
            )
        lines.append("")
        for rank, mode, label, *_ in SETTINGS:
            if (FIG / f"ablation_{rank}_{mode}.png").exists():
                lines.append(f"![{label}](../figures/ablation_{rank}_{mode}.png)")
    else:
        lines.append("(まだ結果が無い)")
    if eq:
        ref = eq["ref_label"]
        lines += [
            f"\n## アブレーション X% ≈ データ削減 Y% (基準 {ref})\n",
            f"アブレーション: ablated − {ref}、データ削減: arm (最良 epoch) − {ref} (どれも水匠 11 に対する Elo の差)。",
            "Y はデータ側の曲線 (単調減少に当てはめ) で同じ低下になる削減率。区間は開始局面のブートストラップ (2,000 回)。\n",
            f"| 順位 | モード | アブレーション X | {ref} からの差 (Elo) | 95% CI | 同等なデータ削減 Y | 95% 区間 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        top = eq["top"]

        def ys(r: dict) -> tuple[str, str]:
            if "y" not in r:
                return "–", "–"
            return fy(r["y"], top), f"[{fy(r['y_ci'][0], top)}, {fy(r['y_ci'][1], top)}]"

        if eq["pooled"]:
            pooled = [
                "**まとめ (4 設定の平均)**: 順位 (教師 / 対局) × モード (zero / random) の 4 本の差を平均した値。",
                "4 本は同じ親・同じ開始局面で、選ぶ特徴量も大きく重なるので独立ではない (区間は同じ再標本で平均してから取る)。\n",
                f"| アブレーション X | {ref} からの差 (Elo, 4 設定の平均) | 95% CI | 同等なデータ削減 Y | 95% 区間 |",
                "| --- | --- | --- | --- | --- |",
            ]
            for r in eq["pooled"]:
                y = ys(r)
                pooled.append(
                    f"| {r['pct']}% | {r['delta']:+.1f} | {fmt_ci(*r['ci'])} | {y[0]} | {y[1]} |"
                )
            pooled += ["", "**設定ごと**:\n", lines.pop(-2), lines.pop(-1)]
            lines += pooled
        for r in eq["abl"]:
            y = ys(r)
            lines.append(
                f"| {RANK_JA[r['rank']]} | {MODE_JA[r['mode']]} | {r['pct']}% | {r['delta']:+.1f} | "
                f"{fmt_ci(*r['ci'])} | {y[0]} | {y[1]} |"
            )
        lines += [
            f"\n| データ削減 arm | 最良 epoch | 削った割合 | {ref} からの差 (Elo) | 95% CI |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in eq["data"]:
            lines.append(
                f"| {r['arm']} | e{r['epoch']} | {r['removed']:g}% | {r['delta']:+.1f} | {fmt_ci(*r['ci'])} |"
            )
        if "epoch" in eq["aux"]:
            d, ci = eq["aux"]["epoch"]
            lines.append(f"| (補助) full-e10: epoch の効果 | e10 | 0% | {d:+.1f} | {fmt_ci(*ci)} |")
        if "noise" in eq["aux"]:
            d, ci = eq["aux"]["noise"]
            lines.append(
                f"| (補助) full-rot-e10 − full-e10: 学習の揺らぎ | e10 | 0% | {d:+.1f} | {fmt_ci(*ci)} |"
            )
        lines.append("\n![ablation vs data](../figures/ablation_vs_data.png)")
    NOTE.write_text("\n".join(lines) + "\n")


def main() -> None:
    abl = plot_ablation()
    eq = equivalence()
    if eq and (eq["data"] or eq["abl"]):
        plot_equivalence(eq)
    write_note(abl, eq)
    print(f"ablation rows {len(abl)}, equivalence {'yes' if eq else 'no'} -> {NOTE}")


if __name__ == "__main__":
    main()
