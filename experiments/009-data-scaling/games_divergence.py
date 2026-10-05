"""アブレーションしたネットの棋譜が、親 (full-e16) の棋譜から何手目で分かれるか (最終アブレーション 24 ネット)。

同じ開始局面・同じ先後の 2 局 (親 / アブレーション) の指し手列を先頭から比べ、最初に違う手の位置を数える。
対局は決定的 (Threads 1, 固定ノード, 毎局 isready) なので、分岐する手は必ずアブレーションした側の手番になる
(これも確かめる)。

出力 (どれも自動生成):
- `notes/divergence.md`: 24 ネットの表と読み方
- `figures/divergence_summary.png`: 分岐の位置 (中央値と四分位) を下位 X% に対して 4 設定で
- `figures/divergence_all.png`: 24 ネットそれぞれの分岐位置 (局の進行度) のヒストグラム
- `figures/divergence_<設定>.png`: 1 ネットの詳細 (手目と進行度のヒストグラム。`--plot` で選ぶ)

    .venv/bin/python experiments/009-data-scaling/games_divergence.py [--plot zs-teach-f10 ...]
"""

from __future__ import annotations

import argparse
import json
import statistics as st

import numpy as np
from plot_results import (
    FIG,
    GAMES,
    GRID,
    HERE,
    INK2,
    MODE_JA,
    RANK_JA,
    SETTINGS,
    SURFACE,
    draw,
    plt,
    style,
)

PARENT = "final-full-e16@48-vs-s11@32-300k-2000p"
REPLAY = "fv-full-e16@48-vs-s11@32-300k"  # 同じネット・同じ FV_SCALE を別のジョブとして対局 (FV_SCALE 格子, 400 ペア, 10-01 06:43〜)
PCTS = (5, 10, 20, 30, 50, 75)
OPENING_PLY = 24  # 互角局面集の局面は 24 手目から指す (sfen の手数)
NOTE = HERE / "notes" / "divergence.md"


def load(name: str) -> dict[tuple[str, bool], dict]:
    d = {}
    for line in (GAMES / name / "games.jsonl").read_text().splitlines():
        r = json.loads(line)
        d[(r["sfen"], r["cand_is_black"])] = r
    return d


def divergence(parent: dict, child: dict) -> dict:
    """k = 開始局面から数えて最初に違う手の位置 (0 = 最初の手)。rel = k / 親の局の手数 (%)。"""
    ks, rel, n, cand_first = [], [], 0, 0
    for key, c in child.items():
        if key not in parent:
            continue
        n += 1
        a, b = parent[key]["moves"].split(), c["moves"].split()
        if a == b:
            continue
        k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        black_first = key[0].split()[1] == "b"  # sfen の 2 番目のフィールド = 手番
        mover_is_black = (k % 2 == 0) == black_first
        cand_first += mover_is_black == key[1]
        ks.append(k)
        rel.append(100 * k / len(a))

    # ペアの得点 (先後 2 局の合計) が親と違うペアの数 (results.md の「得点が変わったペア」と同じ定義)
    def pair_score(d: dict) -> dict[str, float]:
        t: dict[str, float] = {}
        for (sfen, _), r in d.items():
            t[sfen] = t.get(sfen, 0.0) + r["cand_score"]
        return t

    pc, pp = pair_score({k: v for k, v in child.items() if k in parent}), pair_score(parent)
    score_changed = sum(1 for s in pc if pc[s] != pp[s])
    return {"n": n, "ks": ks, "rel": rel, "cand_first": cand_first, "pairs": len(pc),
            "score_changed": score_changed}  # fmt: skip


def quart(v: list[float]) -> tuple[float, float, float]:
    q1, med, q3 = st.quantiles(v, n=4)
    return q1, med, q3


def plot_one(setting: str, d: dict) -> None:
    mode, rank, _ = setting.split("-")
    col = next((s[3] for s in SETTINGS if (s[0], s[1]) == (rank, mode)), INK2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    ply = np.array(d["ks"]) + OPENING_PLY
    rel = np.array(d["rel"])
    for ax, v, bins, xlab, fmt in (
        (axes[0], ply, np.arange(OPENING_PLY, ply.max() + 3, 2), "Move number of the first different move",
         "move {:.0f}"),
        (axes[1], rel, np.arange(0, 102, 2), "Progress of the game at the first different move (%)", "{:.0f}%"),
    ):  # fmt: skip
        style(ax, "", xlab, "Games")
        ax.hist(v, bins=bins, color=col, edgecolor=SURFACE, linewidth=0.6)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.axvline(med, color=INK2, linewidth=1.2, linestyle=(0, (3, 3)))
        ax.set_title(f"median: {fmt.format(med)}  (IQR {fmt.format(q1)} – {fmt.format(q3)})",
                     color=INK2, fontsize=9, loc="left")  # fmt: skip
    fig.suptitle(f"{setting}: where the game record first differs from full-e16 "
                 f"({len(d['ks'])} of {d['n']} games differ)", x=0.01, ha="left", fontsize=11)  # fmt: skip
    fig.text(0.01, 0.01, "Progress = (moves played before the first different move) / "
             "(length of full-e16's game), counted from the opening position (move 24).",
             color=INK2, fontsize=7.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = FIG / f"divergence_{setting.replace('-', '_')}.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(f"wrote {out}")


def plot_summary(res: dict, parent_end: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    panels = (
        (
            axes[0],
            lambda d: [v + OPENING_PLY for v in d["ks"]],
            "Move number of the first different move",
        ),
        (axes[1], lambda d: d["rel"], "Progress of the game at the first different move (%)"),
    )
    for ax, get, ylab in panels:
        style(ax, "", "Ablated features (% of possible, rarest first)", ylab)
        for i, (rank, mode, label, color, marker, ls) in enumerate(SETTINGS):
            q = [quart(get(res[(rank, mode, x)])) for x in PCTS]
            draw(ax, list(PCTS), [m for _, m, _ in q], [a for a, _, _ in q], [b for _, _, b in q],
                 color, marker, ls, label, offset=(i - 1.5) * 0.9)  # fmt: skip
        ax.set_xticks(PCTS)
    axes[0].axhline(parent_end, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    axes[0].text(75, parent_end, f"full-e16's games end (median move {parent_end:.0f})", color=INK2,
                 fontsize=7.5, ha="right", va="bottom")  # fmt: skip
    axes[0].set_ylim(OPENING_PLY - 2, parent_end + 12)
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID, loc="upper right")
    fig.suptitle("Where the game record first differs from full-e16 (median, bars = interquartile range)",
                 x=0.01, ha="left", fontsize=11)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / "divergence_summary.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(f"wrote {out}")


def plot_all(res: dict) -> None:
    fig, axes = plt.subplots(len(SETTINGS), len(PCTS), figsize=(14, 8.5), sharex=True,
                             facecolor=SURFACE)  # fmt: skip
    bins = np.arange(0, 102, 4)
    for r, (rank, mode, label, color, *_) in enumerate(SETTINGS):
        for c, x in enumerate(PCTS):
            ax, d = axes[r][c], res[(rank, mode, x)]
            style(ax, "", "", "")
            ax.hist(d["rel"], bins=bins, color=color, edgecolor=SURFACE, linewidth=0.4)
            med = st.median(d["rel"])
            ax.axvline(med, color=INK2, linewidth=1.0, linestyle=(0, (3, 3)))
            ax.set_title(f"rarest {x}%: median {med:.0f}%  ({100 * len(d['ks']) / d['n']:.0f}% differ)",
                         color=INK2, fontsize=8, loc="left")  # fmt: skip
            if c == 0:
                ax.set_ylabel(label.replace(", ", "\n"), color=INK2, fontsize=8)
            if r == len(SETTINGS) - 1:
                ax.set_xlabel("Progress at first difference (%)", color=INK2, fontsize=8)
    fig.suptitle("Where each game first differs from full-e16 — progress of the game (%), "
                 "histogram of games for each ablated net", x=0.01, ha="left", fontsize=11)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "divergence_all.png"
    fig.savefig(out, dpi=100, facecolor=SURFACE)
    print(f"wrote {out}")


def write_note(res: dict, parent_len: float, replay: tuple[int, int]) -> None:
    lines = [
        "# 棋譜はどこで分かれるか (自動生成)\n",
        "`games_divergence.py` が作る (手で編集しない)。最終アブレーション (full-e16 @ FV_SCALE 48 の 24 ネット, 各 667 ペア = "
        "1,334 局, 水匠 11 相手, 300k ノード) の各局を、同じ開始局面・同じ先後の full-e16 の局と指し手列で比べ、"
        "最初に違う手の位置を数えた。\n",
        "- **手目**: 開始局面 (24 手目) からの通しの手数。**進行度**: 分岐までに指した手数 ÷ full-e16 の局の手数 "
        f"(どちらも開始局面から数える。full-e16 の局は中央値 {parent_len:.0f} 手 = {parent_len + OPENING_PLY - 1:.0f} 手目で終局)",
        f"- **対局の再現性 (対照)**: full-e16 そのものを別のジョブとして約 4 時間前に対局した {replay[1]} 局 (FV_SCALE 格子の対局, "
        f"開始局面は 2,000 ペアの部分集合) は、{replay[0]} 局が最終の対局と**同じ棋譜**だった。"
        "アブレーションしなければ棋譜は変わらないので、下の表の変化はすべてアブレーションによる",
        "- 対局は決定的なので、**最初に違う手は 24 ネット・全局でアブレーションした側の手**だった",
        "- **得点が変わったペア** ([results.md](results.md) と同じ定義): ペアの得点 (先後 2 局の合計) が full-e16 と違うペアの数。"
        "棋譜が変わっても得点は変わらないことが多い\n",
        "![summary](../figures/divergence_summary.png)\n",
        "| 順位 | モード | 下位 % | 棋譜が変わった局 | 分岐の手目 (中央値 [四分位]) | 進行度 (中央値 [四分位]) | 得点が変わったペア |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, mode, *_ in SETTINGS:
        for x in PCTS:
            d = res[(rank, mode, x)]
            q1, med, q3 = quart(d["ks"])
            r1, rm, r3 = quart(d["rel"])
            lines.append(
                f"| {RANK_JA[rank]} | {MODE_JA[mode]} | {x} | {len(d['ks'])} / {d['n']} "
                f"({100 * len(d['ks']) / d['n']:.0f}%) | {med + OPENING_PLY:.0f} [{q1 + OPENING_PLY:.0f}, "
                f"{q3 + OPENING_PLY:.0f}] | {rm:.0f}% [{r1:.0f}, {r3:.0f}] | {d['score_changed']} / {d['pairs']} |"
            )
    lines += [
        "",
        "24 ネットそれぞれの分布 (進行度):\n",
        "![all](../figures/divergence_all.png)\n",
        "1 ネットの詳細 (教師順位・zero・下位 10%):\n",
        "![zs-teach-f10](../figures/divergence_zs_teach_f10.png)",
    ]
    NOTE.write_text("\n".join(lines) + "\n")
    print(f"wrote {NOTE}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plot", nargs="*", default=["zs-teach-f10"], help="詳細図を作る設定 (例 zs-teach-f10)"
    )
    args = ap.parse_args()
    parent = load(PARENT)
    res = {}
    for rank, mode, *_ in SETTINGS:
        for x in PCTS:
            d = divergence(parent, load(f"final-full-e16-{mode}-{rank}-f{x}@48-vs-s11@32-300k"))
            assert d["cand_first"] == len(d["ks"]), (
                rank,
                mode,
                x,
            )  # 分岐は必ずアブレーション側の手
            res[(rank, mode, x)] = d
    # full-e16 の局の手数 (アブレーションと同じ 667 開始局面の局だけ)
    keys = {k for d in [load("final-full-e16-zs-teach-f5@48-vs-s11@32-300k")] for k in d}
    parent_len = st.median(len(parent[k]["moves"].split()) for k in keys if k in parent)
    for s in args.plot:
        mode, rank, f = s.split("-")
        plot_one(s, res[(rank, mode, int(f[1:]))])
    plot_summary(res, parent_len + OPENING_PLY - 1)
    plot_all(res)
    rp = load(REPLAY)
    replay = (
        sum(1 for k, r in rp.items() if k in parent and r["moves"] == parent[k]["moves"]),
        len(rp),
    )
    write_note(res, parent_len, replay)


if __name__ == "__main__":
    main()
