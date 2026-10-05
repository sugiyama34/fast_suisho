"""アブレーションしたネットの棋譜が、親 (full-e16) の棋譜から何手目で分かれるか。

同じ開始局面・同じ先後の 2 局 (親 / アブレーション) の指し手列を先頭から比べ、最初に違う手の位置を数える。
対局は決定的 (Threads 1, 固定ノード, 毎局 isready) なので、分岐する手は必ずアブレーションした側の手番になる
(これも確かめる)。

- 標準出力: 24 ネットの表 (markdown)
- `figures/divergence_<設定>.png`: 1 設定の分岐位置のヒストグラム (手目と、終局までの手数に対する割合)

    .venv/bin/python experiments/009-data-scaling/games_divergence.py [--plot zs-teach-f10]
"""

from __future__ import annotations

import argparse
import json
import statistics as st

import numpy as np
from plot_results import FIG, GAMES, INK2, SETTINGS, SURFACE, plt, style

PARENT = "final-full-e16@48-vs-s11@32-300k-2000p"
PCTS = (5, 10, 20, 30, 50, 75)
OPENING_PLY = 24  # 互角局面集の局面は 24 手目から指す (sfen の手数)


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
    return {"n": n, "ks": ks, "rel": rel, "cand_first": cand_first}


def plot(setting: str, d: dict) -> None:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="zs-teach-f10", help="図にする設定 (例 zs-teach-f10)")
    args = ap.parse_args()
    parent = load(PARENT)
    print(
        "| 設定 | 棋譜が変わった局 | 分岐の手目 (中央値 [IQR]) | 終局までの何 % で分岐 (中央値 [IQR]) |"
    )
    print("| --- | --- | --- | --- |")
    for mode in ("zs", "rs"):
        for rank in ("teach", "match"):
            for x in PCTS:
                s = f"{mode}-{rank}-f{x}"
                d = divergence(parent, load(f"final-full-e16-{s}@48-vs-s11@32-300k"))
                assert d["cand_first"] == len(d["ks"]), s  # 分岐は必ずアブレーション側の手
                q = st.quantiles(d["ks"], n=4)
                r = st.quantiles(d["rel"], n=4)
                print(f"| {s} | {len(d['ks'])} / {d['n']} ({100 * len(d['ks']) / d['n']:.0f}%) | "
                      f"{q[1] + OPENING_PLY:.0f} [{q[0] + OPENING_PLY:.0f}, {q[2] + OPENING_PLY:.0f}] | "
                      f"{r[1]:.0f}% [{r[0]:.0f}, {r[2]:.0f}] |")  # fmt: skip
                if s == args.plot:
                    keep = d
    plot(args.plot, keep)


if __name__ == "__main__":
    main()
