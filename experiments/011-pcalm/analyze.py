"""experiment-011 の最終集計: 同じ開始局面どうしの Elo の差 (開始局面のブートストラップ) と §4.5 の判定。

- 水匠 11 相手の 2,000 ペアの対局どうしを開始局面 (sfen) で対応させ、Elo の差の 95% CI を出す
  (開始局面を復元抽出 10,000 回)。experiment-009 の full-e16 @ 48 (−119.4) との比較もここで出す
- 判定 (hypothesis.md §4.5, 事前に固定): D = 直接対局の Elo (PC-ALM から見た値)、N = ノイズの目安
- 出力: 標準出力に Markdown の表 (report.md に貼る)

使い方: .venv/bin/python experiments/011-pcalm/analyze.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAMES = HERE / "games"
GAMES_009 = HERE.parent / "009-data-scaling" / "games"
B = 10_000

RUNS = {
    "PC-ALM (e12 @ 24 = 48 相当)": GAMES / "final-full-pcalm-T4-gn-e12@24-vs-s11@32-300k-2000p",
    "BP (e16 @ 40)": GAMES / "final-full-bp-lr0.7-lrmin3-e16@40-vs-s11@32-300k-2000p",
    "BP の複製 (e16 @ 40)": GAMES / "noise-full-bp-lr0.7-lrmin3-rot-e16@40-vs-s11@32-300k-2000p",
    "experiment-009 の BP (full-e16 @ 48)": GAMES_009 / "final-full-e16@48-vs-s11@32-300k-2000p",
}
DIRECT = GAMES / "direct-full-pcalm-T4-gn-e12@24-vs-full-bp-lr0.7-lrmin3-e16@40-300k-2000p"
SMALL_REP = GAMES / "tune-s-bp-lr1-e1@48-vs-bp-rep-e1@48-300k"  # 小規模の複製の差 (レシピ vs 複製)


def done(d: Path) -> bool:
    return (d / "queue_done.json").exists() and (d / "summary.json").exists()


def pair_scores(d: Path) -> dict[str, float]:
    tot: dict[str, float] = {}
    cnt: dict[str, int] = {}
    for line in (d / "games.jsonl").read_text().splitlines():
        r = json.loads(line)
        tot[r["sfen"]] = tot.get(r["sfen"], 0.0) + r["cand_score"]
        cnt[r["sfen"]] = cnt.get(r["sfen"], 0) + 1
    return {k: v / 2 for k, v in tot.items() if cnt[k] == 2}


def elo(s: float) -> float:
    s = min(max(s, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / s - 1)


def diff_ci(
    a: dict[str, float], b: dict[str, float], seed: int = 0
) -> tuple[float, float, float, int]:
    keys = sorted(set(a) & set(b))
    n = len(keys)
    va, vb = [a[k] for k in keys], [b[k] for k in keys]
    point = elo(sum(va) / n) - elo(sum(vb) / n)
    rng = random.Random(seed)
    ds = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        ds.append(elo(sum(va[i] for i in idx) / n) - elo(sum(vb[i] for i in idx) / n))
    ds.sort()
    return point, ds[int(0.025 * (B - 1))], ds[int(0.975 * (B - 1))], n


def main() -> None:
    s = {k: json.loads((d / "summary.json").read_text()) for k, d in RUNS.items() if done(d)}
    ps = {k: pair_scores(d) for k, d in RUNS.items() if done(d)}
    print("| ネット | 水匠 11 比 Elo (2,000 ペア, 95% CI) |\n| --- | --- |")
    for k, v in s.items():
        lo, hi = v["elo_ci95"]
        print(f"| {k} | {v['elo']:+.1f} [{lo:+.1f}, {hi:+.1f}] |")
    print(
        "\n| 差 (同じ開始局面, ブートストラップ) | Elo の差 [95% CI] | 開始局面 |\n| --- | --- | --- |"
    )
    pairs = [
        ("PC-ALM (e12 @ 24 = 48 相当)", "BP (e16 @ 40)"),
        ("BP (e16 @ 40)", "experiment-009 の BP (full-e16 @ 48)"),
        ("BP の複製 (e16 @ 40)", "BP (e16 @ 40)"),
        ("PC-ALM (e12 @ 24 = 48 相当)", "experiment-009 の BP (full-e16 @ 48)"),
    ]
    out = {}
    for a, b in pairs:
        if a in ps and b in ps:
            p, lo, hi, n = diff_ci(ps[a], ps[b])
            out[(a, b)] = (p, lo, hi)
            print(f"| {a} − {b} | {p:+.1f} [{lo:+.1f}, {hi:+.1f}] | {n:,} |")
    if done(DIRECT):
        d = json.loads((DIRECT / "summary.json").read_text())
        dlo, dhi = d["elo_ci95"]
        rep = out.get(("BP の複製 (e16 @ 40)", "BP (e16 @ 40)"))
        if rep:
            n_val = max(abs(rep[0]), (rep[2] - rep[1]) / 2)
            n_src = f"本番規模の複製: |{rep[0]:+.1f}| と CI の半幅 {(rep[2] - rep[1]) / 2:.1f} の大きい方"
        else:
            sr = json.loads((SMALL_REP / "summary.json").read_text())
            n_val = max(abs(sr["elo"]), (dhi - dlo) / 2)
            n_src = f"暫定 (複製の対局待ち): 小規模の複製の差 |{sr['elo']:+.1f}| と直接対局の CI の半幅 {(dhi - dlo) / 2:.1f} の大きい方"
        if dlo > 0 and d["elo"] > n_val:
            verdict = "**PC-ALM の方が強い**"
        elif dhi < 0 and abs(d["elo"]) > n_val:
            verdict = "**PC-ALM の方が弱い**"
        else:
            verdict = "**差は検出できない**"
        print(
            f"\n直接対局 D = {d['elo']:+.1f} [{dlo:+.1f}, {dhi:+.1f}] (2,000 ペア, pentanomial {d['pentanomial']})、"
            f"N = {n_val:.1f} ({n_src}) → {verdict}"
        )


if __name__ == "__main__":
    main()
