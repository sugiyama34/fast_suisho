"""games/<name>/games.jsonl を たややん氏の連続対局サマリ形式で集計する。

- レーティング差: 引き分け無効の勝率 p から 400*log10(p/(1-p))。CI は正規近似の
  二項 95% CI を Elo に変換
- 二項検定: 決着局のみ、H0 = 勝率 0.5 の両側 Z 検定
- 無効対局数: エンジン silent crash (issue #18) で破棄・補充されたペア試行数×2
  (log から数える。jsonl には有効局のみ記録されている)

使い方:
    uv run python experiments/007-wrm-factorizer/format_match_summary.py \
        007-wrm-ep20-vs-suisho11 --cand-name 007-wrm-ep20 --base-name Suisho11
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def elo(p: float) -> float:
    return 400.0 * math.log10(p / (1.0 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="games/<name>")
    ap.add_argument("--cand-name", default="candidate")
    ap.add_argument("--base-name", default="Suisho11")
    ap.add_argument("--log", default=None, help="無効対局数を数える log (任意)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(HERE / "games" / args.name / "games.jsonl")]
    cand_w = sum(1 for r in rows if r["cand_score"] == 1.0)
    cand_wb = sum(1 for r in rows if r["cand_score"] == 1.0 and r["cand_is_black"])
    base_w = sum(1 for r in rows if r["cand_score"] == 0.0)
    base_wb = sum(1 for r in rows if r["cand_score"] == 0.0 and not r["cand_is_black"])
    draws = sum(1 for r in rows if r["cand_score"] == 0.5)

    invalid = 0
    if args.log:
        pat = re.compile(r"attempt \d+ failed")
        in_section = False
        for line in open(args.log):
            if line.startswith("[measure] "):
                in_section = args.name.split("-vs-")[0] in line
            elif in_section and pat.search(line):
                invalid += 2  # 1 ペア試行 = 最大 2 局を破棄

    n = cand_w + base_w
    winner, loser = (
        (
            (args.base_name, base_w, base_wb, base_w - base_wb),
            (args.cand_name, cand_w, cand_wb, cand_w - cand_wb),
        )
        if base_w >= cand_w
        else (
            (args.cand_name, cand_w, cand_wb, cand_w - cand_wb),
            (args.base_name, base_w, base_wb, base_w - base_wb),
        )
    )
    p = winner[1] / n
    se = math.sqrt(p * (1 - p) / n)
    z = (winner[1] - n / 2) / math.sqrt(n / 4)

    print("連続対局終了")
    print()
    for name, w, wb, ww in (winner, loser):
        print(f"- {name}")
        print(f"  - 勝ち数: {w}")
        print(f"  - 勝ち数(先手): {wb}")
        print(f"  - 勝ち数(後手): {ww}")
    print(f"- 引き分け: {draws}")
    print(f"- 有効対局数: {len(rows)}")
    print(f"- 無効対局数: {invalid}")
    print("- レーティング差 (引き分け無効)")
    print(f"  - {elo(p):.2f} ({winner[0]} 優位)")
    print(f"  - 95% CI: [{elo(p - 1.96 * se):.1f}, {elo(p + 1.96 * se):.1f}]")
    print("- 二項検定")
    print(f"  - np > 5: {n * 0.5 > 5}")
    print(f"  - Z値: {z:.2f}")
    print(f"  - 有意水準5%: {abs(z) > 1.96}")
    print(f"  - 有意水準1%: {abs(z) > 2.576}")


if __name__ == "__main__":
    main()
