"""full の FV_SCALE を full-e8 の格子対局 (vs 水匠 11@32) から自動で決め、待たせているジョブを解放する。

- 待つもの: games/fv-full-e8@{32,40,48,56,64}-vs-s11@32-300k/queue_done.json (5 本すべて)
- 決め方: Elo (A = full-e8@FV) が最大の FV_SCALE (2026-09-30 ユーザー決定: 選択バイアスは許容)
- やること (match_queue.toml を一時ファイル経由で置き換える):
  1. full-e12 のアブレーション (名前が full-e12-zs- で始まるジョブ, 相手は水匠 11@32) に a_fv = 最良値を入れる
     (アブレーションしたネットは親の FV_SCALE をそのまま使う)
  2. full-e12 / e16 / e20 の FV_SCALE 格子 (最良値 ± 8 の 3 点, vs 水匠 11@32, 400 ペア) を追加
  3. /mnt/nvme1/sugiyama/fv/full.decided に最良値を書く (e12 アブレーションの requires が解ける)

使い方: nohup setsid data/matchenv/bin/python experiments/009-data-scaling/fv_decide.py &
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAMES = HERE / "games"
TOML = HERE / "match_queue.toml"
FLAG = Path("/mnt/nvme1/sugiyama/fv/full.decided")
GRID = (32, 40, 48, 56, 64)


def name(ep: int, fv: int) -> str:
    return f"fv-full-e{ep}@{fv}-vs-s11@32-300k"


def main() -> None:
    while not all((GAMES / name(8, fv) / "queue_done.json").exists() for fv in GRID):
        time.sleep(120)
    elo = {fv: json.loads((GAMES / name(8, fv) / "summary.json").read_text())["elo"] for fv in GRID}
    best = max(elo, key=elo.get)
    print(f"{time.strftime('%F %T')} e8 grid: {elo} -> best FV_SCALE {best}", flush=True)

    s = TOML.read_text()
    blocks = re.split(r"(?m)^\[\[job\]\]\n", s)
    out = [blocks[0]]
    for b in blocks[1:]:
        if re.search(r'(?m)^name = "full-e12-zs-', b) and not re.search(r"(?m)^a_fv", b):
            b = re.sub(r"(?m)^(b = \"suisho11\"\n)", rf"a_fv = {best}\n\1", b)
        out.append("[[job]]\n" + b)
    s = "".join(out)
    s += (
        f"\n# 7. fv_decide.py が自動追加 ({time.strftime('%F %T')}): full-e8 の格子で最良の FV_SCALE = {best}"
        f" (Elo {elo})。\n#    e12/e16/e20 は最良値 ± 8 の 3 点\n"
    )
    for ep, pr in ((12, 20), (16, 30), (20, 30)):
        for fv in (best, best - 8, best + 8):
            s += (
                f'[[job]]\nname = "{name(ep, fv)}"\na = "009-full/{ep:04d}"\na_fv = {fv}\n'
                f'b = "suisho11"\nb_fv = 32\nnodes = 300000\npairs = 400\npriority = {pr}\n'
                f'note = "FV_SCALE 調整 (学習ネット, e8 の最良値 ± 8)"\n\n'
            )
    tmp = TOML.with_suffix(".toml.tmp")
    tmp.write_text(s)
    os.replace(tmp, TOML)
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(f"{best}\n")
    print(f"{time.strftime('%F %T')} released e12 ablations with FV_SCALE {best}", flush=True)


if __name__ == "__main__":
    main()
