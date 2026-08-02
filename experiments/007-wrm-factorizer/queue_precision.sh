#!/bin/bash
# フェーズ2 ステップ2: ctrl 再測定の終了を待ち、wrm ep16/ep20 を 200 ペアへ拡張する
# (match_runner の resume 機能で既存 50 ペアに +150 ペア追記。book は 500 局面あるので
# 51 ペア目以降は未使用の開始局面が使われる)。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="$REPO/experiments/005-bulletou-sojo/match_runner.py"
GAMES_DIR="$REPO/experiments/007-wrm-factorizer/games"
PY="$REPO/data/matchenv/bin/python"

PATTERN="007-ctrl-ep16-remeasure"
echo "[queue precision] waiting for '$PATTERN' ($(date))"
while pgrep -f "$PATTERN" > /dev/null; do sleep 30; done
echo "[queue precision] remeasure finished ($(date))"

for point in 007-wrm-ep16 007-wrm-ep20; do
  ep="${point##*-}"
  dir="$REPO/data/bulletou/checkpoints/007-wrm/00${ep#ep}"
  echo "[measure] $point precision 200 pairs ($(date +%H:%M:%S))"
  "$PY" "$RUNNER" \
    --candidate-evaldir "$dir" \
    --name "$point-vs-suisho11" \
    --games-dir "$GAMES_DIR" \
    --max-pairs 200 \
    --fixed-pairs
done
echo "[queue precision all done] $(date)"
