#!/bin/bash
# フェーズ3 測定: wrmshot (WRM × 単発アニール) の ep16/ep20 相当を 50 ペアで
# スクリーニング。GPU 学習と同時に走らせないこと。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="$REPO/experiments/005-bulletou-sojo/match_runner.py"
GAMES_DIR="$REPO/experiments/007-wrm-factorizer/games"
PY="$REPO/data/matchenv/bin/python"
PAIRS="${PAIRS:-50}"

for ep in 0016 0020; do
  dir="$REPO/data/bulletou/checkpoints/007-wrmshot/$ep"
  name="007-wrmshot-ep${ep#00}-vs-suisho11"
  [ -f "$dir/nn.bin" ] || { echo "[skip] $dir/nn.bin なし"; continue; }
  if [ -f "$GAMES_DIR/$name/summary.json" ] && \
     [ "$(python3 -c "import json; print(json.load(open('$GAMES_DIR/$name/summary.json'))['pairs'])")" -ge "$PAIRS" ]; then
    echo "[skip] $name は測定済み"
    continue
  fi
  echo "[measure] $name ($(date +%H:%M:%S))"
  "$PY" "$RUNNER" \
    --candidate-evaldir "$dir" \
    --name "$name" \
    --games-dir "$GAMES_DIR" \
    --max-pairs "$PAIRS" \
    --fixed-pairs
done
echo "[all done] $(date)"
