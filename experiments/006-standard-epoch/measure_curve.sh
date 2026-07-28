#!/bin/bash
# フェーズ a/c: checkpoint 系列の Elo カーブ測定 (固定 50 ペア、直列)。
#
#   bash experiments/006-standard-epoch/measure_curve.sh a       # 005 (sb=12) 系列
#   bash experiments/006-standard-epoch/measure_curve.sh c-yane  # 006 arm yane (sb=108) 系列
#   bash experiments/006-standard-epoch/measure_curve.sh c-std   # 006 arm std (sb=367) 系列
#
# 対局条件は experiment-005 フェーズ3 と同一。GPU 学習と同時に走らせないこと
# (CPU 競合で movetime 校正が崩れる)。
set -euo pipefail

PHASE="${1:?phase を指定 (a|c-yane|c-std)}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="$REPO/experiments/005-bulletou-sojo/match_runner.py"
GAMES_DIR="$REPO/experiments/006-standard-epoch/games"
PY="$REPO/data/matchenv/bin/python"
PAIRS=50

case "$PHASE" in
  a)      CKPT_BASE="$REPO/data/bulletou/checkpoints/005-main"; PREFIX="bu-ep" ;;
  c-yane) CKPT_BASE="$REPO/data/bulletou/checkpoints/006-yane"; PREFIX="yane-ep" ;;
  c-std)  CKPT_BASE="$REPO/data/bulletou/checkpoints/006-std";  PREFIX="std-ep" ;;
  *) echo "unknown phase: $PHASE" >&2; exit 1 ;;
esac

for n in 1 2 4 8 16; do
  dir=$(printf '%s/%04d' "$CKPT_BASE" "$n")
  if [ ! -f "$dir/nn.bin" ]; then
    echo "[skip] $dir/nn.bin がまだ無い"
    continue
  fi
  name="${PREFIX}${n}-vs-suisho11"
  echo "[measure] $name ($(date +%H:%M:%S))"
  "$PY" "$RUNNER" \
    --candidate-evaldir "$dir" \
    --name "$name" \
    --games-dir "$GAMES_DIR" \
    --max-pairs "$PAIRS" \
    --fixed-pairs
done
echo "[all done] $(date)"
for d in "$GAMES_DIR"/${PREFIX}*/summary.json; do
  echo "== $d"; cat "$d"; echo
done
