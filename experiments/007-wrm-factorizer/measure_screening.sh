#!/bin/bash
# フェーズ2 ステップ1: 5 arm × ep16/ep20 相当 (10 点) の Elo スクリーニング
# (固定 50 ペア、直列、vs Suisho 11)。
#
#   bash experiments/007-wrm-factorizer/measure_screening.sh
#
# 対局条件は experiment-005 フェーズ3 と同一。GPU 学習と同時に走らせないこと
# (CPU 競合で movetime 校正が崩れる)。測定順は情報価値順:
# ctrl (−104 再現確認) → wrm / nofact / both / lr1shot。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="$REPO/experiments/005-bulletou-sojo/match_runner.py"
GAMES_DIR="$REPO/experiments/007-wrm-factorizer/games"
PY="$REPO/data/matchenv/bin/python"
PAIRS=50

# arm:checkpoint の測定列。lr1shot の 0016/0020 は sb1728/sb2160 (= 位置数で
# 他 arm の ep16/ep20 と同一グリッド)。
# 先頭の 006-yane20:0016 は再アンカー点: エンジンバイナリが 006 測定時
# (87857ca8) から再ビルドされている (4881fcf4, 2026-07-31) ため、−104 の
# 基準 net を新エンジン条件で測り直し、006 カーブとの橋渡しに使う
POINTS="
006-reanchor:SPECIAL
ctrl:0016
ctrl:0020
wrm:0016
wrm:0020
nofact:0016
nofact:0020
both:0016
both:0020
lr1shot:0016
lr1shot:0020
"

for point in $POINTS; do
  arm="${point%%:*}"; ep="${point##*:}"
  if [ "$point" = "006-reanchor:SPECIAL" ]; then
    dir="$REPO/data/bulletou/checkpoints/006-yane20/0016"
    name="006-yane-ep16-reanchor-vs-suisho11"
  else
    dir="$REPO/data/bulletou/checkpoints/007-$arm/$ep"
    name="007-${arm}-ep${ep#00}-vs-suisho11"
  fi
  if [ ! -f "$dir/nn.bin" ]; then
    echo "[skip] $dir/nn.bin がまだ無い"
    continue
  fi
  if [ -f "$GAMES_DIR/$name/summary.json" ]; then
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
for d in "$GAMES_DIR"/007-*/summary.json; do
  echo "== $d"; cat "$d"; echo
done
