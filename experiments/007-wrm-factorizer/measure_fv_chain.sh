#!/bin/bash
# FV_SCALE 補正後の再測定チェーン (2026-08-04, たややん氏指摘対応)。直列実行。
#
# 1. repro   : wrm20@28  vs Suisho@40 (200 ペア) — たややん氏計測 (−59) の再現試行
# 2. grid    : wrm20@{22,25,32} vs Suisho@40 (各 100 ペア) — 強度ベースの FV 微調整
# 3. ctrl    : ctrl20@52 vs Suisho@40 (200 ペア) — 適正スケール同士の基準線
# 4. direct  : wrm20@25  vs ctrl20@52 (200 ペア) — 純粋な loss 効果の分離
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RUNNER="$REPO/experiments/005-bulletou-sojo/match_runner.py"
GAMES_DIR="$REPO/experiments/007-wrm-factorizer/games"
PY="$REPO/data/matchenv/bin/python"
WRM="$REPO/data/bulletou/checkpoints/007-wrm/0020"
CTRL="$REPO/data/bulletou/checkpoints/007-ctrl/0020"

run() { # name cand_dir cand_fv base_dir base_fv pairs
  local name="$1" cand="$2" cfv="$3" base="$4" bfv="$5" pairs="$6"
  if [ -f "$GAMES_DIR/$name/summary.json" ] && \
     [ "$(python3 -c "import json; print(json.load(open('$GAMES_DIR/$name/summary.json'))['pairs'])")" -ge "$pairs" ]; then
    echo "[skip] $name は測定済み"; return
  fi
  echo "[measure] $name ($(date +%H:%M:%S))"
  "$PY" "$RUNNER" \
    --candidate-evaldir "$cand" --cand-fv-scale "$cfv" \
    --baseline-evaldir "$base" --base-fv-scale "$bfv" \
    --name "$name" --games-dir "$GAMES_DIR" \
    --max-pairs "$pairs" --fixed-pairs
}

run fv-wrm20s28-vs-suisho40   "$WRM" 28 /home/sugiyama/suisho11 40 200
run fv-wrm20s22-vs-suisho40   "$WRM" 22 /home/sugiyama/suisho11 40 100
run fv-wrm20s25-vs-suisho40   "$WRM" 25 /home/sugiyama/suisho11 40 100
run fv-wrm20s32-vs-suisho40   "$WRM" 32 /home/sugiyama/suisho11 40 100
run fv-ctrl20s52-vs-suisho40  "$CTRL" 52 /home/sugiyama/suisho11 40 200
run fv-wrm20s25-vs-ctrl20s52  "$WRM" 25 "$CTRL" 52 200
echo "[chain all done] $(date)"
