#!/bin/bash
# CPU に対局の負荷をかける: 水匠 11 同士を 300k ノード/手・Threads 1 で W 並列 (match_nodes.py)。
# 1 ワーカー ≈ 1 コア。experiment-009 の対局・特徴量集計と同じ nice -n 10。
# 段階的に増やすときは名前を変えて追加で起動する (例: c1 = 16, c2 = 16 → 合計 32)。
# 棋譜は $STRESS_DIR/games/stress-<name>/ に出し、起動のたびに消す (ratings.md には書かない)。
#
#   bash experiments/009-data-scaling/stress/cpu_load.sh <workers> <name>   # 例: 16 c1
#   止める: bash experiments/009-data-scaling/stress/stop.sh c1
set -euo pipefail
source "$(dirname "$0")/common.sh"
W="${1:?ワーカー数}"
NAME="${2:?名前 (c1, c2, ...)}"
[[ "$W" =~ ^[0-9]+$ ]] && [ "$W" -ge 1 ] && [ "$W" -le 64 ] || { echo "error: workers は 1-64" >&2; exit 1; }
[[ "$NAME" =~ ^c[0-9a-z]+$ ]] || { echo "error: 名前は c で始める (例: c1)" >&2; exit 1; }
if alive "$NAME"; then
  echo "error: $NAME は動作中 (pgid $(pgid_of "$NAME"))" >&2
  exit 1
fi

GAMES="$STRESS_DIR/games"
[ "${DRY_RUN:-0}" = 1 ] || rm -rf "$GAMES/stress-$NAME"
launch "$NAME" nice -n 10 "$REPO/data/matchenv/bin/python" "$HERE/match_nodes.py" \
  --engine "$HOME/engines/009-finny-avx2/YaneuraOu-by-gcc" \
  --candidate-evaldir "$HOME/suisho11" \
  --baseline-evaldir "$HOME/suisho11" \
  --name "stress-$NAME" \
  --games-dir "$GAMES" \
  --nodes 300000 \
  --pairs 10000 \
  --workers "$W"
