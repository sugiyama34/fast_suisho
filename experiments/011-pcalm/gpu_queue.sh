#!/bin/bash
# experiment-011: 1 枚の GPU (MIG スライス) で arm を順に学習する (train_supervised.py 経由)。
#
#   nohup setsid bash experiments/011-pcalm/gpu_queue.sh <gpu> <arm> [<arm> ...] \
#       >> /mnt/D/sugiyama/011/queue-<名前>.log 2>&1 < /dev/null &
#
# - 次の arm を始める前に /mnt/D/sugiyama/011/hold-<gpu> があれば、消えるまで待つ
#   (BENCH などでスライスを空けたいとき: touch で予約 → 今の run が終わると止まる → rm で再開)
# - 既に checkpoint (011-<arm>/NNNN/nn.bin, NNNN = 既定 epoch) がある arm は飛ばす
# - arm が失敗したら次へ進む (ログに [fail] を残す)
set -uo pipefail
GPU="${1:?gpu}"; shift
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HOLD="/mnt/D/sugiyama/011/hold-$GPU"
for ARM in "$@"; do
  while [ -e "$HOLD" ]; do sleep 60; done
  EP=$(python3 "$REPO/experiments/011-pcalm/arms.py" "$ARM" | python3 -c "import json,sys;print(json.load(sys.stdin)['epochs'])") || { echo "[fail] $ARM: arm 名"; continue; }
  if [ -f "$REPO/data/bulletou/checkpoints/011-$ARM/$(printf %04d "$EP")/nn.bin" ]; then
    echo "[skip] $ARM (checkpoint あり) $(date '+%F %T')"; continue
  fi
  echo "[start] $ARM gpu=$GPU $(date '+%F %T')"
  if (cd "$REPO" && "$REPO/.venv/bin/python" experiments/011-pcalm/train_supervised.py --arm "$ARM" --gpu "$GPU"); then
    echo "[done] $ARM $(date '+%F %T')"
  else
    echo "[fail] $ARM rc=$? $(date '+%F %T')"
  fi
done
echo "[queue end] gpu=$GPU $(date '+%F %T')"
