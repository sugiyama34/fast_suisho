#!/bin/bash
# 先行 arm の supervisor 終了を待ってから、指定 arm を同一 GPU で順次学習する。
# フェーズ1 の 5 arm を 2 GPU で無人消化するためのキュー。
#
# 待機は PID ではなくコマンドライン pattern で行う (setsid 経由の起動では
# 呼び出し側が supervisor の実 PID を確実に得られないため — 2026-07-31 の
# 誤発火の教訓)。
#
#   bash queue_arms.sh <wait_arm> <gpu> <arm> [arm...]
set -euo pipefail

WAIT_ARM="${1:?待機対象の先行 arm 名}"
GPU="${2:?GPU index}"
shift 2
[ $# -ge 1 ] || { echo "error: arm を 1 つ以上指定" >&2; exit 1; }

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$REPO/experiments/007-wrm-factorizer/logs"
PATTERN="train_supervised.py --arm $WAIT_ARM"

# 起動レースを避ける: 先行 supervisor がまだ見えないうちは待たない
if ! pgrep -f "$PATTERN" > /dev/null; then
  echo "error: 先行 supervisor ($PATTERN) が見つからない" >&2
  exit 1
fi

echo "[queue gpu=$GPU] waiting for '$PATTERN'; then: $* ($(date))"
while pgrep -f "$PATTERN" > /dev/null; do sleep 30; done
echo "[queue gpu=$GPU] '$PATTERN' finished ($(date))"

cd "$REPO"
for ARM in "$@"; do
  echo "[queue gpu=$GPU] starting arm=$ARM ($(date))"
  uv run python experiments/007-wrm-factorizer/train_supervised.py \
    --arm "$ARM" --gpu "$GPU" > "$LOG_DIR/supervisor-$ARM.log" 2>&1 || {
    echo "[queue gpu=$GPU] arm=$ARM FAILED ($(date))" >&2
    exit 1
  }
  echo "[queue gpu=$GPU] arm=$ARM finished ($(date))"
done
echo "[queue gpu=$GPU] all done ($(date))"
