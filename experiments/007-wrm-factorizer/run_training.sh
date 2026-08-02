#!/bin/bash
# experiment-007 フェーズ1: やねうらお氏 3 提案 (WRM loss / factorizer 無効化 /
# LR スケジュール単発化) の 5 arm 学習。BulletOu 2a8e5ed (+ sm_120 patch) が前提。
#
#   bash experiments/007-wrm-factorizer/run_training.sh <ctrl|wrm|nofact|both|lr1shot> [gpu_index]
#
#   ctrl    : 006 レシピ再現 @ 新コミット (sb=108 × 20ep = 864 億局面, GPU ~10.5h)
#   wrm     : + --win-rate-model --loss-pow-exp 2.5
#   nofact  : + --sfnn-factorizer none
#   both    : wrm + nofact
#   lr1shot : sb=2160 × 1ep (同 864 億局面)。warm restart なしの単発アニール。
#             108 sb ごとに checkpoint 保存 (他 arm の ep 境界と同一グリッド)
#   wrmshot : フェーズ3 勝者併用 arm = lr1shot スケジュール + WRM loss
set -euo pipefail

ARM="${1:?arm を指定 (ctrl|wrm|nofact|both|lr1shot|wrmshot)}"
GPU="${2:-0}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$REPO/data/bulletou/BulletOu/target/release/examples/bulletou"
SOJO="$REPO/data/teacher/sojo/train"
TEST_FILE="$REPO/data/teacher/floodgate/floodgate.hcpe"
LOG_DIR="$REPO/experiments/007-wrm-factorizer/logs"
mkdir -p "$LOG_DIR"

export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SB=108; EPOCHS=20; SAVE_RATE=9999
EXTRA_FLAGS=()
case "$ARM" in
  ctrl)    ;;
  wrm)     EXTRA_FLAGS=(--win-rate-model --loss-pow-exp 2.5) ;;
  nofact)  EXTRA_FLAGS=(--sfnn-factorizer none) ;;
  both)    EXTRA_FLAGS=(--win-rate-model --loss-pow-exp 2.5 --sfnn-factorizer none) ;;
  lr1shot) SB=2160; EPOCHS=1; SAVE_RATE=108 ;;
  wrmshot) SB=2160; EPOCHS=1; SAVE_RATE=108
           EXTRA_FLAGS=(--win-rate-model --loss-pow-exp 2.5) ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

# 教師全 30 ファイルが揃っているか検証 (146.7 億局面)
N_FILES=$(ls "$SOJO"/dlsuisho_unique_0*.psv 2>/dev/null | wc -l)
[ "$N_FILES" -eq 30 ] || { echo "error: teacher files $N_FILES/30" >&2; exit 1; }

echo "[launch] arm=$ARM sb=$SB epochs=$EPOCHS gpu=$GPU $(date)"
"$BIN" \
  --backend cuda-cpp \
  --cuda-cpp-device "$GPU" \
  --arch SFNN_halfka2_1024_7_64_k3k3 \
  --teacher "$SOJO" \
  --test-teacher "$TEST_FILE" \
  --test-positions 300000 \
  --positions-per-superbatch 40000000 \
  --superbatches "$SB" \
  --max-epochs "$EPOCHS" \
  --lr 0.000875 \
  --lr-min 0.000030 \
  --lr-schedule step \
  --optimizer ranger \
  --optimizer-weight-decay 0.0 \
  --save-rate "$SAVE_RATE" \
  --validation-rate 4 \
  --output "$REPO/data/bulletou/checkpoints/007-$ARM" \
  ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
  "${@:3}" 2>&1 | tee -a "$LOG_DIR/$ARM.log"
echo "[exit] arm=$ARM $(date)"
