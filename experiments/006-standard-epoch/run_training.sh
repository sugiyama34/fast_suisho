#!/bin/bash
# フェーズ b: superbatches スケール学習 (2026-07-28 やねうらお氏の実設定公開を反映)。
#
#   bash experiments/006-standard-epoch/run_training.sh <yane|yane20|std> [gpu_index]
#
#   yane  : --superbatches 108 × 16 epoch = 691 億局面 (氏の実設定の再現, GPU ~8.4h)
#   yane20: --superbatches 108 × 20 epoch = 864 億局面 (氏の「20 epochぐらい」指針, ~10.5h)
#   std   : --superbatches 367 × 16 epoch = 2350 億局面 (LR 周期 ≒ 教師 1 周。
#           「sb 上げればもっと強い」の検証, GPU ~28h)
set -euo pipefail

ARM="${1:?arm を指定 (yane|yane20|std)}"
GPU="${2:-0}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$REPO/data/bulletou/BulletOu/target/release/examples/bulletou"
SOJO="$REPO/data/teacher/sojo/train"
TEST_FILE="$REPO/data/teacher/floodgate/floodgate.hcpe"
LOG_DIR="$REPO/experiments/006-standard-epoch/logs"
mkdir -p "$LOG_DIR"

export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

OUT_ARM="$ARM"
EXTRA_FLAGS=()
case "$ARM" in
  yane)   SB=108; EPOCHS=16 ;;
  yane20) SB=108; EPOCHS=20 ;;
  # yane20 の checkpoint 0020 から累積 32 epoch まで継続学習。
  # resume 時の --max-epochs は「累積の上限」として解釈される (12 を渡すと
  # 既に 20 epoch 済みのため即終了する — 2026-07-29 実測)
  yane32) SB=108; EPOCHS=32; OUT_ARM="yane20"; EXTRA_FLAGS=(--resume) ;;
  std)    SB=367; EPOCHS=16 ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

# 教師全 30 ファイルが揃っているか検証 (146.7 億局面)
N_FILES=$(ls "$SOJO"/dlsuisho_unique_0*.psv 2>/dev/null | wc -l)
[ "$N_FILES" -eq 30 ] || { echo "error: teacher files $N_FILES/30" >&2; exit 1; }

echo "[launch] arm=$ARM sb=$SB gpu=$GPU $(date)"
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
  --save-rate 9999 \
  --validation-rate 4 \
  --output "$REPO/data/bulletou/checkpoints/006-$OUT_ARM" \
  ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
  "${@:3}" 2>&1 | tee -a "$LOG_DIR/$ARM.log"
echo "[exit] arm=$ARM $(date)"
