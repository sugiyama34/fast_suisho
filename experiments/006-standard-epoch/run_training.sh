#!/bin/bash
# フェーズ b: standard-epoch 学習 (1 BulletOu-epoch ≒ 教師全 30 ファイル 1 周)。
#
#   bash experiments/006-standard-epoch/run_training.sh std [gpu_index]
#
# --superbatches 367: 367 × 39,976,960 = 146.7 億局面 ≒ 教師 1 周。
# LR step 減衰の周期も 1 standard-epoch に伸びる (「2 日」解釈の第一候補設定)。
# 16 standard-epoch ≒ 2350 億局面 ≒ RTX PRO 5000 で約 28 時間。
set -euo pipefail

ARM="${1:?arm を指定 (std)}"
GPU="${2:-0}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$REPO/data/bulletou/BulletOu/target/release/examples/bulletou"
SOJO="$REPO/data/teacher/sojo/train"
TEST_FILE="$REPO/data/teacher/floodgate/floodgate.hcpe"
LOG_DIR="$REPO/experiments/006-standard-epoch/logs"
mkdir -p "$LOG_DIR"

export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

[ "$ARM" = "std" ] || { echo "unknown arm: $ARM" >&2; exit 1; }

# 教師全 30 ファイルが揃っているか検証 (146.7 億局面)
N_FILES=$(ls "$SOJO"/dlsuisho_unique_0*.psv 2>/dev/null | wc -l)
[ "$N_FILES" -eq 30 ] || { echo "error: teacher files $N_FILES/30" >&2; exit 1; }

echo "[launch] arm=$ARM gpu=$GPU $(date)"
"$BIN" \
  --backend cuda-cpp \
  --cuda-cpp-device "$GPU" \
  --arch SFNN_halfka2_1024_7_64_k3k3 \
  --teacher "$SOJO" \
  --test-teacher "$TEST_FILE" \
  --test-positions 300000 \
  --positions-per-superbatch 40000000 \
  --superbatches 367 \
  --max-epochs 16 \
  --lr 0.000875 \
  --lr-min 0.000030 \
  --lr-schedule step \
  --optimizer ranger \
  --optimizer-weight-decay 0.0 \
  --save-rate 9999 \
  --validation-rate 4 \
  --output "$REPO/data/bulletou/checkpoints/006-std" \
  "${@:3}" 2>&1 | tee -a "$LOG_DIR/$ARM.log"
echo "[exit] arm=$ARM $(date)"
