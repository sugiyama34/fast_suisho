#!/bin/bash
# GPU 1 枚に BulletOu の学習負荷をかける。引数は experiment-009 の run_training.sh と同じ
# (arm p10, sb=108 × 40M, 20 epoch = 止めるまで約 21 時間続く)。1 GPU に 1 run。
# 出力は $STRESS_DIR/gpu/g<N>/ で、起動のたびに消す (experiment-009 の checkpoint には触れない)。
#
#   bash experiments/009-data-scaling/stress/gpu_load.sh <gpu> [<gpu> ...]   # 例: 0 / 0 1 3
#   止める: bash experiments/009-data-scaling/stress/stop.sh g0
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || { echo "usage: $0 <gpu 0-4> [...]" >&2; exit 1; }

CUDA_HOME="${CUDA_HOME:-/mnt/nvme1/sugiyama/cuda-12.8}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
BIN="$REPO/data/bulletou/BulletOu/target/release/examples/bulletou"
TEACHER="$("$REPO/.venv/bin/python" "$HERE/subsets.py" --teacher p10)"

for G in "$@"; do
  [[ "$G" =~ ^[0-4]$ ]] || { echo "error: GPU は 0-4 (指定: $G)" >&2; exit 1; }
  NAME="g$G"
  if alive "$NAME"; then
    echo "skip: $NAME は動作中 (pgid $(pgid_of "$NAME"))"
    continue
  fi
  OUT="$STRESS_DIR/gpu/$NAME"
  [ "${DRY_RUN:-0}" = 1 ] || rm -rf "$OUT"
  CUDA_VISIBLE_DEVICES="$G" launch "$NAME" "$BIN" \
    --backend cuda-cpp \
    --cuda-cpp-device 0 \
    --arch SFNN_halfka2_1024_7_64_k3k3 \
    --teacher "$TEACHER" \
    --test-teacher "$REPO/data/teacher/floodgate/floodgate.hcpe" \
    --test-positions 300000 \
    --test-seed 20260928 \
    --positions-per-superbatch 40000000 \
    --superbatches 108 \
    --max-epochs 20 \
    --lr 0.000875 \
    --lr-min 0.000030 \
    --lr-schedule step \
    --optimizer ranger \
    --optimizer-weight-decay 0.0 \
    --save-rate 9999 \
    --validation-rate 4 \
    --threads 8 \
    --output "$OUT"
done
