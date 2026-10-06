#!/bin/bash
# experiment-011: 学習 1 本。arm 名 → arms.py → BulletOu。
# レシピは experiment-009 の run_training.sh と同一 (sigmoid-MSE, sb=108 × 40M, step LR, ranger)。
# arm ごとに変わるのは教師ファイル (と順序)・LR の最大値/最小値・勾配の計算法 (--credit) だけ。
#
#   bash experiments/011-pcalm/run_training.sh <arm> <gpu> [epochs]
#
#   <arm>: arms.py の文法 (例 s-bp-lr1, s-pcalm-T4-eta0.1, f-bp-rot)
#   <gpu>: CUDA_VISIBLE_DEVICES の値。kajiki では MIG device 2 / 3 の UUID だけ (ALLOWED_GPUS で強制)
#   epochs: 省略時は arm の既定 (小規模 1, 本番 20)。--max-epochs は resume シグネチャに含まれないので延長できる
#
#   SMOKE=1: sb=2 で動作確認 (出力 011-<arm>-smoke/)。BENCH=1: sb=16 × 1 epoch (出力 011-<arm>-bench/)
#   OUT_TAG=<tag>: 出力とログ名の接尾辞
#   BIN_BP / BIN_PCALM: トレーナの場所。bp は改造前の BulletOu、pcalm / pc は改造版 (BulletOu-pcalm)
set -euo pipefail

ARM="${1:?arm を指定 (arms.py の文法)}"
GPU="${2:?GPU を指定 (CUDA_VISIBLE_DEVICES の値)}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$REPO/experiments/011-pcalm"
ARM_VARS="$(python3 "$HERE/arms.py" --shell "$ARM")" || exit 1
eval "$ARM_VARS"
EPOCHS="${3:-$EPOCHS_DEFAULT}"
SB=108
OUT="011-$ARM"
if [ "${SMOKE:-0}" = 1 ]; then SB=2; OUT="011-$ARM-smoke"; fi
if [ "${BENCH:-0}" = 1 ]; then SB=16; EPOCHS=1; OUT="011-$ARM-bench"; fi
OUT="$OUT${OUT_TAG:+-$OUT_TAG}"
BIN_BP="${BIN_BP:-$REPO/data/bulletou/BulletOu/target/release/examples/bulletou}"
BIN_PCALM="${BIN_PCALM:-$REPO/data/bulletou/BulletOu-pcalm/target/release/examples/bulletou}"
if [ "$METHOD" = bp ]; then BIN="$BIN_BP"; else BIN="$BIN_PCALM"; fi
TEST_FILE="$REPO/data/teacher/floodgate/floodgate.hcpe"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"

# kajiki では GPU 0 の MIG device 2 / 3 しか使わない (他の 2 つは他ユーザー)
if [ -z "${ALLOWED_GPUS+x}" ] && [ "$(hostname)" = kajiki ]; then
  ALLOWED_GPUS="MIG-a4e39a6f-6d5d-5203-881e-f4ae3bd988de MIG-0d33d072-e48b-5dd1-8ff1-3d8b3ce6056b"
fi
if [ -n "${ALLOWED_GPUS:-}" ] && [[ " $ALLOWED_GPUS " != *" $GPU "* ]]; then
  echo "error: 許可されていない GPU: $GPU (ALLOWED_GPUS=$ALLOWED_GPUS)" >&2; exit 1
fi

# 全ファイルが揃っていてサイズが正しいことを確認 (ダウンロード途中のファイルを掴まない)
IFS=',' read -r -a FILES <<< "$TEACHER"
for f in "${FILES[@]}"; do
  expected=19558599240
  [[ "$f" == *_029.psv ]] && expected=19558599520
  have=$(stat -L -c%s "$f" 2>/dev/null || echo 0)
  [ "$have" -eq "$expected" ] || { echo "error: $f size=$have expected=$expected" >&2; exit 1; }
done
[ -x "$BIN" ] || { echo "error: トレーナが無い: $BIN" >&2; exit 1; }

if [ "${SMOKE:-0}" = 1 ] || [ "${BENCH:-0}" = 1 ]; then
  rm -rf "$REPO/data/bulletou/checkpoints/$OUT"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[launch] arm=$ARM method=$METHOD files=${#FILES[@]} sb=$SB epochs=$EPOCHS lr=$LR lr_min=$LR_MIN credit=(${CREDIT_ARGS[*]:-bp}) gpu=$GPU out=$OUT bin=$BIN sha256=$(sha256sum "$BIN" | cut -c1-16) $(date)"
"$BIN" \
  --backend cuda-cpp \
  --cuda-cpp-device 0 \
  --arch SFNN_halfka2_1024_7_64_k3k3 \
  --teacher "$TEACHER" \
  --test-teacher "$TEST_FILE" \
  --test-positions 300000 \
  --test-seed 20260928 \
  --positions-per-superbatch 40000000 \
  --superbatches "$SB" \
  --max-epochs "$EPOCHS" \
  --lr "$LR" \
  --lr-min "$LR_MIN" \
  --lr-schedule step \
  --optimizer ranger \
  --optimizer-weight-decay 0.0 \
  --save-rate 9999 \
  --validation-rate 4 \
  --threads 8 \
  --output "$REPO/data/bulletou/checkpoints/$OUT" \
  "${CREDIT_ARGS[@]}" \
  "${@:4}" 2>&1 | while IFS= read -r line; do printf '%(%F %T)T %s\n' -1 "$line"; done \
  | tee -a "$LOG_DIR/$OUT.log"
echo "[exit] arm=$ARM $(date)"
