#!/bin/bash
# experiment-009: 教師データ量スケーリング。全 arm 共通レシピ = experiment-006 の yane 系
# (sigmoid-MSE, WRM なし, sb=108 × 40M, step LR 0.000875→0.00003, ranger)。
# arm ごとに変わるのは --teacher のファイル集合だけ (subsets.py)。
#
#   bash experiments/009-data-scaling/run_training.sh <arm> <gpu> [epochs]
#
#   <gpu>: CUDA_VISIBLE_DEVICES にそのまま渡す値 (GPU 番号 "0" や MIG/GPU の UUID)。
#     プロセス内では常に device 0 として見える。使ってよい GPU はサーバーごとの制約に従う
#     (docs/SETUP.md §9。例: kajiki は MIG device 2/3 のみ)。
#     ALLOWED_GPUS (空白区切り) を設定すると、それ以外の値を拒否する
#   epochs: 全 arm で同じ値を使うこと (既定は EPOCHS_DEFAULT)。--max-epochs は resume
#     シグネチャに含まれないので、途中で止めた run を後から大きい値で再開して延長できる
#
#   SMOKE=1 を付けると sb=2 × epochs で動作確認だけ行う (出力は 009-<arm>-smoke/)。
#     例: SMOKE=1 bash experiments/009-data-scaling/run_training.sh p10 0 1
#
# checkpoint は各 epoch 末 (LR 最小点) に data/bulletou/checkpoints/009-<arm>/NNNN/ に出る
# (--save-rate 9999 + 既定の save-epoch-end → NNNN = epoch 番号)。
set -euo pipefail

ARM="${1:?arm を指定 (subsets.py の一覧)}"
GPU="${2:?GPU を指定 (CUDA_VISIBLE_DEVICES の値)}"
EPOCHS_DEFAULT=20
EPOCHS="${3:-$EPOCHS_DEFAULT}"
SB=108
OUT="009-$ARM"
if [ "${SMOKE:-0}" = 1 ]; then SB=2; OUT="009-$ARM-smoke"; fi
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$REPO/experiments/009-data-scaling"
BIN="$REPO/data/bulletou/BulletOu/target/release/examples/bulletou"
TEST_FILE="$REPO/data/teacher/floodgate/floodgate.hcpe"
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"

if [ -n "${ALLOWED_GPUS:-}" ] && [[ " $ALLOWED_GPUS " != *" $GPU "* ]]; then
  echo "error: 許可されていない GPU: $GPU (ALLOWED_GPUS=$ALLOWED_GPUS)" >&2; exit 1
fi

TEACHER="$("$REPO/.venv/bin/python" "$HERE/subsets.py" --teacher "$ARM")"
# 全ファイルが揃っていてサイズが正しいことを確認 (ダウンロード途中のファイルを掴まない)
IFS=',' read -r -a FILES <<< "$TEACHER"
for f in "${FILES[@]}"; do
  expected=19558599240
  [[ "$f" == *_029.psv ]] && expected=19558599520
  have=$(stat -L -c%s "$f" 2>/dev/null || echo 0)
  [ "$have" -eq "$expected" ] || { echo "error: $f size=$have expected=$expected" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[launch] arm=$ARM files=${#FILES[@]} sb=$SB epochs=$EPOCHS gpu=$GPU out=$OUT $(date)"
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
  --lr 0.000875 \
  --lr-min 0.000030 \
  --lr-schedule step \
  --optimizer ranger \
  --optimizer-weight-decay 0.0 \
  --save-rate 9999 \
  --validation-rate 4 \
  --threads 8 \
  --output "$REPO/data/bulletou/checkpoints/$OUT" \
  "${@:4}" 2>&1 | tee -a "$LOG_DIR/$ARM.log"
echo "[exit] arm=$ARM $(date)"
