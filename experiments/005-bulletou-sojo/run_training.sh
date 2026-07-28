#!/bin/bash
# BulletOu 学習ランチャ (実端末で実行する — Claude Code セッション内には GPU デバイスがない)
#
#   使い方:  bash experiments/005-bulletou-sojo/run_training.sh <arm> [gpu_index]
#     arm = smoke | main | var-b | var-c
#       smoke : 2M 局面 × 2 sb × 1 epoch のパイプライン確認 (数分)
#       main  : やねうら氏レシピ本番 (40M × 12 sb × 最大 16 epoch)
#       var-b : main と同一だが教師ファイル順を 8 ファイル回転 (run 間分散の推定用)
#       var-c : 同じく 4 ファイル回転 (分散推定の 3 点目)
#     gpu_index 省略時: smoke/main=0, var-b=1, var-c=2
#
#   長時間 run は tmux / nohup 推奨:
#     tmux new -s bulletou-main 'bash experiments/005-bulletou-sojo/run_training.sh main 0'
set -eu

ARM="${1:?arm を指定 (smoke|main|var-b)}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$REPO/data/bulletou/BulletOu/target/release/examples/bulletou"
SOJO="$REPO/data/teacher/sojo"
# 検証セット: 山岡氏の floodgate 評価用データセット (2017-2018, R3500+, 856,923 局面)。
# やねうら氏の私有 test hcpe (fg2021 系) に最も近い公開代替 (2026-07-27 ユーザー指定)
TEST_FILE="$REPO/data/teacher/floodgate/floodgate.hcpe"
LOG_DIR="$REPO/experiments/005-bulletou-sojo/logs"
mkdir -p "$LOG_DIR"

export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 共通ハイパーパラメータ (やねうら氏共有コマンドの値)
COMMON=(
  --backend cuda-cpp
  --arch SFNN_halfka2_1024_7_64_k3k3
  --lr 0.000875
  --lr-min 0.000030
  --lr-schedule step
  --optimizer ranger
  --optimizer-weight-decay 0.0
  --save-rate 9999
  --validation-rate 1
  --test-teacher "$TEST_FILE"
)

train_list() {  # $1=先頭に持ってくるファイル番号 (回転)。001-016 を順に並べる
  local start="$1" out="" i n
  for i in $(seq 0 15); do
    n=$(printf '%03d' $(( (start - 1 + i) % 16 + 1 )))
    out="${out:+$out,}$SOJO/train/dlsuisho_unique_${n}.psv"
  done
  echo "$out"
}

case "$ARM" in
  smoke)
    GPU="${2:-0}"
    ARGS=(
      --teacher "$TEST_FILE"
      --test-positions 50000
      --positions-per-superbatch 2000000
      --superbatches 2
      --max-epochs 1
      --output "$REPO/data/bulletou/checkpoints/005-smoke"
      --cuda-cpp-device "$GPU"
    )
    ;;
  main)
    GPU="${2:-0}"
    ARGS=(
      --teacher "$(train_list 1)"
      --test-positions 300000
      --positions-per-superbatch 40000000
      --superbatches 12
      --max-epochs 16
      --output "$REPO/data/bulletou/checkpoints/005-main"
      --cuda-cpp-device "$GPU"
    )
    ;;
  var-b)
    GPU="${2:-1}"
    ARGS=(
      --teacher "$(train_list 9)"
      --test-positions 300000
      --positions-per-superbatch 40000000
      --superbatches 12
      --max-epochs 16
      --output "$REPO/data/bulletou/checkpoints/005-var-b"
      --cuda-cpp-device "$GPU"
    )
    ;;
  var-c)
    GPU="${2:-2}"
    ARGS=(
      --teacher "$(train_list 5)"
      --test-positions 300000
      --positions-per-superbatch 40000000
      --superbatches 12
      --max-epochs 16
      --output "$REPO/data/bulletou/checkpoints/005-var-c"
      --cuda-cpp-device "$GPU"
    )
    ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

echo "[launch] arm=$ARM gpu=$GPU $(date)"
"$BIN" "${COMMON[@]}" "${ARGS[@]}" "${@:3}" 2>&1 | tee -a "$LOG_DIR/$ARM.log"
echo "[exit] arm=$ARM rc=${PIPESTATUS[0]} $(date)"
