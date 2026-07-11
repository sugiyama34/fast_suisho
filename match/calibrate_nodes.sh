#!/bin/bash
# 固定 movetime で1手あたり何ノード探索できるかを互角局面サンプルで校正する。
# 対局条件の「1手約2Mノード」合わせに使う (docs/PLAN.md フェーズ3参照)。
#
# Usage: ./calibrate_nodes.sh [movetime_ms] [threads] [n_positions]

set -euo pipefail
ENGINE=${ENGINE:-/home/sugiyama/YaneuraOu/source/YaneuraOu-by-gcc}
EVALDIR=${EVALDIR:-/home/sugiyama/suisho11}
BOOK=${BOOK:-"$(dirname "$0")/../data/books/start_sfens_ply24.txt"}
MOVETIME=${1:-240}
THREADS=${2:-16}
NPOS=${3:-10}

[ -x "$ENGINE" ] || { echo "error: engine not found or not executable: $ENGINE" >&2; exit 1; }
[ -f "$EVALDIR/nn.bin" ] || { echo "error: nn.bin not found in: $EVALDIR" >&2; exit 1; }
[ -f "$BOOK" ] || { echo "error: book not found: $BOOK" >&2; exit 1; }

echo "engine sha256 : $(sha256sum "$ENGINE" | cut -d' ' -f1)"
echo "eval sha256   : $(sha256sum "$EVALDIR/nn.bin" | cut -d' ' -f1)"

# 決定的サンプリング: 局面数に応じた等間隔ストライド
BOOK_LINES=$(wc -l < "$BOOK")
STRIDE=$(( BOOK_LINES / NPOS ))
[ "$STRIDE" -ge 1 ] || STRIDE=1
mapfile -t POS < <(awk -v s="$STRIDE" 'NR % s == 1' "$BOOK" | head -"$NPOS")

# 注意: stdout を読まずに sleep で歩調を合わせる素朴な実装。
# ロードや探索が想定より遅れた場合は後段の件数チェックで fail する (無言のバイアスにしない)。
mapfile -t NODES < <(
  {
    printf 'setoption name EvalDir value %s\n' "$EVALDIR"
    printf 'setoption name Threads value %s\n' "$THREADS"
    printf 'setoption name USI_Hash value 1024\n'
    printf 'setoption name USI_OwnBook value false\n'
    printf 'isready\n'
    sleep 15   # 評価関数ロード待ち (135MB, コールドキャッシュ余裕込み)
    for p in "${POS[@]}"; do
      printf 'usinewgame\n'
      printf 'position %s\n' "$p"
      printf 'go movetime %s\n' "$MOVETIME"
      sleep "$(awk -v m="$MOVETIME" 'BEGIN{print m/1000 + 0.6}')"
    done
    printf 'quit\n'
  } | "$ENGINE" 2>&1 | grep -B1 '^bestmove' | grep -oE 'nodes [0-9]+' | awk '{print $2}' | sort -n
)

COUNT=${#NODES[@]}
if [ "$COUNT" -ne "${#POS[@]}" ]; then
  echo "error: expected ${#POS[@]} search results but parsed $COUNT" >&2
  echo "       (エンジンのロード/探索がタイムテーブルからずれた可能性。再実行するか sleep を増やすこと)" >&2
  exit 1
fi

sum=0
for v in "${NODES[@]}"; do sum=$(( sum + v )); done
mean=$(( sum / COUNT ))
if (( COUNT % 2 == 1 )); then
  median=${NODES[$(( COUNT / 2 ))]}
else
  median=$(( (NODES[COUNT/2 - 1] + NODES[COUNT/2]) / 2 ))
fi

echo "threads=$THREADS movetime=${MOVETIME}ms positions=$COUNT"
echo "nodes/move: median=$median mean=$mean min=${NODES[0]} max=${NODES[$((COUNT-1))]}"
