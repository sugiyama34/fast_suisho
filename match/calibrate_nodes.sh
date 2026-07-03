#!/bin/bash
# 固定 movetime で1手あたり何ノード探索できるかを互角局面サンプルで校正する。
# 対局条件の「1手約2Mノード」合わせに使う (docs/PLAN.md フェーズ3参照)。
#
# Usage: ./calibrate_nodes.sh [movetime_ms] [threads] [n_positions]

set -euo pipefail
ENGINE=${ENGINE:-/home/sugiyama/YaneuraOu/source/YaneuraOu-by-gcc}
EVALDIR=${EVALDIR:-/home/sugiyama/suisho11}
BOOK=${BOOK:-"$(dirname "$0")/books/start_sfens_ply24.txt"}
MOVETIME=${1:-240}
THREADS=${2:-16}
NPOS=${3:-10}

[ -x "$ENGINE" ] || { echo "error: engine not found or not executable: $ENGINE" >&2; exit 1; }
[ -f "$EVALDIR/nn.bin" ] || { echo "error: nn.bin not found in: $EVALDIR" >&2; exit 1; }
[ -f "$BOOK" ] || { echo "error: book not found: $BOOK" >&2; exit 1; }

echo "engine sha256 : $(sha256sum "$ENGINE" | cut -d' ' -f1)"
echo "eval sha256   : $(sha256sum "$EVALDIR/nn.bin" | cut -d' ' -f1)"

# 決定的サンプリング: 3000行おき
mapfile -t POS < <(awk 'NR % 3000 == 1' "$BOOK" | head -"$NPOS")

{
  printf 'setoption name EvalDir value %s\n' "$EVALDIR"
  printf 'setoption name Threads value %s\n' "$THREADS"
  printf 'setoption name USI_Hash value 1024\n'
  printf 'setoption name USI_OwnBook value false\n'
  printf 'isready\n'
  sleep 8   # 評価関数ロード待ち
  for p in "${POS[@]}"; do
    printf 'usinewgame\n'
    printf 'position %s\n' "$p"
    printf 'go movetime %s\n' "$MOVETIME"
    sleep "$(awk -v m="$MOVETIME" 'BEGIN{print m/1000 + 0.4}')"
  done
  printf 'quit\n'
} | "$ENGINE" 2>&1 | grep -B1 '^bestmove' | grep -oE 'nodes [0-9]+' | awk -v t="$THREADS" -v m="$MOVETIME" '
  {n[NR]=$2; sum+=$2}
  END{
    if (NR == 0) { print "error: no search results parsed" > "/dev/stderr"; exit 1 }
    asort(n)
    printf "threads=%s movetime=%sms positions=%d\n", t, m, NR
    printf "nodes/move: median=%d mean=%d min=%d max=%d\n", n[int((NR+1)/2)], sum/NR, n[1], n[NR]
  }'
