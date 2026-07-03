#!/bin/bash
# NPS thread-scaling benchmark: Suisho 11 nn.bin x YaneuraOu V9.60 (SFNN_halfka2_1024-7-64-k3k3)
#
# やねうら王の bench コマンド (デフォルト4局面) を movetime 固定で回し、
# スレッド数ごとの Nodes/second を計測する。
#
# Usage: ./bench_nps.sh [movetime_ms]
# Env:   ENGINE, EVALDIR, THREADS_LIST, HASH_MB を上書き可能

set -euo pipefail

ENGINE=${ENGINE:-/home/sugiyama/YaneuraOu/source/YaneuraOu-by-gcc}
EVALDIR=${EVALDIR:-/home/sugiyama/suisho11}
MOVETIME=${1:-10000}
HASH_MB=${HASH_MB:-1024}
THREADS_LIST=${THREADS_LIST:-"1 2 4 8 16"}

echo "engine   : $ENGINE"
echo "eval     : $EVALDIR ($(sha256sum "$EVALDIR/nn.bin" | cut -c1-16)...)"
echo "movetime : ${MOVETIME}ms x 4 positions, hash ${HASH_MB}MB"
echo

printf '%-8s %-14s %-14s %s\n' "threads" "nps" "nodes" "time_ms"
for t in $THREADS_LIST; do
  out=$(printf 'setoption name EvalDir value %s\nbench %s %s %s default movetime\nquit\n' \
          "$EVALDIR" "$HASH_MB" "$t" "$MOVETIME" | "$ENGINE" 2>&1)
  nps=$(echo "$out"   | grep -oE 'Nodes/second    : [0-9]+'  | grep -oE '[0-9]+' | tail -1)
  nodes=$(echo "$out" | grep -oE 'Nodes searched  : [0-9]+'  | grep -oE '[0-9]+' | tail -1)
  ms=$(echo "$out"    | grep -oE 'Total time \(ms\) : [0-9]+' | grep -oE '[0-9]+' | tail -1)
  printf '%-8s %-14s %-14s %s\n' "$t" "${nps:-ERR}" "${nodes:-ERR}" "${ms:-ERR}"
done
