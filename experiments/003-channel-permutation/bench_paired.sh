#!/bin/bash
# 対計測 (experiment-003): オリジナル / permuted の 1 スレッド bench を交互に実行し、
# ペアごとの NPS 差を出す。周波数ドリフト・温度などの時間変動をペア内で相殺する。
#
# Usage: ./bench_paired.sh [pairs] [movetime_ms]
# Env:   ENGINE, EVALDIR_A (オリジナル), EVALDIR_B (permuted), HASH_MB

set -euo pipefail

ENGINE=${ENGINE:-/home/sugiyama/engines/baseline-000/YaneuraOu-by-gcc}
EVALDIR_A=${EVALDIR_A:-/home/sugiyama/suisho11}
EVALDIR_B=${EVALDIR_B:-/home/sugiyama/engines/perm-003}
PAIRS=${1:-10}
MOVETIME=${2:-10000}
HASH_MB=${HASH_MB:-1024}

echo "engine        : $ENGINE ($(sha256sum "$ENGINE" | cut -c1-16)…)"
echo "eval A (orig) : $EVALDIR_A/nn.bin ($(sha256sum "$EVALDIR_A/nn.bin" | cut -c1-16)…)"
echo "eval B (perm) : $EVALDIR_B/nn.bin ($(sha256sum "$EVALDIR_B/nn.bin" | cut -c1-16)…)"
echo "conditions    : 1 thread, movetime ${MOVETIME}ms x 4 positions, hash ${HASH_MB}MB, ${PAIRS} pairs (interleaved)"
echo "cpu           : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
echo "governor      : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)" \
     "/ boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unknown)"
echo "loadavg(pre)  : $(cut -d' ' -f1-3 /proc/loadavg)"
echo

run_once() { # $1=evaldir -> echoes nps
  local out nps
  out=$(printf 'setoption name EvalDir value %s\nbench %s 1 %s default movetime\nquit\n' \
          "$1" "$HASH_MB" "$MOVETIME" | "$ENGINE" 2>&1)
  nps=$(echo "$out" | grep -oE 'Nodes/second    : [0-9]+' | grep -oE '[0-9]+' | tail -1 || true)
  echo "${nps:-ERR}"
}

printf '%-6s %-12s %-12s %s\n' "pair" "nps_orig" "nps_perm" "delta_pct"
for i in $(seq "$PAIRS"); do
  a=$(run_once "$EVALDIR_A")
  b=$(run_once "$EVALDIR_B")
  if [ "$a" = ERR ] || [ "$b" = ERR ]; then
    printf '%-6s %-12s %-12s %s\n' "$i" "$a" "$b" ERR
    continue
  fi
  printf '%-6s %-12s %-12s %s\n' "$i" "$a" "$b" \
         "$(awk -v a="$a" -v b="$b" 'BEGIN { printf "%+.3f%%", (b - a) / a * 100 }')"
done
echo
echo "loadavg(post) : $(cut -d' ' -f1-3 /proc/loadavg)"
