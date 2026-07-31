#!/bin/bash
# 対計測 (experiment-008): baseline / finny の 2 バイナリで bench を交互に実行し、
# ペアごとの NPS 差を出す。周波数ドリフト・温度などの時間変動をペア内で相殺する。
# experiment-003 の bench_paired.sh の binary A/B 版 (あちらは同一バイナリで評価関数 A/B)。
#
# Usage: ./bench_paired.sh [pairs] [movetime_ms]
# Env:   ENGINE_A (baseline), ENGINE_B (finny), EVALDIR, THREADS, HASH_MB

set -euo pipefail

ENGINE_A=${ENGINE_A:-/home/sugiyama/engines/baseline-000/YaneuraOu-by-gcc}
ENGINE_B=${ENGINE_B:-/home/sugiyama/engines/finny-008/YaneuraOu-by-gcc}
EVALDIR=${EVALDIR:-/home/sugiyama/suisho11}
PAIRS=${1:-5}
MOVETIME=${2:-10000}
THREADS=${THREADS:-1}
HASH_MB=${HASH_MB:-1024}

echo "engine A      : $ENGINE_A ($(sha256sum "$ENGINE_A" | cut -c1-16)…)"
echo "engine B      : $ENGINE_B ($(sha256sum "$ENGINE_B" | cut -c1-16)…)"
echo "eval          : $EVALDIR/nn.bin ($(sha256sum "$EVALDIR/nn.bin" | cut -c1-16)…)"
echo "conditions    : ${THREADS} thread(s), movetime ${MOVETIME}ms x 4 positions, hash ${HASH_MB}MB, ${PAIRS} pairs (interleaved)"
echo "cpu           : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
echo "governor      : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)" \
     "/ boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unknown)"
echo "loadavg(pre)  : $(cut -d' ' -f1-3 /proc/loadavg)"
echo

run_once() { # $1=engine -> echoes nps
  local out nps
  out=$(printf 'setoption name EvalDir value %s\nbench %s %s %s default movetime\nquit\n' \
          "$EVALDIR" "$HASH_MB" "$THREADS" "$MOVETIME" | "$1" 2>&1)
  nps=$(echo "$out" | grep -oE 'Nodes/second    : [0-9]+' | grep -oE '[0-9]+' | tail -1 || true)
  echo "${nps:-ERR}"
}

printf '%-6s %-12s %-12s %s\n' "pair" "nps_base" "nps_finny" "delta_pct"
for i in $(seq "$PAIRS"); do
  a=$(run_once "$ENGINE_A")
  b=$(run_once "$ENGINE_B")
  if [ "$a" = ERR ] || [ "$b" = ERR ]; then
    printf '%-6s %-12s %-12s %s\n' "$i" "$a" "$b" ERR
    continue
  fi
  printf '%-6s %-12s %-12s %s\n' "$i" "$a" "$b" \
         "$(awk -v a="$a" -v b="$b" 'BEGIN { printf "%+.3f%%", (b - a) / a * 100 }')"
done
echo
echo "loadavg(post) : $(cut -d' ' -f1-3 /proc/loadavg)"
