#!/bin/bash
# NPS thread-scaling benchmark: Suisho 11 nn.bin x YaneuraOu V9.60 (SFNN_halfka2_1024-7-64-k3k3)
#
# やねうら王の bench コマンド (デフォルト4局面) を movetime 固定で REPEATS 回ずつ回し、
# スレッド数ごとの Nodes/second の中央値と範囲を計測する。
# 比較の妥当性のため、エンジン/評価関数の sha256 と CPU 周波数条件もヘッダに記録する。
#
# Usage: ./bench_nps.sh [movetime_ms]
# Env:   ENGINE, EVALDIR, THREADS_LIST, HASH_MB, REPEATS を上書き可能

set -euo pipefail

ENGINE=${ENGINE:-/home/sugiyama/YaneuraOu/source/YaneuraOu-by-gcc}
EVALDIR=${EVALDIR:-/home/sugiyama/suisho11}
MOVETIME=${1:-10000}
HASH_MB=${HASH_MB:-1024}
THREADS_LIST=${THREADS_LIST:-"1 2 4 8 16"}
REPEATS=${REPEATS:-3}

[ -x "$ENGINE" ] || { echo "error: engine not found or not executable: $ENGINE" >&2; exit 1; }
[ -f "$EVALDIR/nn.bin" ] || { echo "error: nn.bin not found in: $EVALDIR" >&2; exit 1; }

echo "engine        : $ENGINE"
echo "engine sha256 : $(sha256sum "$ENGINE" | cut -d' ' -f1)"
echo "eval sha256   : $(sha256sum "$EVALDIR/nn.bin" | cut -d' ' -f1)"
echo "conditions    : movetime ${MOVETIME}ms x 4 positions, hash ${HASH_MB}MB, repeats ${REPEATS}"
echo "cpu           : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
echo "governor      : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)" \
     "/ boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unknown)"
echo "loadavg(pre)  : $(cut -d' ' -f1-3 /proc/loadavg)"
echo

run_bench_once() { # $1=threads -> echoes "nps nodes" or "ERR ERR"
  local out nps nodes
  if ! out=$(printf 'setoption name EvalDir value %s\nbench %s %s %s default movetime\nquit\n' \
               "$EVALDIR" "$HASH_MB" "$1" "$MOVETIME" | "$ENGINE" 2>&1); then
    echo "error: engine exited nonzero at threads=$1; output was:" >&2
    echo "$out" >&2
    exit 1
  fi
  # grep 不一致 (出力形式変更など) は ERR 表示で続行させる
  nps=$(echo "$out"   | grep -oE 'Nodes/second    : [0-9]+' | grep -oE '[0-9]+' | tail -1 || true)
  nodes=$(echo "$out" | grep -oE 'Nodes searched  : [0-9]+' | grep -oE '[0-9]+' | tail -1 || true)
  echo "${nps:-ERR} ${nodes:-ERR}"
}

printf '%-8s %-12s %-22s %s\n' "threads" "nps_median" "nps_min-max" "nodes_median"
for t in $THREADS_LIST; do
  nps_runs=()
  nodes_runs=()
  for _ in $(seq "$REPEATS"); do
    read -r nps nodes < <(run_bench_once "$t")
    nps_runs+=("$nps")
    nodes_runs+=("$nodes")
  done
  if printf '%s\n' "${nps_runs[@]}" | grep -q ERR; then
    printf '%-8s %-12s %-22s %s\n' "$t" ERR "$(IFS=,; echo "${nps_runs[*]}")" ERR
    continue
  fi
  mapfile -t nps_sorted < <(printf '%s\n' "${nps_runs[@]}" | sort -n)
  mapfile -t nodes_sorted < <(printf '%s\n' "${nodes_runs[@]}" | sort -n)
  mid=$(( REPEATS / 2 ))
  printf '%-8s %-12s %-22s %s\n' "$t" "${nps_sorted[$mid]}" \
         "${nps_sorted[0]}-${nps_sorted[$((REPEATS-1))]}" "${nodes_sorted[$mid]}"
done
echo
echo "loadavg(post) : $(cut -d' ' -f1-3 /proc/loadavg)"
