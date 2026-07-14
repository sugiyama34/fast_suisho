#!/bin/bash
# perf プロファイル取得: Suisho 11 nn.bin × やねうら王 bench (実験 002)
#
# エンジンを FIFO 経由で起動し、nn.bin 読み込み完了 (readyok) を確認してから
# perf をアタッチして bench 区間のみをサンプリングする (LEB128 ローダのコストを
# プロファイルに混入させないため)。
#   pass 1: perf record (cycles:u, dwarf call-graph) → perf.data + テキストレポート
#   pass 2: perf stat  (IPC / キャッシュ / dTLB ミス — huge pages 検討の判断材料)
#
# 前提: kernel.perf_event_paranoid <= 2
#   (一時変更: sudo sysctl kernel.perf_event_paranoid=1 — 再起動でデフォルトに戻る)
#
# Usage: ./profile.sh [movetime_ms]
# Env:   ENGINE, EVALDIR, HASH_MB, THREADS, FREQ, OUTDIR を上書き可能

set -euo pipefail

ENGINE=${ENGINE:-$HOME/engines/profile-002/YaneuraOu-by-gcc}
EVALDIR=${EVALDIR:-$HOME/suisho11}
MOVETIME=${1:-10000}
HASH_MB=${HASH_MB:-1024}
THREADS=${THREADS:-1}
FREQ=${FREQ:-397}
OUTDIR=${OUTDIR:-$(cd "$(dirname "$0")" && pwd)}

paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)
if [ "$paranoid" -gt 2 ]; then
  echo "error: kernel.perf_event_paranoid=$paranoid — 非特権 perf が無効です" >&2
  echo "  sudo sysctl kernel.perf_event_paranoid=1  を実行してから再試行してください" >&2
  exit 1
fi
[ -x "$ENGINE" ] || { echo "error: engine not found or not executable: $ENGINE" >&2; exit 1; }
[ -f "$EVALDIR/nn.bin" ] || { echo "error: nn.bin not found in: $EVALDIR" >&2; exit 1; }

{
  echo "engine        : $ENGINE"
  echo "engine sha256 : $(sha256sum "$ENGINE" | cut -d' ' -f1)"
  echo "eval sha256   : $(sha256sum "$EVALDIR/nn.bin" | cut -d' ' -f1)"
  echo "conditions    : movetime ${MOVETIME}ms x 4 positions, hash ${HASH_MB}MB, threads ${THREADS}, sample ${FREQ}Hz"
  echo "cpu           : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
  echo "governor      : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)" \
       "/ boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unknown)"
  echo "paranoid      : $paranoid"
  echo "loadavg(pre)  : $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "date          : $(date -Iseconds)"
} | tee "$OUTDIR/profile_env.txt"

ENGINE_PID=
ENGINE_LOG=
FIFO=

cleanup() {
  [ -n "$ENGINE_PID" ] && kill "$ENGINE_PID" 2>/dev/null || true
  [ -n "$FIFO" ] && rm -f "$FIFO" || true
}
trap cleanup EXIT

launch_engine() { # $1 = engine log file → ENGINE_PID / fd 3 (stdin) をセット
  ENGINE_LOG=$1
  FIFO=$(mktemp -u "${TMPDIR:-/tmp}/usi_fifo.XXXXXX")
  mkfifo "$FIFO"
  "$ENGINE" < "$FIFO" > "$ENGINE_LOG" 2>&1 &
  ENGINE_PID=$!
  exec 3>"$FIFO"
  printf 'setoption name EvalDir value %s\nisready\n' "$EVALDIR" >&3
  for _ in $(seq 300); do   # nn.bin ロード待ち (最大 150 秒)
    grep -q '^readyok' "$ENGINE_LOG" && return 0
    kill -0 "$ENGINE_PID" 2>/dev/null || { echo "error: engine died during load:" >&2; cat "$ENGINE_LOG" >&2; exit 1; }
    sleep 0.5
  done
  echo "error: readyok timeout" >&2; exit 1
}

run_bench_and_wait() {
  printf 'bench %s %s %s default movetime\n' "$HASH_MB" "$THREADS" "$MOVETIME" >&3
  local deadline=$(( (MOVETIME / 1000 + 10) * 4 + 60 ))
  for _ in $(seq "$deadline"); do
    grep -q 'Nodes/second' "$ENGINE_LOG" && return 0
    kill -0 "$ENGINE_PID" 2>/dev/null || { echo "error: engine died during bench:" >&2; tail "$ENGINE_LOG" >&2; exit 1; }
    sleep 1
  done
  echo "error: bench timeout" >&2; exit 1
}

shutdown_engine() {
  printf 'quit\n' >&3
  exec 3>&-
  wait "$ENGINE_PID" 2>/dev/null || true
  ENGINE_PID=
  rm -f "$FIFO"; FIFO=
  grep -E 'Nodes (searched|/second)' "$ENGINE_LOG" | tail -2 || true
}

echo
echo "== pass 1/2: perf record (cycles:u, dwarf call-graph) =="
launch_engine "$OUTDIR/engine_record.log"
perf record -F "$FREQ" -e cycles:u --call-graph dwarf,8192 -o "$OUTDIR/perf.data" -p "$ENGINE_PID" &
PERF_PID=$!
sleep 1   # perf のアタッチ完了待ち
run_bench_and_wait
kill -INT "$PERF_PID"
wait "$PERF_PID" || true
shutdown_engine

echo
echo "== pass 2/2: perf stat (IPC / cache / dTLB) =="
launch_engine "$OUTDIR/engine_stat.log"
perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,dTLB-loads,dTLB-load-misses \
  -p "$ENGINE_PID" -o "$OUTDIR/perf_stat.txt" &
PERF_PID=$!
sleep 1
run_bench_and_wait
kill -INT "$PERF_PID"
wait "$PERF_PID" || true
shutdown_engine

echo
echo "== レポート生成 =="
perf report --stdio --no-children --percent-limit 0.3 -i "$OUTDIR/perf.data" > "$OUTDIR/perf_flat.txt"
perf report --stdio --children --percent-limit 1.0 -i "$OUTDIR/perf.data" > "$OUTDIR/perf_graph.txt"
echo "done:"
ls -lh "$OUTDIR"/perf.data "$OUTDIR"/perf_flat.txt "$OUTDIR"/perf_graph.txt "$OUTDIR"/perf_stat.txt
