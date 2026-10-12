#!/bin/bash
# MIG2: PC (α=0) の小規模 T=4 / T=32 (queue PID 121476) と、その調整対局が終わったら、
# PC の最良の T (T = 4, 8, 16, 32 の調整対局で Elo 最大, 規則) で本番規模 (E=20) を学習する (ユーザー決定 10-08)
G=/home/sugiyama/fast_suisho/experiments/011-pcalm/games
while kill -0 121476 2>/dev/null; do sleep 60; done
for T in 4 32; do
  until [ -f "$G/tune-s-pc-T$T-gn-e1x@24-vs-bp-rep-e1@48-300k/queue_done.json" ]; do
    [ -f "$G/tune-s-pc-T$T-gn-e1x@24-vs-bp-rep-e1@48-300k/queue_failed.json" ] && { echo "tune T$T failed; stop"; exit 1; }
    sleep 60
  done
done
BEST=$(python3 - <<'PY'
import json
G="/home/sugiyama/fast_suisho/experiments/011-pcalm/games"
r={T: json.load(open(f"{G}/tune-s-pc-T{T}-gn-e1x@24-vs-bp-rep-e1@48-300k/summary.json"))["elo"] for T in (4, 8, 16, 32)}
print(max(r, key=r.get))
PY
)
echo "$(date '+%F %T') PC best T=$BEST"
cd /home/sugiyama/fast_suisho
export BIN_PCALM=/mnt/D/sugiyama/011/bin/bulletou-pcalm-d2ae90002b25
exec bash experiments/011-pcalm/gpu_queue.sh MIG-a4e39a6f-6d5d-5203-881e-f4ae3bd988de "f-pc-T$BEST-gn"
