#!/bin/bash
# experiment-011: 学習速度の計測 (1 sb = 約 4,000 万局面あたりの実時間)。arm を順に BENCH=1 で走らせ、
# ログの [progress] 行の時刻差の中央値 (最初の 1 本は起動を含むので除く) を表にする (docs/SETUP.md §8)。
#
#   BENCH_SB=6 bash experiments/011-pcalm/bench.sh <gpu> <arm> [<arm> ...]
#
# 結果は experiments/011-pcalm/logs/bench-<日時>.tsv。同じスライスで他の学習を動かさないこと
set -uo pipefail
GPU="${1:?gpu}"; shift
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$REPO/experiments/011-pcalm"
export BENCH=1 BENCH_SB="${BENCH_SB:-6}"
OUT="$HERE/logs/bench-$(date +%Y%m%d-%H%M%S).tsv"
mkdir -p "$HERE/logs"
printf 'arm\tsec_per_sb_median\tn\tsamples\n' > "$OUT"
for ARM in "$@"; do
  LOG="$HERE/logs/011-$ARM-bench.log"
  rm -f "$LOG"
  bash "$HERE/run_training.sh" "$ARM" "$GPU" > /dev/null 2>&1 || echo "[fail] $ARM" >&2
  grep -a 'progress\]\|checkpoint\]' "$LOG" | sed 's/\x1b\[[0-9;]*m//g' \
    | awk -v arm="$ARM" '{split($2,t,":"); s=t[1]*3600+t[2]*60+t[3]; if (p) d[n++]=s-p; p=s}
      END {asort(d); m=(n%2)?d[(n+1)/2]:(d[n/2]+d[n/2+1])/2; out=""; for(i=1;i<=n;i++) out=out d[i] ","; printf "%s\t%.1f\t%d\t%s\n", arm, m, n, out}' >> "$OUT"
  tail -1 "$OUT"
  rm -rf "$REPO/data/bulletou/checkpoints/011-$ARM-bench"
done
echo "wrote $OUT"
