#!/bin/bash
# 負荷を止める。
#
#   bash experiments/009-data-scaling/stress/stop.sh            # 緊急停止: 全負荷を即座に SIGKILL
#   bash experiments/009-data-scaling/stress/stop.sh g4 c3      # 指定した負荷だけ (SIGTERM → 5 秒後に SIGKILL)
#
# 引数なしのときは pid ファイルのグループを止めたあと、取りこぼし (pid ファイルの無いもの) として
# 自分の bulletou・やねうら王・stress 用の match_nodes.py も PID で止める。
# 名前の照合はプロセス名 (pgrep -x) と、自分自身に一致しない正規表現 ([.]) で行う。
set -uo pipefail
source "$(dirname "$0")/common.sh"

kill_group() {  # <name> <signal>
  local g
  g="$(pgid_of "$1")"
  [ -n "$g" ] && kill "-$2" -- "-$g" 2>/dev/null
}

if [ $# -eq 0 ]; then
  for f in "$PID_DIR"/*.pid; do
    [ -e "$f" ] || continue
    kill_group "$(basename "$f" .pid)" KILL
  done
  me="$(id -u)"
  left="$( (pgrep -u "$me" -x bulletou; pgrep -u "$me" -x YaneuraOu-by-gc;
            pgrep -u "$me" -f 'match_nodes[.]py .*--name stress-') | sort -u | tr '\n' ' ')"
  # shellcheck disable=SC2086
  [ -n "${left// /}" ] && kill -KILL $left 2>/dev/null
  rm -f "$PID_DIR"/*.pid
  event "STOP ALL (SIGKILL) extra_pids=[${left% }]"
else
  for name in "$@"; do
    if alive "$name"; then
      kill_group "$name" TERM
    else
      echo "$name: 動いていない"
    fi
  done
  for _ in $(seq 50); do
    busy=0
    for name in "$@"; do alive "$name" && busy=1; done
    [ "$busy" = 0 ] && break
    sleep 0.1
  done
  for name in "$@"; do
    alive "$name" && kill_group "$name" KILL
    rm -f "$PID_DIR/$name.pid"
  done
  event "stop $*"
fi

sleep 1
echo "残っている GPU プロセス:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | sed 's/^/  /'
echo "残っている bulletou / やねうら王: $( (pgrep -x bulletou; pgrep -x YaneuraOu-by-gc) | wc -l)"
