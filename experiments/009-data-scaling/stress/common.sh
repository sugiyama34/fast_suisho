# suzuki の電力の負荷試験 (2026-10-05) の共通設定。各スクリプトから source する。
# 実行時の状態 (pid ファイル・ログ・学習と対局の出力) はすべて STRESS_DIR に置き、
# experiment-009 の checkpoint (/mnt/nvme1/sugiyama/checkpoints) と棋譜 (games/) には触れない。

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$REPO/experiments/009-data-scaling"
STRESS_DIR="${STRESS_DIR:-/mnt/nvme1/sugiyama/stress}"
PID_DIR="$STRESS_DIR/pids"
LOG_DIR="$STRESS_DIR/logs"
EVENTS="$STRESS_DIR/events.log"
mkdir -p "$PID_DIR" "$LOG_DIR"

# 開始・停止の記録 (monitor.py の CSV と時刻で突き合わせる)
event() { printf '%(%F %T)T %s\n' -1 "$*" | tee -a "$EVENTS"; }

# 負荷 <name> のプロセスグループ ID (pid ファイルが無い・別の起動 (再起動前) のものなら空)。
# pid ファイルは「グループ ID boot_id」。ブレーカーで落ちて再起動したあとに古い pid ファイルの
# 番号で無関係なプロセスを止めないよう、boot_id が今と違えば無視する
BOOT_ID="$(cat /proc/sys/kernel/random/boot_id)"
pgid_of() {
  local g b
  { read -r g b < "$PID_DIR/$1.pid"; } 2>/dev/null || return 0
  [ "$b" = "$BOOT_ID" ] && echo "$g"
  return 0
}

# 負荷 <name> が動いているか (グループに生きたプロセスが 1 つでもあるか)
alive() {
  local g
  g="$(pgid_of "$1")"
  [ -n "$g" ] && kill -0 -- "-$g" 2>/dev/null
}

# 新しいセッション (= 新しいプロセスグループ) でコマンドを起動し、グループ ID を pid ファイルに書く。
# 子プロセス (やねうら王) も同じグループに入るので、stop.sh はグループごと止められる。
# DRY_RUN=1 ならコマンドを表示するだけで起動しない
launch() {
  local name="$1"
  shift
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "[dry-run] $name: $*"
    return
  fi
  local pidfile="$PID_DIR/$name.pid"
  rm -f "$pidfile"
  setsid bash -c 'echo "$$ $(cat /proc/sys/kernel/random/boot_id)" > "$0"; exec "$@"' "$pidfile" "$@" >> "$LOG_DIR/$name.log" 2>&1 < /dev/null &
  for _ in $(seq 50); do
    [ -s "$pidfile" ] && break
    sleep 0.1
  done
  [ -s "$pidfile" ] || { echo "error: $name の起動を確認できない (ログ: $LOG_DIR/$name.log)" >&2; return 1; }
  event "start $name pgid=$(pgid_of "$name")"
}
