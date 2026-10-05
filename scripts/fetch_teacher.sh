#!/bin/bash
# 学習用データの取得 (HuggingFace)
#
#   - 奏乗教師データ: washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique
#     全 30 ファイル (計 14,668,949,437 局面 / 約 587 GB, PackedSfenValue 40B/局面)
#     → data/teacher/sojo/train/dlsuisho_unique_NNN.psv
#       (.bin → .psv にリネーム。BulletOu は拡張子で PackedSfenValue を判別する)
#   - 検証セット: takaoyamaoka/floodgate.hcpe (856,923 局面, hcpe 38B/局面)
#     → data/teacher/floodgate/floodgate.hcpe
#
# 使い方:
#   bash scripts/fetch_teacher.sh            # 取得 (サイズ一致のファイルはスキップ, 再開可)
#   bash scripts/fetch_teacher.sh --verify   # 取得後に sha256 を全ファイル検証 (587GB を読むので時間がかかる)
#   SOJO_ORDER="001 011 ..." bash scripts/fetch_teacher.sh   # 取得順の指定 (下記)
#
# 大容量ディスクに置く場合は data/teacher/sojo を先にシンボリックリンクにしておく
# (例: ln -s /mnt/D/<user>/teacher/sojo data/teacher/sojo)。スクリプトはリンク先に書く。
#
# sha256 は HF の LFS oid (= ファイル全体の sha256) を scripts/teacher_sojo.sha256 に固定したもの。
# 元は experiment-005/006 で使った data/teacher/sojo/download.sh (git 管理外) を移植した。
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOJO_BASE="https://huggingface.co/datasets/washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique/resolve/main"
FG_URL="https://huggingface.co/datasets/takaoyamaoka/floodgate.hcpe/resolve/main/floodgate.hcpe"
FG_SHA256="fb9d60b283ade32cb5c5715fe27042476bc7169cc46b6d62a2d85975c1a945ac"
FG_SIZE=32563074
SOJO_DIR="$REPO_ROOT/data/teacher/sojo/train"
FG_DIR="$REPO_ROOT/data/teacher/floodgate"
SHA_LIST="$REPO_ROOT/scripts/teacher_sojo.sha256"
PARALLEL="${PARALLEL:-4}"

VERIFY=0
[ "${1:-}" = "--verify" ] && VERIFY=1

mkdir -p "$SOJO_DIR" "$FG_DIR"

# 1 ファイル取得: fetch <url> <dst> <expected_size>
fetch() {
  local url="$1" dst="$2" expected="$3"
  local have=0
  [ -f "$dst" ] && have=$(stat -c%s "$dst")
  if [ "$have" -eq "$expected" ]; then
    echo "[skip] $(basename "$dst") already complete"
    return 0
  fi
  for attempt in 1 2 3 4 5; do
    curl -sS -L -C - --retry 5 --retry-delay 10 --speed-limit 100000 --speed-time 60 \
      -o "$dst" "$url" && break
    echo "[retry $attempt] $(basename "$dst")"
    sleep 20
  done
  have=$(stat -c%s "$dst" 2>/dev/null || echo 0)
  if [ "$have" -eq "$expected" ]; then
    echo "[done] $(basename "$dst") ($(date +%H:%M:%S))"
  else
    echo "[FAIL] $(basename "$dst") size=$have expected=$expected"
    return 1
  fi
}

fetch_sojo() {
  local num="$1"
  # 029 だけ 7 局面 (280B) 大きい (HF API 実測。40B/局面の倍数で正当)
  local expected=19558599240
  [ "$num" = "029" ] && expected=19558599520
  fetch "$SOJO_BASE/dlsuisho_unique_${num}.bin" "$SOJO_DIR/dlsuisho_unique_${num}.psv" "$expected"
}
export -f fetch fetch_sojo
export SOJO_BASE SOJO_DIR

echo "[start] $(date)"
status=0
fetch "$FG_URL" "$FG_DIR/floodgate.hcpe" "$FG_SIZE" || status=1
# SOJO_ORDER (空白区切りのファイル番号) で取得順を変えられる。experiment-009 は
# 小さいサブセットから揃うよう NESTED 順で取る (experiments/009-data-scaling/subsets.py):
#   SOJO_ORDER="001 011 021 006 016 026 003 013 023 008 018 028 004 014 024 002 012 022 009 019 029 005 015 025 007 017 027 010 020 030"
if [ -n "${SOJO_ORDER:-}" ]; then
  printf '%s\n' $SOJO_ORDER
else
  seq -f '%03g' 1 30
fi | xargs -P "$PARALLEL" -I{} bash -c 'fetch_sojo {}' || status=1
echo "[end] $(date)"

if [ "$VERIFY" -eq 1 ]; then
  echo "[verify] sha256 (floodgate)"
  echo "$FG_SHA256  floodgate.hcpe" | (cd "$FG_DIR" && sha256sum -c -) || status=1
  echo "[verify] sha256 (奏乗 30 files)"
  (cd "$SOJO_DIR" && sha256sum -c "$SHA_LIST") || status=1
fi

exit "$status"
