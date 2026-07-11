#!/bin/bash
# 開始局面集を data/books/ (git管理外) にダウンロードして検証する。
#
# start_sfens_ply24.txt — やねうら王互角局面集2025 (24手目局面集)
#   - 出典: https://github.com/yaneurao/YaneuraOu/releases/tag/BalancedPositions2025
#   - 解説: https://yaneuraou.yaneu.com/2025/07/29/yaneuraou-balanced-position-collection-2025/
#   - License: MIT (配布元による)
#   - 生成方法 (配布元記載): 水匠10で1局面2億ノード探索し、評価値の絶対値が
#     50以下の24手目局面を抽出したもの。30,053局面、1行1 SFEN
#   - 注意: 行は辞書順ソートされており隣接行は近似重複が多い。
#     先頭N行抽出は禁止、等間隔ストライドで抽出すること (docs/PLAN.md 参照)
#
# 配布オリジナルは CRLF (sha256 f93098ab6c0cf5eab5f55aaffd48da4421934e08d4c2fcfba68f7b60853ac0a2)。
# USI コマンドへの \r 混入を防ぐため LF に正規化して保存する。

set -euo pipefail

URL="https://github.com/user-attachments/files/21482992/start_sfens_ply24.txt"
SHA256_LF="a11e3ae7efd34f4c7ad8a7e42c0f610239927970259a57915f06a644cee8d90d"
DEST_DIR="$(dirname "$0")/../data/books"
DEST="$DEST_DIR/start_sfens_ply24.txt"

if [ -f "$DEST" ] && echo "$SHA256_LF  $DEST" | sha256sum -c --quiet 2>/dev/null; then
  echo "ok: $DEST already present and verified"
  exit 0
fi

mkdir -p "$DEST_DIR"
echo "downloading $URL ..."
curl -sL "$URL" -o "$DEST.tmp"
sed -i 's/\r$//' "$DEST.tmp"   # CRLF -> LF 正規化

if ! echo "$SHA256_LF  $DEST.tmp" | sha256sum -c --quiet; then
  echo "error: sha256 mismatch after download+normalize; leaving $DEST.tmp for inspection" >&2
  exit 1
fi
mv "$DEST.tmp" "$DEST"
echo "ok: $(wc -l < "$DEST") positions -> $DEST"
