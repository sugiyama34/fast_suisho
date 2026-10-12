#!/bin/bash
# experiment-011: suzuki から転送したファイル (pack_on_suzuki.sh の出力) を kajiki で検証して所定の場所に置く。
# **kajiki で実行する**。
#
#   bash experiments/011-pcalm/xfer/install_on_kajiki.sh
#
# 1. MANIFEST.sha256 で転送の破損を確認する
# 2. 既知の sha256 (experiment-009 の report.md / notes/ratings.md) と照合し、正しいファイルかを確認する
# 3. 配置 (同じファイルシステムは mv、エンジンだけコピー):
#      エンジン   → ~/engines/009-finny-avx2/YaneuraOu-by-gcc
#      checkpoint → /mnt/D/sugiyama/checkpoints/009-full/NNNN/   (= data/bulletou/checkpoints/009-full/)
#      棋譜       → experiments/009-data-scaling/games/<対局名>/  (009 のツールと同じ相対位置, git 管理外)
#      loss 標本  → /mnt/D/sugiyama/loss_samples/
# IN (既定 /mnt/D/sugiyama/xfer-011) で転送先を変えられる。既にあるファイルは上書きしない (止まる)。
set -euo pipefail

IN="${IN:-/mnt/D/sugiyama/xfer-011}"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT=/mnt/D/sugiyama/checkpoints
cd "$IN"

echo "[1/3] MANIFEST.sha256 の検証"
sha256sum -c --quiet MANIFEST.sha256

echo "[2/3] 既知の sha256 との照合"
check() {  # check <ファイル> <期待する sha256 (先頭だけでも可)>
  local got; got=$(sha256sum "$1" | cut -c1-${#2})
  [ "$got" = "$2" ] || { echo "error: sha256 が違う: $1 ($got != $2)" >&2; exit 1; }
  echo "  ok $1"
}
check engine/009-finny-avx2/YaneuraOu-by-gcc 423b6b1cd3477321d0a049632d7966301981d158ef381e0fa14f04012995b94e
check checkpoints/009-full/0016/nn.bin b68d6d7b320de52675a4e08163022999a883d5c02823b988d406f278e36c8239
check checkpoints/009-full/0012/nn.bin a834c15b1ad2
check checkpoints/009-full/0020/nn.bin 711247c13c4b

echo "[3/3] 配置"
place() {  # place <相対パス> <置き先> (mv)
  [ -e "$2" ] && { echo "error: 既にある: $2" >&2; exit 1; }
  mkdir -p "$(dirname "$2")"
  mv "$1" "$2"
  echo "  $2"
}
dst="$HOME/engines/009-finny-avx2/YaneuraOu-by-gcc"
[ -e "$dst" ] && { echo "error: 既にある: $dst" >&2; exit 1; }
install -D -m 755 engine/009-finny-avx2/YaneuraOu-by-gcc "$dst" && echo "  $dst"
for f in checkpoints/009-full/*/*; do place "$f" "$CKPT/${f#checkpoints/}"; done
for f in games-009/*/*; do place "$f" "$REPO/experiments/009-data-scaling/games/${f#games-009/}"; done
if compgen -G "loss_samples/*" > /dev/null; then
  for f in loss_samples/*; do place "$f" "/mnt/D/sugiyama/loss_samples/${f#loss_samples/}"; done
fi
cp MANIFEST.sha256 "$REPO/experiments/011-pcalm/xfer/MANIFEST-from-suzuki.sha256"
echo "ok: sha256 の一覧を experiments/011-pcalm/xfer/MANIFEST-from-suzuki.sha256 に保存 (レポートに記録する)"
