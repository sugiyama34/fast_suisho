#!/bin/bash
# experiment-011: suzuki にしか無いファイルを 1 つのディレクトリに集め、sha256 の一覧 (MANIFEST.sha256) を作る。
# **suzuki で実行する**。転送と kajiki 側の配置は experiments/011-pcalm/hypothesis.md §5 と install_on_kajiki.sh。
#
#   bash pack_on_suzuki.sh
#   (このファイルが suzuki に無いときは: git -C ~/fast_suisho fetch origin exp/011-pcalm &&
#    git -C ~/fast_suisho show origin/exp/011-pcalm:experiments/011-pcalm/xfer/pack_on_suzuki.sh | bash)
#
# 環境変数で場所を変えられる: REPO (既定 ~/fast_suisho), CKPT (既定 /mnt/nvme1/sugiyama/checkpoints),
# OUT (既定 /mnt/nvme1/sugiyama/xfer-011), WITH_LOSS=0 で loss の標本 (B_s0, 約 360 MB) を省く。
# 同じファイルシステム上のものはハードリンク (コピーしない)、それ以外はコピーする。元のファイルは変更しない。
set -euo pipefail

REPO="${REPO:-$HOME/fast_suisho}"
CKPT="${CKPT:-/mnt/nvme1/sugiyama/checkpoints}"
OUT="${OUT:-/mnt/nvme1/sugiyama/xfer-011}"
WITH_LOSS="${WITH_LOSS:-1}"
GAME="final-full-e16@48-vs-s11@32-300k-2000p"

mkdir -p "$OUT"
put() {  # put <元のパス> <OUT からの相対パス>
  local src="$1" rel="$2"
  [ -f "$src" ] || { echo "error: 見つからない: $src" >&2; exit 1; }
  mkdir -p "$OUT/$(dirname "$rel")"
  ln -f "$src" "$OUT/$rel" 2>/dev/null || cp -p "$src" "$OUT/$rel"
}

# 1. 対局エンジン (experiment-009 の AVX2 + finny.patch ビルド, sha256 423b6b1c…)
put "$HOME/engines/009-finny-avx2/YaneuraOu-by-gcc" engine/009-finny-avx2/YaneuraOu-by-gcc
# 2. BP の基準 (e16) と FV_SCALE の端の追加対局用 (e12, e20) の nn.bin
for ep in 0012 0016 0020; do
  put "$CKPT/009-full/$ep/nn.bin" "checkpoints/009-full/$ep/nn.bin"
done
# 3. M0 (勾配の比較) 用の学習済み重み (畳み込み前の f32, 各 2.2 GB)
for ep in 0001 0016; do
  put "$CKPT/009-full/$ep/state.bin" "checkpoints/009-full/$ep/state.bin"
done
# 4. full-e16 @ 48 の 2,000 ペアの棋譜 (エンジンの同一性確認と、開始局面ごとの対応のある比較に使う)
for f in "$REPO/experiments/009-data-scaling/games/$GAME"/*; do
  put "$f" "games-009/$GAME/$(basename "$f")"
done
# 5. (任意) held-out loss の標本 B_s0
if [ "$WITH_LOSS" = 1 ]; then
  for f in /mnt/nvme1/sugiyama/loss_samples/B_s0.*; do
    put "$f" "loss_samples/$(basename "$f")"
  done
fi

(cd "$OUT" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
cat "$OUT/MANIFEST.sha256"
du -sh --apparent-size "$OUT"
echo "ok: $OUT (次は手元の端末から kajiki へ転送。hypothesis.md §5)"
