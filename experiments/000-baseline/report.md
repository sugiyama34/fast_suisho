# 000-baseline: Suisho 11 × やねうら王 V9.60 ベースライン

日付: 2026-07-02

## 結果サマリ

- やねうら王 V9.60 (commit `9133c527`) を `YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3`
  edition でビルドし、Suisho 11 `nn.bin` の読み込み・探索を確認した
- 初期局面 3 秒探索 (アドホック単発計測): depth 23, 約 600 kNPS (1スレッド, ログ値 nps 600373),
  bestmove 2g2f。cp 103 は upperbound (fail 境界値) であり確定評価値ではない
- 正式な NPS ベースラインは固定局面集 / bench で別途計測する (下記 TODO)。
  この単発値を高速化の比較基準にしないこと

## ビルドコマンド

```sh
cd ../YaneuraOu/source
make -j16 normal YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3 \
     COMPILER=g++ TARGET_CPU=AVX2 PYTHON=python3
# → source/YaneuraOu-by-gcc
```

- `PYTHON=python3` 必須 (デフォルトの `python` は存在しない)
- arch header は `nnue_arch_gen.py` が自動生成する。suisho11 付属の `sfnnwop-1536.h` と
  実質同一 (差分: GetName 文字列と ac_sqr_0_out の memset のみ) であることを確認済み

## 既知の注意点

1. **hash mismatch 警告は無害**: nn.bin 内の arch 文字列は旧名 `HalfKA(Friend)`
   (hash 0x5f134cb9 系)、V9.60 は `HalfKA2` (0x5f234cb9) に改名済み。
   次元数 (131949→1024x2) と index 割当は同一で、評価値も正常。
   警告 (`NNUE hash mismatch: expected 1008745266 got 1008746012`) は無視してよい。
   警告の原因はファイル内 hash (0x3c203e1c) とエンジン側定数 (0x3c203b32,
   `evaluate_nnue.h`) の不一致。自作 serializer では元ファイルの hash を保持すると
   バイト一致は保てるが警告は残る (無害)。エンジン期待値の hash を書けば警告は
   消えるが、元ファイルとのバイト一致は失われる。どちらかを選ぶこと。
   なお現行 V9.60 ソースで hash `0x5f134cb9` は旧 HalfKA ではなく **HalfKA1**
   (138510 次元、index 割当が異なる別 feature) を指すため、hash 値から現行ソースの
   feature class を逆引きしないこと。
2. **USI 操作時は stdin を閉じない**: `go` 直後に EOF を送ると探索が即中断され
   nodes 0 の bestmove が返る。ハーネスは対話的にパイプを維持すること。
3. 定跡 `book/standard_book.db` は未配置。対局時は `USI_OwnBook false` にするか
   互角局面集を使う。

## USI 動作確認ログ (抜粋)

```
id name YaneuraOu NNUE 9.60git 64AVX2
info string loading eval file : /home/sugiyama/suisho11/nn.bin
readyok
info depth 23 seldepth 37 multipv 1 score cp 103 upperbound nodes 1130504 nps 600373 time 1883 pv 2g2f 8c8d
bestmove 2g2f ponder 8c8d
```

## TODO (次フェーズ)

- [ ] 固定局面集での NPS 計測 (1 / 16 スレッド) をスクリプト化
- [ ] nn.bin parser (tools/) 実装 → フェーズ1
