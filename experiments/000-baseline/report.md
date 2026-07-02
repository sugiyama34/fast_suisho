# 000-baseline: Suisho 11 × やねうら王 V9.60 ベースライン

日付: 2026-07-02

## 結果サマリ

- やねうら王 V9.60 (commit `9133c527`) を `YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3`
  edition でビルドし、Suisho 11 `nn.bin` の読み込み・探索を確認した
- 初期局面 3 秒探索: depth 23, **約 610 kNPS (1スレッド)**, bestmove 2g2f (cp +103)

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
   自作 serializer で書き出す際は元ファイルの hash をそのまま保持すれば警告も消える。
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
