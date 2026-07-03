# 000-baseline: Suisho 11 × やねうら王 V9.60 ベースライン

日付: 2026-07-02

## 結果サマリ

- やねうら王 V9.60 (commit `9133c527`) を `YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3`
  edition でビルドし、Suisho 11 `nn.bin` の読み込み・探索を確認した
- 初期局面 3 秒探索 (アドホック単発計測): depth 23, 約 600 kNPS (1スレッド, ログ値 nps 600373),
  bestmove 2g2f。cp 103 は upperbound (fail 境界値) であり確定評価値ではない
- 正式な NPS ベースラインは固定局面集 / bench で別途計測する (下記 TODO)。
  この単発値を高速化の比較基準にしないこと

## 参照アーティファクトの fingerprint

以降の全実験はこの 2 つを基準物とする。実験前に sha256 が一致することを確認すること
(nn.bin はリポジトリ外・root 所有の可変ファイルであり、git では追跡されない):

```
a78b7f889843037d344f482623b3febd124ead5c1f34f134d9f1c2c78cd0f829  ../suisho11/nn.bin
312b1368348e83fe546f45983e49bd354636609b1ad4857a033c4ab0e4aafff6  ../YaneuraOu/source/YaneuraOu-by-gcc
```

- ビルド環境: g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0。コンパイラが変わると
  同一ソース・同一フラグでもバイト一致・性能一致は保証されない
- `source/YaneuraOu-by-gcc` は Makefile の in-place 出力であり、**次のビルドで上書きされる**。
  ベースラインバイナリのアーカイブ: `~/engines/baseline-000/YaneuraOu-by-gcc` (sha256 同上)。
  ベースライン再計測時はアーカイブ側を `ENGINE=` で指定するか sha256 を確認すること
  (`bench_nps.sh` はヘッダにエンジン sha256 を出力する)

※ 報告中の hash 0x3c203e1c はアーキテクチャ hash であり、重みの fingerprint ではない。

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

## NPS ベースライン (bench, 2026-07-03)

`./bench_nps.sh 10000` (やねうら王 bench デフォルト4局面 × movetime 10秒 × 3回,
hash 1024MB, 中央値と min–max を記録):

| threads | NPS 中央値 | min–max | scaling |
|---|---|---|---|
| 1 | 467,764 | 465,951–470,802 | 1.00× |
| 2 | 940,238 | 931,355–957,435 | 2.01× |
| 4 | 1,879,826 | 1,863,572–1,900,876 | 4.02× |
| 8 | 3,698,334 | 3,693,286–3,758,488 | 7.91× |
| 16 | **7,177,147** | 7,094,389–7,201,109 | 15.34× |

- **最大 NPS ≈ 7.18M (16スレッド)**。EPYC 7302 は 16 物理コア / SMT なしのため 16 が上限
- run 間ばらつきは最大 ±1.5% 程度。5% を下回る速度差の主張にはこの分散を考慮すること
- スケーリングはほぼ線形 → 探索は綺麗に並列化されており、評価関数速度が支配的
- 計測条件: governor `schedutil`, boost 有効, アイドル状態のマシンで実施。
  比較計測は必ず同条件 (同 governor / 他負荷なし) で行い、`bench_nps.sh` ヘッダの
  条件出力を結果と一緒に保存すること
- 1スレッド値 (468k) が冒頭のアドホック値 (600k) より低いのは、bench 局面集に
  評価リフレッシュの多い複雑な中盤局面を含むため。**高速化の比較基準はこの bench 値を使う**

## TODO (次フェーズ)

- [x] 固定局面集での NPS 計測をスクリプト化 (`bench_nps.sh`)
- [ ] nn.bin parser (tools/) 実装 → フェーズ1
