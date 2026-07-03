# fast_suisho

Suisho 11 (水匠11) NNUE 評価関数の高速化・軽量化実験リポジトリ。

## Goal

- Suisho 11 の NNUE ネットワーク (`nn.bin`) に枝刈り (pruning)・量子化・構造縮小などを適用して評価を高速化し、探索 NPS を稼ぐことで棋力向上を狙う
- オリジナル NNUE vs 高速化 NNUE の自己対局 (最大1000局) で棋力を検証する

## Layout

```
fast_suisho/
├── networks/       # nn.bin 派生物 (バイナリは git 管理外)
├── tools/          # nn.bin の parse / prune / 再パック用スクリプト
├── match/          # 自己対局ハーネスと対局設定
├── experiments/    # 実験ごとの結果・レポート
└── docs/           # 計画・調査メモ
```

## External dependencies (siblings, not committed)

| Path | What | Pinned |
|---|---|---|
| `../YaneuraOu` | やねうら王 engine source (github.com/yaneurao/YaneuraOu) | commit `9133c527` (V9.60) |
| `../suisho11`  | Suisho 11 `nn.bin` (135 MB) + `sfnnwop-1536.h` | - |

## Architecture of Suisho 11 nn.bin

`SFNN_halfka2_1024-7-64-k3k3` 相当:

- Input features: HalfKA2 (friend side)
- Feature transformer: 1024 dims (PSQT なし / SFNNwoPSQT)
- LayerStacks: 9 (king3 × king3)
- Network per stack: fc0 1024→7(+1) sparse → CReLU/SqrCReLU → fc1 14→64 → CReLU → fc2 64→1

やねうら王の Makefile はこのアーキテクチャ名からヘッダを自動生成できるため、
本家ソースの改変は不要。

## Baseline engine build

```sh
cd ../YaneuraOu/source
make -j16 normal YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3 \
     COMPILER=g++ TARGET_CPU=AVX2 PYTHON=python3
```
