# nn.bin 統計 (tools/analyze_nn.py 生成)

- 入力: `../suisho11/nn.bin`
- arch 文字列: `ModelType=SFNNWithoutPsqt;Features=HalfKA(Friend)[131949->1024x2],Network=AffineTransform[1<-64](ClippedReLU[64](AffineTransform[64<-7](ClippedReLU[7](AffineTransform[7<-2048](InputSlice[2048(0:2048)]))))){LayerStack=9}`
## FeatureTransformer

- weights: shape (131949, 1024) (feature × channel), int16
- weights 範囲: [-231, 251], 平均 0.002, 標準偏差 4.89
- biases 範囲: [-102, 90], 平均 8.0

| 条件 | 割合 |
|---|---|
| \|w\| ≤ 0 | 28.13% |
| \|w\| ≤ 1 | 60.72% |
| \|w\| ≤ 2 | 74.09% |
| \|w\| ≤ 4 | 87.16% |
| \|w\| ≤ 8 | 95.36% |
| \|w\| ≤ 16 | 98.71% |

## FT 出力チャネル重要度 (L1 ノルム)

- チャネル数: 1024
- L1 ノルム範囲: [148,459, 1,978,123], 中央値 282,797
- 最大/最小比: 13.3×
- 上位 512 チャネルが全 L1 質量に占める割合: 60.57%

## LayerStacks (9 スタック)

| layer | shape (out × in, padding 除外) | 重み範囲 | ゼロ割合 |
|---|---|---|---|
| fc_0 | 8 × 1024 | [-128, 127] | 4.96% |
| fc_1 | 64 × 14 | [-127, 127] | 3.71% |
| fc_2 | 1 × 64 | [-90, 127] | 0.87% |

- fc_1 のパディング列 (入力 14〜31) が全て 0: True
