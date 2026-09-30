# experiment-009 の結果 (自動生成)

`plot_results.py` が対局結果から作る (手で編集しない)。すべて水匠 11 (FV_SCALE 32) との対局、300k ノード。
**基準は常に full** (最良 epoch・最良 FV_SCALE)。対局は決定的なので、アブレーションで指し手が変わらない局は
full と同じ棋譜になり、同じ開始局面どうしの差は対応のある比較になる。

## 稀な特徴量のアブレーション (provisional: full-e12 @ FV_SCALE 40)

![all settings](../figures/ablation_merged.png)

| 順位 | モード | 下位 % | Elo vs 水匠 11 | 95% CI | 同じ局面での full-e12 | 局面数 | 棋譜が変わったペア |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 教師 | zero | 5 | -136.5 | [-162, -112] | -140.0 | 400 | 115 |
| 教師 | zero | 10 | -143.6 | [-170, -118] | -140.0 | 400 | 190 |
| 教師 | zero | 20 | -134.4 | [-160, -110] | -140.0 | 400 | 257 |
| 教師 | zero | 30 | -130.4 | [-157, -105] | -140.0 | 400 | 238 |
| 対局 | zero | 5 | -131.9 | [-158, -108] | -140.0 | 400 | 119 |

![Training-dist. rank, zero](../figures/ablation_teach_zs.png)
![Match-dist. rank, zero](../figures/ablation_match_zs.png)
