# experiment-009 の結果 (自動生成)

`plot_results.py` が対局結果から作る (手で編集しない)。Elo は水匠 11 (FV_SCALE 32) 相手、300k ノード。
「対応のある差」は親と同じ開始局面のペアごとの得点差から計算した Elo 差 (対局は決定的なので、
アブレーションで指し手が変わらない局は親と同じ棋譜になる)。

## 稀な特徴量のアブレーション (provisional: full-e12 @ FV_SCALE 0, 400 pairs)

![all settings](../figures/ablation_merged.png)

| 順位 | モード | 下位 % | Elo vs 水匠 11 | 95% CI | 親との対応のある差 | 95% CI | 棋譜が変わったペア |
| --- | --- | --- | --- | --- | --- | --- | --- |

