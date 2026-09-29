# experiment-009 作業ノート

実験の途中で分かったこと・決めたことを、トピックごとに記録する。最終的な `report.md` は
これらをまとめて書く。各ノートは「分かった時点で追記」し、日付を付ける。

| ノート | 内容 |
| --- | --- |
| [decisions.md](decisions.md) | 仮説の再設定 (2026-09-29) と設計上の決定事項・期限 |
| [bulletou-internals.md](bulletou-internals.md) | BulletOu `2a8e5ed` のソース調査: FT の factorizer・初期化・checkpoint 形式・nn.bin 形式・Ranger の挙動 |
| [feature-counts.md](feature-counts.md) | 特徴量 (one-hot) の出現回数の統計: 教師データ側・対局側、分布間の距離 (TV / JS) |
| [ablation.md](ablation.md) | アブレーション用ツール (`ablate/`) の仕様と検証、アブレーションの結果 |
| [server-suzuki.md](server-suzuki.md) | 新サーバー suzuki の環境構築と計測値 (学習スループット・対局速度) |
| [ratings.md](ratings.md) | 対局結果の一覧 (`match_queue.py` が自動で追記) と対局キューの使い方 |
