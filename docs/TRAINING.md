# 学習トラックの概要

作成: 2026-09-28

本リポジトリには 2 つの独立したトラックがある:

- **高速化トラック**: Suisho 11 の評価関数を速くする。計画は `docs/PLAN.md`
- **学習トラック**: BulletOu と奏乗教師データで評価関数を学習し、水匠 11 級の強さを目指す。
  本書がその入口

`docs/PLAN.md` は高速化トラック専用で、学習トラックの実験は意図的に載せていない
(2026-07-31 ユーザー決定)。両トラックは `experiments/` の通し番号を共有する。
次の実験番号は `experiments/` と open PR のブランチ名を確認してから決める。

## 実験一覧

| 実験 | 内容 | 結論 | 状態 |
| --- | --- | --- | --- |
| experiment-004 (`004-wandb-verify`) | WandB 統合の検証 | 運用規約は `docs/wandb-guide.md` | マージ済み (PR #11) |
| experiment-005 (`005-bulletou-sojo`) | やねうらお氏共有レシピ (sb=12 × 16 epoch) の再現 | 水匠 11 比 −293 Elo。テンプレート値では学習量が桁違いに不足 | マージ済み (PR #14) |
| experiment-006 (`006-standard-epoch`) | 学習量と LR 周期 (superbatches) のスケーリング | 氏の実設定 (sb=108 × 16) で −104 を再現。強さは教師 4〜5 周でピーク、以降は低下 | マージ済み (PR #15) |
| experiment-007 (`007-wrm-factorizer`) | WRM loss / factorizer 無効化 / LR 単発アニール | FV_SCALE 補正後、WRM loss の効果は +54 Elo。最良 net は水匠 11 比 −40〜−60 | PR #19 (open) |
| experiment-009 (`009-data-scaling`) | 教師データ量のアブレーション (100〜10%, 定跡除外は experiment-010) | 未着手。kajiki は GPU 律速で遅く、新サーバーで学習する | PR #23 (open) |

詳細は各実験フォルダの `report.md` を読むこと。

## 必ず押さえておく知見

1. **「epoch」は 2 つの意味がある** (experiment-005 で定義)
   - **BulletOu-epoch**: `--superbatches N` × 40M 局面の LR サイクル。データ量とは無関係
   - **standard-epoch (周)**: 教師全体 1 周 = 14,668,949,437 局面
2. **floodgate accuracy は強さの代理指標として使えない** (experiment-006, 007)。
   accuracy が同じでも Elo が 150 以上違う例、+54 Elo の差が accuracy で +0.15pp にしか
   現れない例がある。**レシピや checkpoint の選択は対局で行う**
3. **FV_SCALE を評価関数ごとに合わせる** (experiment-007 追記)。やねうら王は生評価値を
   `FV_SCALE` (既定 16) で割ってから探索に使い、適正値は net ごとに違う。
   experiment-005〜007 本文の Elo は両陣営 16 固定で測っており、~200 Elo 級の
   アーティファクトを含む。推定値は WRM 系 ≈ 25、sigmoid-MSE 系 ≈ 52、水匠 11 = 40。
   推定方法は `experiments/007-wrm-factorizer/fit_fv_scale.py` (PR #19 ブランチ)
4. **周回しすぎると弱くなる** (experiment-006)。sb=108 では 4.7 周がピークで、
   9.4 周では −263 まで悪化した
5. **学習速度は実時間で測る** (experiment-009 準備で確認)。BulletOu 進捗行の `pos/s` は
   GPU へのカーネル投入時間だけを数えることがあり、実時間の 6 倍の値が出た例がある
   (`docs/SETUP.md` §8)
6. **experiment-009 (データアブレーション系列) は PR #19 (experiment-007) の知見に依存しない**
   (2026-09-28 ユーザー決定)。レシピは experiment-006 のもの (sigmoid-MSE) を使い、
   対局は固定ノード数で行う
7. 対局条件は `docs/PLAN.md` フェーズ 3 のもの (movetime 240ms, 16 スレッド, hash 1024MB,
   互角局面集のストライド 500 局面)。新サーバーでは**ノード数の再校正が必要**
   (`match/calibrate_nodes.sh`)。CPU が変わると同じ movetime でもノード数が変わる

## 現時点の到達点

- 最良 net: experiment-007 の wrm arm, epoch 20
  (`data/bulletou/checkpoints/007-wrm/0020/nn.bin`, sha256 `0c325ad2…`)
- 適正 FV_SCALE (wrm 25〜28 / 水匠 40) で水匠 11 に対し **約 −40〜−60 Elo**
- 学習レシピ: BulletOu `2a8e5ed` + `--win-rate-model --loss-pow-exp 2.5`。
  完全なコマンドは `experiments/007-wrm-factorizer/run_training.sh`

## 環境とデータ

セットアップ手順は `docs/SETUP.md` (教師データの取得、BulletOu のビルド、対局用 venv)。

- WandB: entity `suisho`。experiment-005 は project `suisho-test`、
  experiment-006/007 は project `20260728_BulletOu_with_Sojo_data`
  (group = 実験名)。experiment-009 以降のデータアブレーション系列は project
  `data_ablation_study` (2026-09-28 作成)。`.claude/settings.json` の既定 project もこれに
  切り替えた。API キーはファイルに書かず、起動シェルの環境変数でだけ渡す (`docs/SETUP.md` §1)
- 学習の起動は `experiments/<実験>/train_supervised.py` 経由が標準
  (トレーナと W&B run のライフサイクルを一体で管理する)
- 旧開発機では GPU が Claude Code の sandbox から見えなかったため、学習は実端末から
  起動していた。2026-09-28 に sandbox を無効化したので、新サーバーでは
  セッション内から起動できるか確認すること
