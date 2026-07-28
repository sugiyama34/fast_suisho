# 006-standard-epoch: 学習量 (standard-epoch) を伸ばせば奏乗レシピは水匠級に到達するか

**日付:** 2026-07-28  ·  **状態:** running

用語 (experiment-005 で定義): **BulletOu-epoch** = `--superbatches N`×40M 局面の
LR サイクル (005 では 4.8 億局面)。**standard-epoch** = 教師データ全体 1 周
(奏乗全 30 ファイル = 約 146.7 億局面)。

## 問い

experiment-005 は共有コマンドのテンプレート値 (16 BulletOu-epoch ≒ 0.52
standard-epoch) で Elo −293 だった。学習量を standard-epoch 単位で伸ばしたとき、
「学習量 → Elo」カーブはどう振る舞い、水匠 (Suisho 11) 級 (Elo ≧ −30) に到達するか。

## 予測

- Elo は学習量に対して概ね log-linear に改善し、1 standard-epoch (147 億局面) で
  −150〜−100、16 standard-epoch (2350 億局面 ≒ 「2 日くらい」相当) で −50〜0 に
  達する (到達すればやねうら氏の報告と整合)
- 1 standard-epoch を超えた周回学習でもカーブは (勾配を落としつつ) 改善を続ける

## 決定指標

各 checkpoint の vs Suisho 11 Elo (フェーズ3 条件、**固定 50 ペア = 100 局**の
点推定 ± 95% CI。カーブ用に SPRT は使わない)。最終判定のみ experiment-005 と同一の
pentanomial SPRT (elo0=−30, elo1=0, α=β=0.05)。

## ベースライン

- Suisho 11 `nn.bin` (対局相手)
- experiment-005 の 2^n BulletOu-epoch checkpoint 系列 (同一レシピの序盤カーブ)

## セットアップ

- **フェーズ a**: 005-main の checkpoint 0001/0002/0004/0008/0016
  (= 1/2/4/8/16 BulletOu-epoch = 4.8〜76.8 億局面) を固定 50 ペアで測定
- **フェーズ b**: standard-epoch 学習 — `--superbatches 367` (367×40M ≒ 146.8 億 ≒
  教師 1 周。LR step 減衰の周期も 1 standard-epoch に伸びる) + `--max-epochs 16`、
  教師は奏乗全 30 ファイル (ファイル名順)、検証は floodgate.hcpe (005 と同一)。
  supervisor (`train_supervised.py` 系) で W&B 追跡
- **フェーズ c**: フェーズ b の 1/2/4/8/16 standard-epoch checkpoint をフェーズ a と
  同条件で測定
- 対局条件は experiment-005 と完全同一 (movetime 240ms・16 threads・hash 1024・
  互角局面集ストライド 500・同一エンジンバイナリ)。CPU 競合を避けるため
  **対局と GPU 学習は同時に走らせない** (校正条件の維持)
- 学習は BulletOu の固定シードで決定的 (単一 run で足りる。データ順感度は 005 で
  誤差レベルと確認済み)

## 交絡・リスク

- やねうら氏の実設定 (superbatches / max-epochs / LR 周期) は未確認のまま
  「1 epoch = 教師 1 周」解釈を第一候補として走らせる (たややん氏が氏に確認中。
  判明した時点で設定を合わせた run に切り替え)
- 周回学習 (>1 standard-epoch) では教師の再利用が始まる (過学習・改善鈍化の可能性
  はカーブとして観察対象)
- フェーズ a (BulletOu-epoch 系列) とフェーズ c (standard-epoch 系列) は LR
  スケジュール周期が異なるため、同一局面数でも同 Elo とは限らない (それ自体が
  スケジュール効果の観察になる)
