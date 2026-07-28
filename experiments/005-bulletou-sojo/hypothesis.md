# 005-bulletou-sojo: BulletOu + 奏乗教師データで WCSC36 水匠級の評価関数が再現できるか

**日付:** 2026-07-27  ·  **状態:** done (2026-07-28 対局完了 — report.md 参照。
判定: 共有コマンドの数値そのままでは再現失敗、Elo −221。学習量の乖離が主因とみられる)

## 問い

やねうら氏の報告「BulletOu を使い、奏乗の教師データで 2 日くらい学習したら
WCSC36 の水匠と同じくらい強くなった」を、本人が共有したコマンド (レシピ) で再現できるか。

- トレーナ: [BulletOu](https://github.com/yaneurao/BulletOu) (`9577f08`, cuda-cpp backend)
- 教師データ: [washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique](https://huggingface.co/datasets/washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique)
  (PackedSfenValue 40B/局面, 30 ファイル × 488,964,981 局面 = 約 146.7 億局面, 587 GB)
- アーキテクチャ: `SFNN_halfka2_1024_7_64_k3k3` (= フェーズ0 ベースラインと同じ)

## 予測

学習は正常に収束し (train/test loss 単調減少)、得られた `nn.bin` は
Suisho 11 (WCSC36 水匠相当と仮定) に対し **Elo 差 -50〜0 程度** に収まる。
「同じくらい強い」の再現には成功側に倒れると予想するが、教師・テスト分布の差
(下記) により完全一致は期待しない。

## 決定指標

最終判定: **フェーズ3 プロトコル (docs/PLAN.md) による自己対局の Elo 差**
(学習ネット vs Suisho 11 `nn.bin`、同一エンジン・同一持ち時間 movetime 240ms、
互角局面 500 ペア、pentanomial SPRT)。

- **再現成功**: Elo 差の 95% 信頼区間が -30 Elo 以上と交差 (≒「同じくらい強い」)
- **再現失敗**: 95% CI 上限が -30 Elo を下回る

学習中の追跡指標 (中間モニタ、判定には使わない):
`test_value_loss` / `test_value_accuracy`。検証セットは
[takaoyamaoka/floodgate.hcpe](https://huggingface.co/datasets/takaoyamaoka/floodgate.hcpe)
(floodgate 2017-2018, 両者 R3500+, 856,923 局面。300k サンプル/検証イベント。
2026-07-27 ユーザー指定 — 当初案の奏乗 030 held-out から変更)。

## ベースライン

Suisho 11 の `nn.bin` (フェーズ0 で動作確認済みのもの)。同一アーキテクチャ・
同一エンジンバイナリなので、差は評価関数の重みのみ。

## セットアップ

- 検証変数: 「やねうら氏レシピの学習で水匠級ネットができるか」(単一 run の成否検証であり ablation ではない)
- 学習コマンド: `command.md` / `run_training.sh` 参照。sb = 40M 局面 × 12 sb/epoch ×
  最大 16 epoch = 76.8 億局面 (early stop あり: epoch 末検証で loss も accuracy も
  改善しなければ打ち切り)
- 教師: 奏乗 001–016 (78.2 億局面)。BulletOu は学習時再シャッフルなし・ファイル名順に消費
- テスト: floodgate.hcpe (上記。教師と分布が独立な out-of-distribution 検証)
- シード: BulletOu の重み初期化は**固定シード埋め込み** (0x5071_e001 等) で、
  同一コマンドの再実行は決定的。run 間分散を測るには教師ファイル順の回転
  (var-b アーム) を使う
- 最安の反証 run: smoke アーム (2M 局面 × 2 sb × 1 epoch) でパイプライン全体
  (loader / CUDA / 検証 / wandb 同期) を先に確認する

## 交絡・リスク (やねうら氏環境との差分)

1. **テストセットが本人と異なる**: 氏の `test20231010_fg2021_dls5_ryfc20_ev8250k825.hcpe`
   は非公開 (floodgate 2021 系とみられる)。公開の floodgate.hcpe (2017-2018) で代替する。
   同系統 (floodgate 実戦局面) だが年代・評価値注釈が違うため test loss の絶対値は
   比較不能。early stop の発火条件も氏の run とズレうる (学習局面数が変わりうる)
2. **教師の消費範囲**: 氏がどのファイル群をどの順で置いたか不明。本実験は 001–016 固定
3. **GPU**: RTX PRO 5000 Blackwell (48GB) — 氏の環境と異なる。学習結果は同一コマンドなら
   決定的だが、wall-clock (「2日くらい」) は環境依存
4. **BulletOu の版**: 氏の実行時点のコミット不明。本実験は `9577f08` (2026-07-27 clone)
5. ~~**「WCSC36 の水匠」= Suisho 11 の仮定**~~ → **確認済み (2026-07-27)**:
   本リポジトリの Suisho 11 (`SFNN_halfka2_1024_7_64_k3k3`) が WCSC36 の水匠である
   (たややん氏の発言によるとのこと。ユーザー確認)。ベースライン同定の懸念は解消
