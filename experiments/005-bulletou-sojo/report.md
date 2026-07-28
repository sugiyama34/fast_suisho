# 005-bulletou-sojo: BulletOu + 奏乗教師データで WCSC36 水匠級の評価関数が再現できるか

**判定:** **再現失敗 (共有コマンドの数値のまま、では)** — 学習は健全に収束したが、
得られたネットは Suisho 11 に対し **Elo −221 (95% CI −365 〜 −124)**、SPRT は
H0 (−30 Elo より弱い) を受理。ただし共有コマンドは `--superbatches` /
`--max-epochs` に「ここ調整」注記付きの**テンプレート**であり、そのままの値では
学習量が氏の「2 日」に全く届かない (下記解釈)。**氏の実際の 2 日分計算量での
再現可否は未検証** — 学習曲線は 16 epoch 時点でまだ改善中で、学習量を
桁で増やす追実験が次のステップ。

## TL;DR

やねうら氏共有のレシピ (BulletOu, `SFNN_halfka2_1024_7_64_k3k3`, 40M 局面 × 12 sb ×
16 epoch, Ranger, step LR) を奏乗教師データ 78 億局面で実行した。RTX PRO 5000
(Blackwell) では **1 run = 68 分** (2.32M pos/s) で完走し、floodgate.hcpe に対する
test accuracy は **69.7%** に到達。教師ファイル順を回転した 3 アームの差は
accuracy ±0.13pp と小さく、結果はデータ順にほぼ不感。しかしフェーズ3 自己対局
(vs Suisho 11, 校正済み movetime 240ms) は 32 ペア 64 局で **14/64 (21.9%)**、
Elo −221 で SPRT 棄却。「共有された数値そのままでは水匠級にならない」が結論。
テンプレートの学習量 (7.68 億×10 局面 = 68 分) と「2 日」の乖離が主因とみられる。

## 仮説

`hypothesis.md` 参照。決定指標はフェーズ3 プロトコルの自己対局 Elo 差
(pentanomial SPRT, elo0=-30 / elo1=0)。「再現成功」= 95% CI が -30 Elo 以上と交差。

## セットアップ

- 学習: BulletOu `9577f08` (cuda-cpp, ローカルパッチで sm_120 SASS 追加 — 数値経路不変)。
  コマンドは `command.md` / `run_training.sh`
- 教師: 奏乗 001–016 (16 × 488,964,981 = 78.2 億局面, PSV)。再シャッフルなし
- 検証: [takaoyamaoka/floodgate.hcpe](https://huggingface.co/datasets/takaoyamaoka/floodgate.hcpe)
  856,923 局面から毎 sb 300k サンプル (ユーザー指定。やねうら氏の私有 fg 系テストの公開代替)
- アーム: main (ファイル順 001→016) / var-b (009 起点回転) / var-c (005 起点回転)。
  BulletOu は初期化シード固定・再シャッフルなしのため、ファイル順回転が run 間分散の代理
- ハード: RTX PRO 5000 Blackwell 48GB × 3 (1 run = 1 GPU で 3 アーム並列)
- 記録: W&B `suisho/suisho-test` (per-sb train/test loss, accuracy, LR) + ローカル JSONL

## 結果 (学習フェーズ)

| Arm | floodgate acc (final) | floodgate loss (final) | train loss (final) | 16 epoch 完走 |
| --- | --- | --- | --- | --- |
| main  | **69.72%** | 0.07736 | 0.03348 | ✅ (early stop なし) |
| var-b | 69.46% | 0.07818 | 0.03221 | ✅ |
| var-c | 69.56% | 0.07866 | 0.03217 | ✅ |

![training curves](figures/training_curves.png)

![test accuracy zoom](figures/test_accuracy_zoom.png)

- 学習は安定 (NaN・発散なし)。毎 epoch の LR warm restart (step schedule) による
  のこぎり状の一時悪化を挟みつつ、test loss/accuracy は 16 epoch 通して改善が持続
  → `--max-epochs 16` は「まだ伸びる」側で打ち切られており、レシピの
  「ここ調整」ポイント (`--superbatches` / `--max-epochs`) には増やす余地がある
- アーム間差 (データ順感度): accuracy 差 max 0.26pp、test loss 差 max 0.0013。
  単一 run の結論としても実質的に頑健
- スループット 2.32M pos/s/GPU。1 run 68 分 (train 3306s + 検証/保存等)

## 結果 (強さ測定 — フェーズ3)

| 項目 | 値 |
| --- | --- |
| 対局数 | 32 ペア / 64 局 (SPRT が受理し打ち切り) |
| 得点 (main 側) | **14 / 64 (21.9%)** — 先手 10/32, 後手 4/32 |
| pentanomial (ペア得点 0/0.5/1/1.5/2) | [20, 2, 6, 2, 2] |
| Elo 差 (vs Suisho 11) | **−221** (95% CI **−365 〜 −124**) |
| SPRT (elo0=−30, elo1=0, α=β=0.05) | LLR −3.56 ≤ −2.94 → **H0 受理 (−30 Elo より弱い)** |
| 終局理由 | 詰み 47, 連続王手等の千日手負け 12, 千日手引き分け 4, 千日手勝ち 1 |
| 平均手数 | 109.3 手 (開始局面 24 手目から) |

条件: 同一エンジンバイナリ (`87857ca8…`, 2026-07-21 ビルド)、EvalDir のみ差し替え。
`go movetime 240`, Threads 16, USI_Hash 1024, ponder/定跡なし, 直列。
校正: median 1,974,173 nodes/move (PLAN 目標 ~2M と整合)。
開始局面: 互角局面集2025 の等間隔ストライド (`start_sfens_500.txt`)、先後入替。
生データ: `games/main-vs-suisho11/games.jsonl` (git 管理外)。

対局中にエンジンプロセスの散発的な silent crash が 2 回発生 (stderr 出力なし、
両陣営同一バイナリ、発生時はペアを再走して回復)。結果への系統的影響はないと判断。

## 解釈

1. **共有コマンドの数値そのままでは「WCSC36 水匠と同等」は再現しない。**
   −221 Elo は明確な差で、判定基準 (CI が −30 と交差) を大きく外れる
2. **主因は学習量の乖離とみられる。** 共有コマンドは `--superbatches 12` /
   `--max-epochs 16` に「ここ調整」注記が付いた**テンプレート**であり、この値では
   7.68 億×10 局面 = 本環境で 68 分・氏の環境でも数時間で終わる。「2 日くらい学習」
   という記述と桁で合わない。氏の実走は学習量が 1〜2 桁大きかった可能性が高い
   (教師ストリームは EOF で wrap するため epoch を増やすだけで学習量は稼げる)
3. その読みと整合して、**floodgate accuracy は 16 epoch 時点でまだ単調改善中**
   (69.7%、頭打ちの兆候なし)。学習の器 (アーキテクチャ・データ・トレーナ) は健全で、
   スケジュールだけが足りていないという形
4. 学習自体は極めて安定・決定的で、データ順の影響も誤差レベル (3 アーム差 0.26pp)。
   「BulletOu + 奏乗データ」というパイプラインの有効性自体はこの実験で確立できた

## 次のステップ

- **学習量スケール実験** (experiment-006 候補): 同一レシピで `--max-epochs` を
  10〜20 倍 (例: 160〜320 epoch ≒ 77〜154 億×10 局面 ≒ GPU 11〜23 時間/run)、
  もしくは「2 日」相当まで伸ばし、epoch ごとの floodgate accuracy と
  対 Suisho 11 Elo の関係を測る。3 GPU で長さ違いの 3 点を並列に取ると効率的
- 途中 checkpoint (0001〜0016 保存済み) vs Suisho 11 の Elo を数点測れば
  「学習量 → Elo」の外挿カーブが先に安く得られる
- やねうら氏に実際の `--superbatches` / `--max-epochs` / 総学習局面数を確認できると
  仮説の裏取りになる

## 妥当性への脅威

- **Suisho 11 の floodgate accuracy 参照値が未取得** (本エンジンビルドに静的 eval
  コマンドがなく、BulletOu の eval 例は旧 checkpoint 形式専用)。69.7% が
  「水匠級」からどれだけ離れているかの直接比較はできていない — 強さ判定は
  対局 Elo のみに依拠
- **テストセットが本人と異なる** (floodgate 2017-2018 vs 氏の非公開 fg2021 系):
  loss/accuracy の絶対値は氏の値と比較できない。early stop 挙動もズレうる
  (今回は 16 epoch 完走なので early stop 差は顕在化せず)
- **教師の消費範囲・順序が不明** (本実験は 001–016 固定 + 回転 2 アーム)
- **BulletOu の版差**: 氏の実行コミット不明。本実験 `9577f08`
- SPRT 実装は自作 (GSPRT 近似 + 分散下限 + 最小 32 ペア)。1 ペア即断バグを
  修正済み (修正前に棄却された pair 0 の 2 敗は games.jsonl に残置し再集計で使用)

## 再現

- リポジトリ commit: `7d25aaa` + 本実験フォルダ (experiments/005-bulletou-sojo/)
- データ取得: `data/teacher/sojo/download.sh` / floodgate.hcpe は HF から直接
- 学習: `bash experiments/005-bulletou-sojo/run_training.sh main 0` (GPU 必須。
  Claude Code sandbox は GPU 不可視のため sandbox off か実端末で — issue #13 参照)
- 対局: `data/matchenv/bin/python experiments/005-bulletou-sojo/match_runner.py
  --candidate-evaldir data/bulletou/eval/main --name main-vs-suisho11`
  (cshogi は numpy<2 制約のため専用 venv `data/matchenv/`)
