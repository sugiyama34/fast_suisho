# 006-standard-epoch: 学習量とLR周期 (superbatches) を伸ばせば奏乗レシピはどこまで強くなるか

**日付:** 2026-07-28 (やねうらお氏の実設定公開 [同日] を受けて改訂)  ·  **状態:** running

用語 (experiment-005 で定義): **BulletOu-epoch** = `--superbatches N`×40M 局面の
LR サイクル (005 では 4.8 億局面)。**standard-epoch** = 教師データ全体 1 周
(奏乗全 30 ファイル = 約 146.7 億局面)。

## 前提の更新 (2026-07-28, やねうらお氏本人のコメント)

> 1 epochでlr下がりきるスケジューリングなので、superbatch 上げないとlr下がるの
> 速すぎて収束の仕方が悪いです。また、その状態で20 epochぐらい回さないといかんです。
> それで、そうすると(sb=108で16epochにて)accuracy自体は水匠11Plusを超えるのですが、
> (正確に計測したところ)強さは水匠11PlusよりR100ぐらい低いっぽいです。
> sb上げれば、もっとつよなるはずです。

- **氏の実設定が判明**: `--superbatches 108` × 16 BulletOu-epoch = **約 691 億局面
  ≒ 4.7 standard-epoch** (本環境 GPU で約 8.4 時間)。LR 減衰周期は 1 BulletOu-epoch
  = 43.2 億局面
- **元の主張は本人により訂正**: 「水匠と同じくらい」→ 正確な計測では
  **水匠11Plus より R100 ほど低い** (accuracy は上回る)。005 の否定的結果と
  矛盾しなくなった
- 予測として「sb をさらに上げれば強くなる」が示された

## 問い

1. **氏の実設定 (sb=108 × 16ep) を再現すると、本当に Suisho 11 比 ≈ −100 Elo 域に
   到達するか** (訂正後の主張の検証)
2. **sb をさらに上げる (sb=367 ≒ standard-epoch 周期) と、氏の予測どおり
   さらに強くなるか**
3. 学習量 → Elo カーブ (BulletOu-epoch 系列 / 各 sb 系列) はどう振る舞うか

## 予測

- sb=108 再現 run は floodgate accuracy で 005 (sb=12) を明確に上回り、
  Elo は **−150〜−50** に入る (氏の「R100 低い」と整合すれば検証成功)
- sb=367 run は sb=108 を上回る (氏の予測の検証)。差の大きさは未知

## 決定指標

各 checkpoint の vs Suisho 11 Elo (フェーズ3 条件、**固定 50 ペア = 100 局**の
点推定 ± 95% CI。カーブ用に SPRT は使わない)。最終判定のみ experiment-005 と同一の
pentanomial SPRT (elo0=−30, elo1=0, α=β=0.05)。

## ベースライン

- Suisho 11 `nn.bin` (対局相手)
- experiment-005 の 2^n BulletOu-epoch checkpoint 系列 (同一レシピの序盤カーブ)

## セットアップ

- **フェーズ a**: 005-main の checkpoint 0001/0002/0004/0008/0016
  (= 1/2/4/8/16 BulletOu-epoch, sb=12 系列 = 4.8〜76.8 億局面) を固定 50 ペアで測定
  (「LR 周期が短すぎる場合」の参照カーブ)
- **フェーズ b — 2 アーム並列 (GPU 0/1)**:
  - **arm `yane20`**: `--superbatches 108` + `--max-epochs 20` = 864 億局面 (GPU ~10.5h)。
    学習は決定的なので、この run の **0016 checkpoint = 氏の実設定 (sb=108×16ep) の
    net そのもの**、0020 = 氏の「20 epoch ぐらい」指針の net。1 run で両方得る
  - **arm `std`** (「sb 上げればもっと強い」の検証): `--superbatches 367`
    (LR 周期 ≒ 教師 1 周) + `--max-epochs 16` = 2350 億局面 ≒ 16 standard-epoch
    (GPU ~28h)
  - 共通: 教師 = 奏乗全 30 ファイル、検証 = floodgate.hcpe、supervisor で W&B 追跡
    (W&B project: `20260728_BulletOu_with_Sojo_data`, 2026-07-28 ユーザー作成)
- **フェーズ c**: yane20 の 1/2/4/8/16/20、std の 1/2/4/8/16 BulletOu-epoch
  checkpoint をフェーズ a と同条件で測定 (sb=12 / 108 / 367 の 3 本のカーブが揃う)。
  **最終判定**: yane20 の 0016 checkpoint (= 氏の実設定 net) に対し SPRT
  (elo0=−150, elo1=−50; 「R100 低い」の検証) と、arm std 最終 vs yane20 最終の直接比較
- 対局条件は experiment-005 と完全同一 (movetime 240ms・16 threads・hash 1024・
  互角局面集ストライド 500・同一エンジンバイナリ)。CPU 競合を避けるため
  **対局と GPU 学習は同時に走らせない** (校正条件の維持)
- 学習は BulletOu の固定シードで決定的 (単一 run で足りる。データ順感度は 005 で
  誤差レベルと確認済み)

## 交絡・リスク

- **「水匠11Plus」と本実験ベースライン (Suisho 11 = WCSC36 水匠) の同一性が未確認**。
  別物なら「R100 低い」の照合に数十 Elo のずれが乗りうる (要ユーザー確認)
- 氏の sb=108 run の他の設定 (教師ファイル構成・test set・BulletOu 版) は依然不明
- 周回学習 (>1 standard-epoch) では教師の再利用が始まる (過学習・改善鈍化はカーブと
  して観察対象)
- sb=12/108/367 の 3 系列は LR スケジュール周期が異なるため、同一局面数でも
  同 Elo とは限らない (それ自体がスケジュール効果の観察になる)
