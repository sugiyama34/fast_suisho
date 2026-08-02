# 007-wrm-factorizer: やねうらお氏の 3 提案 (WRM loss / factorizer 無効化 / LR スケジュール) は Elo を改善するか

**日付:** 2026-07-31  ·  **状態:** planned

## 背景 (2026-07-31, やねうらお氏のコメント)

> 1. たぬきさんの実験にあるように tatara と同じ `--loss-pow-exp 2.5` オプションを
>    指定すると改善するかも。例: `--win-rate-model --loss-pow-exp 2.5`
> 2. 1 epoch の sb を大きくとるほど accuracy が改善することがわかっています
>    (1sb=40M pos として、36sb よりは 108sb、108sb よりは 324sb、…)。おそらく
>    いまのデフォルトの lr スケジューリングでは lr 下げるのが速すぎるのだと
>    思います。教師データも、もっとあると過学習しにくいこともわかっています。
> 3. L1 factorizer がデフォルトで share (一応共有する) になっとるのですが、
>    これ外したほうが最終的には高い accuracy になることがわかっています。
>    例: `--sfnn-factorizer none`

experiment-006 までの到達点: 氏の実設定 (sb=108 × 16 BulletOu-epoch) の再現 net が
**Elo −104** (vs Suisho 11)。強さのピークは教師 4〜5 周、また同一 run 内の
エポック軸では accuracy が 70.39% で飽和したまま Elo が −104 → −263 と悪化する
(= accuracy は過学習を検出できない) ことを観測した。

**006 の知見の射程に関する注意 (2026-07-31 ユーザー指摘)**: 006 の
「acc≥0.70 の 8 点で r = +0.01」は **範囲制限 (selection bias) の影響下にある** —
この 8 点の accuracy 分布は ~70.2〜70.4% と幅 0.2pp しかなく、説明変数側の分散が
ほぼゼロなので r ≈ 0 は機械的な帰結に近い。頑健なのは「同一 accuracy で Elo が
大きく異なる点が存在する (ep16 vs ep32)」= accuracy は Elo を**決定しない**、
という部分だけであり、「レシピ変更で accuracy が 70.4% の壁を超えたとき Elo も
上がるか」は観測範囲外で、**未検証**である。

**本実験の核心的な問い**: 氏の 3 提案の根拠はいずれも **accuracy の改善**である。
提案が実際に accuracy を既存プラトー (70.4%) の外へ押し上げるなら、それは 006 の
範囲制限の外側の初めてのデータ点になる。つまり本実験は (a) 提案の Elo 効果の検証で
あると同時に、(b) 「レシピ間の accuracy 差は Elo を予測するか」の初のクリーンな
テストでもある。

## コードの前提 (2026-07-31 調査済み)

- `--win-rate-model` / `--loss-pow-exp` は現在の手元 BulletOu (`9577f08`) には
  **存在しない**。upstream `origin/shogi-support` の **`2a8e5ed`**
  ("Use tatara-style WRM loss options") で追加された (手元から +16 commits)
- `--loss-pow-exp` は `--win-rate-model` 併用時のみ有効。tatara デフォルトは 2.0、
  氏の推奨は 2.5。loss は `|WRM(net) − WRM(target)|^p`
- `--sfnn-factorizer` のデフォルトは **`shared`** (006 の −104 net もこれで学習
  されている)。`none` で全 factorizer 項を無効化
- ⚠️ `9577f08 → 2a8e5ed` の差分には新フラグ以外の変更も含まれる
  ("Optimize SFNN CUDA train-step hot path", PSV `.bin` 扱い変更等)。
  **新コミットでの再現コントロール arm が必須** (下記)
- ビルド時は local sm_120 patch (`crates/cuda_cpp/build.rs`) の再適用が必要

## 問い

1. `--win-rate-model --loss-pow-exp 2.5` は、sb=108 レシピの Elo を改善するか
2. `--sfnn-factorizer none` は同レシピの Elo を改善するか
3. **LR スケジュールの単発アニール化** (warm restart を廃し、全学習予算で
   1 回だけ lr → lr-min に減衰) は Elo を改善するか — 氏の「デフォルトの
   lr スケジューリングでは lr 下げるのが速すぎる」という機構仮説を、
   sb 拡大 (氏の対症療法) ではなくスケジュール自体の変更で直接検証する
   (2026-07-31 ユーザー提案)
4. 上記で効いた lever は加法的か (併用が単独より良いか)

「教師データを増やす」(提案 2 後半) は本実験のスコープ外 — 奏乗は unique 147 億
局面で頭打ちであり、データ拡充は別実験 (教師生成 or 追加データセット) として扱う。

## 予測

- **注意 (2026-07-31 ユーザー指摘)**: 氏の報告はいずれも **accuracy についてのみ**
  であり、Elo (rating) の改善報告ではない。したがって wrm / nofact の Elo 効果は
  中立予測 (−20〜+30 Elo、accuracy-only に終わる可能性が相応にある)
- **lr1shot (単発アニール) が本命**: LR スケジュールは最適化そのものを変えるため
  accuracy だけでなく実戦強さに効く機構的経路が明確。006 の「強さは 4〜5 周で
  ピーク後に悪化」も warm restart 下の観測であり、アニール終端をピーク近傍に
  合わせる単発スケジュールは +20〜+60 Elo を予測
- 副次予測: 各 arm の (Δaccuracy, ΔElo) の対応関係そのものが
  「レシピ間 accuracy 差の proxy 性能」の新データになる。accuracy が上がって
  Elo が上がらないケースも、上がるケースも、どちらも報告価値がある

## 決定指標

- **一次**: vs Suisho 11 の Elo (フェーズ3 条件: movetime 240ms ≒ 2M nodes/move・
  16 threads・hash 1024・互角局面集ストライド 500)
- **最終判定**: 新レシピ最良 checkpoint vs **006-yane ep16 net (−104)** の
  **直接対局** (pentanomial SPRT, elo0=0, elo1=+50, α=β=0.05)。
  絶対スケール較差 (vs Suisho 11 経由) より直接対局のほうが分散が半分で済む
- accuracy / test loss は checkpoint 選別には使わない (006: 同一 accuracy で
  Elo が大きく異なる)。ただし各 arm の accuracy は必ず記録し、
  (Δaccuracy, ΔElo) の対応をレシピ間 proxy 性能の新データとして報告する

## ベースライン

1. **006-yane20 の 0016 checkpoint** (−104, CI −171〜−43) — 旧コミット・旧レシピの実測点
2. **arm `ctrl`** (新設): BulletOu `2a8e5ed` + 006 と同一フラグの再学習 —
   コミット更新自体の影響を分離するコントロール。ctrl ≈ −104 なら旧実測点と接続できる

## セットアップ

### フェーズ 0: BulletOu 更新 + smoke (GPU ~10 分)

- `2a8e5ed` へ更新、sm_120 patch 再適用、`cargo build --release --example bulletou`
- sb=1 × 1ep の smoke run で新旧フラグの受理と学習開始を確認

### フェーズ 1: 5 arm (総局面数 864 億で統一, GPU 計 ~52h, 2 GPU で ~26h)

全 arm 共通: 006 `run_training.sh` と同一 (教師 = 奏乗 30 ファイル、
検証 = floodgate.hcpe、arch `SFNN_halfka2_1024_7_64_k3k3`、lr 0.000875、
ranger)。sb=108 系は 20ep とし、ピーク領域 (教師 4〜5 周 = ep14〜17 付近) を
確実に跨ぐ。

| arm | 変更点 | GPU 時間 |
| --- | --- | --- |
| `ctrl` | なし (006 再現 @ 新コミット, sb=108 × 20ep) | ~10.5h |
| `wrm` | `--win-rate-model --loss-pow-exp 2.5` | ~10.5h |
| `nofact` | `--sfnn-factorizer none` | ~10.5h |
| `both` | wrm + nofact | ~10.5h |
| `lr1shot` | `--superbatches 2160 --max-epochs 1 --save-rate 108` | ~10.5h |

**`lr1shot` の設計**: 総局面数は他 arm と同一 (2160 × 40M = 864 億 ≒ 教師 5.9 周)
だが、LR 減衰周期を全学習予算に一致させ **warm restart を 1 度も行わない**
(step schedule は `--superbatches` から gamma を自動計算し、epoch 末で
lr = lr-min に到達する)。アニール終端 (5.9 周) は 006 のピーク (4〜5 周) の
直後に来る設計。`--save-rate 108` で 108 sb (= 43.2 億局面) ごとに checkpoint を
保存し、他 arm の「ep k」と同一グリッドで比較できるようにする。
これは ctrl と「LR スケジュールのみが違う」対照であり、氏の sb ラダー
(108 → 324 → …) の極限を 1 本で先取りする。中間案 (sb=108 のまま
`--lr-step-gamma` で減衰だけ遅くする = restart は維持) は、lr1shot が効いた場合の
分解実験として保留

### フェーズ 2: Elo 測定 (学習と非同時実行)

1. **スクリーニング**: 各 arm の ep16 / ep20 相当 (10 点。lr1shot は checkpoint
   1728sb / 2160sb) × 固定 50 ペア (CI ±70 級)。ctrl が −104 と整合するかをまず確認
2. **精密測定**: スクリーニング上位 2 点に +150 ペア (計 200 ペア, CI ±35 級)
   vs Suisho 11
3. **最終判定**: 最良 checkpoint vs 006-yane ep16 の直接 SPRT (上記)

### フェーズ 3 (条件付き): 勝者 lever の併用 (GPU ~10.5h/本)

フェーズ 2 で ctrl 比 +30 Elo 以上の lever が複数あれば、それらを併用した 1 本を
追加 (例: lr1shot + wrm)。単独勝者が lr1shot のみなら、分解実験
(sb=108 + `--lr-step-gamma` 遅延 = 「減衰速度 vs restart 有無」の切り分け) を
優先候補とする。勝者がなければ実施しない

## コスト見積り

- GPU: フェーズ1 = ~52h (2 GPU 並列で実時間 ~26h)、フェーズ3 = ~10.5h/本
- 対局: スクリーニング 500 ペア + 精密 300 ペア + SPRT (可変) ≒ 006 の全測定
  (846 ペア) と同規模。学習終了後に実施 (CPU 競合回避)
- 前提: `/sandbox` off (GPU アクセス)、W&B project は settings.json で
  ローカル切替 (コミットしない)

## 交絡・リスク

- **コミット更新の交絡**: 新フラグ検証は必ず同一コミット内の arm 比較 (vs `ctrl`)
  で行う。006 の −104 との比較は ctrl の再現性確認を経て参考値扱い
- **CI 解像度**: 200 ペアでも ±35 Elo。+20 Elo 級の改善は直接対局 SPRT でしか
  解像できない。「差なし」の結論は「±35〜50 Elo 級の差はない」の意味に限定する
- **ピーク位置の移動**: 新 loss / 新スケジュールで最適 checkpoint が 4〜5 周から
  ずれる可能性。特に lr1shot は restart なしのため過学習の力学自体が変わり、
  終端 (5.9 周) より手前が最良になり得る。ep16/ep20 相当の 2 点スクリーニングで
  大きなずれは検出できるが、細かい位置は追加測定が要る
- **lr1shot の総予算とアニール終端の交絡**: lr1shot は「終端 = 5.9 周」に固定した
  1 点のみの検証であり、終端位置の最適化はしない (効いた場合の後続実験とする)
- **WRM loss と lambda の相互作用**: lambda はデフォルト 1.0 (純 eval) のまま固定。
  tatara 側の推奨 lambda が異なる場合、その差は本実験では扱わない
- エンジン silent crash (006 で 2〜6%/ペア) は respawn+retry で吸収 (既存 harness)

## 再現 (実行時に確定)

- 学習: `run_training.sh` (006 版を拡張して本フォルダに作成予定)
- 測定: `experiments/005-bulletou-sojo/match_runner.py` / `match_sprt.py` (既存)
- BulletOu: `2a8e5ed` + sm_120 patch (パッチ内容は metadata.json に記録予定)
