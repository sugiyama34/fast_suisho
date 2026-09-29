# loss の測定ツール (`loss_eval/`, 2026-09-30)

`decisions.md` の「loss の測り方」の実装。全 arm・全ネットで共通の固定標本 (奏乗の局面と奏乗の評価値) の上で、
**nn.bin (やねうら王が読む量子化ネット)** の loss を CPU で測り、局面の稀さで層別する。
アブレーションしたネットは nn.bin しか無いので、checkpoint も含めて nn.bin で測る。GPU は使わない。

## 構成

| ファイル | 内容 |
| --- | --- |
| `loss_eval/src/main.rs` | Rust 製 (`loss_eval`)。BulletOu `2a8e5ed` の `bulletou_lib` を path 参照 (CUDA なし)。サブコマンド `features` / `scan` / `eval` |
| `loss_eval/make_samples.py` | 固定標本を作る (`/mnt/nvme1/sugiyama/loss_samples/`) |
| `loss_eval/loss_eval.py` | ネット 1 つの loss を標本ごと・層ごとに JSON に書く。`--diff` で 2 ネットの対応のある差 |
| `loss_eval/validate_engine.py` | 順伝播を対局用やねうら王の静的評価値と局面ごとに突き合わせる |

ビルド: `docs/SETUP.md` のツールチェインで `cd experiments/009-data-scaling/loss_eval && cargo build --release -j 8`
(`target-cpu=x86-64-v3`)。バイナリ sha256 は各 JSON の `tool_sha256` に記録する。

## 順伝播 (やねうら王 SFNN と同一の整数演算)

特徴抽出は `ShogiHalfKa2::map_features`、layer stack の番号は `ShogiKingRankBucket::<9>` をそのまま呼ぶ
(学習・validation と同じ関数)。k3k3 の規則は BulletOu (`crates/bulletou_lib/src/game/outputs.rs`) と
やねうら王 (`evaluate_nnue.cpp` の `stack_index_for_nnue`) で同じ:

- 自玉の段 = 手番側から見た段 (一段目 = 0)、相手玉の段 = 相手から見た段
- stack = `{0,0,0,3,3,3,6,6,6}[自玉の段] + {0,0,0,1,1,1,2,2,2}[相手玉の段]`
  (両玉とも自陣の下 3 段なら 8。標本の約 83% が stack 8)

整数演算 (`SFNN_halfka2_1024-7-64-k3k3.h` と各層の実装から):

1. FT: 各視点で bias + 40 行の和。エンジンは読み込み時に FT の重みとバイアスを 2 倍して i16 で累積する
   (`scale_weights`)。累積値 `a` を前半 512 は `[0, 254]`、後半 512 は上だけ 254 でクリップし、
   `floor(a0·a1 / 512)` (負は 0) を u8 出力にする。手番側 512 + 非手番側 512 = 1024。
   i16 のオーバーフローは検出して数える (これまで 0 件)
2. `fc_0` (1024 → 8, i8 重み, i32 バイアス)。出力 `y0[0..7]` から
   SqrClippedReLU `min(127, y0² >> 19)` と ClippedReLU `clamp(y0 >> 6, 0, 127)` を並べた 14 次元
3. `fc_1` (14 → 64) → `clamp(y1 >> 6, 0, 127)` → `fc_2` (64 → 1)。出力 = `fc_2` + `y0[7]` (shortcut)
4. 生出力 `raw` のスケールは QA·QB = 127 × 64 = 8128。**logit = raw / 8128** が学習時の出力 (sigmoid の前) に対応する。
   エンジンの評価値は `raw / FV_SCALE` (0 方向に切り捨て, ±31753 でクリップ)

`--state state.bin` を付けると、checkpoint の畳み込み前の f32 重みを BulletOu の CPU validation
(`cuda_cpp_sfnn_weights_for_cpu_validation`: FT の factorizer と layer stack の shared factorizer を畳み込む) と
同じ手順で読み、`fast_sfnn.rs` の scalar forward で float の logit も計算する (検証用)。

## loss の定義

BulletOu の validation (`test_value_loss`, `validate.rs` の `compute_sign_accuracy_with_loss`) と同じ:

- `target = λ·sigmoid(score / 290) + (1 − λ)·result` (λ = 1.0 なので勝敗は使わない。`--lambda` で変更可)
- `loss = (sigmoid(logit) − target)²` を **|score| < 32000 の record で平均** (|score| ≥ 32000 は学習で重み 0)
- 学習ログの `train_value_loss` は重み 0 の record もバッチの分母に含める (`loss_sigmoid_mse_reduce_kernel` の
  `/ batch`) ので、同じネット・同じ局面集合でも約 0.951 倍の値になる。本ツールは validation の定義
- **MAE (cp)** = 同じ record 上の `|290·logit − score|` の平均 (教師のスケール)。対局時のエンジンの表示値
  (FV_SCALE 16 で `508·logit`) とは 1.75 倍違う
- 集計は float64。Rust 側も BulletOu の関数そのもので loss を表示する (f32 の逐次加算なので 100 万件では
  相対 6 × 10⁻⁴ ほどずれる。JSON の `bulletou_validate_loss_f32_unweighted` は照合用)

## 検証

### 1. BulletOu の test loss の再現 (float と量子化の差)

`009-p10-smoke/0001` (sb 2 × 40M 局面だけ学習した smoke。nn.bin sha256 `466008e6…`)、
`--test-teacher floodgate.hcpe --test-positions 300000 --test-seed 20260928` (BulletOu の
`read_random_teacher_positions` を呼ぶので標本は学習時と同一):

| 順伝播 | accuracy | loss |
| --- | --- | --- |
| BulletOu の学習ログ (`learn.log`, GPU, float) | 0.6398600 | 0.08677422 |
| 本ツール float (`state.bin`) | **0.6398600** (191,958 / 300,000) | **0.08677422** |
| 本ツール: nn.bin の重みを逆量子化 + float の活性 | 0.6379433 | 0.08988249 |
| 本ツール 量子化 (nn.bin, エンジンと同一) | 0.6376600 | 0.08990008 |

- float は学習ログと 8 桁一致 → 標本の抽出・特徴・stack・factorizer の畳み込み・loss の式が BulletOu と同一
- 量子化との差 (+0.0031) は**ほぼ全部が重みの丸め** (活性の丸めの寄与は 0.00002)。logit の差は平均 0.143
  (41 cp)、最大 0.446。この smoke ネットは L1 の重みがほぼ初期値のままで、`|w| × 64` の中央値 1.75、
  64 倍して丸めたときの相対 RMS 誤差が 11% (L2: 2.8%, L3: 4.8%) あるため。十分学習したネットでは
  重みが大きくなり相対誤差は小さくなるはずだが、float との差は checkpoint ごとに `--state` で確かめられる
- 本ツールの loss は「エンジンが実際に使うネット」の loss であり、学習ログの test loss とは量子化の分だけ違う

### 2. やねうら王との局面ごとの一致

`validate_engine.py`: 対局用バイナリ `~/engines/009-finny-avx2/YaneuraOu-by-gcc` (sha256 `423b6b1c…`) に
`position sfen …` → `e` (`Eval::evaluate()` の値を表示する USI 拡張コマンド) を送り、FV_SCALE = 1 (生出力) と
16 (対局時) の両方で比較:

| ネット | 局面 | 不一致 |
| --- | --- | --- |
| 009-p10-smoke/0001 | floodgate 標本から 200 | 0 / 0 (FV_SCALE 1 / 16) |
| 009-p10-smoke/0001 | A_s1 から 500 | 0 / 0 |
| 009-p10-smoke/0001 | C_s1_rare から 500 (稀な局面) | 0 / 0 |
| 同 zero-full5 アブレーション (下記) | C_s1_rare から 500 | 0 / 0 |
| 水匠 11 (`~/suisho11/nn.bin`, sha256 `a78b7f88…`, 同じ形の nn.bin) | A_s1 / C_s1_rare から 500 ずつ | 0 / 0 |

- FV_SCALE = 1 では |raw| > 31753 がクリップされる (500 局面中 135〜228)。それらも FV_SCALE = 16
  (クリップなし, `trunc(raw / 16)`) で一致している
- 標本・floodgate のどのネットでも FT の i16 累積のオーバーフローは 0 件

### 3. アブレーション

`select_features.py --arm full --percent 5` (full の教師出現回数で下位 5% = 6,153 特徴量, 最大出現回数 14,297)、
`ablate.py --mode zero-specific` で `009-p10-smoke/0001` をアブレーション (nn.bin sha256 `51529579…`)。
`loss_eval.py --diff` で元のネットと比べると:

- 出力が変わった record は**すべて**アブレーション対象の特徴量を含む (A_s1: 対象を含む 1,831 record のうち
  1,827 が変化、対象を含まないのに変化した record は 0。B_s0: 1,893 中 1,888, 0)。対象を含むのに変わらない
  数 record は、smoke ネットでは `w_specific` がほぼ初期値で 0 にしても丸めた値がほとんど変わらず、
  その小さな変化も途中のクリップや整数の切り捨てで消えるため
- 層別: `min_count` ≥ 10⁵ の層は出力が完全に同一 (Δloss = 0)。[10⁴, 10⁵) は 42,962 中 946 record だけ変化、
  10⁴ 未満はほぼ全 record が変化
- 一様標本では稀な層が薄く差は検出できないが、層別標本 (C, D) では smoke ネットでもアブレーションによる
  loss の増加が稀な層だけに有意に出る (Δloss = アブレーション − 元, 対応のある差, ± は SE):

| min_count | C_s1_rare Δloss | 変化した record | D_s0_rare Δloss | 変化した record |
| --- | --- | --- | --- | --- |
| [1, 10) | +8.8 × 10⁻⁵ ± 8.7 × 10⁻⁵ | 216 / 221 | +1.4 × 10⁻⁴ ± 1.2 × 10⁻⁴ | 206 / 207 |
| [10, 10²) | +9.3 × 10⁻⁵ ± 1.7 × 10⁻⁵ | 5,160 / 5,212 | +4.0 × 10⁻⁵ ± 1.5 × 10⁻⁵ | 5,013 / 5,070 |
| [10², 10³) | +9.5 × 10⁻⁵ ± 0.7 × 10⁻⁵ | 24,556 / 24,804 | +8.4 × 10⁻⁵ ± 0.6 × 10⁻⁵ | 24,562 / 24,802 |
| [10³, 10⁴) | +1.43 × 10⁻⁴ ± 0.13 × 10⁻⁴ | 99,823 / 100,000 | +1.30 × 10⁻⁴ ± 0.13 × 10⁻⁴ | 99,843 / 100,000 |
| [10⁴, 10⁵) | +4.2 × 10⁻⁶ ± 2.2 × 10⁻⁶ | 2,242 / 100,000 | +1.3 × 10⁻⁶ ± 2.3 × 10⁻⁶ | 2,224 / 100,000 |
| ≥ 10⁵ | 0 (完全に同一) | 0 | 0 | 0 |
| 母集団全体 (重み付き) | +2.9 × 10⁻⁷ ± 1.0 × 10⁻⁷ | | +1.5 × 10⁻⁷ ± 1.0 × 10⁻⁷ | |

  (元の smoke ネットの層ごとの loss は 0.03〜0.13 程度。smoke は動作確認用で、数値そのものに意味は無い。
  十分学習したネットでの差の大きさは未測定)
- 出力: `/mnt/nvme1/sugiyama/loss_eval/009-p10-smoke-0001/`, `.../009-p10-smoke-0001-zero-full5/`,
  差は `.../diff-009-p10-smoke-0001-zero-full5.json`

## 標本 (`/mnt/nvme1/sugiyama/loss_samples/`)

| 名前 | ファイル | 抽出 | record 数 | loss に使う (\|score\| < 32000) | psv の sha256 |
| --- | --- | --- | --- | --- | --- |
| A_s1 | 001/011/021 (S1, 全 arm が学習 = train loss) | 一様, seed 20260930 | 1,000,000 | 951,753 | `87e6a52e…` |
| B_s0 | 010/020/030 (S0, full 以外で held-out) | 一様, seed 20260931 | 1,000,000 | 951,708 | `c4f255b2…` |
| C_s1_rare | 001/011/021 | 層別, seed 20260932 | 630,237 | 630,237 | `44496ecb…` |
| D_s0_rare | 010/020/030 | 層別, seed 20260933 | 630,079 | 630,079 | `26d1b1ee…` |

- 一様標本: 3 ファイルの連結 (1,466,894,943 record) から numpy 1.26.4 の `Generator(PCG64).choice(replace=False)`。
  |score| ≥ 32000 の record (4.8%) も標本には入っていて、loss からは除く
- 層別標本: `loss_eval scan` で 3 ファイルを全走査し、|score| < 32000 の record を `min_count` の log10 ビンに分けて、
  ビンごとに `splitmix64(seed ^ splitmix64(record 番号))` が小さい 100,000 個を取る (ビン内で一様・スレッド数に
  よらず決定的)。10³ 未満のビンは母集団全部が入る。record ごとの `design_weight` = 母集団のビンの大きさ / 標本数。
  JSON の値はこの重みで重み付けした母集団の推定値
- `min_count` = その局面の 80 個 (両視点 40 個ずつ) の特徴量の、full の教師 (30 ファイル, loss 重み > 0) での
  出現回数の最小値。全 record で特徴量はちょうど 80 個、loss に使う record の `min_count` は必ず ≥ 1
- 出現回数の配列: `loss_samples/full_counts.both.npy` (sha256 `10bd00da…`)。標本作成時の `loss_eval` は
  sha256 `6ac2d2bf…` (その後の変更は nn.bin の説明文字列の検査と空入力の扱いだけ)。A / B は作り直しても psv・feat の sha256 が
  同一になることを確認した

`min_count` のビンごとの record 数 (loss に使う record。層別標本の括弧内は母集団 = S1 / S0 の全 record):

| min_count | A_s1 | B_s0 | C_s1_rare | D_s0_rare |
| --- | --- | --- | --- | --- |
| [1, 10) | 0 | 0 | 221 (221) | 207 (207) |
| [10, 10²) | 2 | 5 | 5,212 (5,212) | 5,070 (5,070) |
| [10², 10³) | 14 | 12 | 24,804 (24,804) | 24,802 (24,802) |
| [10³, 10⁴) | 663 | 697 | 100,000 (982,484) | 100,000 (984,387) |
| [10⁴, 10⁵) | 42,962 | 42,848 | 100,000 (6,306 万) | 100,000 (6,307 万) |
| [10⁵, 10⁶) | 247,496 | 248,333 | 100,000 (3.64 億) | 100,000 (3.64 億) |
| [10⁶, 10⁷) | 481,241 | 480,008 | 100,000 (7.05 億) | 100,000 (7.04 億) |
| [10⁷, 10⁸) | 173,278 | 173,590 | 100,000 (2.54 億) | 100,000 (2.54 億) |
| [10⁸, 10⁹) | 6,097 | 6,215 | 100,000 (904 万) | 100,000 (904 万) |

- **稀な層の局面は形勢の大差がついた終盤**: `min_count` < 10⁴ の局面は平均 |score| 3,212 cp・平均手数 193、
  ≥ 10⁸ の局面は 345 cp・38 手 (A_s1)。層によって loss・MAE の水準が大きく違うので、ネット間の比較は
  同じ層の中で (できれば `--diff` の対応のある差で) 行う
- 教師の |score| は大きい (A_s1 で平均 1,896 cp, 中央値 1,484 cp, 26% が 3,000 cp 超)。MAE (cp) はこれらの
  局面に支配される。sigmoid を通す loss は大差の局面をほぼ無視する

## 速度

8 スレッド (suzuki, EPYC 7543)。

- 順伝播: 100 万局面あたり約 2.2 秒 (約 45 万局面/秒)。nn.bin の読み込み 0.4 秒。
  `loss_eval.py` 1 回 (A〜D の 326 万局面) で約 11 秒
- float (`--state`) の順伝播は約 30 万局面/秒
- 標本作成 (一度だけ): A, B は乱択読み出しで各 2〜3 分 (I/O 律速)、C, D は 3 ファイルの全走査で各約 3.5 分

## 使い方

```sh
# 標本 (一度だけ。作成済み)
data/matchenv/bin/python experiments/009-data-scaling/loss_eval/make_samples.py
# ネット 1 つ (checkpoint でもアブレーション済みでも nn.bin を渡す)
data/matchenv/bin/python experiments/009-data-scaling/loss_eval/loss_eval.py \
    --net data/bulletou/checkpoints/009-full/0012/nn.bin          # → /mnt/nvme1/sugiyama/loss_eval/009-full-0012/
data/matchenv/bin/python experiments/009-data-scaling/loss_eval/loss_eval.py \
    --net /mnt/nvme1/sugiyama/ablated/<名前>/nn.bin                 # → .../loss_eval/ablated-<名前>/
# 2 ネットの対応のある差 (a − b, 同じ record 上)
data/matchenv/bin/python experiments/009-data-scaling/loss_eval/loss_eval.py \
    --diff /mnt/nvme1/sugiyama/loss_eval/<a> /mnt/nvme1/sugiyama/loss_eval/<b> --out diff.json
# float との比較・BulletOu の test loss の再現 (checkpoint のみ)
experiments/009-data-scaling/loss_eval/target/release/loss_eval eval --net <ckpt>/nn.bin --state <ckpt>/state.bin \
    --test-teacher data/teacher/floodgate/floodgate.hcpe --test-positions 300000 --test-seed 20260928 [--dequant]
```

出力 (`/mnt/nvme1/sugiyama/loss_eval/<label>/`):

- `loss.json`: 標本ごとに `overall` (n, loss, loss_se, mae_cp) と `by_rarest_log10` (`min_count` の log10 ビン)、
  `by_n_lt_1e4` (出現回数 10⁴ 未満の特徴量の個数)。各行に n。層別標本 (C, D) は `design_weight` で重み付けした
  母集団の推定値で、`n_eff` (実効標本数) も付く
- `<標本>.q.npy`: record ごとの生出力 (i32)。後から別の層別 (arm 自身の出現回数、対局側の出現回数など) を
  するときは、標本の `feat.npy` と組み合わせて Python だけで計算できる (順伝播をやり直す必要は無い)

## 注意

- 一様標本 A / B では稀な層がとても薄い (100 万 record 中 `min_count` < 10⁴ は約 680、< 10³ は約 16)。
  稀な特徴量は局面の側から見ても稀なため。仮説の主検定 (稀な層での arm 間の差) には層別標本 C / D を使う
- 層の境界は full の教師出現回数で固定している。arm ごとの出現回数で層別したいときは `feat.npy` から計算する
- loss は Elo の代理にならない (`decisions.md`)。確認用の指標として読む
