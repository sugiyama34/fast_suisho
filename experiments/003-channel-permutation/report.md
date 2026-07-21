# 003-channel-permutation: FT 出力チャネル並べ替え (activation-sparsity permutation)

日付: 2026-07-21
状態: ✅ 完了

## 仮説

- **問い**: FT 出力チャネル (pairwise-mul ペア単位) を「同時にゼロになりやすいものが
  同じ 4ch チャンクに固まる」よう並べ替えると、fc_0 sparse カーネルのスキップ率が上がり
  NPS が向上するか
- **予測**: fc_0 は全サイクルの 6.8% (experiment-002 実測)、うち重み積和部 ~5.5%。
  チャンク非ゼロ率を 3 割下げられれば **+1〜2% NPS**。experiment-000 実測の
  run 間ばらつき ±1.5% と同オーダーなので、REPEATS を増やして計測する
- **判定指標**: `bench_nps.sh` の 1 スレッド NPS 中央値 (movetime 10s × 4局面 × REPEATS=9)。
  副指標 = チャンク非ゼロ率 (実測マスクから計算する機構レベルの指標。NPS がノイズに
  埋もれてもこちらで機構の成否を判定できる)
- **ベースライン**: オリジナル `../suisho11/nn.bin`
  (sha256 `a78b7f88…`, experiments/000-baseline/report.md 記載のもの)。
  エンジンは両側とも `~/engines/baseline-000/YaneuraOu-by-gcc` (同一バイナリ) を使う —
  weights-only 実験なのでビルド差は原理的に混入しない
- **合格条件 (精度)**: 等価変換なので、固定局面集 500 点で評価値が元ネットと
  **完全一致**すること (フェーズ3 軽量プロトコル)。一致しなければ実装バグであり不合格

## 背景 (experiment-002 で確定した設計情報)

- fc_0 (`AffineTransformSparseInputExplicit<1024,8>`) は uint8 入力 1024 本を
  **int32 単位 = 連続 4 チャネルのチャンク**として扱い、`find_nnz_explicit` が
  チャンク内のいずれかが非ゼロならチャンク全体を処理する
- fc_0 の入力 z は FT の pairwise-mul 出力: 視点 p ごとに
  z^p_j = clamp(a^p_j)·clamp(a^p_{j+512})/128 (j = 0..511)、連結 z = [z^stm; z^opp]
- したがって**並べ替えの自由度は FT ペア (j, j+512) 単位**の置換 σ (512 要素)。
  z の両半分 (stm/opp) には同じ σ が適用されるため、チャンク構成も両半分で共通
- 上限見積り: fc_0 6.8% のうち 3 割削減で **+2% NPS 程度** (experiment-002 結論 3)。
  効果は小さいが、精度損ゼロ・weights-only で最も安価。**weights-only 加工パイプライン
  (NNUEModel 編集 → serialize → 検証) の練習台**を兼ねる

## 等価変換の定義

σ を [0,512) 上の置換とし、`perm.npy` は「新位置 p に置く旧ペア番号」`perm[p]` を持つ
(new[:, p] = old[:, perm[p]])。idx = concat(perm, perm+512) として:

- FT: weights (131949, 1024) の**列**を `w[:, idx]` に、biases (1024,) を `b[idx]` に置換
- fc_0 (全 9 スタック): weights (8, 1024) の**入力列**を `w[:, idx]` に置換
- 他の層・hash・arch 文字列は不変。ペア内の積は可換なのでペア内 (j ↔ j+512) の
  入れ替えも等価だが、チャンク構成に影響しないため行わない

## 手順

1. **活性化統計の収集**: やねうら王に一時パッチ (`ft_stats.patch`) を当て、fc_0 入力の
   非ゼロビットマスク (1024 bit/サンプル) をサンプリング記録する計測ビルドを作る。
   互角局面集からストライド抽出した 500 局面 (`sfens_stats_500.txt`, NR%60==1) を
   1 スレッドで `go nodes 20000` ずつ探索し、探索中に実際に評価される局面分布で
   統計を取る。パッチは収集後に revert する (対象 = 計測専用ビルドのみ。
   検証・ベンチはパッチと無関係のアーカイブ済み baseline-000 バイナリを使う)
2. **σ の最適化** (`optimize_permutation.py`): マスクを半分ずつ 512 bit の
   ペア空間サンプルに分解し、「チャンク (4 ペア) が全ゼロになる確率の和」を
   貪欲構築 + 山登り交換で最大化。identity との比較でチャンク非ゼロ率の予測改善を出す
3. **適用** (`apply_permutation.py`): tools/nnue で nn.bin を読み、上記の等価変換を
   適用して `nn.bin` (permuted) を書き出す。ランダム入力に対する fc_0 出力一致の
   代数チェック内蔵
4. **検証 (フェーズ3 軽量プロトコル)**:
   - 評価値完全一致: 検証用 500 局面 (`sfens_verify_500.txt`, NR%60==31 — 統計収集とは
     別系列) で `go nodes 50000` (Threads=1, 決定的) の最終 info 行 (score/nodes/pv,
     time・nps は除去) と bestmove を diff。V9.60 の `eval` コマンドは未実装スタブ
     だったため、静的評価値の直接比較の代わりに決定的探索の完全一致で検証する
     (探索中に評価される全局面 — 差分更新・refresh 両経路 — で評価値が 1 つでも
     ずれれば score/pv/nodes が一致しなくなるため、こちらの方が網羅性は高い)
   - NPS: `bench_nps.sh` (ENGINE=baseline-000, REPEATS=9, THREADS_LIST="1 16") を
     オリジナル / permuted の両 EvalDir で実行
   - 機構確認: 計測ビルドで permuted ネットのマスクを再収集し、チャンク非ゼロ率の
     実測低下が予測と合うか確認

## 結果

### 活性化統計

計測ビルド (`ft_stats.patch` 適用、sha256 `87857ca8…`、パッチは収集後 revert 済み) で
`sfens_stats_500.txt` の 500 局面を Threads=1・`go nodes 20000` ずつ探索し、
16 call に 1 回サンプリングで **534,627 サンプル** (= 1,069,254 半ベクトル) を収集:

- fc_0 入力チャネルの非ゼロ率 (平均): **29.2%**
  (experiment-001 の重みゼロ率とは別物。実入力の活性化スパース性)
- マスクファイル sha256: orig `1f02713d…` / permuted 再収集 `e4e82847…`
  (68 MB のためリポジトリには置かない。`collect` 手順で再現可能)

### 最適化 (チャンク非ゼロ率)

`optimize_permutation.py` (貪欲構築 + 山登り交換 60,000 回、最適化は 15 万サンプル、
評価は全サンプル。ログ: `optimize_log.txt`):

| 並べ替え | チャンク非ゼロ率 |
|---|---|
| identity (元の並び) | 0.7436 |
| 貪欲構築 | 0.5448 |
| 貪欲 + 山登り交換 (採用 = `perm.npy`) | **0.5376** |

相対削減 **27.7%**。fc_0 の重み積和部は全サイクルの ~5.5% (experiment-002) なので、
NPS への予測効果は **+1.5% 程度** (5.5% × 27.7%)

### 評価値完全一致 (等価変換の検証)

- `apply_permutation.py` 内蔵の代数チェック (W_new @ z[idx] == W @ z、全 9 スタック) 合格
- permuted nn.bin はファイルサイズが元とバイト数一致 (135,285,230 bytes —
  LEB128 長は値の多重集合で決まるため並べ替え不変)。sha256 `738cbaa2…`
- **決定的探索の完全一致**: `sfens_verify_500.txt` の 500 局面 (統計収集とは別系列) ×
  `go nodes 50000` (Threads=1, baseline-000 エンジン) で、最終 info 行
  (depth/seldepth/score/nodes/pv) と bestmove が **全局面で diff ゼロ**。
  探索木全体 (~25M ノードの評価、差分更新・refresh 両経路) で評価値が一致している
- **機構確認**: permuted ネットで活性化統計を再収集すると、サンプル数は完全に同一
  (探索が同一である傍証) のまま、チャンク非ゼロ率の実測が **0.5376** — 予測と一致

### NPS

`bench_nps.sh` (baseline-000 エンジン, movetime 10s × 4局面 × REPEATS=9,
`bench_orig.txt` / `bench_perm.txt`):

| threads | オリジナル | permuted | Δ |
|---|---|---|---|
| 1 | 468,822 (464,616–475,486) | 472,288 (468,459–477,626) | **+0.74%** |
| 16 | 7,211,686 (7,134,803–7,260,723) | 7,203,284 (7,121,315–7,251,249) | −0.12% |

中央値差 +0.74% (1T) は run 間ばらつき (±1.5%) の内側なので、時間変動をペア内で
相殺する**交互対計測** (`bench_paired.sh`, 12 ペア × 1T movetime 10s,
`bench_paired.txt`) を追加実施:

- ペア差の平均 **+0.38%** (SE 0.38, 95% CI おおよそ −0.5〜+1.2%)、中央値 **+0.70%**、
  正のペア 7/12 — **統計的に有意ではない**
- ペア差の標準偏差は 1.3% で、ペア化してもばらつきは大きい (bench の movetime 方式は
  実行ごとのノード数変動が支配的)。+0.4% 級の効果を有意に検出するには
  ~100 ペア (2 時間超) が必要であり、開発機スクリーニングとしては打ち切りが妥当

## 結論

1. **等価変換は完全に成立** — 決定的探索 500 局面で diff ゼロ (評価値完全一致)。
   weights-only パイプライン (NNUEModel 編集 → serialize → 検証) の練習台としての
   目的は達成
2. **機構レベルの効果は予測どおり確定**: fc_0 のチャンク非ゼロ率 0.7436 → 0.5376
   (相対 −27.7%)。permuted ネットでの実測が最適化時の予測と一致
3. **NPS 効果は +0.4〜0.7% 程度 (1T) で、開発機の計測ノイズと同水準**。
   experiment-002 の上限見積り (+2%)・本レポートの予測 (+1.5%) と矛盾しない小効果。
   find_nnz 走査 (1.3%) は削減されないこと、スキップで浮くのは積和スループットの
   一部であることから、実効がこの水準に留まるのは想定内
4. **採用判断**: 精度リスクゼロ・NPS は非負 (16T でも差なし) なので、
   **今後のネットは最終パッケージング工程として σ を適用する価値がある**。
   ただし開発機での単独効果は小さいため、c8a (AVX-512, 192 コア) での再計測を
   もって最終判断とする。σ の合成は任意のネットに対して `apply_permutation.py` を
   流すだけで済む (活性化統計はネットが変われば要再収集)
5. **今後の実験への含意**: finny tables (フェーズ2 項目 1) や FT 幅縮小の後段でも
   fc_0 の相対コストが変わるため、σ 適用の期待値はそのとき再評価する

## 妥当性への脅威

- bench の 4 局面と統計収集の互角局面 500 点は分布が異なる (むしろ汎化の確認になって
  いるが、bench 局面に過適合した σ ではないことに注意)
- 順次実行の bench_orig/bench_perm には時間順の系統誤差 (温度・周波数) が乗りうる。
  交互対計測で補ったが、いずれも 10 分オーダーの計測であり ±0.4% 級の結論が限界
- 開発機 (Zen 2 AVX2) の結果であり、c8a (Zen 5 AVX-512) では find_nnz が
  `_mm512_cmpgt_epi32_mask` 系になるなどコスト構造が変わる

## 再現手順

```sh
# 1. 計測ビルド (パッチ適用 → ビルド → revert)
cd ../YaneuraOu && patch -p1 < ../fast_suisho/experiments/003-channel-permutation/ft_stats.patch
cd source && make -j16 normal YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3 \
     COMPILER=g++ TARGET_CPU=AVX2 PYTHON=python3
cp YaneuraOu-by-gcc ~/engines/ftstats-003/YaneuraOu-ftstats
cd .. && patch -R -p1 < ../fast_suisho/experiments/003-channel-permutation/ft_stats.patch

# 2. 活性化統計収集 (Threads=1, 16 call に 1 サンプル)
cd ../fast_suisho
FT_STATS_FILE=/tmp/masks_orig.bin FT_STATS_EVERY=16 .venv/bin/python \
  experiments/003-channel-permutation/usi_drive.py --engine ~/engines/ftstats-003/YaneuraOu-ftstats \
  --evaldir ~/suisho11 --sfens experiments/003-channel-permutation/sfens_stats_500.txt \
  --mode search --nodes 20000

# 3. σ 最適化 → 適用
.venv/bin/python experiments/003-channel-permutation/optimize_permutation.py \
  --masks /tmp/masks_orig.bin --out experiments/003-channel-permutation/perm.npy --swap-iters 60000
PYTHONPATH=. .venv/bin/python experiments/003-channel-permutation/apply_permutation.py \
  --nn ~/suisho11/nn.bin --perm experiments/003-channel-permutation/perm.npy \
  --out ~/engines/perm-003/nn.bin

# 4. 検証 (評価値完全一致 = 決定的探索 diff、機構確認、NPS)
.venv/bin/python experiments/003-channel-permutation/usi_drive.py --engine ~/engines/baseline-000/YaneuraOu-by-gcc \
  --evaldir ~/suisho11 --sfens experiments/003-channel-permutation/sfens_verify_500.txt \
  --mode searchlog --nodes 50000 > /tmp/searchlog_orig.txt   # EvalDir を差し替えてもう一度 → diff
ENGINE=~/engines/baseline-000/YaneuraOu-by-gcc EVALDIR=~/engines/perm-003 \
  THREADS_LIST="1 16" REPEATS=9 ./experiments/000-baseline/bench_nps.sh 10000
./experiments/003-channel-permutation/bench_paired.sh 12 10000
```

## アーティファクト fingerprint (sha256)

```
a78b7f889843037d344f482623b3febd124ead5c1f34f134d9f1c2c78cd0f829  ../suisho11/nn.bin (オリジナル)
738cbaa230363e6adfc96e6c7fa428f963aab4bd1db6592868885b93ba9ae35d  ~/engines/perm-003/nn.bin (permuted)
312b1368348e83fe546f45983e49bd354636609b1ad4857a033c4ab0e4aafff6  ~/engines/baseline-000/YaneuraOu-by-gcc (検証・NPS 用)
87857ca87f8674f4722812d9ac8608f23503fb4d1b8e651b0371db8527f82240  ~/engines/ftstats-003/YaneuraOu-ftstats (計測ビルド)
1f02713df34d11e889ca52b047141e2495eb4515ecd915d30cdb7c7c3d070455  masks_orig.bin (68 MB, 非コミット)
e4e82847ae9c4cc9f3ac4083e68a618294bf1e4066f645f57214cbfb9a315923  masks_perm.bin (68 MB, 非コミット)
```
