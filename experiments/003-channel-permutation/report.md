# 003-channel-permutation: FT 出力チャネル並べ替え (activation-sparsity permutation)

日付: 2026-07-21 (開始)
状態: 🏃‍♀️ 進行中

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

(🏃‍♀️ 計測中 — bench_orig.txt / bench_perm.txt)

## 結論

(未記入)
