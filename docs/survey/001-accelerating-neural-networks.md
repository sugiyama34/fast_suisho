# サーベイ 001: ニューラルネットワーク高速化手法 (Accelerating Neural Networks) の NNUE への適用

日付: 2026-07-11
対象: Suisho 11 NNUE (`SFNN_halfka2_1024-7-64-k3k3`) × やねうら王 V9.60 / AVX2 CPU 推論

## 0. 目的とスコープ

NN 推論高速化の代表的手法 (量子化・プルーニング・蒸留・低ランク分解・条件付き計算・実装最適化) を概観し、
本プロジェクトの NNUE 評価関数に適用した場合の **期待 NPS 向上 / 精度 (レーティング) リスク / 実装コスト** を整理する。
フェーズ2 の実験候補の選定と優先順位付け (docs/PLAN.md) の根拠資料とする。

**戦略上の位置づけ (2026-07-11 追記)**: 本プロジェクトのゴールは単なる高速化ではない。
「高速化手法で NPS が上がり、精度低下を織り込んでもレーティングが上がる」ことが確認できたら、
**まず現行より大きなネットワークを訓練して精度の余裕 (headroom) を作り、それを圧縮して速度を回収する**
二段構えで、最終的に強い評価関数を得る (§4 参照)。

## 1. 前提: 対象ネットワークのコスト構造

### 1.1 アーキテクチャとメモリフットプリント

- **FeatureTransformer (FT)**: 入力 131,949 次元 (HalfKA2, 玉位置×盤上駒) → 出力 1024×2 視点。
  重みは int16。**重み行列だけで 131,949 × 1024 × 2 byte ≈ 258 MiB** — 実行時メモリのほぼ全てが FT。
- **LayerStacks ×9** (`k3k3` = 玉位置 3×3 バケット): fc_0 (2048→7ch, sparse-input 実装済み) →
  二乗活性 → fc_1 (→64ch) → fc_2 (→1)。重みは int8。合計数百 KB 程度で FT に比べ無視できる規模
  (正確な次元・活性構成はフェーズ1 の parser で確定する)。
- ディスク上は FT 部が LEB128 圧縮で 135 MB。**LEB128 はロード時に展開されるためディスクサイズのみの話**であり、
  実行速度には関係しない。

### 1.2 評価 1 回のコスト内訳

NNUE の設計上、コストは 3 つに分かれる:

1. **差分更新 (accumulator update)**: 1 手で変化する特徴は 2〜4 個。該当する FT 重み列
   (1024 × int16 = 2 KiB/列) を加減算するだけ。安価だが**毎ノード発生**し、
   258 MiB の重みテーブルへのランダムアクセスなのでキャッシュ/メモリ帯域が効く。
2. **全計算 (refresh)**: 自玉が動く (またはバケットをまたぐ) と、その視点の accumulator を作り直す。
   盤上駒 ≈ 38 特徴 × 2 KiB ≈ 76 KiB の読み込み+加算。頻度は低いが 1 回が重い。
3. **後段 FC**: 数万 MAC 規模で軽量。fc_0 は入力活性 (clipped ReLU 後) の 0 が多いことを利用した
   sparse-input 実装が既に入っている。

→ **高速化の主戦場は FT (幅・ビット幅・更新/refresh 頻度)**。後段 FC を削っても NPS はほぼ動かない。
ただし比率は実測していないため、フェーズ2 冒頭で `perf` によるプロファイル取得を行い確定させる。

### 1.3 ハードウェア前提 (EPYC 7302, Zen 2)

- AVX2 (256bit) まで。**AVX-512 なし、VNNI なし** → int8 内積は `vpmaddubsw` + `vpmaddwd` の 2 命令構成。
- L3 128 MiB (16 MiB × 8 CCX) < FT 重み 258 MiB。玉位置の局所性で実効ヒット率は上がるが、
  refresh・更新のメモリ帯域は無視できない → **ビット幅削減 = 帯域削減**が効く余地がある。

### 1.4 NPS とレーティングの換算 (トレードオフの物差し)

チェス/将棋の経験則で **探索速度 2 倍 ≈ +70 Elo 前後** (時間・条件で変動)。対数スケールなので:

| NPS 向上 | 期待 Elo |
|---|---|
| +10% | ≈ +10 |
| +25% | ≈ +15〜22 |
| +50% | ≈ +30〜40 |
| +100% | ≈ +70 |

**手法の採否は「NPS による Elo 利得 − 精度低下による Elo 損失」の符号で決まる**。
精度低下の Elo 換算は静的指標 (評価値相関) からは予測しにくいため、最終判定は PLAN.md フェーズ3 の
自己対局 (SPRT) で行う。

## 2. 手法カタログ

### 2.1 量子化 (Quantization)

**現状**: NNUE は既に量子化済みネットワークである (FT: int16、後段: int8。Stockfish 系の標準スキーム)。
つまり「FP32→int8」の定番の伸びしろは既に消化されており、残る選択肢はさらなる低ビット化。

- **FT の int8 化**: 重み 258→129 MiB、帯域・キャッシュ圧が半減。差分更新の SIMD 幅も 2 倍。
  - リスク: FT は第 1 層で精度感度が高く、Stockfish も FT を int16 に留めている。
    加算だけで桁あふれしない設計 (per-channel スケール、飽和加算) が必要。
  - 実装: エンジン側 SIMD カーネルの書き換え (エンジン改造・大)。量子化自体は PTQ
    (post-training quantization) でまず試し、精度が落ちるなら QAT (quantization-aware training) へ。
- **int4 / 3値 / 2値**: LLM 系で流行だが、AVX2 にはニブル演算がなくデコードコストが載る。
  FT は「積和」ではなく「列の加算」なので低ビット化の演算利得も薄い。帯域目的なら int8 で十分。**優先度低**。
- **量子化スケールの再最適化**: 現行スキームのスケール定数 (クリップ範囲) を Suisho 11 の実際の
  重み/活性分布に合わせて再調整する。weights-only で試せる可能性があり安価 (PLAN 既載)。

参考: [nnue-pytorch docs (quantization 節)](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)、
量子化サーベイ [Gholami+ 2021 (arXiv:2103.13630)](https://arxiv.org/abs/2103.13630)、
低ビット量子化サーベイ [arXiv:2505.05530](https://arxiv.org/abs/2505.05530)

### 2.2 構造的プルーニング / 幅削減 (Structured pruning)

チャネル (ニューロン) 単位で削るため、**密行列演算のまま比例的に速くなる**。NNUE と最も相性が良い。

- **FT 幅の縮小 (1024 → 768 / 512)**: 本命。差分更新・refresh・fc_0 が全て比例して軽くなる。
  - チャネル選択の重要度基準: 重み列の L1 ノルム、活性化統計 (実局面での出力分散)、
    Taylor 展開近似 (損失への寄与)。L1 ノルムは最も簡便で、fine-tune 前提なら十分とされる。
  - 削っただけでは精度が落ちるので **蒸留 / fine-tune による回復が必須** (§2.4)。
    文献では「刈って fine-tune」は少データ・少エポックでもかなり回復する
    ([structured pruning survey, arXiv:2303.00566](https://arxiv.org/abs/2303.00566))。
  - 実装コスト小: arch header の生成パラメータ変更 + nn.bin 差し替えで済む (エンジン本体は不変)。
- **後段 fc の刈り込み (fc_1 64ch など)**: コスト比率が小さいので NPS 効果は数 % 未満。優先度低。
- **LayerStacks の統合 (9→少数)**: 実行時は玉位置で **1 バケットしか評価しない**ため、
  統合しても評価コストは不変 = **NPS 目的では効果なし** (数百 KB のメモリ削減のみ)。
  PLAN の優先度低の判断は妥当で、速度目的からは候補から外してよい。

### 2.3 スパース性の活用 (非構造プルーニング・並べ替え)

- **重みの非構造 (要素単位) スパース化**: 密 SIMD カーネルでは 0 が混ざっても 1 命令も減らない。
  スパース専用カーネルはインデックス処理のオーバーヘッドで、NNUE 規模の小さい層では逆効果。**非推奨**。
- **活性スパース性の活用 — チャネル並べ替え (permutation)**: fc_0 の sparse-input 実装は
  「FT 出力のうち 0 でないブロックだけ処理する」方式。FT の出力チャネルを並べ替えて
  **0 になりやすいチャネルを同じ処理ブロックに固める**と、スキップ効率が上がる。
  - nnue-pytorch には `ft_optimize` (FT permutation) として実装があり、Stockfish では数 % の speedup 実績。
  - **等価変換なので精度損はゼロ**。FT 出力順と fc_0 入力順を同時に並べ替えた nn.bin を作るだけで、
    **エンジン改造不要 (weights-only)** — フェーズ1 の parser ができれば即試せる。
  - 前提: やねうら王側 sparse 実装のブロック単位 (何 ch 単位でスキップ判定するか) の確認が必要。
- **活性スパース率を高める訓練** (L1 正則化で FT 出力の 0 を増やす): 再訓練とセットで permutation の
  効果を増幅する発展形。

### 2.4 知識蒸留 (Knowledge distillation)

縮小ネットワークの精度回復の標準手段であり、**§2.2 と §4 の前提インフラ**。

- teacher = Suisho 11 (評価値 or 浅い探索値)、student = 縮小ネット。
  nnue-pytorch 系トレーナ (将棋向けは tanuki- 系の移植が存在) で、教師評価値と勝敗の混合 (λ) で学習。
- **品質は学習局面の分布と量に支配される** (数億局面規模)。PLAN 記載の通り、
  コーパス確保 (公開教師データ or 自己対局生成) が着手条件。
  参考: [Study of the Proper NNUE Dataset (arXiv:2412.17948)](https://arxiv.org/abs/2412.17948)
- 事例: Stockfish の smallnet (L1-128) は大ネットの評価値でフィルタ/生成したデータで訓練され、
  dual-net 構成の部品として +Elo に寄与 ([SF PR #4915](https://github.com/official-stockfish/Stockfish/pull/4915))。
- 原典: [Hinton+ 2015 (arXiv:1503.02531)](https://arxiv.org/abs/1503.02531)

### 2.5 低ランク分解 (Low-rank factorization)

FT を W ≈ U·V (131949×r, r×1024) に分解すると重みメモリは大幅に減る (r=256 で 258→67 MiB) が、
**評価時に r→1024 の密射影 (視点あたり ~26万 MAC) が毎ノード追加**される。
NNUE の差分更新は元々「列を数本足すだけ」で極めて安価なため、**演算面では明確に損**。
NNUE の構造 (第 1 層 = 疎入力の埋め込み表) が既に「これ以上ないほど推論効率が良い」ためであり、**優先度低**。
フェーズ1 の解析で FT 行列の特異値分布を見る (低ランク性の有無を知る) こと自体は診断として有用。

### 2.6 条件付き計算 / Dual net (大小ネットの使い分け)

- **Stockfish の dual NNUE** ([PR #4915](https://github.com/official-stockfish/Stockfish/pull/4915)):
  形勢が大きく離れた局面は小ネット (L1-128)、僅差の局面のみ大ネット (L1-1024+) で評価。
  **speedup ≈ +10.8%、約 +20 Elo** の実績。精度が要る局面にだけ精度を払う、という条件付き計算の成功例。
- 将棋への適用: エンジン改造 (中〜大) + 小ネットの蒸留が必要。蒸留パイプライン確立後の中期候補。
- 既存の LayerStacks (material/玉位置バケット) も条件付き計算の一種と見なせる。

### 2.7 エンジン・ビルドレベルの最適化 (ネットワーク不変)

精度リスクゼロで NPS だけ動く系。**比較のフェアネスに注意**: これらを高速化版だけに適用すると
ネットワーク手法の効果と混ざるため、採用する場合はベースラインにも適用して再計測する。

- **PGO (profile-guided optimization) ビルド**: Stockfish では標準で数 % の speedup。
  やねうら王 Makefile の対応有無を確認し、なければ `-fprofile-generate/use` を手動適用。
- **Huge pages (透過的 THP / explicit)**: 258 MiB の FT テーブルへのランダムアクセスは TLB ミスが載る。
  large page 化で数 % 級の改善余地。やねうら王の対応状況を確認。
- **prefetch / メモリレイアウト**: 差分更新する重み列の software prefetch、視点 2 本の interleave 等。
  エンジン改造 (小〜中) だが効果は要実測。
- コンパイラ/フラグ: `-march=native`、LTO の確認 (既定で有効かも含め)。

### 2.8 適用しにくい・対象外の手法

- **GPU オフロード**: αβ 探索はノード粒度が細かくレイテンシ律速。GPU 向けは dlshogi 系 (MCTS+DNN) の路線で、
  本プロジェクトのスコープ外。
- **weight clustering / ハフマン符号化** ([Deep Compression, arXiv:1510.00149](https://arxiv.org/abs/1510.00149)):
  ディスク/配布サイズ削減のみで実行時は展開が必要。LEB128 で既に同等の役割を果たしている。
- **NAS (アーキテクチャ探索)**: 1 実験 = 1 訓練+1 対局検証が高価な本環境では探索予算が現実的でない。
  手動の幅バリエーション (§2.2) で十分。

## 3. まとめ: 期待効果 × リスク × コスト一覧

| 手法 | NPS 期待 | 精度リスク | 実装コスト | 蒸留/訓練 | 備考 |
|---|---|---|---|---|---|
| チャネル並べ替え (§2.3) | +1〜5% | **ゼロ** (等価変換) | weights-only | 不要 | parser 完成後すぐ試せる |
| PGO / huge pages (§2.7) | +数% | ゼロ | ビルドのみ | 不要 | ベースラインにも適用して再計測 |
| 量子化スケール再調整 (§2.1) | 0〜数% | 小 | weights-only 〜 header | 不要 | 効果は不確か |
| **FT 幅縮小 768/512 (§2.2)** | **+15〜50%** | 中 (蒸留で回復) | header + nn.bin | **必須** | 本命 |
| Dual net (§2.6) | ~+10% | 小 | エンジン改造 (中) | 必須 | SF で +20 Elo 実績 |
| FT int8 化 (§2.1) | +10〜30%? | 中 | エンジン改造 (大) | 推奨 (QAT) | 帯域律速なら効く |
| fc 刈り込み (§2.2) | <+3% | 小 | header + nn.bin | 推奨 | 効果小 |
| LayerStacks 統合 (§2.2) | ~0% | 中 | header + nn.bin | 必須 | **速度目的では無意味** |
| 低ランク FT (§2.5) | マイナス見込み | 未知 | エンジン改造 (大) | 必須 | 診断 (SVD) のみ実施 |
| 非構造スパース (§2.3) | マイナス見込み | — | — | — | 非推奨 |

## 4. 戦略: 「大きく訓練してから圧縮する」(Train Large, Then Compress)

文献上、**同じ推論コストなら「大きく訓練して圧縮したモデル」が「最初から小さく訓練したモデル」に勝つ**
ことが繰り返し報告されている:

- [Li+ 2020 "Train Large, Then Compress" (arXiv:2002.11794)](https://arxiv.org/abs/2002.11794) —
  大モデルを訓練→量子化/プルーニングした方が、同予算の小モデル直接訓練より高精度。
- [Zhu & Gupta 2017 (arXiv:1710.01878)](https://arxiv.org/abs/1710.01878) — large-sparse > small-dense。
- [Lottery Ticket (arXiv:1803.03635)](https://arxiv.org/abs/1803.03635) — 良い部分ネットは大ネットの中から見つかる。
- チェス実例: Stockfish は L1 を 256→…→3072 と拡大し続けつつ、蒸留由来の smallnet を dual 構成で併用。

本プロジェクトへの適用 (二段構え):

1. **第一段 (検証)**: Suisho 11 (1024) を teacher に、縮小 (768/512) を student として圧縮パイプラインを検証。
   「NPS 増 − 精度損」が正 (レーティング向上) になる手法と圧縮率の感覚を掴む。
2. **第二段 (本番)**: より大きなネット (例: FT 1536/2048) を訓練して精度の余裕を確保し、
   検証済みの圧縮手法で 1024 相当以下まで落とす。
   圧縮に必要なインフラ (コーパス + トレーナ) は第一段と完全に共通なので、追加コストは大ネットの訓練のみ。
   - リスク: Suisho 11 は高度にチューニングされており、それを上回る大ネットの訓練自体が難しい。
     大ネットが Suisho 11 を超えられなくても、第一段の成果 (Suisho 11 の直接圧縮) は独立に価値を持つ。

## 5. 推奨実験順序 (PLAN.md フェーズ2 への反映)

1. **プロファイル取得** (`perf`): FT 更新 / refresh / 後段 / 探索部の実測コスト比率を確定 — 全手法の効果予測の土台
2. **精度リスクゼロ系**: チャネル並べ替え (weights-only)、PGO/huge pages (ベースライン込み再計測)
3. **蒸留パイプライン整備**: コーパス確保 → nnue-pytorch 系トレーナ移植 → Suisho 11 の再現蒸留 (等幅) で検証
4. **FT 幅縮小** (768 → 512): 本命。静的相関 → SPRT で判定
5. **中期**: dual net、FT int8 化 (2〜4 の結果を見て)
6. **第二段**: 大ネット訓練 → 圧縮 (§4)

各実験は `experiments/NNN-name/` に置き、検証は PLAN.md フェーズ3 のプロトコル
(bench_nps.sh / 静的相関 / SPRT・pentanomial) に従う。

## 6. 参考文献

### NNUE / チェス・将棋エンジン
- [nnue-pytorch docs (NNUE の設計・量子化・SIMD 実装の一次資料)](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
  ([HTML 版](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html))
- [Stockfish PR #4915: Dual NNUE with L1-128 smallnet](https://github.com/official-stockfish/Stockfish/pull/4915)
- [Chessprogramming wiki: NNUE](https://www.chessprogramming.org/NNUE) / [Stockfish NNUE](https://www.chessprogramming.org/Stockfish_NNUE)
- [Study of the Proper NNUE Dataset (arXiv:2412.17948)](https://arxiv.org/abs/2412.17948)
- [やねうら王ブログ: NNUE 評価関数の学習方法](https://yaneuraou.yaneu.com/2018/12/30/nnue%E8%A9%95%E4%BE%A1%E9%96%A2%E6%95%B0%E3%81%AE%E5%AD%A6%E7%BF%92%E6%96%B9%E6%B3%95%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6/)
- [Qhapaq: NNUE 関数のネットワーク構造](https://qhapaq.hatenablog.com/entry/2018/06/02/221612)

### 高速化・圧縮の一般文献
- [Gholami+ 2021: A Survey of Quantization Methods for Efficient Neural Network Inference (arXiv:2103.13630)](https://arxiv.org/abs/2103.13630)
- [Low-bit Model Quantization for DNNs: A Survey (arXiv:2505.05530)](https://arxiv.org/abs/2505.05530)
- [Structured Pruning for Deep CNNs: A Survey (arXiv:2303.00566)](https://arxiv.org/abs/2303.00566)
- [Han+ 2015: Deep Compression (arXiv:1510.00149)](https://arxiv.org/abs/1510.00149)
- [Hinton+ 2015: Distilling the Knowledge in a Neural Network (arXiv:1503.02531)](https://arxiv.org/abs/1503.02531)
- [Li+ 2020: Train Large, Then Compress (arXiv:2002.11794)](https://arxiv.org/abs/2002.11794)
- [Zhu & Gupta 2017: To prune, or not to prune (arXiv:1710.01878)](https://arxiv.org/abs/1710.01878)
- [Frankle & Carbin 2018: The Lottery Ticket Hypothesis (arXiv:1803.03635)](https://arxiv.org/abs/1803.03635)
