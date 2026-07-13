# Suisho 11 NNUE (SFNN_halfka2_1024-7-64-k3k3) のネットワーク構造

日付: 2026-07-12

Suisho 11 `nn.bin` のネットワーク構造・各層の定義 (数式)・量子化仕様・ファイルレイアウトを
まとめる。やねうら王 V9.60 (`9133c527`) のソース
(`nnue_feature_transformer.h`, `architectures/SFNN_halfka2_1024-7-64-k3k3.h`,
`layers/*_explicit.h`, `evaluate_nnue.cpp`) を根拠とし、実測値は
`experiments/001-nn-analysis/` の解析結果を引用する。
パーサ実装 (`tools/nnue/`) との対応も末尾に示す。

## 全体像

```
入力: HalfKA2<Friend> 特徴量 131,949 次元 × 2 視点 (手番側 / 相手側)
        │
        ▼
[1] FeatureTransformer (131949 → 1024, 視点ごと)     int16, LEB128 圧縮
        │
        ▼
[2] pairwise multiply 活性化 (1024 → 512, 視点ごと) → 連結 2×512 = 1024ch (uint8)
        │
        ▼
[3-7] LayerStack ×9 (両玉の段で 1 本選択; 各スタックは以下)
        ├─ [3] fc_0   AffineTransformSparseInput 1024 → 8   (hidden 7 + shortcut 1)
        ├─ [4] ac_0     ClippedReLU [7]      ┐ パラメータなし
        ├─ [5] ac_sqr_0 SqrClippedReLU [7]   ┘ 連結 → 14ch
        ├─ [6] fc_1   AffineTransform 14 → 64  + ClippedReLU
        └─ [7] fc_2   AffineTransform 64 → 1   + shortcut 加算
        │
        ▼
[8] eval = out / FV_SCALE
```

量子化の規約: 重みはスケール 64 (= $2^6$, `kWeightScaleBits = 6`)、
活性値は uint8 の $[0, 127]$ が実数の $[0.0, \sim 1.0]$ に対応する。

## [0] 入力エンコード — HalfKA2(Friend)

視点 $p \in \lbrace \text{手番側}, \text{相手側} \rbrace$ ごとに、局面を **ちょうど 40 個の 1** を持つ
二値スパースベクトル $x^p \in \lbrace 0,1 \rbrace^{131949}$ ($131949 = 81 \times 1629$) で表す
(盤上・持ち駒の全駒 1 つにつき 1 特徴。**両玉を含む** — HalfKA の "A" = All pieces):

$$
\text{active}(x^p) = \lbrace \text{index}(\text{sq}_{\text{自玉}},\ \text{BonaPiece}(i)) \mid i \in \text{全 40 駒} \rbrace
$$

各視点は自玉の位置を基準に特徴量を張る。

### index 関数の定義

`half_ka2.cpp` の `MakeIndex` そのもの:

$$
\text{index}(k, p) = 1629 \cdot k + p', \qquad
p' = \begin{cases}
p - 81 & (p \ge 1629 \text{ 、つまり相手玉}) \\
p & (\text{それ以外})
\end{cases}
$$

- $k$ = 自玉のマス (0–80)。後手視点では盤を 180° 回転した正規化座標を使う
  (エンジンは視点別の駒リスト `piece_list_fb` / `piece_list_fw` を持つ)
- $p$ = **BonaPiece** — 「どの駒がどこに (持ち駒なら何枚目か)」を 1 つの整数に
  割り当てた Bonanza 流の駒状態番号 (`evaluate.h`)。`f_` = 自分側 / `e_` = 相手側
- 81 マス × 1629 駒状態の one-hot 格子を平坦化したものが特徴量番号になる。
  **全特徴量が自玉の位置で条件付けられる** のが "HalfK" の意味であり、
  自玉が動くと全 active index が変わる = accumulator の全再計算 (refresh) が必要になる理由

### BonaPiece の内訳 (この build: DISTINGUISH_GOLDS 無効)

| 範囲 | 内容 |
|---|---|
| 0 | `BONA_PIECE_ZERO` (駒なし。active にならない) |
| 1–89 | **持ち駒**: 駒種×所有者ごとにブロックを持ち、**n 枚目ごとに別 id** (`id = base + n − 1`)。歩 3 枚なら 3 特徴が立つ。例: 自分の持ち歩 1–18, 相手の持ち歩 20–37, 自分の持ち香 39–42, … (0 枚目用の隙間あり: 19, 38, 43, …) |
| 90–1547 | **盤上の駒 (玉以外)**: (駒種×所有者) ごとに 81 幅のブロック、`id = base + マス` (１一 = 0 … ９九 = 80)。順に 歩/香/桂/銀/金/角/馬/飛/竜 × f/e。**馬・竜は独立ブロック**だが、と金・成香・成桂・成銀は**金のブロックに合流** (`DISTINGUISH_GOLDS` 無効のため。これが `fe_end` = 1548 の理由) |
| 1548–1628 | **玉の平面**: `f_king + マス`。enum 上は相手玉に 1629–1709 が割り当てられているが、`MakeIndex` が 81 を引いて**この平面に折り畳む** (自玉の特徴は必ず対角成分 $p' - 1548 = k$ になるため衝突しない)。幅が 1710 でなく 1629 で済むのはこのため |

例 (先手視点): 自玉が５九、先手の歩が７六、後手玉が５一なら
(先手歩の盤上ブロック base = 90)、
歩の特徴 = $1629 \cdot \text{sq}(59) + (90 + \text{sq}(76))$、
後手玉の特徴 = $1629 \cdot \text{sq}(59) + (1548 + \text{sq}(51))$ — これを全 40 駒分
並べたものが $\text{active}(x^p)$ である。

## [1] FeatureTransformer — スパース affine, 131949 → 1024 (視点ごと)

$$
a^p = b^{FT} + \sum_{i \in \text{active}(x^p)} W^{FT}_{i,\cdot} \qquad a^p \in \mathbb{Z}^{1024}\ (\text{int16})
$$

- $W^{FT}$: int16, shape $(131949, 1024)$ feature-major。 $b^{FT}$: int16 $(1024)$。
  nn.bin の LEB128 圧縮部はここ (生 int16 なら約 258 MiB)
- 探索中はゼロから再計算せず、指し手による特徴の増減で**差分更新** (差分計算) する。
  玉が動いた視点のみ全再計算 (refresh)。これが FT が評価コストの大半を占めると
  見込まれる理由 (実測は フェーズ2 の 0 で行う)
- エンジン実装の注意: 読み込み時に FT の重み・バイアスを 2 倍する (`scale_weights`)。
  本ドキュメントの数式は**ファイル上の重み単位** (パーサが見る値) で書いており、
  エンジン内の ×2 は次段の除数 (512 = 4×128) で相殺される

### なぜ int16 か (出力は [0,127] にクランプされるのに)

クランプは**活性化関数**であり、読み出し時にだけ適用される。格納側は
クランプ前の総和を正確に持つ必要がある:

1. **総和は int8 に収まらない**: 40 行 + バイアスの和は日常的に $[-128, 127]$ を超える
2. **差分更新はクランプと両立しない**: `acc += W[追加] − W[削除]` を undo するには
   非可逆なクランプ済み値では情報が足りない。常に正確な未クランプ和を保持する必要がある
3. **単一の重みですら int8 を超える**: 実測レンジ $[-231, 251]$
   (`experiments/001-nn-analysis/stats.md`)

## [2] pairwise multiply 活性化 — 1024 → 512 (視点ごと)

$a^p$ を前半 $u = a^p_{0..511}$ / 後半 $v = a^p_{512..1023}$ に分け:

$$
z^p_j = \left\lfloor \frac{\mathrm{clamp}(u_j, 0, 127) \cdot \mathrm{clamp}(v_j, 0, 127)}{128} \right\rfloor \qquad z^p \in \lbrace 0,\dots,126 \rbrace^{512}\ (\text{uint8})
$$

手番側を先に連結する: $z = [z^{\text{stm}}; z^{\text{opp}}] \in \text{uint8}^{1024}$
(stm = side to move 手番側, opp = opponent 相手側。fc_0 の入力は手番に依らず
「自分の視点が先」になり、1 つのネットワークで両手番を対称に評価できる)。
パラメータなし。積の形のため片方が 0 にクランプされると出力が 0 になり、
$z$ は高スパースになる — 次層がこれを利用する。

## [3] fc_0 — sparse-input affine, 1024 → 8

$$
s = B^{(0)} + W^{(0)} z \qquad s \in \mathbb{Z}^{8}\ (\text{int32})
$$

- $W^{(0)}$: int8 $(8, 1024)$、 $B^{(0)}$: int32 $(8)$
- "sparse input" = AVX2 カーネルは $z$ の非ゼロチャンク (4 バイトブロック単位) の
  インデックスリストを作り、ゼロブロックを完全にスキップする。
  **チャネル並べ替え (フェーズ2 の 0) はこのブロック粒度に合わせて設計すること**
- 出力の役割分担: $s_0..s_6$ = hidden、 $s_7$ = **shortcut**
  (後段を迂回する線形評価項)

## [4][5] $s_{0..6}$ への並列活性化 (各 7 次元)

ClippedReLU (`ac_0`):

$$
h_j = \mathrm{clamp}\left(\left\lfloor s_j / 2^6 \right\rfloor,\ 0,\ 127\right)
$$

SqrClippedReLU (`ac_sqr_0`):

$$
q_j = \min\left(127,\ \left\lfloor s_j^2 / 2^{19} \right\rfloor\right) \qquad (19 = 2 \cdot 6 + 7)
$$

2 乗系統は、わずか 7 ユニットの層に安価に非線形表現力を足すための仕掛け。

## [6] fc_1 — affine, 14 → 64

入力は連結 $[q_0..q_6,\ h_0..h_6] \in \text{uint8}^{14}$:

$$
r = B^{(1)} + W^{(1)} [q; h] \qquad r \in \mathbb{Z}^{64}\ (\text{int32})
$$

続けて ClippedReLU (`ac_1`):

$$
y_j = \mathrm{clamp}\left(\left\lfloor r_j / 2^6 \right\rfloor,\ 0,\ 127\right)
$$

$W^{(1)}$: int8。論理形状 $(64, 14)$、ファイル上は $(64, 32)$ に列パディング
(列 14〜31 は全ゼロ)。

### なぜ $r$ は int32 か

理論上界: $14 \times 127 \times 127 \approx 226{,}000$ で int16 (32,767) を大きく超える。
実ネットの最悪ケース実測 (9 スタック、符号別に最悪入力を仮定):
**pre-activation $r \in [-47{,}062,\ +51{,}588]$**、バイアス単体でも $[-14{,}166,\ 8{,}154]$。
飽和 int16 演算での代用も不可 (部分和の飽和は順序依存で、
$40000 - 35000 = 5000$ が $32767 - 35000 = -2233$ になる類の誤差が出る)。
なお AVX2 カーネルは `maddubs` (uint8×int8 → int16 ペア和) → `madd` (int32 拡張) で
自然に int32 蓄積になる。活性値の上限が 255 でなく 127 なのは、この int16 中間の
ペア和上界を $2 \times 127 \times 127 = 32{,}258 < 32{,}767$ に収めるための設計。

## [7] fc_2 — affine, 64 → 1 + shortcut 加算

$$
o = B^{(2)} + W^{(2)} y \qquad (\text{int32 スカラー}), \qquad \text{out} = o + s_7
$$

shortcut $s_7$ は**シフトなし**で加算される ($o$ と同じ 64 倍スケールに載っている)。

## [8] 最終スコア

$$
\text{eval} = \left\lfloor \frac{\text{out}}{F} \right\rfloor
$$

$F$ = `FV_SCALE` (エンジンオプション。デフォルト 16)。

## バケット選択 (9 スタックのどれを使うか)

層 [3]〜[7] は 9 セット (LayerStacks) 存在し、局面ごとに**ちょうど 1 本**だけ評価される。
選択は両玉の段 (rank): 各玉の段を 3 ゾーンに割り当て (k3k3)、 $3 \times 3 = 9$ バケット。
FT ([1][2]) は全スタック共有 — LayerStacks の統合が NPS に効かない理由はこれ
(`docs/PLAN.md` 候補 6)。

## ファイルレイアウト (バイト順)

| セクション | 内容 |
|---|---|
| header | version `0x7AF32F16` (u32) + hash (u32) + arch 文字列長 (u32) + arch 文字列 |
| FT | section hash (u32) + LEB128 ブロック (biases 1024×i16) + LEB128 ブロック (weights 131949×1024×i16, feature-major) |
| LayerStack ×9 | 各: section hash (u32) + fc_0 (bias 8×i32 + w 8×1024×i8) + fc_1 (bias 64×i32 + w 64×**32**×i8) + fc_2 (bias 1×i32 + w 1×64×i8) |

- LEB128 ブロック = magic `COMPRESSED_LEB128` + u32 (圧縮バイト数) + signed LEB128 本体
- affine 重みはパディング入力次元込みの row-major。エンジン内の scrambled index は
  メモリ配置のみでディスク形式は線形
- ヘッダの arch 文字列は訓練時の旧名 (`HalfKA(Friend)`, `AffineTransform[7<-2048]` 等) の
  まま。hash 不一致警告が出るが無害 (`experiments/000-baseline/report.md` 既知の注意点 1)

## パーサとの対応 (`tools/nnue/`)

| 数式 | パーサのフィールド |
|---|---|
| $W^{FT}, b^{FT}$ [1] | `NNUEModel.feature_transformer.weights / .biases` |
| $W^{(0)}, B^{(0)}$ [3] | `LayerStack.fc_0.weights / .biases` |
| $W^{(1)}, B^{(1)}$ [6] | `LayerStack.fc_1.weights / .biases` (パディング列込み) |
| $W^{(2)}, B^{(2)}$ [7] | `LayerStack.fc_2.weights / .biases` |

活性化 [2][4][5] とバケット選択はパラメータを持たないため nn.bin には現れない。
