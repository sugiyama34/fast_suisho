# experiment-011: PC-ALM は backprop より強い評価関数を学習できるか

作成: 2026-10-06 / ブランチ `exp/011-pcalm` / サーバー **kajiki** (GPU 0 の MIG device 2 / 3 のみ)
状態: 設計確定 (2026-10-06 ユーザー決定, §7)。**主結果は 2026-10-10 に確定: PC-ALM の方が弱い (直接対局 −80.5 Elo)。[report.md](report.md)**。
本番規模の BP の複製 (ノイズ床) と参照の PC の本番規模は集計中

## 1. 問い

- PC-ALM (Augmented Lagrangian Predictive Coding, Seely & Gould, Sakana AI, 2026,
  [arXiv:2605.31022](https://arxiv.org/abs/2605.31022), [解説](https://pub.sakana.ai/pc-alm/),
  [MIT の JAX 参照実装](https://github.com/SakanaAI/pc-alm) `660747f`) で学習した
  `SFNN_halfka2_1024-7-64-k3k3` は、backprop (BP) で学習したものより強いか
- 「強い」= 対局のレーティングが高い。各手法とも最良の設定 (FV_SCALE を含む) で、**同じ教師データ・同じ学習局面数**で比べる
- 副次の軸: 学習の実時間コスト (1 局面あたり)
- 基準 (BP の最良): experiment-009 の full-e16 @ FV_SCALE 48 = 水匠 11 (FV_SCALE 32) 比 **−119.4 ± 11.3 Elo**
  (300k ノード, 2,000 ペア, `experiments/009-data-scaling/report.md`)

## 2. PC-ALM の要点 (論文と参照実装から)

ネットを「各層の出力 `h_i` を独立変数とし、層の式を制約にした最適化問題」に持ち上げ (lift)、
制約に Lagrange 乗数 `λ_i` と二次のペナルティを付けた拡張 Lagrangian を使う:

$$
\mathcal{L}_\rho(h,\theta,\lambda) = \ell\bigl(y, \hat{y}(h_{L-1})\bigr) + \sum_{i=1}^{L-1} \lambda_i^\top r_i + \frac{\rho}{2}\sum_{i=1}^{L-1} \lVert r_i \rVert^2,
\qquad r_i = h_i - f_i(h_{i-1};\theta_i)
$$

ミニバッチごとに、順伝播の値で `h` を初期化し `λ = 0` から始めて、局所的な更新を `T` 回繰り返したあと重みを 1 回更新する
(論文の Algorithm 1):

$$
\begin{aligned}
h_i &\leftarrow h_i - \eta_h \nabla_{h_i} \mathcal{L}_\rho \\
\lambda_i &\leftarrow \lambda_i + \alpha\, r_i \\
\theta_i &\leftarrow \theta_i - \eta_\theta \nabla_{\theta_i} \mathcal{L}_\rho(h,\theta,\lambda)
\end{aligned}
$$

1 行目 (primal) と 2 行目 (dual) を交互に `T − 1` 回、最後に primal をもう 1 回行い、3 行目 (重み) を 1 回行う。

- `∇_{h_i}` は隣の層 (`i−1`, `i`, `i+1`) だけに依存する (層局所)。`α = 0` は通常の predictive coding (PC)
- 重みの勾配は「BP の重み勾配で、逆伝播の随伴 `δ_i` を合成信号 `−(λ_i + ρ r_i)` に、前段の順伝播の活性を収束後の `h_{i−1}` に置き換えたもの」
- **平衡点では BP の勾配と一致する** (付録 A 命題 2: 実行可能な KKT 点では `λ_i = −δ_i`、重み勾配 = BP 勾配。
  線形ネットでは PC-ALM の反復がその点に収束することも示されている)。有限の `T` では BP からずれる
- 既定値: `ρ = 1`, `α = 1`, `T = 2L`, `η_h = 1/λ_max` (推論の線形化作用素の最大固有値。参照実装は表で持つ)。
  安定条件はモードごとに `η_h σ² (2ρ + α) < 4`
- 論文の主張は「**BP に匹敵する**」(深さ 1000 の MNIST で BP の約 2pp 以内、Fashion-MNIST の幅 8〜128 × 深さ 8〜128 の格子で BP と同等)。
  BP を上回る課題は報告されていない。主な動機は局所学習 (脳・ニューロモルフィック) で、GPU 上の高速化や性能向上ではない
- PC が BP より悪くなるのは「深くて細い」ネット。深さ 8 では幅によらず PC も BP とほぼ同じ (論文 Fig. 2)

参照実装 (`pcalm/inference.py`, 約 1,100 行) は密な残差 MLP + MSE + Adam のみ。推論の 1 ステップは
エネルギーの `jax.grad` で、バッチ平均のエネルギーに対する状態の学習率をバッチサイズ倍にして 1 サンプルあたりの `η_h` に戻している。
疎な入力・重み共有・layer stack・量子化は扱っていない。

## 3. 本ネットに当てはめると何が変わるか (実現可能性)

### 3.1 計算グラフ (BulletOu `2a8e5ed` の CUDA カーネルから)

| 段 | 計算 | 幅 |
| --- | --- | --- |
| FT | 視点 p ごとに `a_p = crelu(W0 x_p + b0)`。`W0` は両視点で共有、`x_p` は HalfKA2 の約 40 個 + factorizer の仮想特徴量 | 2 × 1,024 |
| pairwise | `c = [a_p[j] · a_p[j+512] · 127/128]` を両視点ぶん連結 | 1,024 |
| L1 | `z1 = W1[b] c + b1[b]` (+ shared factorizer) | 8 (= 7 + shortcut 1) |
| L2 入力 | `u = [crelu(z1[0:7]² · 127/128), crelu(z1[0:7])]` | 14 |
| L2 | `z2 = crelu(W2[b] u + b2[b])` (+ shared factorizer) | 64 |
| L3 | `out = W3[b] z2 + b3[b] + z1[7]` (+ shared factorizer) | 1 |
| loss | `(sigmoid(out) − target)²`, `target = sigmoid(score / 290)` (lambda 1.0) | |

`crelu` は 0〜1 へのクリップ、`b` は k3k3 の layer stack 番号 (9 組)。パラメータの 99.9% は FT。

### 3.2 持ち上げる状態 (lift) と制約

主案は **parametric 層の出力 3 つ**を状態にする (L = 4 ブロック):

| 状態 | 幅 | 制約 (残差 `r_i = h_i − f_i(h_{i−1})`) |
| --- | --- | --- |
| `h1 = c` | 1,024 | `c − pair(crelu(W0 x + b0))` (FT と pairwise を 1 ブロックとする) |
| `h2 = z1` | 8 | `z1 − L1_b(c)` |
| `h3 = z2` | 64 | `z2 − crelu(L2_b(u(z1)))` |
| 出力 | 1 | 制約なし。`ℓ(sigmoid(L3_b(z2) + z1[7]), target)` |

- shortcut (`z1[7]` → 出力) で計算グラフは DAG になるが、`∇_{z1}` が `r3` と出力の両方から信号を受けるだけ (論文付録 A の DAG への拡張)
- layer stack と shared factorizer は「サンプルごとに選ばれた実効重み」として扱う。重み勾配の分配は BP と同じ
- **FT の前活性 `W0 x + b0` は推論中に変わらない** (入力 `x` と重みは固定) ので、疎な FT は順伝播で 1 回計算するだけ。
  `T` 回の推論は幅 1,024 の要素ごとの演算と小さな密層 (1,024 → 8 → 14 → 64 → 1) だけで済む
- 代案: `h1 = (a_stm, a_nstm)` (pairwise の手前, 幅 2,048)。推論のメモリ転送が 2 倍になる。M0 で主案が不安定なときだけ使う
- `λ` はミニバッチごとに 0 から始めるので checkpoint に持たない。**`state.bin` と `nn.bin` の形式は変わらない**。
  量子化・nn.bin の書き出し・やねうら王側は一切変更なし

### 3.3 1 局面あたりの学習コスト (見積もり, 要実測)

- kajiki の MIG 1g.24gb スライスで BP は 1 sb (4,000 万局面 = 610 ステップ) に 57〜60 秒 → **1 ステップ約 95 ms** (batch 65,536)
- PC-ALM の追加分は推論 1 回あたり、`h1`・`λ1`・予測値の読みと `h1`・`λ1` の書き (約 5 × 65,536 × 1,024 × 4 B ≈ 1.3 GB) が支配的。
  スライスの帯域を全体の 1/4 (約 0.45 TB/s) と仮定すると約 3 ms/回。重み勾配の計算は BP の逆伝播を置き換えるだけ
- したがって **PC-ALM / BP ≈ 1 + T × 3 ms / 95 ms**: `T = 2` で 1.06 倍、`T = 8` (= 2L) で約 1.25 倍、`T = 16` で約 1.5 倍、`T = 32` で約 2 倍。
  代案の lift (幅 2,048) では追加分が 2 倍。カーネルを融合しない素朴な実装ではさらに 2〜3 倍悪化しうる
- GPU メモリの追加は約 1 GB (BulletOu は 1 run 約 5.2 GB, スライスは 24 GB)
- 実測は M1 の `BENCH` で行う (進捗行の時刻差 = 実時間。進捗行の `pos/s` は使わない, `docs/SETUP.md` §8)

### 3.4 実装の選択肢

| 案 | 内容 | 長所 | 短所 |
| --- | --- | --- | --- |
| **A. BulletOu を改造 (推奨)** | `--credit bp\|pcalm` と `--pcalm-steps/-alpha/-rho/-eta-h` を追加。順伝播の後に推論ループ (残差・状態の勾配・dual 更新) のカーネルを挟み、既存の逆伝播カーネル列 (L3 → L2 → L1 → pairwise → 疎な L0) の各段の入力勾配バッファを合成信号 `−(λ_i + ρ r_i)` で上書きし、活性を収束後の状態にして重み勾配だけを使う | BP arm は experiment-009 のレシピそのもの (データ経路・Ranger・factorizer・出力が同一)。速度も BulletOu 並み | 31,700 行の CUDA/Rust に手を入れる。融合カーネルの中で信号を差し替えるので、検証 (下記 M1) が必須。見積もり 1〜1.5 日 (検証込み) |
| B. 新トレーナ (PyTorch) で両手法 | psv ローダ、HalfKA2 + factorizer、9 stack + shared factorizer、Ranger、出力をすべて書き直す | 柔軟。自動微分で検証が楽 | BP が BulletOu と同等になる保証がなく、トレーナの差が交絡する。速度は 2〜5 倍遅い見込みで、kajiki の 2 スライスでは本番規模 (E=20) が 1 run 1 週間以上 |

- 案 A の BP arm は改造前と数値経路が同じであることを確認する (`--credit bp` で改造前と同じ loss 曲線)
- 案 B の部品の一部は小規模の検証 (M0) にだけ使う: 特徴量は experiment-009 の `loss_eval features`
  (Rust, psv → HalfKA2 u32 [N, 80] + stack 番号)、`state.bin` の読み書きと nn.bin 出力は `ablate/nnbin.py`
- nodchip/nnue-pytorch (`~/nnue-pytorch`) は旧アーキテクチャ (HalfKA 1024-8-96) 用で SFNN には使えない

### 3.5 事前の見立て (Claude)

**PC-ALM が BP を上回る見込みは低い**。結論は「差が検出できない、コストは 1.25〜2 倍」になる公算が大きい。理由:

1. PC-ALM は BP の勾配を局所的な力学で近似する手法で、平衡点では BP の勾配そのもの。「最良の設定」は BP に近づく方向にある
2. 本ネットは 4 ブロックしかない。論文でも深さ 8 では PC すら BP と同じで、PC-ALM の利点 (深くて細いネットでの信号の減衰を防ぐ) が効く領域ではない
3. BP と違う結果を生む唯一の経路は有限の `T` でのずれ (PC の prospective configuration, Song et al. 2024)。
   その利点が報告されているのは小バッチ・オンライン・継続学習で、本実験 (batch 65,536、教師 146.7 億局面) の領域ではない。
   experiment-009 では教師を半分にしても −6 Elo で、データ効率の改善が効く余地が小さい
4. ずれは勾配の誤差でもある。Ranger は勾配の大きさを重みごとに正規化するので、FT への信号が弱まる分はある程度吸収されるが、向きのずれは残る

ユーザー (10-06): 理論上は向かないことに同意する。ただし将棋 AI では実践が理論を裏切ることが多いので、見立てによらず実測で決める。
したがって途中で打ち切らず全段を行い (§4.2)、PC-ALM の調整は BP からずれる領域に寄せる (§4.3)。

## 4. 設計 (2026-10-06 ユーザー決定を反映, §7)

### 4.1 arm

| arm | 内容 | 学習 |
| --- | --- | --- |
| `bp-best` | experiment-009 の full-e16 @ FV_SCALE 48 (sha256 `b68d6d7b…`, 水匠 11 比 −119.4)。§4.4 の端の追加対局で e12 / e20 の方が良ければそれに替える。M2 の調整で BP のレシピが変わった場合は、新レシピを kajiki で E=20 学習し直したもの | 既存 (または M3 で 37 時間) |
| `pcalm-best` | experiment-009 と同じレシピ (奏乗 30 ファイル・同じ順序、sb=108 × 4,000 万、LR 0.000875 → 0.00003 step、Ranger、sigmoid-MSE、E=20) で、勾配だけ PC-ALM (M2 で選んだ `T`, `α`, `ρ`, LR) | M3 |
| `bp-rep` | ノイズ床。experiment-009 のレシピの BP を kajiki で、**教師の順序を回転** (016 始まり, experiment-009 の `full-rot` 案) して E=16 まで | M3 (29.5 時間) |
| `pc` | 通常の PC (`α = 0`)。小規模の arm (§4.3)。小規模で `pcalm-best` の候補に勝ったときだけ本番規模 (E=20) も学習する | M2 (+ 条件付きで M3) |

- BulletOu は学習中に再シャッフルせず、教師はファイル番号順に読む。したがって `pcalm-best` と full-e16 は**ステップごとに同じ局面列**を見る
  (違いは勾配の計算法と GPU だけ)。「同じ教師データ」より強い対応である
- 同じシード・同じ順序の再実行では GPU の非決定性しか測れず、学習 1 本ごとの揺らぎを過小評価する。そのため `bp-rep` は順序を回す
- `bp-rep` は「別の GPU (A100 → Blackwell) で BP を学習し直しただけの差」も含む。`pcalm-best` も kajiki で学習するので、比較の物差しとして適切

### 4.2 段階

**途中で打ち切らない** (ユーザー決定: 将棋 AI では実践が理論を裏切ることが多いので、§3.5 の見立てによらず全段を実測する)。
各段の終わりに結果をユーザーに報告するが、指示が無い限り次の段に進む。

| 段 | 内容 | GPU | 期間 (目安) |
| --- | --- | --- | --- |
| **M0** | Python (PyTorch) の float 参照実装 (順伝播は BulletOu と同一、PC-ALM の推論、BP は自動微分)。PC-ALM と BP の重み勾配の層ごとの cosine とノルム比を `T ∈ {1, 2, 3, 4, 8, 16, 32, 64}` × `α ∈ {0, 0.5, 1, 1.5}` × `ρ ∈ {0.5, 1, 2}` で、初期値と学習済み `state.bin` (experiment-009 の full-e1 / full-e16) の両方で測る。推論作用素の `λ_max` (べき乗法) から `η_h` を決め、学習の進行で変わるかも見る。M1 の検証の基準にもなる。極小規模の Python での学習は行わない (ユーザー決定) | 1 時間未満 | 約 0.5 日 |
| **M1** | BulletOu の改造 (案 A)。**対応するのは本実験の構成だけ** (SFNN k3k3 + shared factorizer + cuda-cpp。それ以外で `--credit pcalm` を指定したらエラー)。検証: (1) Python の順伝播が BulletOu の float 順伝播と一致、(2) 同じミニバッチで勾配が M0 と相対誤差 1e-4 以内、(3) `--credit bp` が改造前と一致、(4) `T` を大きくすると BP の勾配に近づく。`BENCH` で BP と PC-ALM (`T = 2, 3, 4, 8, 16`)・PC (`T` = 32 まで) の 1 sb あたりの実時間を同じスライスで測る。安定性の smoke (実データで `η_h` が発散しないか) | 約 3 時間 | 1〜1.5 日 |
| **M2** | 小規模の調整 (§4.3) と `pc` arm。最後に両手法の最良どうしの直接対局 400 ペア (参考) | 約 42 スライス時間 | 1〜1.5 日 |
| **M3** | 本番規模の学習: スライス A で `pcalm-best` (E=20)、スライス B で `bp-rep` (E=16)。`pc` が本番に進む場合は `bp-rep` の後にスライス B で | 69〜86 (+ `pc` 40〜55) スライス時間 | 2〜3 日 (+ 約 2 日) |
| **M4** | 対局 (§4.4, §4.5)。選択対局は M3 の学習中に checkpoint ができ次第始める | なし | 学習終了後 0.5〜1 日 |

### 4.3 ハイパーパラメータ調整 (両手法とも 8 run, 対局で選ぶ)

- 単位: 1 BulletOu-epoch (108 sb × 4,000 万 = 43.2 億局面, LR 1 周期)。教師は p30 の 9 ファイル
  (001 011 021 006 016 026 003 013 023, 約 44 億局面 = 約 1 周)。全 run で同じファイル・同じ順序 (BP の複製だけ回転)
- **選択は対局だけで行う** (ユーザー決定。loss は記録するが選択には使わない)。各 run の e1 を **BP の複製 `s-bp-lr1-rot`**
  (候補ではない) の e1 と 300 ペア対局し、Elo 最大のものを選ぶ (誤差内でも最大値)。`bp-recipe` (= `s-bp-lr1`) も同じ相手と対局する
  (10-06 変更: 当初案の「`bp-recipe` を相手にする」では `bp-recipe` だけが 0 に固定され非対称になるため, [notes/decisions.md](notes/decisions.md))。
  相手を水匠 11 にしないのは、e1 は水匠 11 より約 300 Elo 弱く、近い強さの相手の方が区間が狭いため (300 ペアで約 ±28 対 ±39 Elo)。
  FV_SCALE は小規模では調整せず両方 48 に固定。開始局面は experiment-009 の 300 ペアの集合 (2,000 ペアの部分集合)。
  エンジンは kajiki のビルド (`a5f519d2`) で調整対局の系列を揃え、`ulimit -s 8192` で起動する

| 手法 | 8 run の内訳 |
| --- | --- |
| PC-ALM | (A) `T ∈ {2, 3, 4, 8}` (`α = 1`, `ρ = 1`, LR はレシピ値) の 4 本 → (B) 最良の `T` で `α = 1.5` と `ρ = 2` の 2 本 → (C) 最良の (`T`, `α`, `ρ`) で LR × {0.5, 2} の 2 本 |
| BP | LR (最大値) × {0.5, 0.7, 1 (= `bp-recipe`), 1.4, 2} の 5 本 → 最良の LR で lr-min × {0.3, 3} の 2 本 → `bp-recipe` の複製 (教師の順序を回転) 1 本 |
| `pc` (参照, 予算外) | `T ∈ {4, 8, 16, 32}` (`α = 0`, LR はレシピ値) の 4 本。PC は信号が FT に届くまでに PC-ALM より多くの `T` が要る |

- PC-ALM の格子は **BP からずれる領域** (少ない `T`、大きい `α`、`ρ`) に寄せる。`T` を大きくすると BP の勾配に収束するだけなので、
  実践で理論と違う結果が出るとすればずれのある領域だから
- `η_h` は M0 の `λ_max` から (`α`, `ρ`) ごとに決め、調整はしない
- **非対称性 (明記する)**:
  - PC-ALM は調整の次元が多い (`T`, `α`, `ρ`, LR) ので、同じ 8 run でも次元あたりの探索は BP より粗い (PC-ALM に不利)
  - レシピ (sb=108、LR の形、epoch 数) は experiment-005/006 で BP 向けに探したもの。PC-ALM はそれを借りる (PC-ALM に不利)。LR の倍率の 2 点で一部補う
  - BP の複製 1 本は調整ではなくノイズの推定に使う (BP の調整は実質 7 本)
- BP の最良がレシピ値 (LR × 1, lr-min × 1) から変わったら、`bp-best` は新レシピで kajiki で E=20 学習し直す (ユーザー決定, M3 で +37 時間)。
  M3 の `bp-rep` の枠をそれに使うので、本番規模のノイズ床は無くなる (§4.5 の `N` は小規模の複製と CI で代用する)

### 4.4 checkpoint と FV_SCALE の選び方 (experiment-009 と同一の手順)

「同じ学習局面数」は **両手法に同じ予算 (E ≤ 20 = 最大 864 億局面、同じファイル順) を与え、同じ選択手順で予算内の最良を選ぶ**、と読む
(ユーザー決定)。experiment-009 の BP の最良 (e16) も e20 までの学習から選んだものだからである。e16 どうし (691 億局面ちょうど) の比較は副次の結果として出す。

experiment-009 の full と同じ規則を `pcalm-best` (と、本番に進んだ場合の `pc`) にそのまま当てる (計 21 点 × 400 ペア, 相手は水匠 11 @ 32):

| epoch | FV_SCALE |
| --- | --- |
| e5, e8 | 32, 40, 48, 56, 64 |
| e12 | `f − 8`, `f`, `f + 8` (`f` = e8 の格子で最良の値) |
| e16, e20 | `f − 8`, `f`, `f + 8`, `f + 16` |

- 21 点の Elo 最大を最良とし、2,000 ペアで測り直す (experiment-009 と同じ。選択に使った 400 ペアは 2,000 ペアの部分集合なので、わずかに上振れしうる)
- **格子の端が最良になったら外側に 1 点足す** (両手法に共通の規則, ユーザー決定)。experiment-009 の BP では e12 の最良 (48, −119.6) と
  e20 の最良 (56, −115.2) が格子の上端だった (e5 / e8 / e16 は内側)。公平のため BP 側にも `full-e12 @ 56` と `full-e20 @ 64`
  (各 400 ペア) を足す。e12 @ 48 は e16 @ 48 と 400 ペアで約 7 Elo しか違わないので、どちらかが最良になる可能性はある。
  そのときは `bp-best` をそれに替えて 2,000 ペアで測り直す (+2,000 ペア)。experiment-009 の −119.4 とその開始局面ごとの対応はそのまま使えなくなる
- PC-ALM は出力のスケールが BP と違いうるので、e8 の格子の最良が 32 か 64 なら、そちら側に 8 刻みで広げる

### 4.5 最終評価と判定規則

対局条件は experiment-009 と同一: やねうら王 V9.60 + `finny.patch` (AVX2, **experiment-009 と同じバイナリ** sha256 `423b6b1c…` を suzuki から複製)、
1 手 300,000 ノード固定、Threads 1、Hash 256 MB、互角局面集 ply24 からストライド抽出 (2,000 ペア = experiment-009 の full と同じ開始局面)、
相手の水匠 11 は FV_SCALE 32。ツールは experiment-009 の `match_nodes.py` / `match_queue.py` を流用する (キュー設定とパスだけ 011 用に分ける)。

- エンジンの同一性: 複製したバイナリで full-e16 @ 48 の experiment-009 の棋譜から約 50 ペアを kajiki で再生し、全局の指し手が一致することを確かめてから
  −119.4 とその対応を使う。一致しなければ kajiki で full-e16 を 2,000 ペア測り直す

| 対局 | ペア |
| --- | --- |
| `pcalm-best` vs 水匠 11 @ 32 | 2,000 |
| `bp-rep` e16 @ 48 vs 水匠 11 @ 32 (ノイズ床) | 2,000 |
| **`pcalm-best` vs `bp-best` (直接対局, 同じ開始局面)** | 2,000 |
| (再利用) `bp-best` vs 水匠 11 @ 32 = −119.4 | experiment-009 の 2,000 ペア |

判定 (事前に固定する):

- `D` = 直接対局の Elo (`pcalm-best` から見た値, pentanomial の 95% CI)
- `N` = max(`|Elo(bp-rep) − Elo(full-e16)|`, その対応のある差の 95% CI の半幅)。学習 1 本ごとの揺らぎの目安
  (水匠 11 相手, 同じ開始局面)。複製は 1 本だけなので、観測した差が偶然 0 に近くても判定が甘くならないよう、CI の半幅を下限にする。
  `bp-rep` が無い場合 (§4.3 で BP を学習し直したとき) は、小規模の複製の差と直接対局の CI の半幅の大きい方
- **PC-ALM の方が強い**: `D` の 95% CI の下限 > 0 かつ `D > N`
- **PC-ALM の方が弱い**: `D` の 95% CI の上限 < 0 かつ `|D| > N`
- それ以外: **差は検出できない** (CI 付きで報告)
- 副次: 水匠 11 相手の Elo の差 (開始局面のブートストラップ)、e16 どうしの比較、1 局面あたりの実時間の比、held-out loss、`pc` の結果

### 4.6 計算量と日程 (kajiki のみ, ユーザー決定)

kajiki: MIG 1g.24gb × 2 (BP は 1 スライス約 0.65M 局面/秒、2026-09-28 実測)、CPU 16 コア (他ユーザーが常時約 2 コア使用)。
対局の並列数は学習中 W=8、学習していないとき W=12。

| 段 | GPU (スライス時間) | 対局 (ペア) | 実時間 |
| --- | --- | --- | --- |
| 準備 (§5) | – | 約 100 (エンジンの同一性確認) | 教師の取得 約 3.5 時間 (M0 と並行) |
| M0 | 1 未満 | – | 約 0.5 日 (10-06 に実装と初期値での測定まで済み) |
| M1 | 約 3 | – | 1〜1.5 日 (10-06 に実装と検証まで済み, [notes/m0.md](notes/m0.md)) |
| M2 | BP 8 × 1.85 + PC-ALM 8 × 約 2.1 + PC 4 × 約 2.6 ≈ 42 | 20 × 300 + 400 = 6,400 (学習と並行) | 1〜1.5 日 |
| M3 | `pcalm-best` 37 × 1.06〜1.5 = 39〜56、`bp-rep` 29.5 | – | 2〜3 日 |
| M4 | – | 8,400 (選択) + 800 (BP の端) + 2,000 × 3 = 15,200 (`bp-best` が替われば +2,000) | 学習終了後 0.5〜1 日 |
| 報告 | | | 約 0.5 日 |
| 計 | 約 115〜135 | 約 21,700 | **6〜8 日** |

- 条件付きの追加: `pc` が本番に進めば GPU +40〜55 スライス時間と対局 +10,400 ペアで約 +2 日。BP をレシピ変更で学習し直す場合は `bp-rep` の枠を使うので実時間はほぼ変わらない
- 実装 (M0 + M1) の見積もりは当初 1〜2 週間としていたが、人の作業量の目安で見積もっていたので改めた。律速はビルドと検証の実行、GPU の計測
- 対局の速度は suzuki の実測 (300k ノードで 1 局 42 秒/コア) からの推定で、W=12 で約 500 ペア/時、W=8 で約 340 ペア/時。kajiki で最初に測る

## 5. kajiki の準備

| もの | 状態 (2026-10-06) | 対応 |
| --- | --- | --- |
| 奏乗の教師 30 ファイル (587 GB) | **取得中** (10-06 16:36 開始、約 46 MB/s で約 3.5 時間) | `/mnt/D/sugiyama/teacher/sojo/train` に置き `data/teacher/sojo` をシンボリックリンクにした (`/` は空き 218 GB)。順序は p30 の 9 ファイル → S0 (010 020 030) → 残り。ログ `/mnt/D/sugiyama/logs/fetch_teacher.log`。揃ったら `scripts/fetch_teacher.sh --verify` |
| `floodgate.hcpe` | 取得済み (sha256 一致) | `data/teacher/floodgate/` |
| checkpoint 置き場 | 作成済み | `/mnt/D/sugiyama/checkpoints` (`data/bulletou/checkpoints` からリンク)。M2・M3 で約 60〜80 個 = 約 140〜180 GB |
| suzuki にしか無いもの | 未 | 下の手順でユーザーが転送する |
| BulletOu | `2a8e5ed` + sm_120 用の変更でビルド済み (sha256 `fa67f210…`, CUDA 13.1) | 改造版は別ディレクトリでビルドし、改造前のバイナリは BP 用に残す |
| PyTorch (M0) | 無い | 専用の venv (sm_120 対応の版) |

### suzuki から kajiki への転送

持ってくるもの (計 約 5 GB):

| ファイル (suzuki) | 用途 | sha256 |
| --- | --- | --- |
| `~/engines/009-finny-avx2/YaneuraOu-by-gcc` | 対局エンジン (experiment-009 と同一) | `423b6b1c…` |
| `/mnt/nvme1/sugiyama/checkpoints/009-full/0016/nn.bin` | `bp-best` | `b68d6d7b…` |
| `/mnt/nvme1/sugiyama/checkpoints/009-full/{0012,0020}/nn.bin` | FV_SCALE の端の追加対局 (§4.4) | `a834c15b1ad2…`, `711247c13c4b…` |
| `/mnt/nvme1/sugiyama/checkpoints/009-full/{0001,0016}/state.bin` | M0 の学習済み重み (各 2.2 GB) | 転送時に記録 |
| `~/fast_suisho/experiments/009-data-scaling/games/final-full-e16@48-vs-s11@32-300k-2000p/` | エンジンの同一性確認と、開始局面ごとの対応のある比較 | 転送時に記録 |
| `/mnt/nvme1/sugiyama/loss_samples/B_s0.*` (任意, 約 360 MB) | held-out loss の記録用 | 転送時に記録 |

手順:

1. **suzuki で**集める (元のファイルは変更しない。同じファイルシステム上はハードリンク):

   ```sh
   cd ~/fast_suisho && git fetch origin exp/011-pcalm
   git show origin/exp/011-pcalm:experiments/011-pcalm/xfer/pack_on_suzuki.sh | bash
   # → /mnt/nvme1/sugiyama/xfer-011/ と MANIFEST.sha256
   ```

2. **手元の端末から**転送する (suzuki と kajiki の両方に ssh できる端末。データは端末に保存されず通過するだけ):

   ```sh
   ssh suzuki 'tar -C /mnt/nvme1/sugiyama -cf - xfer-011' | ssh kajiki 'tar -C /mnt/D/sugiyama -xf -'
   # 代わりに: scp -3 -r suzuki:/mnt/nvme1/sugiyama/xfer-011 kajiki:/mnt/D/sugiyama/
   ```

   suzuki から kajiki へ直接 ssh できるなら、suzuki で `rsync -a --info=progress2 /mnt/nvme1/sugiyama/xfer-011 kajiki:/mnt/D/sugiyama/` が速い

3. **kajiki で**検証して置く (Claude が実行してもよい):

   ```sh
   bash experiments/011-pcalm/xfer/install_on_kajiki.sh
   ```

   MANIFEST の sha256 と、experiment-009 のレポートに記録された sha256 の両方を確認してから、エンジンを `~/engines/009-finny-avx2/`、
   checkpoint を `/mnt/D/sugiyama/checkpoints/009-full/`、棋譜を `experiments/009-data-scaling/games/` に置く

4. 終わったら suzuki の `/mnt/nvme1/sugiyama/xfer-011` は消してよい (ハードリンクなので元のファイルは残る)

## 6. 依存と規約

- 学習レシピ・対局・FV_SCALE の手順は experiment-009 (マージ済み, PR #23) のもの。PR #19 (experiment-007) の WRM loss と
  `fit_fv_scale.py` は使わない
- 対局エンジンは experiment-009 と同じく experiment-008 の `finny.patch` を当てたビルド (experiment-009 のバイナリそのもの)。パッチの出所は
  PR #17 のブランチだが、experiment-009 の対局プロトコルの再利用であり、PR #17 の内容に新たに依存するものではない (ユーザー承認 10-06)
- W&B: entity `suisho`、project **`pcalm_vs_backprop`** (ユーザーが作成)、group = `011-pcalm`。API キーは起動シェルの環境変数だけ
- 実験番号: experiment-010 は定跡の実験 (`experiments/009-data-scaling/hypothesis.md` §8) に予約済みなので飛ばし、011 を使う
- GPU は kajiki の GPU 0 の MIG device 2 / 3 だけ (UUID で指定。`docs/SETUP.md` §9)

## 7. 決定事項 (2026-10-06 ユーザー)

| 項目 | 決定 |
| --- | --- |
| 実施 | 行う。理論上は PC-ALM が本ネットに向かないことに同意するが、将棋 AI では実践が理論を裏切ることが多いので実測で決める |
| サーバー | **kajiki のみ** (学習も対局も)。GPU は MIG device 2 / 3 |
| 教師の取得 | すぐに始めてよい (`/mnt/D`) |
| エンジン | experiment-009 のバイナリ (`423b6b1c…`, finny.patch 入り) を suzuki から複製し、棋譜の再生で同一性を確認 |
| 「同じ学習局面数」 | 同じ予算 (E ≤ 20) + 同じ選択手順。e16 どうしは副次 |
| 打ち切り | **行わない** (全段を実行し、各段で報告) |
| 調整の選び方 | **対局だけ** (各 run 300 ペア, 誤差内でも Elo 最大) |
| 調整の規模 | 両手法とも 8 run |
| 通常の PC | 小規模の arm。PC-ALM に勝ったときだけ本番規模も |
| BP の複製 | 行う (教師の順序を回転, E=16) |
| FV_SCALE の端 | 両手法に同じ規則。BP にも e12 @ 56 と e20 @ 64 を追加 |
| BP のレシピが調整で変わったら | 本番規模で学習し直す (`bp-rep` の枠を使う) |
| 実装の範囲 | 極小規模の Python 学習は行わない (勾配の比較だけ)。BulletOu の改造は本実験の構成だけに対応 |
| suzuki のファイル | Claude が一覧と手順を用意し、ユーザーが転送する (§5) |
| W&B | project `pcalm_vs_backprop` |
| 文書 | この設計を commit し、draft PR を出す |
