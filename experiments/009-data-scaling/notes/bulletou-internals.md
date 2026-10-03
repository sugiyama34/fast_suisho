# BulletOu `2a8e5ed` の内部 (ソース調査, 2026-09-29)

コードを読んだだけで、実行による確認はしていない (水匠 11 の nn.bin の読み取り専用パースを除く)。
パスは BulletOu リポジトリからの相対。`B` = `examples/bulletou.rs`,
`CU` = `crates/cuda_cpp/cpp/bulletou_cuda_backend.cu`, `LIB` = `crates/cuda_cpp/src/lib.rs`。

## 特徴量 HalfKA2

- 次元 81 (自玉のマス) × 1,629 (駒入力) = **131,949**。1 局面・1 視点あたり 40 個 (玉 2 枚 + 38 駒)
- index = `kb*1629 + pack(bp)`、`pack(bp) = bp >= 1629 ? bp-81 : bp` (相手玉を自玉 plane に畳む)。
  kb = 自玉のマス (後手視点は `80 - sq`)。持ち駒は「k 枚目」ごとに 1 特徴量
  (`crates/bulletou_lib/src/game/inputs/shogi_halfka.rs:522-605`)
- やねうら王の `Features::HalfKA2` (hash `0x5f234cb8`, `source/eval/nnue/features/half_ka2.h`) と
  index の計算式が一致する。教師側と対局側の集計を同じ index で突き合わせられる
- `k3k3` は入力特徴量ではなく **layer stack の選択**: 自玉の段 (3 区分) × 相手玉の段 (3 区分) で
  9 組の L1/L2/L3 から 1 つを選ぶ。FT は全 stack 共通

## FT の factorizer (HalfKA2 では常に有効)

- 学習中は 131,949 行に加え、駒入力ごとの仮想特徴量 1,629 行 (`w_factor[P]`, 玉位置によらない)
  を持つ。特徴量 (K, P) が発火すると仮想特徴量 P も発火する。無効にするフラグは無い
  (`B` の `virtual_rows()` が HalfKA2 で固定値を返す)
- export 時に畳み込む: `nn.bin` の行 (K, P) = `w_specific[K,P] + w_factor[P]`
  (`fold_sfnn_halfka2_piece_factorized_l0w`, `B`:9862-9888)
- `w_factor[P]` の勾配は 81 玉位置分の和 (`CU`:2259-2283)。したがって `w_factor` はほぼ稀にならない
- `--sfnn-factorizer` (既定 `shared`) は layer stack 側の factorizer で、FT とは別物

## 初期化

- `w_specific`: 一様分布 [-h, h], h = 1/√131949 ≈ **0.002753**。固定シード `0x5f11_e001` の
  xorshift64 で生成 (`B`:8474-8588, 10042-10046, 10446-10466)。**実行ごとにビット単位で同一**
  (`--seed` は `nerf` サブコマンド用で学習には効かない)
- `w_factor`: **0**。FT bias: 同じ一様分布 (シード `0x5f11_e002`)
- export の量子化スケールは 127 なので、0.00275 × 127 ≈ 0.35 → 0 に丸まる。
  **一度も学習されていない `w_specific` の行は nn.bin 上ですべて 0** になり、`w_factor` だけが残る。
  したがって「`w_specific` を 0 にする」と「`w_specific` を初期値に戻す」は nn.bin 上で同一

## checkpoint (`<output>/NNNN/`)

- `nn.bin`, `state.bin`, `teacher.txt`, `dataloader_pos.txt`, `learn.log`
- `state.bin` = レコードの連結。各レコードは `id` + `\n` + u64 LE 個数 + f32 LE × 個数
  (`crates/bulletou_lib/src/value/yaneuraou_kppt.rs:109,168`)
- **畳み込み前の重みを持つ**: `nnue/weights/l0w` = (131,949 + 1,629) × 1024 の f32、行優先。
  行 0..131948 が `w_specific`、行 131949 + P が `w_factor[P]`。
  他に `nnue/{momentum,velocity,slow}/<id>` と `nnue/step_ranger/<id>`
- **checkpoint から nn.bin だけを書き出すツールは無い**。アブレーション用に
  「state.bin を読む → 行を書き換える → 畳み込む → 量子化 → nn.bin を書く」ツールを自作する

## nn.bin 形式 (`B`:9543-9757)

- ヘッダ: version `0x7AF32F16`, hash, desc 長 + desc, FT hash
- FT: bias ブロックと weight ブロック。どちらも `"COMPRESSED_LEB128"` + u32 サイズ + signed LEB128 の i16
  (スケール 127)。PSQT は無い。可変長なので行単位のその場書き換えはできない (全体をデコード→再エンコード)
- layer stack × 9: 各 10,600 バイト (L1 i8 8×1024, L2 i8 64×32, L3 i8 64, bias は i32)
- 水匠 11 の nn.bin (135,285,230 バイト) の内訳: FT weight 131,949 × 1024 個の i16 のうち
  1 バイトで符号化されるもの 135,043,027 個、2 バイト 72,749 個。ほぼ全値が [-64, 63] に収まるため
  270 MB ではなく 135 MB になる

## Ranger (唯一のオプティマイザ)

- cuda-cpp では Ranger (RAdam + Lookahead) だけ。beta1 0.99, beta2 0.999, Lookahead k=6 α=0.5,
  重みは ±1.98 にクリップ (`crates/trainer/src/optimiser/ranger.rs:176-199`)
- **FT の全行を毎ステップ更新する (疎な更新ではない)**: 勾配の無い行も m ← 0.99m, v ← 0.999v で
  減衰しながら `lr·m/(√v+ε)` だけ動き続ける。Lookahead も 6 ステップごとに全行にかかる
  (`LIB`:5422-5429, `CU`:351-474)。稀な特徴量の行は、最後に発火した後も古い momentum で動く
- SGD にしたい場合: RAdam の `n_sma_threshold` (現在 5.0 固定) を十分大きくすると分母を使わない
  更新 (= momentum SGD) になる。1 行の変更で可能 (今回は Ranger のままと決定)

## 教師の読み込み

- `.psv` は全レコードを順に読む。手数や評価値でのスキップは無い
- **|score| ≥ 32000 のレコードは除外されず、loss の重みが 0** になる (`--score-drop-abs` 既定 32000)。
  勾配を生まないので、特徴量の出現回数の集計ではこれらを別に数える (`feature-counts.md`)
