# 探索中の特徴量発火回数の計測 (やねうら王 FEATCOUNT ビルド)

水匠 11 同士の対局で、探索中に NNUE を実際に計算した評価ごとに HalfKA2 特徴量 (131,949 個,
両視点) の発火回数を数える。教師側の集計 (`../feature_count/`, `feature-counts.md`) と同じ index。

## ビルド

- ソース: `~/YaneuraOu` (V9.60 `9133c527` + experiment-008 `finny.patch`) を `~/YaneuraOu-featcount` に
  コピーし、`featcount.patch` (sha256 `f390f9f9…fbad432`) を当てる。計測コードはすべて `#if defined(FEATCOUNT)` の中
- `make -j16 normal YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3 COMPILER=g++ TARGET_CPU=AVX2 PYTHON=python3 EXTRA_CPPFLAGS=-DFEATCOUNT`
- バイナリ: `~/engines/009-featcount/YaneuraOu-by-gcc`
  sha256 `a0483118e87c4d54891345cb5bad0fced3a6239f2d344a5d84ff18ca073157a2` (g++ 13.3.0, suzuki)。
  対局用の `~/engines/009-finny-avx2/` とは別物で、そちらは変更していない

## 数える場所

`evaluate_nnue.cpp` の `NNUE::ComputeScore()` で、`accumulator.computed_score` による早期 return の後
(= Transform + Propagate を実行する直前)。

- この版には eval hash (`USE_EVAL_HASH`) がコンパイルされておらず、評価値のキャッシュは
  `computed_score` (同じ StateInfo で 2 回目以降の evaluate()) だけ。TT に staticEval があって
  evaluate() を呼ばない場合もここに来ない。したがって「ネットワークを実際に計算した回数」だけを数える
- `refresh == true` の呼び出し (`Position::set()` からの `compute_eval()`, 探索外) はヒストグラムに入れず
  `set_evals` として別に数える
- active index は差分計算の状態を使わず、局面から `RawFeatures::AppendActiveIndices()`
  (全計算と同じ関数) で両視点ぶん計算し直す。1 評価 = 80 カウント。40 個でない視点は `bad` に数える
- ヒストグラムはスレッドごと (thread_local) で、dump 時に合算する
- オーバーヘッド: 1 スレッドの NPS で約 8.6% 減 (922.8k → 843.5k, 3M ノード × 4 局面 × 3 回の中央値,
  1 コアに固定・交互計測)

## USI 拡張コマンド

| コマンド | 動作 |
| --- | --- |
| `featdump <path>` | 合算ヒストグラムを `.npy` (`<u8`, shape (131949,)) で書き、`info string featdump path … evals N set_evals M sum S bad B` を返す |
| `featreset` | 0 に戻す |
| `featstat` | 回数だけ表示 |
| `featsfen <sfen>` | (検証用) その局面を 1 評価として数え、両視点の index を表示 |

環境変数 `FEATCOUNT_OUT` があれば `quit` 時にそのパスへ dump する (stderr に結果)。

## 検証 (2026-09-29)

- `validate_engine.py`: 互角局面集から乱数で 0〜120 手進めた 2,000 局面で、`featsfen` の index と
  cshogi での再計算 (`../feature_count/validate.py` の `halfka2_features`) を比較 → **不一致 0**。
  featdump したヒストグラムも Python 側の集計と完全一致 (合計 160,000 = 2,000 × 80)
- 対局中の全評価で `sum == 80 × evals`, `bad == 0` (`match_featcount.py` が毎局確認する)
- 探索を変えないこと: 300k ノード・20 開始局面で、対局用バイナリ (`match_nodes.py`, 40 局 = 先後両方) と
  計測版 (`match_featcount.py`, 20 局) の棋譜が **40 局すべて一致**

## 対局と集計

- `../match_featcount.py`: 1 開始局面 1 局、`search.npy` (探索中の評価) と `played.npy`
  (棋譜上の局面, cshogi で計算) を `games/featcount/<name>/` に出す
- `compare.py <run_dir>`: 教師 (30 ファイルの `both` の和) との TV / JS、教師で稀な特徴量の数など
