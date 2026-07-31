# 008-finny-tables: AccumulatorCache (finny tables) の移植

日付: 2026-07-31

## 結果サマリ (TL;DR)

- Stockfish の AccumulatorCaches (通称 finny tables) をやねうら王 V9.60 の旧NNUE
  (HalfKA2 1024) に移植した。**精度損ゼロの等価変換** — 決定的探索 (500 局面 ×
  50k ノード, 1スレッド) の searchlog がベースラインとバイト単位で完全一致
- ついでに `update_accumulator()` が前局面の Accumulator (約 4KB) を**値渡しでコピー**
  していた問題 (experiment-002 の memmove 費目の主因) を参照渡しに修正 (これも等価変換)
- NPS (1スレッド, 対計測):
  - finny tables 単体: **+28.3〜29.8%**
  - コピー修正の上乗せ: +7.4〜10.0%
  - **合計 +38.7〜41.4%** (refresh 全消しの理論上限 +43% にほぼ到達)
- 16スレッド (暫定): **+36.3〜37.0%** (理論上限 +39%)
- **計測条件の注意**: 対計測の途中から同一マシンで GPU 学習ジョブ (bulletou ×2〜3,
  約 13 コア消費) が動いていた。ペア内交互実行により相対値は一貫しているが、
  メモリ帯域競合下ではメモリ系削減の効果が過大に出る方向のバイアスがありうる。
  特にコピー修正の上乗せ分と 16T の値はアイドル環境での再計測が望ましい (c8a では必須)

## 仮説と目的

- **問い**: FT refresh (experiment-002 実測で全サイクルの 30.1%、差分更新の 3.4 倍) を
  「玉マスごとにキャッシュした accumulator との差分適用」に置き換えると NPS はどれだけ
  上がるか。評価値は完全一致を維持できるか
- **予測**: refresh 全消しの Amdahl 上限は 1T +43% / 16T +39% (experiment-002)。
  finny tables は refresh を「エントリ保存時から動いた駒だけの差分」に置き換えるため
  大半を回収できる見込み。等価変換なので評価値は完全一致するはず
- **判定指標**: フェーズ3 軽量プロトコル (等価変換用) — (1) 決定的探索 searchlog の
  完全一致 (`usi_drive.py --mode searchlog`)、(2) `bench` NPS の対計測

## 設計

パッチ全文: `finny.patch` (やねうら王 V9.60 `9133c527` に適用)。変更 5 ファイル・+141 行。

### キャッシュ構造 (`nnue_accumulator.h`)

```
AccumulatorCacheEntry {                    // 視点 × 視点から見た自玉のマス ごと
    int16_t   accumulation[1024];          // キャッシュされた accumulator
    BonaPiece pieces[PIECE_NUMBER_NB=40];  // それを構成した駒リスト
    bool      computed;
}
AccumulatorCache { entries[2][SQ_NB=81]; uint32_t generation; }
inline thread_local AccumulatorCache accumulator_cache;   // 約 354 KiB/スレッド
```

エントリの不変条件: `accumulation == biases + Σ column(MakeIndex(sq_k, pieces[i]))`。
これは駒番号 (PieceNumber) の割り当ての同一性に依存しないので、別のゲーム・別探索木で
作られたエントリもそのまま差分の基点として正しい (クリア不要)。

### refresh の置き換え (`nnue_feature_transformer.h`)

`refresh_perspective_with_cache(pos, perspective)` を追加し、従来の全計算 2 箇所を
これに置き換えた:

1. `refresh_accumulator()` — 両視点の全計算 (前局面の accumulator が無いとき)
2. `update_accumulator()` の reset 分岐 — 自玉が動いた視点の全計算

処理: 現局面の駒リスト (`piece_list_fb/fw`, 40 スロット) とエントリ保存時の駒リストを
スロット単位で比較し、値が違うスロットだけ旧値の列を減算・新値の列を加算して
エントリを最新化し、エントリを pos の accumulator に memcpy する。従来の全計算は
biases + 40 列の加算 (258 MiB の重みテーブルから 80KB ストリーム) だったのが、
典型的には数駒分の差分 + memcpy 2KB ×2 になる。

**等価性の根拠**: int16 の加算は 2 の補数の mod 2^16 で可換・結合的なので、
「biases + 現在の 40 駒の列の総和」に至る加減算の順序・経路によらず結果はビット一致する。

### ガードと無効化

- キャッシュ実装は `RawFeatures == FeatureSet<HalfKA2<kFriend>>` (refresh trigger 1 個)
  のときだけ `if constexpr` で有効化。他の特徴量セット (halfkp 等) は従来コードのまま
- 評価関数パラメータの読み直し (`LoadAndShare`) で世代カウンタを進め、スレッドローカル
  キャッシュ側が世代不一致を検出したら全エントリを無効化 (nn.bin 差し替え対応)。
  世代カウンタは relaxed で、**読み直し中に並行して評価が走らないこと** (USI では
  isready 中は探索停止) を前提とする。学習器など別経路でこの前提を破ると、旧重みで
  作られたエントリ + 新重みの差分列という壊れた accumulator がキャッシュに残りうる
- `HalfKA2::MakeIndex` はキャッシュ側からも呼ぶため定義を .cpp からヘッダに移動 (同一定義)

### 同梱した等価修正 2 件

1. **`update_accumulator()` の値渡しコピー除去**: `const auto prev_accumulator = ...` が
   Accumulator 丸ごと (約 4KB) のコピーだったのを `const auto&` に。experiment-002 で
   memmove 3.6% のうち約 3.5pt が update/refresh 経路の accumulator コピーと
   特定されていたものの主因
2. **`reset[2]` の未初期化読み取り修正**: `AppendChangedIndices` は `dirty_num == 0`
   のとき `reset[]` を書かずに return するため、未初期化のまま参照されうる。
   実際には `do_null_move` は StateInfo を memcpy するため `dirtyPiece` は親のものを
   継承し (`dirty_num` は 1 か 2)、`dirty_num == 0` はルート局面のみ = 常に refresh
   経路に入るので、この読み取りは現行コードでは到達しないと思われる (レビューで確認)。
   防御的に `{false, false}` で初期化しておく (仮に到達してゴミが true に転ぶと、
   ベースライン実装では biases のみの accumulator になり評価値が壊れる箇所だった)

## セットアップ

- ベース: やねうら王 V9.60 commit `9133c527` + `finny.patch`、
  edition `YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3`, COMPILER=g++, TARGET_CPU=AVX2
  (000-baseline と同一フラグ)。評価関数: Suisho 11 `nn.bin` (`a78b7f88…`, 000-baseline と同一)
- ビルド・適用手順:

  ```sh
  cd ../YaneuraOu
  git apply ../fast_suisho/experiments/008-finny-tables/finny.patch
  cd source
  make clean YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3 \
       COMPILER=g++ TARGET_CPU=AVX2 PYTHON=python3
  make -j16 normal YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3 \
       COMPILER=g++ TARGET_CPU=AVX2 PYTHON=python3
  cd .. && git checkout -- source/eval   # 計測後に復元
  ```

- バイナリのアーカイブ (`~/engines/finny-008/`):

  ```
  4881fcf42afd6aef31f3db825e2d2901b55ac5e7e76bc9085fe89aeb27b8ee6c  YaneuraOu-by-gcc            (最終版: finny + コピー修正)
  8e077b178654c7ddf95aee624d15b87ce66b956c644a692f4480d007719d90f5  YaneuraOu-by-gcc-finny-only (finny のみ。効果分離の計測用)
  ```

- 比較相手: `~/engines/baseline-000/YaneuraOu-by-gcc` (`312b1368…`)

## 検証: 評価値の完全一致

experiment-003 で確立した決定的探索 searchlog 方式 (V9.60 の `eval` USI コマンドは
未実装スタブのため、Threads=1 固定ノード探索の出力一致で全評価局面の一致を検証する):

```sh
.venv/bin/python experiments/003-channel-permutation/usi_drive.py \
  --engine <binary> --evaldir ~/suisho11 \
  --sfens experiments/003-channel-permutation/sfens_verify_500.txt \
  --mode searchlog --nodes 50000
```

ベースライン vs finny-only、ベースライン vs 最終版 (finny+コピー修正) の双方で
**500 局面すべての出力 (最終 info 行 + bestmove) がバイト単位一致** (`diff` 空)。

## NPS 結果

対計測 (`bench_paired.sh`): 2 バイナリを交互に実行し、周波数ドリフト等をペア内で相殺。
bench はやねうら王デフォルト 4 局面 × movetime 10 秒, hash 1024MB。生ログは
`bench_paired_*.txt` (エンジン sha256・governor・loadavg を記録)。
注: ログ中のエンジンのパス名は計測時点のもので、アーカイブの最終名と異なる場合がある
(finny-only は計測時 `YaneuraOu-by-gcc`、アーカイブでは `-finny-only` に改名)。
ログ先頭の sha256 prefix (`8e07…` = finny-only / `4881…` = 最終版) が正である。

### 1 スレッド

| 比較 | ペア数 | delta | ログ |
|---|---|---|---|
| baseline → finny-only | 5 | **+28.3〜29.8%** | `bench_paired_1t.txt` |
| finny-only → +コピー修正 | 5 | +7.4〜10.0% | `bench_paired_1t_AvsB.txt` |
| baseline → 最終版 | 5 | **+38.7〜41.4%** | `bench_paired_1t_B.txt` |

- 分解の整合: 1.29 × 1.09 ≈ 1.41 ✓
- 絶対値の目安: baseline ~455k → 最終版 ~630k NPS (1T)
- 理論上限 (refresh 全消し +43%) との比較: 初回タッチの全計算・残差分・エントリ⇔
  accumulator 間の memcpy が残るため上限までは届かないはずだが、コピー修正と合わせて
  ほぼ上限に到達した

### 16 スレッド (暫定)

| 比較 | ペア数 | delta | ログ |
|---|---|---|---|
| baseline → 最終版 | 3 | **+36.3〜37.0%** | `bench_paired_16t_B_provisional.txt` |

計測時は GPU 学習ジョブが約 13 コアを消費しており (loadavg 47〜67)、絶対 NPS は
大きく低下している (baseline 4.2M vs アイドル時 7.18M)。相対値はペア内で一貫するが
**暫定値**とする。

## 妥当性への脅威

1. **同時実行負荷**: 1T 計測もペア 3 以降は学習ジョブと同時実行。finny-only の
   +29% は負荷前のペア (および負荷前の単発 3 秒 bench +28.6%) と一致しており堅い。
   一方**コピー修正の +7〜10% は帯域競合下の計測のみ**で、アイドル環境では
   experiment-002 の費目 (~2pt) からは +3% 前後にとどまる可能性がある。
   合計値のアイドル環境予想は +33〜41% の幅を見ておく
2. bench 4 局面は refresh の多い中盤寄り (000-baseline 記載)。実対局の局面分布では
   効果がこれより小さい可能性がある。対局条件での効果はレーティングに直結する
   NPS 増として現れるので、必要ならフェーズ3 フル手順 (自己対局) で確認できるが、
   等価変換なのでプロトコル上は不要
3. キャッシュは thread_local 約 354 KiB/スレッド。L2 512KB の EPYC 7302 では
   ワーキングセットへの圧迫は観測範囲で問題なし (NPS が上がっているのが証拠)。
   c8a (Zen 5, 192 コア) では L2/L3 構成が異なるため再計測時に確認する

## 今後

- **c8a での再計測** (AVX-512 ビルド + アイドル環境): 効果の確定はそちらで。
  帯域競合が増える環境ほど refresh 削減の価値は上がる方向 (experiment-002 の結論 3)
- experiment-003 の channel permutation (weights-only) と本パッチ (エンジン側) は独立に
  合成可能。permuted net との組み合わせ検証は c8a セッションで
- FT 幅縮小 (PLAN フェーズ2 の 2) の Amdahl 上限は refresh がほぼ消えた分だけ低下する。
  優先度の再評価が必要 (差分更新 8.9% + fc_0 6.8% が次の費目)
