# 002-profile: perf によるコスト比率の実測 (FT 更新 / refresh / 後段 FC / 探索部)

日付: 2026-07-13

## 結果サマリ (TL;DR)

- **評価関数 (`ComputeScore`) が全サイクルの 51.0%**。うち **FT refresh が単独で 30.1%**
  と最大の費目で、FT 差分更新 (8.9%) の **3.4 倍**
- fc_0 (sparse affine 1024→8) は 6.8%。チャネル並べ替えの NPS 上限は **+2% 程度**
  (サーベイ予測 +1〜5% の下寄りと確定)
- dTLB ミスは 2.4M/s で huge pages の期待値は数% 級。IPC 2.24 で全体としては
  帯域完全律速ではないが、キャッシュミス (13.7%/refs) は refresh の 258 MiB
  テーブルストリーミングと整合
- **戦略への含意: refresh 削減が最大のテコ** (全消し理論上限 +43% NPS)。
  Stockfish の AccumulatorCaches (finny tables) 相当のエンジン改造を
  フェーズ2 候補に昇格させるべき (精度損ゼロ・実測 30% を直接叩く)

## 仮説と目的

フェーズ2 の全手法の効果予測は「評価関数のどこにサイクルが使われているか」の実測に
依存する (PLAN フェーズ2 の 0、サーベイ §2.3/§2.7)。本実験はその土台となる
プロファイルを取得する。

- **問い**: bench 探索中の CPU サイクルは FT 差分更新 / FT refresh / 後段 FC
  (fc_0〜fc_2) / 探索部 (指し手生成・TT・その他) にどう配分されているか。
  また FT の 258 MiB テーブルへのアクセスは dTLB / キャッシュミス律速か
  (huge pages・int8 化の判断材料)
- **予測**: 評価関数 (FT + FC) が全体の 40〜60%、うち FT (更新+refresh) が
  評価コストの大半を占める (サーベイの一般論。比率は未実測のため本実験で確定する)
- **判定指標**: perf の cycles:u サンプル比率 (関数/インライン行単位)。
  これ自体は比較実験ではないため baseline との差分はない。
  成果物は「手法候補ごとの理論上限 (Amdahl) を計算できる比率表」

## セットアップ

- エンジン: 000-baseline と同一ソース (やねうら王 V9.60 commit `9133c527`)・同一フラグ
  + `EXTRA_CPPFLAGS='-g'`、リンク時の strip (`-Wl,-s`) のみ無効化した profiling ビルド。
  `-g` はコード生成に影響しないため性能は baseline と同一のはず → **パリティ検証済み**
  (下記)。アーカイブ: `~/engines/profile-002/YaneuraOu-by-gcc`

  ```
  sha256: 215f0e758840620c24516d56f3042e9a987bc890923f7575639886cce0bccd4c
  ```

- ビルド手順 (YaneuraOu checkout にて。Makefile の `LDFLAGS += -Wl,-s` 行を
  一時的にコメントアウトし、ビルド後に `git checkout source/Makefile` で復元):

  ```sh
  make clean YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3 \
       COMPILER=g++ TARGET_CPU=AVX2 PYTHON=python3
  make -j16 normal YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3 \
       COMPILER=g++ TARGET_CPU=AVX2 PYTHON=python3 EXTRA_CPPFLAGS='-g'
  ```

- **NPS パリティ検証** (2026-07-13, `bench_nps.sh 10000`, 1スレッド×3回):
  中央値 **469,489** (464,948–473,087)。baseline の 467,764 (±1.5% 分散) と一致 →
  このビルドのプロファイルは baseline の挙動を代表する
- 計測ワークロード: 000-baseline と同一の bench (デフォルト4局面 × movetime 10秒)、
  **1 スレッド** (アトリビューションを明瞭にするため。000 でスケーリングがほぼ線形
  = スレッド間干渉が小さいことを確認済みなので、1 スレッドの比率で代表させる)
- 計測方法: `profile.sh` 参照。readyok 後に perf をアタッチし bench 区間のみ採取
  (LEB128 ローダの混入なし)。pass 1 = `perf record` (cycles:u, dwarf call-graph),
  pass 2 = `perf stat` (IPC / cache / dTLB)

## 実行方法

```sh
# 前提 (要 sudo, 再起動でデフォルトに戻る):
#   sudo sysctl kernel.perf_event_paranoid=1
./experiments/002-profile/profile.sh          # 約 2.5 分 (40s bench × 2 pass + ロード)
```

生成物: `profile_env.txt` (計測条件), `perf_flat.txt` (関数別 self%),
`perf_graph.txt` (call-graph), `perf_stat.txt` (ハードウェアカウンタ),
`perf.data` (git 管理外の生データ — 追加分析用)。

## 留意点 (分析時)

- LTO により FT の `UpdateAccumulator` / `RefreshAccumulator` や各 FC 層の
  `Propagate` は独立シンボルとして残っていない (NNUE 前向き計算は
  `Eval::NNUE::ComputeScore` に集約)。`-g` の debug info があるため、
  perf report のインライン展開 / srcline 単位の集計で内訳を分離する
- AMD Zen 2 (EPYC 7302) は LBR 非対応・バイナリはフレームポインタ省略ビルドのため、
  call-graph は dwarf アンワインドを使う
- dTLB 系イベントは AMD ではマッピングされない場合がある (`<not supported>` は既知)

## 結果

計測: 2026-07-13 21:08 (paranoid=1 に一時緩和、アイドル状態、governor schedutil)。
16,172 サンプル / bench 4局面 × 10 秒 × 1 スレッド。0.3% ≈ 48 サンプルなので
表の数値は ±0.5 ポイント程度の量子化誤差を含む。

### サイクル配分 (cycles:u, 全体比)

評価関数 `Eval::NNUE::ComputeScore` (self 51.0%) の内訳 (インラインフレーム集計。
intrinsic フレームは親関数に合算):

| 費目 | 全体比 | 備考 |
|---|---|---|
| **FT refresh_accumulator** | **30.1%** | 全費目中最大。差分更新の 3.4 倍 |
| FT update_accumulator (差分更新) | 8.9% | |
| fc_0 `AffineTransformSparseInputExplicit<1024,8>` | 6.8% | うち find_nnz+POPCNT 1.3% |
| FT Transform (CReLU/pairwise-mul/pack) | 1.0% | |
| fc_1 `AffineTransformExplicit<14,64>` | 0.6% | fc_2 は 0.3% 未満で個別検出されず |
| ComputeScore 残余 (0.3% 未満の細片) | 3.6% | |

探索部ほか (49.0%) の主な費目:

| 費目 | 全体比 |
|---|---|
| MovePicker::next_move (うち partial_insertion_sort 3.2%) | 8.6% |
| do_move | 8.3% |
| search\<PV\> | 4.8% |
| memmove (libc — StateInfo/accumulator コピー) | 3.6% |
| correction_value | 2.6% |
| mate_1ply (先後計) | 1.9% |
| TT probe | 1.6% |
| see_ge | 1.3% |
| その他 (bitboard・指し手生成・undo 等) | ~16% |

### ハードウェアカウンタ (perf stat, 42.1 秒)

| 指標 | 値 | 解釈 |
|---|---|---|
| IPC | 2.24 | 全体としては演算も回っており、完全な帯域律速ではない |
| branch-miss | 1.08% | 良好 (探索コードとして普通) |
| L1d ミス率 | 16.1% | 高め。FT テーブルストリーミングと整合 |
| cache-misses / cache-references | 13.7% (5.42G / 39.5G) | LLC 相当のミスが毎秒 129M 回 |
| dTLB-load-misses | 100.8M (2.4M/s) | ページウォーク ~30–60 cyc として全体の 2–5% → huge pages の期待値は数% 級 (サーベイ予測どおり) |

### fc_0 sparse 実装のスキップ粒度 (channel permutation の設計情報)

`affine_transform_sparse_input_explicit.h` を確認:

- fc_0 は uint8 入力 1024 本を **int32 単位 = 連続 4 チャネルのチャンク**として扱い、
  `find_nnz_explicit<256>` がチャンク内 4 チャネルのいずれかが非ゼロなら
  そのチャンクを処理対象に採る (AVX2: `_mm256_cmpgt_epi32` + movemask)
- 重みもチャンク単位 (`i * 8 出力 * 4`) でアクセスされる
- → **並べ替えの目標は「同時にゼロになりやすいチャネルを同じ 4ch チャンクに固める」**こと。
  なお fc_0 入力は FT の pairwise-mul 出力 (512×2 視点) なので、並べ替えの自由度は
  FT チャネルのペア (j, j+512) 単位。実装詳細は次実験 (003) で確定する

## 結論

1. **最大のテコは FT refresh (30.1%)**。予測では「FT = 更新系が大半」としていたが、
   実測は refresh ≫ 差分更新 (3.4:1)。HalfKA 系は玉移動のたびに視点丸ごと再計算になる
   ため、玉が動く将棋の中盤で refresh が支配的になる。**新候補として
   Stockfish の AccumulatorCaches (finny tables) 相当** — 玉マスごとに accumulator を
   キャッシュし refresh を「キャッシュとの差分」に置き換える等価変換 (精度損ゼロ) —
   をフェーズ2 候補に昇格させるべき。理論上限 (refresh 全消し) は **+43% NPS**、
   finny tables の実効でも大半を回収できる見込み
2. **FT 幅縮小 (1024→512) の Amdahl 上限は +25%** (FT 系合計 40.0% の半減として)。
   PLAN の期待値 (+25%) と一致。蒸留前提の優先度は変わらず
3. **チャネル並べ替え (次実験) の上限は小さい**: fc_0 全体で 6.8%、うち重み積和部
   ~5.5% の 3 割を削れても **+2% NPS 程度**。ただし精度損ゼロ・weights-only で
   最も安価なので、パイプライン (並べ替え → serialize → NPS/評価値一致検証) の
   練習台として先行実施する価値はある
4. **huge pages / PGO は数% 級**の見込み (dTLB 実測から)。採用時はベースラインにも
   適用して再計測 (PLAN の注意どおり)
5. 探索部では MovePicker (8.6%, うち挿入ソート 3.2%) と do_move (8.3%) が大きいが、
   本プロジェクトのスコープ (評価関数の高速化) 外。memmove 3.6% には accumulator
   コピーが含まれ、FT 幅縮小で比例して減る副次効果がある

## 妥当性への脅威

- bench デフォルト 4 局面は「refresh の多い複雑な中盤」を含む (000-baseline 記載)。
  実対局の局面分布では refresh 比率がこれより低い可能性があり、refresh 30% は
  この workload における値。finny tables 等 refresh 系施策の対局効果は
  フェーズ3 プロトコルで別途検証する
- 1 スレッド計測。16 スレッドではメモリ帯域・LLC 競合で FT 系 (メモリ重い費目) の
  比率がさらに上がる方向のはずで、FT 系施策の効果を過小評価しこそすれ過大評価はしない
- インライン集計の細片 (0.3% 未満) 3.6% 分は費目に配分していない
