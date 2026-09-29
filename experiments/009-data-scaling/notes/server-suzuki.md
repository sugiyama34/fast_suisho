# サーバー suzuki の環境と計測値

## スペック

| 項目 | 値 |
| --- | --- |
| CPU | AMD EPYC 7543 × 2 ソケット = **64 物理コア, SMT 無効**, AVX2 (AVX-512 なし)。NUMA 16 ノード、4 コア (1 CCD) ごとに L3 32 MB を共有 |
| RAM | 1 TB |
| GPU | A100 × 5 (0, 1, 3: 80 GB / 2, 4: 40 GB)。**全部使ってよい** (ユーザー, 2026-09-29) |
| ドライバ | 580.159.03 (CUDA 13.0 まで) |
| ディスク | `/` NVMe 1.8 TB (空き約 510 GB), `/mnt/nvme1` NVMe 7 TB (空き約 2.7 TB, 他ユーザーと共用) |
| sudo | なし |

## 環境構築 (2026-09-29)

- 教師・checkpoint は `/mnt/nvme1/sugiyama/` に置き、`data/teacher/sojo` と
  `data/bulletou/checkpoints` をシンボリックリンクにした
- 奏乗 30 ファイル: 約 3 時間で取得 (6 並列, 約 60 MB/s)。**30 ファイルとも sha256 一致**
- CUDA toolkit 12.8.1 を runfile でユーザー領域 `/mnt/nvme1/sugiyama/cuda-12.8` に入れた
  (`/usr/local/cuda` は無い)。`CUDA_HOME` で指定する
- Rust 1.97.1: `data/toolchains/` (SETUP.md §6 と同じ)。uv: `~/.local/bin`
- 水匠 11: `~/suisho11/nn.bin` (sha256 一致) と `sfnnwop-1536.h`

## やねうら王 (2026-09-29)

- V9.60 (`9133c527`) + experiment-008 の `finny.patch` (ブランチ `origin/experiment/008-finny-tables`
  から取得。このブランチには無い。patch sha256 `a9fb9eba…0922625`)、g++ 13.3.0
- **対局に使うビルド: AVX2** `~/engines/009-finny-avx2/YaneuraOu-by-gcc`
  sha256 `423b6b1cd3477321d0a049632d7966301981d158ef381e0fa14f04012995b94e`
- ZEN3 ビルド (`~/engines/009-finny-zen3/`, sha256 `ccbfd9ca…`) は 1 スレッドで約 1.5% 速いだけ
  (ピン留め・交互計測)。16 局で AVX2 版と同じ指し手。AVX2 を使う
- 水匠 11 を読み込んで `readyok`。NPS (`bench_nps.sh`, 5 秒 × 4 局面 × 3 回の中央値):
  1 スレッド 825k, 8 スレッド 7.28M。1 CCD の 4 コアすべてで同時に動かしても 1 本あたり約 4% 減
- 互角局面集 ply24 (30,053 局面, sha256 一致)、対局用 venv `data/matchenv` (cshogi 0.7.8, numpy 1.26.4)

## 対局 (`match_nodes.py`, 2026-09-29)

- smoke: 水匠 11 同士、10k / 30k ノードでエラーなし
- **固定ノード + Threads 1 の対局は完全に決定的**: 並列数 W やワーカーの割り当て、ビルド (AVX2 / ZEN3)
  を変えても 16 局すべて同一。ハーネスが毎局 `isready` を送り、置換表と探索履歴が消えるため。
  同じエンジン同士では先後を入れ替えても同じ棋譜になる (同一ネット同士の対局は 1 局面 1 棋譜)
- 変更: `match_runner.play_game` に指し手の出力 (`moves_out`) を追加。`match_nodes.py` は各局に
  `moves` と `sec` を記録し、再開時に開始局面が一致しなければ停止する (`--pairs` を変えて再開すると
  局面の等間隔抽出がずれるため)
- 速度 (Threads 1, W=16, GPU 計測と同時):

| ノード数/手 | 局数 | 平均 秒/局 | 秒/手 | 平均手数 | W=48 での推定 ペア/時 |
| --- | --- | --- | --- | --- | --- |
| 100k | 64 | 15.4 | 0.120 | 128 | 約 5,600 |
| 300k | 64 | 42.1 | 0.347 | 121 | 約 2,050 |
| 1M | 32 | 171.6 | 1.200 | 143 | 約 500 |
| 2M (推定) | – | 約 320 | – | – | 約 270 |

- 1 ペア = 1 コア程度。学習中は BulletOu のローダ (1 run 8 スレッド) と合わせて 64 コアを超えないよう
  W ≈ 32〜40 にする
- ワーカーあたりの局数が少ないと最も遅い局で全体が決まる (1M: 理想 343 秒に対し 733 秒)

## BulletOu (2026-09-29)

- `2a8e5ed` + `bulletou-multiarch.patch`, CUDA 12.8 / Rust 1.97.1 でビルド。sm_80 / sm_90 / sm_120 の SASS と
  compute_80 PTX を埋め込み (`cuobjdump` で確認)。sha256 `f9f1f593fbad1d0fa6bdec0c538b84b84f5eff3b34f4f99508fd0fb44c7720da`
- smoke (`SMOKE=1`, supervisor `--smoke`) とも正常終了。**同一設定の smoke 2 回で test accuracy が
  0.639623 / 0.639860 と一致しない** (学習はビット単位では再現しない)
- `--threads 8` を指定してもログ上は `loader_threads=24` (CPU 使用は 1 run 約 2.9 コアで問題なし)

### スループット (p10, `BENCH=1`, 16 sb, 進捗行の時刻差, 最初の 1 本を除く)

| 構成 | 秒/sb (中央値) | 1 run の局面/秒 | 合計 局面/秒 |
| --- | --- | --- | --- |
| 1 run (GPU 0) | 35 | 1.14M | 1.14M |
| 5 run 同時 (GPU 0〜4) | 80 GB: 34.5 / 40 GB: 35.5 | 1.16M / 1.13M | 約 5.7M |
| 2 run を GPU 0 に相乗り | 80 | 0.50M | 1.00M (1 run より 12% 悪い) |

- **GPU 律速**。1 run のメモリは 5.2 GB で 40 GB カードにも余裕で載る (電力上限 250 W のため約 2% 遅い)。
  CPU (特徴量集計 16 スレッド・やねうら王と同時) やディスクの影響は見えない
- **1 GPU に 1 run、5 並列**が最良。20 epoch (864 億局面 = 2,160 sb) で約 21 時間
- 進捗行の `pos/s` (約 6.1M) は実時間の速度ではない (kajiki と同じ)
