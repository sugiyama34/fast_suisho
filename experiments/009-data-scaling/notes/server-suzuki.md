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

## BulletOu

(スループット計測中。終わったら追記する)
