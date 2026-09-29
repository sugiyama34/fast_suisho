# 新サーバーのセットアップ手順

作成: 2026-09-28 (旧開発機からの移行に合わせて作成)

新しいマシンで fast_suisho の実験 (高速化トラック・学習トラックの両方) を再開するための
チェックリスト。旧開発機ではリポジトリの外 (兄弟ディレクトリ・`data/`・`~/engines/`) に
置いていたものが多く、それらは `git clone` では移らない。本書はそれを全部列挙する。

## 0. ディレクトリ配置

旧開発機では以下の配置だった。スクリプトや既定値がこの相対配置・絶対パスを前提にしている:

```
~/fast_suisho/            # 本リポジトリ (高速化トラック用の clone)
~/2_fast_suisho/          # 本リポジトリの 2 つ目の clone (学習トラック用。data/ はこちらにあった)
~/YaneuraOu/              # やねうら王 (兄弟ディレクトリ, git 管理外)
~/suisho11/               # Suisho 11 の nn.bin (兄弟ディレクトリ)
~/engines/                # ビルド済みエンジンの保管場所 (実験ごとのサブディレクトリ)
```

- 2 つの clone は同じ GitHub リポジトリ (`sugiyama34/fast_suisho`)。トラックごとに
  作業ツリーを分けていただけなので、新サーバーでは 1 つにまとめてもよい
- **`/home` が狭い場合**: 教師 (587 GB) と checkpoint (1 個 2.3 GB × epoch 数 × arm 数) は
  大容量ディスクに置き、`data/teacher/sojo` と `data/bulletou/checkpoints` をシンボリック
  リンクにする。BulletOu もスクリプトもリンクを辿る (2026-09-28 kajiki で確認):

```sh
mkdir -p /mnt/<disk>/<user>/teacher/sojo/train /mnt/<disk>/<user>/checkpoints
mkdir -p data/teacher data/bulletou
ln -s /mnt/<disk>/<user>/teacher/sojo data/teacher/sojo
ln -s /mnt/<disk>/<user>/checkpoints data/bulletou/checkpoints
```
- **絶対パスの既定値に注意**: `experiments/005-bulletou-sojo/match_runner.py` は
  `--engine` の既定値が `/home/sugiyama/YaneuraOu/source/YaneuraOu-by-gcc`、
  `--baseline-evaldir` の既定値が `/home/sugiyama/suisho11`。ホームディレクトリが
  変わる場合は引数で明示する

## 1. 基本ツール

| ツール | 旧開発機での版 | 用途 |
| --- | --- | --- |
| g++ | 13.3.0 | やねうら王のビルド (clang++ は未使用) |
| python3 + uv | Python 3.12 / uv 0.11 | `uv sync` でプロジェクト環境を作る |
| CUDA toolkit | 13.3 (driver 610.43.02) | BulletOu のビルドと学習 |
| Rust | 1.97.1 | BulletOu のビルド |
| gh | - | PR 操作。`gh auth login` で sugiyama34 として認証 |
| codex | 0.145.0 | コードレビュー用 (`codex exec '...'`)。任意 |
| wandb | pyproject の依存 | 実験トラッキング。`docs/wandb-guide.md` 参照 |

```sh
git clone git@github.com:sugiyama34/fast_suisho.git ~/fast_suisho
cd ~/fast_suisho
uv sync
bash scripts/install-hooks.sh   # git の pre-commit hook を有効化
```

- `install-hooks.sh` が未実行だと、Claude Code のセッション開始時に警告が出る
  (`.claude/hooks/check-git-hooks.sh`)
- Bash 経由で変更した `.py` は `.claude/hooks/python-lint-bash.sh` が ruff で検査する
  (検査のみ。修正はしない)
- **コミット署名**: 旧開発機では SSH 署名 (`gpg.format=ssh`, `commit.gpgsign=true`) を
  使っていた。新サーバーでも鍵を用意して設定する
- **`scripts/install-hooks.sh` はユーザーが実行する**。中身は `git config core.hooksPath` で、
  リポジトリの hook (`block-dangerous-git.sh`) が Claude からの `git config` を止めるため
- **CUDA toolkit の版**: `nvidia-smi` 右上の「CUDA Version」(ドライバが対応する上限) 以下の
  toolkit を使う。`nvcc` は PATH に無いことが多いので `/usr/local/cuda/bin` を足す
  (kajiki では toolkit 13.1 / driver 590.48 で問題なし)
- **WandB**: API キーは `wandb login` ではなく環境変数で渡す運用 (`~/.netrc` への平文保存を
  避けるため)。詳細は `docs/wandb-guide.md`。**キーはリポジトリにも `~/.claude/settings.json`
  にも書かない** (2026-09-28 ユーザー方針)。Claude Code を起動するシェルでだけ export する。
  先頭に半角スペースを付けると bash の履歴に残らない (Ubuntu 既定の `HISTCONTROL=ignoreboth`):

```sh
 export WANDB_API_KEY=...        # 先頭の半角スペースに注意 (または: read -rs WANDB_API_KEY && export WANDB_API_KEY)
cd ~/fast_suisho && claude --continue
```

  キーが無い環境では experiment-009 以降の supervisor は offline で記録する。後でキーのある
  端末から `uv run wandb sync wandb/offline-run-*` で送る

## 2. やねうら王 (エンジン)

```sh
git clone https://github.com/yaneurao/YaneuraOu.git ~/YaneuraOu
cd ~/YaneuraOu && git checkout 9133c527     # V9.60 に固定
cd source
make -j"$(nproc)" normal YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfka2_1024-7-64-k3k3 \
     COMPILER=g++ TARGET_CPU=AVX2 PYTHON=python3
```

- `PYTHON=python3` は必須 (Makefile がアーキテクチャ名からヘッダを自動生成する)
- AVX-512 のあるマシンでは `TARGET_CPU` の変更を検討する。**ただしベースラインとの比較では
  必ず両方を同じビルド条件にする** (`docs/PLAN.md` フェーズ 2 の方針)。
  使える値は `AVX512VNNI` / `AVX512` / `AVXVNNI` / `AVX2` / `ZEN3` など (Makefile 冒頭の一覧)。
  kajiki (Xeon Gold 6526Y) では `AVX512VNNI` 版が 3 秒の簡易計測で AVX2 版より約 8% 遅かった
  (負荷のある共用機での 1 回計測なので参考値)。ビルドし直すときは `make clean` を忘れない
  (別 TARGET_CPU の .o が残っていると混ざる)
- **対局に使っていたバイナリは finny tables 版**: experiment-007 以降の対局は
  experiment-008 の `finny.patch` を当てたビルド (sha256 `4881fcf4…`) を使っている。
  同じ条件で対局するなら、パッチを当ててからビルドする:

```sh
cd ~/YaneuraOu
git apply ~/fast_suisho/experiments/008-finny-tables/finny.patch
cd source && make clean && make -j"$(nproc)" normal ...(上と同じ引数)
```

  同じ g++ 版でもマシンが違えば sha256 は一致しない可能性が高い。等価性の確認方法は
  experiment-008 のレポート (決定的探索の searchlog 比較) を参照
- ビルドしたバイナリは `~/engines/<名前>/` に保管し、sha256 をレポートに記録する

## 3. Suisho 11 評価関数

- `~/suisho11/nn.bin` (135 MB) と `sfnnwop-1536.h`
- sha256: `a78b7f889843037d344f482623b3febd124ead5c1f34f134d9f1c2c78cd0f829`
- **入手元**: ユーザーの Google Drive フォルダに保管されている。新マシンにはユーザーが
  共有リンクを渡すので、`~/suisho11/` に置いてから sha256 を確認する。ダウンロードは gdown で
  できる (gdown 6.x は `--fuzzy` を廃止。共有 URL をそのまま渡せばよい):

```sh
mkdir -p ~/suisho11 && cd ~/suisho11
uvx gdown '<nn.bin の共有 URL>'
uvx gdown '<sfnnwop-1536.h の共有 URL>'
```

  sha256 の確認:

```sh
echo "a78b7f889843037d344f482623b3febd124ead5c1f34f134d9f1c2c78cd0f829  nn.bin" | (cd ~/suisho11 && sha256sum -c -)
```

## 4. 対局用データ

```sh
bash match/fetch_books.sh    # 互角局面集 → data/books/start_sfens_ply24.txt (sha256 検証あり)
```

対局ハーネスは cshogi を使い、cshogi が numpy<2 を要求するため**プロジェクトの venv とは
別の venv** を使う (プロジェクトは numpy>=2):

```sh
uv venv --python 3.12 data/matchenv
uv pip install --python data/matchenv/bin/python cshogi==0.7.8 numpy==1.26.4
```

## 5. 学習トラック: 教師データ

```sh
bash scripts/fetch_teacher.sh            # 奏乗 30 ファイル + floodgate.hcpe (再開可)
bash scripts/fetch_teacher.sh --verify   # 取得後に sha256 を全ファイル検証
```

| データ | 置き場所 | サイズ |
| --- | --- | --- |
| 奏乗教師 (PackedSfenValue) | `data/teacher/sojo/train/dlsuisho_unique_NNN.psv` | 約 587 GB (30 ファイル) |
| floodgate.hcpe (検証セット) | `data/teacher/floodgate/floodgate.hcpe` | 約 33 MB |

- experiment-006 / 007 の学習スクリプトは 30 ファイル全部が揃っていないと起動しない。
  experiment-009 の `run_training.sh` は arm が使うファイルだけを (サイズまで) 確認する
- `SOJO_ORDER` で取得順を指定できる。experiment-009 は小さいサブセットから揃う順
  (`experiments/009-data-scaling/hypothesis.md` §4) で取ると、全部揃う前に学習を始められる
- ダウンロードは HuggingFace から 4 並列。回線次第で半日〜1 日かかる。
  旧開発機から直接コピーできるならその方が速い

## 6. 学習トラック: BulletOu (トレーナ)

```sh
git clone https://github.com/yaneurao/BulletOu.git data/bulletou/BulletOu
cd data/bulletou/BulletOu
git checkout 2a8e5ed     # experiment-007 で使用 (shogi-support ブランチ)。005/006 は 9577f08
git apply ../../../experiments/005-bulletou-sojo/bulletou-sm120.patch
```

- `bulletou-sm120.patch` は Blackwell (sm_120) 向けの SASS を追加するだけのパッチ。
  数値経路は変わらない。上流の既定は sm_75 のみで、無しでも起動時 JIT で動くが遅い。
  **このパッチは sm_120 専用で、A100 (sm_80) 等では動かない** (明示した `-gencode` だけが
  埋め込まれ、compute_120 の PTX は古い GPU で JIT できない)
- **GPU が Blackwell 以外なら `experiments/009-data-scaling/bulletou-multiarch.patch` を使う**
  (sm120 パッチの代わりに当てる。sm_80 / sm_90 / sm_120 の SASS + compute_80 PTX)。
  kajiki で CUDA 13.1 によるビルドと `cuobjdump` での埋め込み確認まで済み (A100 実機は未確認)。
  sm_120 は nvcc 12.8 以上、sm_90 は 11.8 以上が必要。`nvcc --version` がそれより古ければ、
  パッチを当てた後に `crates/cuda_cpp/build.rs` から該当する `-gencode` 行を消す:

```sh
git apply ../../../experiments/009-data-scaling/bulletou-multiarch.patch   # sm120 パッチの代わり
```
- Rust ツールチェインは旧開発機ではリポジトリ内 (`data/toolchains/`) に閉じ込めていた:

```sh
export RUSTUP_HOME=$PWD/data/toolchains/rustup CARGO_HOME=$PWD/data/toolchains/cargo
export PATH=$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda
cd data/bulletou/BulletOu
cargo build --release -p bulletou_lib --features cuda-cpp-backend --example bulletou
```

- 生成物: `data/bulletou/BulletOu/target/release/examples/bulletou`
- 学習の起動には `LD_LIBRARY_PATH=/usr/local/cuda/lib64` が必要
  (各実験の `run_training.sh` が設定する)。toolkit が `/usr/local/cuda` 以外にある場合は、
  上のビルドでは `/usr/local/cuda` をその場所に読み替え、学習時は `CUDA_HOME` を export する
  (experiment-009 の `run_training.sh` は `${CUDA_HOME:-/usr/local/cuda}/lib64` を使う)。
  suzuki: `/mnt/nvme1/sugiyama/cuda-12.8` (CUDA 12.8, sudo なしでユーザー領域に導入)
- checkpoint は `data/bulletou/checkpoints/<実験>-<arm>/NNNN/nn.bin` に出る。
  旧開発機では全 checkpoint で約 500 GB あった。ディスクは教師データと合わせて
  **最低 1.2 TB 程度**を見込む
- BulletOu `2a8e5ed` の挙動 (2026-09-28 ソースで確認):
  - `--teacher` はファイル・ディレクトリ・それらのカンマ区切りを受け付ける (glob は不可)。
    ディレクトリ内はパス名の辞書順、明示リストはその順で読む。シンボリックリンクは辿る
  - 学習中の再シャッフルはない。教師ストリームは EOF で先頭に戻る (epoch 境界では戻らない)。
    重み初期化のシードは固定 (`--seed` は無い)
  - `--threads` を既定の 4 のままにすると内部で `コア数×2` (最大 24) に置き換わる。
    共用機では明示的に 8 などを指定する
  - 「epoch 末 validation が改善しなければ停止」は `--lr-schedule plateau` 専用。step では発動しない
  - checkpoint 1 個 = nn.bin 135 MB + state.bin 2.2 GB

## 7. 旧開発機から持ってくると便利なもの

必須ではないが、再生成に時間がかかるもの:

| もの | 旧開発機のパス | 理由 |
| --- | --- | --- |
| experiment-007 の推奨 net | `~/2_fast_suisho/data/bulletou/checkpoints/007-wrm/0020/nn.bin` (sha256 `0c325ad2…`) | 学習トラックの現時点の最良 net。再学習は GPU で約 12 時間 |
| experiment-006 の基準 net | `~/2_fast_suisho/data/bulletou/checkpoints/006-yane20/` | 比較の基準線 |
| エンジンバイナリ | `~/engines/` | 各実験の再現用。新マシンでは再ビルドして sha256 を記録し直してもよい |
| Suisho 11 | `~/suisho11/` | ユーザーの Google Drive から共有される (上記 3)。コピーは不要 |

## 8. 動作確認

1. `uv run pytest` が通る
2. やねうら王が Suisho 11 を読み込んで `isready` → `readyok` を返す
   (`experiments/000-baseline/bench_nps.sh` でベンチも取れる)
3. `nvidia-smi` で GPU が見える。BulletOu の smoke run
   (`experiments/005-bulletou-sojo/run_training.sh smoke`) が数分で完走する
4. 新サーバーの基準 NPS を取り直す。**旧開発機の NPS と直接比較しない**
   (CPU が違えば NPS の絶対値は変わる。比較は常に同一マシン上の A/B で行う)
5. **学習スループットを実時間で測る** (学習計画の前提になる)。`BENCH=1` で sb=16 × 1 epoch を
   走らせ、ログの進捗行 `[progress] ... sb k/16` の**時刻差**から 1 sb (約 4000 万局面) の秒数を出す
   (最初の 1 本は起動時間を含むので捨てる)。進捗行の `pos/s` は GPU へのカーネル投入時間しか
   数えないことがあり、kajiki では実時間の 6 倍の値が出た。1 run 単独と、実際に並べる本数で
   同時に走らせた場合の両方を測る。同時計測では `OUT_TAG` で出力とログを分ける:

```sh
# 1 run 単独 (p10 の 3 ファイル。ログは experiments/009-data-scaling/logs/009-p10-bench.log)
BENCH=1 bash experiments/009-data-scaling/run_training.sh p10 0
# 2 run 同時 (GPU 0 と 1)
BENCH=1 OUT_TAG=g0 bash experiments/009-data-scaling/run_training.sh p10 0 &
BENCH=1 OUT_TAG=g1 bash experiments/009-data-scaling/run_training.sh p10 1 &
wait
# 1 sb あたりの秒数 (行頭の時刻から)
grep -a 'progress\]' experiments/009-data-scaling/logs/009-p10-bench-g0.log \
  | awk '{split($2,t,":"); s=t[1]*3600+t[2]*60+t[3]; if (p) print s-p; p=s}'
```

## 9. サーバーのスペック

`docs/PLAN.md` の「環境メモ」は旧開発機のスペック。移行したサーバーの値はここに追記する。

### kajiki (2026-09-28 に一時使用。学習には不向きと判断)

| 項目 | 値 |
| --- | --- |
| CPU | Intel Xeon Gold 6526Y, 16 コア 16 スレッド, AVX-512 (VNNI / BF16 / FP16) あり |
| RAM | 251 GB |
| GPU | RTX PRO 6000 Blackwell Max-Q (96 GB) を MIG 1g.24gb ×4 に分割。**使ってよいのは MIG device 2 / 3 のみ** (0 / 1 は他ユーザー)。起動時は `CUDA_VISIBLE_DEVICES=<MIG の UUID>` |
| ディスク | `/` SATA SSD 1.8 TB (空き約 230 GB), `/mnt/D` HDD 20 TB (空き 15 TB, 実測 275 MB/s) |
| 学習速度 | MIG スライスあたり約 0.65M 局面/秒で GPU 律速 (1 スライスに 2 run 載せても合計は増えない)。864 億局面の 1 run に約 37 時間 |

詳細な計測は `experiments/009-data-scaling/hypothesis.md` §7。
