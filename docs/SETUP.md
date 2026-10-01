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

- **git hook**: `scripts/install-hooks.sh` は `core.hooksPath=.githooks` を設定し、
  commit 時に `.githooks/pre-commit` (秘密情報スキャナ) が動くようにする。新サーバーで
  これを忘れてスキャナが無効のままになったことがあるため、Claude Code の SessionStart hook
  (`.claude/hooks/check-git-hooks.sh`) が未設定を検出するとセッション開始時に警告する
  (ブロックはしない)。Claude は `git config` を実行できない (`block-dangerous-git.sh` が
  ブロックする) ので、警告が出たら**ユーザーが**上記スクリプトを実行する
- **Python の lint**: CI (`.github/workflows/lint.yml`) は `uv run ruff check` と
  `uv run ruff format --check` を走らせる。Claude Code の PostToolUse hook が 2 つある:
  - `python-lint.sh` (Edit / Write): 編集した `.py` を ruff で自動修正する
  - `python-lint-bash.sh` (Bash): Bash 経由 (heredoc, `sed -i`, 書き換えスクリプト) で
    変更された `.py` を**検査のみ**する (ファイルは変更しない。複数ステップのスクリプト編集の
    途中で整形すると壊れるため)。`git status` に出る変更済み・未追跡の `.py` のうち、
    前回の検査以降に mtime が更新されたものだけを対象にするので、同じファイルを繰り返し
    報告しない。問題があれば ruff の出力と修正コマンドを Claude に返す
  - 同じ Bash 呼び出しの中で commit まで済ませた `.py` は `git status` に出ないため検出
    できない。最終的な防衛線は CI
- **コミット署名**: 旧開発機では SSH 署名 (`gpg.format=ssh`, `commit.gpgsign=true`) を
  使っていた。新サーバーでも鍵を用意して設定する
- **WandB**: API キーは `wandb login` ではなく環境変数で渡す運用 (`~/.netrc` への平文保存を
  避けるため)。詳細は `docs/wandb-guide.md`

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
  必ず両方を同じビルド条件にする** (`docs/PLAN.md` フェーズ 2 の方針)
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
  共有するので、`~/suisho11/` に置いてから sha256 を確認する:

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

- experiment-006 以降の学習スクリプトは 30 ファイル全部が揃っていないと起動しない
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
  GPU の世代が違う場合は `-gencode` の値をその GPU に合わせる
- Rust ツールチェインは旧開発機ではリポジトリ内 (`data/toolchains/`) に閉じ込めていた:

```sh
export RUSTUP_HOME=$PWD/data/toolchains/rustup CARGO_HOME=$PWD/data/toolchains/cargo
export PATH=$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda
cd data/bulletou/BulletOu
cargo build --release -p bulletou_lib --features cuda-cpp-backend --example bulletou
```

- 生成物: `data/bulletou/BulletOu/target/release/examples/bulletou`
- 学習の起動には `LD_LIBRARY_PATH=/usr/local/cuda/lib64` が必要
  (各実験の `run_training.sh` が設定する)
- checkpoint は `data/bulletou/checkpoints/<実験>-<arm>/NNNN/nn.bin` に出る。
  旧開発機では全 checkpoint で約 500 GB あった。ディスクは教師データと合わせて
  **最低 1.2 TB 程度**を見込む

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

## 9. 新サーバーのスペック

(移行後に記入: CPU / コア数 / AVX-512 の有無 / RAM / GPU / ディスク容量)。
`docs/PLAN.md` の「環境メモ」は旧開発機のスペックなので、新サーバーの値はここか
PLAN の環境メモに追記する。
