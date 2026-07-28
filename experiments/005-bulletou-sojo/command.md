# 起動手順

## 環境

- GPU 学習は **Claude Code セッション内から起動できない** (サンドボックス/名前空間に
  `/dev/nvidia*` が無い)。学習本体は**実端末**で起動する
- wandb 同期・監視・分析はセッション内 (サンドボックス) から行う
- BulletOu バイナリ: `data/bulletou/BulletOu/target/release/examples/bulletou`
  (ビルド済み。再ビルドは下記)

## 学習の起動 (実端末で)

```bash
# 1) スモークテスト (数分, GPU0)
bash experiments/005-bulletou-sojo/run_training.sh smoke

# 2) 本番 (約 2 日, GPU0)。tmux 推奨
tmux new -s bulletou-main 'bash experiments/005-bulletou-sojo/run_training.sh main 0'

# (任意) 分散推定アーム: 教師ファイル順を 8 回転した main (GPU1)
tmux new -s bulletou-varb 'bash experiments/005-bulletou-sojo/run_training.sh var-b 1'
```

実効値: sb = 39,976,960 局面 (40M を batch 65536 の倍数へ切り捨て)、
1 epoch = 12 sb ≒ 4.80 億局面、16 epoch ≒ 76.8 億局面 (early stop あり)。

## wandb 同期 (セッション内で Claude が起動)

```bash
uv run python experiments/005-bulletou-sojo/wandb_sync.py --arm main
```

## 再ビルド手順 (参考)

```bash
export RUSTUP_HOME=$PWD/data/toolchains/rustup CARGO_HOME=$PWD/data/toolchains/cargo
export PATH=$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH CUDA_PATH=/usr/local/cuda
cd data/bulletou/BulletOu
cargo build --release -p bulletou_lib --features cuda-cpp-backend --example bulletou
```

## 出力

- checkpoint: `data/bulletou/checkpoints/005-<arm>/` (`summary-learn.log` + `0NNN/nn.bin`)
- 学習 stdout ログ: `experiments/005-bulletou-sojo/logs/<arm>.log`
- メトリクス JSONL: `experiments/005-bulletou-sojo/metrics-*.jsonl` (wandb 二重記録)
- 教師データ: `data/teacher/sojo/` (`download.sh` で取得。train 001–016 + test 030)
