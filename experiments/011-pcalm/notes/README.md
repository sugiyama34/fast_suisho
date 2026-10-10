# experiment-011 作業ノート

結果は [../report.md](../report.md)、設計は [../hypothesis.md](../hypothesis.md)。

- [decisions.md](decisions.md): 実行中の決定の記録
- [m0.md](m0.md): 参照実装・勾配の検証 (CUDA と Python、KKT 点)・勾配の向きと推論の刻み η_h・gn・int8 の書き出し
- `ratings.md`: 対局結果 (match_queue.py が自動で追記)
- `replay-check-summary.json`: kajiki のエンジンで 009 の棋譜を指し直した結果 (100 局一致)

スクリプト (`experiments/011-pcalm/`):

| もの | 役割 |
| --- | --- |
| `arms.py` / `run_training.sh` / `gpu_queue.sh` / `train_supervised.py` | arm 名 → BulletOu の学習 (W&B `pcalm_vs_backprop`) |
| `export_rescaled.py` / `rescale_watch.py` / `quant_check.py` | PC 系の出力側 1/2 の書き出しと、量子化の確認 |
| `match_queue.py` / `select_queue.py` | 対局のキューと、本番規模の選択対局・2,000 ペア・直接対局の自動追加 |
| `replay_check.py` / `analyze.py` | エンジンの同一性の確認 / 最終の集計と判定 |
| `start-full-pc.sh` / `start-full-bp-rep.sh` | MIG の空きに PC の本番規模・BP の複製を流す起動スクリプト |
