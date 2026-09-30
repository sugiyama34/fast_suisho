# 対局結果 (レーティング)

`match_queue.py` が対局 1 本ごとに 1 行ずつ追記する (手で書き換えない。備考の追記は可)。

- 条件: 固定ノード (`go nodes N`)、各エンジン Threads 1、hash 256 MB、やねうら王
  `~/engines/009-finny-avx2/YaneuraOu-by-gcc` (sha256 `423b6b1c…`)。開始局面は互角局面集 ply24 から
  ペア数に応じた等間隔ストライド (ペア数が同じなら全対局で同じ局面集合)。1 ペア = 先後入替 2 局
- **Elo は A から見た値** (A = `match_nodes.py` の candidate, B = baseline)。95% CI は pentanomial
- pentanomial は A のペア得点 0 / 0.5 / 1 / 1.5 / 2 の件数。勝/分/負は A から見た局数
- **FV_SCALE**: ジョブで `a_fv` / `b_fv` を指定したネットは表記に `@FV<値>` を付ける。無指定は既定の 16。
  09-30 以降は評価関数ごとに対局で調整した値を使う (`notes/decisions.md`)。
- A / B の表記: `full-e8` = `009-full/0008` の checkpoint、`s11` = 水匠 11、それ以外は
  アブレーションしたネット (`/mnt/nvme1/sugiyama/ablated/<名前>/nn.bin`、作り方は
  `nn.bin.json` と `rare.npy.json`)。sha256 は nn.bin の先頭 12 桁

## 対局キュー (`match_queue.py`)

- ジョブは `match_queue.toml` に `[[job]]` を追記するだけでよい (キューが 1 分ごとに読み直す)。
  書き方はファイル先頭のコメント。ネットは checkpoint (`"009-p50/0012"`)、`"suisho11"`、
  アブレーション指定 (checkpoint + `select_features.py` の引数 + モード。`/mnt/nvme1/sugiyama/ablated/<名前>/`
  に必要時に作る) のいずれか。checkpoint がまだ無いジョブは出来るまで待つ
- 実行できるジョブのうち priority 最小のものから 1 本ずつ、W = 40 ワーカー (`nice -n 10`) で実行する
- 状態: `games/<name>/queue_done.json` (完了)、`queue_failed.json` (失敗。消せば再試行)。
  ログ `/mnt/nvme1/sugiyama/logs/match_queue.log`
- 停止: `setsid` でプロセスグループの先頭になっているので、`kill -- -<PID>` でグループごと
  (match_nodes.py とエンジンも) 止める。再起動は下のコマンド (途中の対局は完結ペア単位で再開)
  ```
  nohup setsid nice -n 10 data/matchenv/bin/python experiments/009-data-scaling/match_queue.py \
      >> /mnt/nvme1/sugiyama/logs/match_queue.log 2>&1 < /dev/null &
  ```
- 対局側 (探索ノード) のカウントを使うアブレーションは、`search.npy` が集計途中でも存在するので、
  集計完了後に確定版を別名で置き、そのパスを `select` の `--counts` と `requires` に書く

## 結果

| 日付 | 対局名 | A | B | ノード | ペア | Elo ± 95% CI | pentanomial | 勝/分/負 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 09-30 13:19 | fv-s11@40-vs-s11@16-300k | s11@FV40 | s11 | 300,000 | 400 | +67.3 ± 23.6 [+44.0, +91.2] | 57/10/183/23/127 | 457/39/304 | A a78b7f889843; B a78b7f889843; FV_SCALE 調整 (水匠 11 自己対局) |
| 09-30 13:45 | fv-s11@24-vs-s11@16-300k | s11@FV24 | s11 | 300,000 | 400 | +47.2 ± 24.0 [+23.4, +71.5] | 71/9/183/15/122 | 442/24/334 | A a78b7f889843; B a78b7f889843; FV_SCALE 調整 (水匠 11 自己対局) |
