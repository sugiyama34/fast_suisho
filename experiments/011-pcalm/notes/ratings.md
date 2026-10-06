# 対局結果 (レーティング)

`match_queue.py` が対局 1 本ごとに 1 行ずつ追記する (手で書き換えない。備考の追記は可)。

- 条件: 固定ノード (`go nodes N`)、各エンジン Threads 1、hash 256 MB、やねうら王 V9.60 + finny.patch。
  エンジンの sha256 は行ごとに備考に記録する (既定は experiment-009 と同じ `423b6b1c…`)。開始局面は互角局面集 ply24 から
  ペア数に応じた等間隔ストライド (ペア数が同じなら全対局で同じ局面集合)。1 ペア = 先後入替 2 局
- **Elo は A から見た値** (A = `match_nodes.py` の candidate, B = baseline)。95% CI は pentanomial
- pentanomial は A のペア得点 0 / 0.5 / 1 / 1.5 / 2 の件数。勝/分/負は A から見た局数
- **FV_SCALE**: ジョブで `a_fv` / `b_fv` を指定したネットは表記に `@FV<値>` を付ける。無指定は既定の 16
- A / B の表記: `s-bp-lr1-e1` = `011-s-bp-lr1/0001` の checkpoint、`s11` = 水匠 11。sha256 は nn.bin の先頭 12 桁

## 対局キュー (`match_queue.py`)

- ジョブは `match_queue.toml` に `[[job]]` を追記するだけでよい (キューが 1 分ごとに読み直す)。ネットは checkpoint
  (`"011-s-bp-lr1/0001"`) か `"suisho11"`。checkpoint がまだ無いジョブは出来るまで待つ
- 実行できるジョブのうち priority 最小のものから 1 本ずつ、W ワーカー (`nice -n 10`, kajiki は学習中 8 / それ以外 12) で実行する
- 状態: `games/<name>/queue_done.json` (完了)、`queue_failed.json` (失敗。消せば再試行)。
  ログ `/mnt/D/sugiyama/011/match_queue.log`
- 停止: `setsid` でプロセスグループの先頭になっているので、`kill -- -<PID>` でグループごと
  (match_nodes.py とエンジンも) 止める。再起動は下のコマンド (途中の対局は完結ペア単位で再開)
  ```
  nohup setsid nice -n 10 data/matchenv/bin/python experiments/011-pcalm/match_queue.py \
      >> /mnt/D/sugiyama/011/match_queue.log 2>&1 < /dev/null &
  ```

## 結果

| 日付 | 対局名 | A | B | ノード | ペア | Elo ± 95% CI | pentanomial | 勝/分/負 | 備考 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10-06 21:08 | tune-s-bp-lr1-e1@48-vs-bp-rep-e1@48-300k | s-bp-lr1-e1@FV48 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -11.0 ± 26.6 [-37.7, +15.5] | 70/11/149/8/62 | 281/19/300 | A 9ee1877a7161; B ffe7a326a889; engine a5f519d2; M2: bp-recipe (BP のレシピ値)。複製との差 = ノイズの目安 |
