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
| 10-06 22:56 | tune-s-bp-lr0.5-e1@48-vs-bp-rep-e1@48-300k | s-bp-lr0.5-e1@FV48 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | +5.8 ± 27.2 [-21.4, +33.0] | 66/10/145/6/73 | 296/18/286 | A 1715f0892153; B ffe7a326a889; engine a5f519d2; M2: BP の LR 調整 |
| 10-06 23:37 | tune-s-pcalm-T2-e1@48-vs-bp-rep-e1@48-300k | s-pcalm-T2-e1@FV48 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -3600.0 ± 1010.8 [-3600.0, -1578.5] | 300/0/0/0/0 | 0/0/600 | A b5523621ae6a; B ffe7a326a889; engine a5f519d2; M2: PC-ALM 段階 A (T, α=1, ρ=1, η 自動, ε 較正); **書き出しの失敗 (int8 の範囲外): L3 の実効重みの 82% が |w| > 127/64 で切り詰められ、量子化後の一致率 53.9% (float では 75.3%)。600 局すべて詰みで負け。PC-ALM のネットの強さを表さない。出力側 1/2 の書き出しで測り直す (notes/m0.md)** |
| 10-07 00:53 | tune-s-bp-lr1.4-e1@48-vs-bp-rep-e1@48-300k | s-bp-lr1.4-e1@FV48 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -6.4 ± 27.0 [-33.4, +20.6] | 70/11/146/6/67 | 285/19/296 | A edb9ee59fea5; B ffe7a326a889; engine a5f519d2; M2: BP の LR 調整 |
| 10-07 02:25 | tune-s-pcalm-T2-gn-e1@48-vs-bp-rep-e1@48-300k | s-pcalm-T2-gn-e1@FV48 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -109.5 ± 28.8 [-138.9, -81.4] | 122/11/128/6/33 | 200/17/383 | A bbbba4907bc9; B ffe7a326a889; engine a5f519d2; M2: PC-ALM 段階 A (T, α=1, ρ=1, η 自動, gn) |
| 10-07 03:22 | tune-s-pcalm-T2-gn-e1x@24-vs-bp-rep-e1@48-300k | /mnt/D/sugiyama/011/rescaled/011-s-pcalm-T2-gn-e1@FV24 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -66.8 ± 28.7 [-96.0, -38.6] | 104/11/131/3/51 | 235/16/349 | A a17962bc1a1a; B ffe7a326a889; engine a5f519d2; M2: PC-ALM 段階 A (gn)。出力側 1/2 の書き出し (k=2), FV_SCALE 24 = 元の 48 相当 |
| 10-07 04:13 | tune-s-pcalm-T2-e1x@24-vs-bp-rep-e1@48-300k | /mnt/D/sugiyama/011/rescaled/011-s-pcalm-T2-e1@FV24 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -253.8 ± 34.8 [-290.7, -221.2] | 191/11/88/1/9 | 107/12/481 | A dfb7cc0114f0; B ffe7a326a889; engine a5f519d2; 参考: 旧方式 (固定 epsilon)。出力側 1/2 の書き出し (k=2)。書き出しのままの対局は全敗 (int8 の範囲外) |
| 10-07 05:09 | tune-s-pcalm-T4-gn-e1x@24-vs-bp-rep-e1@48-300k | /mnt/D/sugiyama/011/rescaled/011-s-pcalm-T4-gn-e1@FV24 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -59.0 ± 27.8 [-87.2, -31.6] | 99/4/144/5/48 | 245/9/346 | A ceea607d1de6; B ffe7a326a889; engine a5f519d2; M2: PC-ALM 段階 A (gn)。出力側 1/2 の書き出し (k=2), FV_SCALE 24 = 元の 48 相当 |
| 10-07 06:00 | tune-s-bp-lr0.7-e1@48-vs-bp-rep-e1@48-300k | s-bp-lr0.7-e1@FV48 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | +13.3 ± 26.9 [-13.5, +40.3] | 60/16/138/13/73 | 296/31/273 | A 0ae5b0d1cf11; B ffe7a326a889; engine a5f519d2; M2: BP の LR 調整 |
| 10-07 06:52 | tune-s-bp-lr2-e1@48-vs-bp-rep-e1@48-300k | s-bp-lr2-e1@FV48 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -49.0 ± 27.5 [-76.7, -21.8] | 93/6/142/10/49 | 249/18/333 | A 1d4b8b9a119f; B ffe7a326a889; engine a5f519d2; M2: BP の LR 調整 |
| 10-07 07:44 | tune-s-pcalm-T8-gn-e1x@24-vs-bp-rep-e1@48-300k | /mnt/D/sugiyama/011/rescaled/011-s-pcalm-T8-gn-e1@FV24 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -83.2 ± 26.2 [-109.8, -57.5] | 100/8/157/3/32 | 223/13/364 | A c02dfa29218b; B ffe7a326a889; engine a5f519d2; M2: PC-ALM 段階 A (gn)。出力側 1/2 の書き出し (k=2), FV_SCALE 24 = 元の 48 相当 |
| 10-07 08:35 | tune-s-pc-T8-gn-e1x@24-vs-bp-rep-e1@48-300k | /mnt/D/sugiyama/011/rescaled/011-s-pc-T8-gn-e1@FV24 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -61.4 ± 28.5 [-90.4, -33.3] | 103/6/135/5/51 | 240/15/345 | A 5e48269041cd; B ffe7a326a889; engine a5f519d2; M2: PC (α=0) の参照 arm (gn)。出力側 1/2 の書き出し (k=2), FV_SCALE 24 |
| 10-07 09:52 | tune-s-pcalm-T3-gn-e1x@24-vs-bp-rep-e1@48-300k | /mnt/D/sugiyama/011/rescaled/011-s-pcalm-T3-gn-e1@FV24 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -72.8 ± 27.9 [-101.2, -45.4] | 105/5/142/5/43 | 232/12/356 | A 6fdff611eeec; B ffe7a326a889; engine a5f519d2; M2: PC-ALM 段階 A (gn)。出力側 1/2 の書き出し (k=2), FV_SCALE 24 = 元の 48 相当 |
| 10-07 10:44 | tune-s-pc-T16-gn-e1x@24-vs-bp-rep-e1@48-300k | /mnt/D/sugiyama/011/rescaled/011-s-pc-T16-gn-e1@FV24 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -33.1 ± 25.7 [-59.0, -7.6] | 76/9/159/8/48 | 261/21/318 | A 9323dc343599; B ffe7a326a889; engine a5f519d2; M2: PC (α=0) の参照 arm (gn)。出力側 1/2 の書き出し (k=2), FV_SCALE 24 |
| 10-07 12:11 | tune-s-bp-lr0.7-lrmin0.3-e1@48-vs-bp-rep-e1@48-300k | s-bp-lr0.7-lrmin0.3-e1@FV48 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -20.3 ± 26.9 [-47.3, +6.5] | 73/18/141/7/61 | 270/25/305 | A 299ece31ef16; B ffe7a326a889; engine a5f519d2; M2: BP 段階 2 (最良の LR ×0.7 で lr-min の倍率) |
| 10-07 13:14 | tune-s-pcalm-T4-r2-gn-e1x@24-vs-bp-rep-e1@48-300k | /mnt/D/sugiyama/011/rescaled/011-s-pcalm-T4-r2-gn-e1@FV24 | s-bp-lr1-rot-e1@FV48 | 300,000 | 300 | -63.2 ± 27.3 [-91.0, -36.3] | 97/8/146/4/45 | 239/14/347 | A 985436eb2015; B ffe7a326a889; engine a5f519d2; M2: PC-ALM 段階 B (T=4)。出力側 1/2 の書き出し (k=2), FV_SCALE 24 |
