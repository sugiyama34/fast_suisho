# 決定の記録 (experiment-011)

設計時のユーザー決定は [../hypothesis.md](../hypothesis.md) §7。ここには実行中の決定を時刻順に残す。

| 日時 | 決定 | 理由 | 決めた人 |
| --- | --- | --- | --- |
| 10-06 | ユーザーの確認を待たずに進める (確認が要らないものは進める) | ユーザー指示 | ユーザー |
| 10-06 | 対局は `ulimit -s 8192` で起動する | kajiki のシェルは `ulimit -s` が unlimited で、やねうら王の探索スレッドのスタックが小さくなり、深い再帰で SIGSEGV になる局面がある (300k ノード, 互角局面集 3756 番の後手番の対局で毎回同じ手で落ちる)。8 MB で解消を確認。`docs/SETUP.md` §9 に追記 | Claude |
| 10-06 | M2 の調整対局は **全候補を BP の複製 (`s-bp-lr1-rot`) の e1 と対局**させる (bp-recipe も含む)。両方 FV_SCALE 48、300 ペア、300k ノード | 当初案の「bp-recipe の e1 を相手にする」では bp-recipe だけが 0 に固定され、ノイズを持つ他の候補が「誤差内でも最大値」の規則でほぼ必ず勝つ (非対称)。複製は候補ではないので全候補を対称に測れる。同程度の強さの相手なので水匠 11 相手より区間が狭い | Claude (advisor の指摘) |
| 10-06 | M2 の調整対局のエンジンは kajiki のビルド (`yane-9133c527-finny-avx2`, sha256 `a5f519d2…`) で統一する | 調整対局は 1 つの系列の中の相対比較。009 のバイナリ (suzuki から複製待ち) の到着時期によらず系列内で揃える。M4 (本番) は 009 のバイナリ | Claude |
| 10-06 | PC-ALM の `η_h` は推論の作用素の λ_max の**最大値** (全サンプル) の逆数を基準にし、学習済みの重みでも測り直す。M2 の各 PC-ALM run の e1 と M3 の各 epoch の checkpoint で λ_max と勾配の向きを測り、安定の範囲を外れたら報告する | 不安定な η では NaN にならずに壊れた勾配 (cosine 0.1〜0.3, 大きさ 5〜20 倍) のまま学習が進む。打ち切りをしない設計なので、測って記録する (advisor の指摘) | Claude |
| 10-06 | 学習済みの重みで λ_max が 3.4 → 58 に増え、η = 1/λ_max の規則で固定の η は使えない。**η を自動設定** (`--pcalm-eta-auto`, ミニバッチごとに `1 / (1.05 × λ_max の推定)`) にする。α = 1.5 の arm は倍率 0.75 | m0.md「刻み η の自動設定」 | Claude |
| 10-06 | PC-ALM の状態を順伝播からの差で持つ実装に変更 (float32 の桁落ちで L1 の勾配に 4〜7% の誤差があった) | m0.md「数値誤差」 | Claude |
| 10-06 | **PC-ALM / PC の arm は Ranger の epsilon を較正した値にする** (1e-7 × FT の勾配の大きさの比 PC-ALM / BP, BP の e1 で測定, `arms.py` の `EPS_PCALM`)。調整の次元にはしない | PC-ALM の勾配は BP の 1/100〜1/1000 で、1e-7 のままでは FT の更新が epsilon に潰され、信用割当と関係のない理由で負ける (advisor の指摘を受けて記録) | Claude |
| 10-06 | M2 の PC-ALM は最適化したバイナリ (`bulletou-pcalm-a847a642d7da`) で学習する。float64 の参照との差 2e-5 以下を確認済み | 速度 | Claude |
| 10-06 | `s-bp-lr1` と `s-bp-lr1-rot` の W&B run が「failed」になっているのは、学習中に `run_training.sh` を書き換えたため最後の `echo` が壊れたもの (学習と checkpoint は正常終了)。以後スクリプトは一時ファイルに書いて `mv` で置き換える | 事故の記録 | Claude |
| 10-06 | W&B: ユーザーが kajiki で `wandb login` (~/.netrc) を行い、環境変数には置かない方針。`train_supervised.py` は ~/.netrc の資格情報も見て online で記録する。offline だった 2 run は `wandb sync` 済み | ユーザー | ユーザー |
| 10-06 | suzuki からのファイル (009 のエンジン `423b6b1c…`, full-e12/16/20 の nn.bin, full-e1/16 の state.bin, full-e16 @ 48 の棋譜, B_s0) を受け取り、sha256 を確認して配置 (`xfer/MANIFEST-from-suzuki.sha256`) | ユーザーが転送 | ユーザー / Claude |
| 10-06 | W&B の run 名を人が読める形にする (ユーザー: 少し長くてよいが長すぎない)。`011 <small/full> <BP/PC-ALM/PC> <既定と違う設定>` (例: `011 small PC-ALM T=4`, `011 small BP lr×0.5`, `011 full BP replicate`)。全設定と arm 名は config に残る。既存の run も API で改名 | ユーザー | ユーザー |
