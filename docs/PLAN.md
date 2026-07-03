# 実験計画: Suisho 11 NNUE の高速化

日付: 2026-07-02 (立案)

## フェーズ 0: ベースライン確立

1. やねうら王を `SFNN_halfka2_1024-7-64-k3k3` アーキテクチャでビルド (AVX2, g++)
2. Suisho 11 `nn.bin` を読み込ませ、正常動作を確認 (`isready` → bench)
3. ベースライン NPS を計測・記録 (シングルスレッド / 16スレッド)

## フェーズ 1: ネットワーク解析ツール

`tools/` に Python (numpy) で nn.bin の parser / serializer を実装:

- ファイル形式: version + hash + architecture 文字列 + FeatureTransformer + LayerStacks×9 の各層 (int8 weights / int32 biases)
- **FeatureTransformer 部は LEB128 圧縮**: `COMPRESSED_LEB128` マジックを nn.bin の
  offset 235 / 1284 で確認済み (YaneuraOu `nnue_common.h` の `read_leb_128` 参照)。
  展開後が int16 weights/biases。ファイルが 135 MB (生 int16 なら約 270 MB) なのはこのため
- 往復変換 (parse → serialize) でバイト一致することを確認するテストを書く
  (LEB128 を bit-exact に再エンコードする必要がある点に注意)
- 重み分布の統計 (ゼロ近傍割合、チャネルごとの L1 ノルム等) を可視化

## フェーズ 2: 高速化実験 (候補)

効果が見込める順:

1. **Feature Transformer 幅の構造的縮小** (1024 → 768 / 512)
   - 評価コストの大半は FT。出力チャネルを L1 ノルム等で重要度順に選抜して削る
   - 削るだけでは精度が落ちるため、蒸留 (元ネットワークの出力を教師に fine-tune) が前提
   - GPU 3× RTX PRO 5000 が使えるので nnue-pytorch 系トレーナを移植して蒸留する
   - **前提条件: 学習局面コーパスの確保**。蒸留品質は局面分布に支配される
     (nnue-pytorch 系は数億局面規模を想定)。着手前に「入手源 (公開教師データ or
     自己対局生成) / 規模 / 生成コスト」を確定させること。小規模なアドホック
     コーパスで蒸留して「FT 縮小は精度損が大きい」と結論するのはデータ起因の
     誤帰属であり禁止
2. **fc 層のスパース化** (fc_0 は既に sparse-input 実装。fc_1 64ch の刈り込み)
3. **量子化スケールの見直し / SIMD 実装の最適化** (エンジン側の改造。arch header 差し替えで済む範囲)
4. **LayerStacks の統合** (9 → 少数) — 精度影響が大きいので優先度低

各実験は `experiments/NNN-name/` に config・結果・レポートを置く。

## フェーズ 3: 検証

1. **静的検証**: 固定局面集で元ネットとの評価値相関 / 一致率を測る (安価なスクリーニング)
2. **自己対局**: オリジナル vs 高速化版 (対局条件は 2026-07-03 に確定・実測校正済み)
   - **1000局** (500 開始局面 × 先後入れ替え)
   - **開始局面**: やねうら王互角局面集2025 の 24手目局面集
     `data/books/start_sfens_ply24.txt` (git管理外。`match/fetch_books.sh` で
     ダウンロード+sha256検証+LF正規化して配置する。30,053局面, MIT License,
     水匠10の2億ノード探索で |評価値| ≤ 50 を抽出したもの。
     出典: github.com/yaneurao/YaneuraOu releases/BalancedPositions2025)
   - **開始局面の抽出方法 (固定)**: 局面集は辞書順ソート済みで隣接行が近似重複のため、
     先頭 500 行ではなく**等間隔ストライド抽出** (`NR % 60 == 1` で 501 点 → 先頭 500 点) を使う。
     抽出した 500 局面のリストは実験フォルダにコミットして固定する
   - **持ち時間**: 1手固定 movetime **240ms・16スレッド・hash 1024MB**、1ゲームずつ直列実行。
     ユーザー指定の「1手約1/3秒でオリジナルNNUEが約200万ノード」に相当するよう実測校正した値
     (互角局面10点サンプルで中央値 2.03M nodes/move。333ms だと 2.80M で40%超過のため修正)。
     両者同一時間で対局するため、高速化版はノード数増として利得を得る (これが測りたい効果)
   - **エンジン設定の固定 (校正の前提条件)**: `USI_Ponder=false` (ponder 禁止 —
     有効だと両エンジンが同時探索し 32 スレッド/16 コアで校正が無効になる)、
     ハーネスは **`go movetime 240` の形式**で時間を渡す (byoyomi 形式だと
     `NetworkDelay`/`NetworkDelay2` (デフォルト約 120/1120ms) が差し引かれ
     ノード数が校正値を大きく下回る。byoyomi しか使えないハーネスの場合は
     NetworkDelay=0, NetworkDelay2=0 を設定した上で再校正)、MultiPV=1。
     本番ハーネス経由で `match/calibrate_nodes.sh` 相当の再校正を対局開始前に必ず行う
   - 代替構成: 8スレッド×500ms (中央値 2.10M nodes/move, アイドル時計測) で2並列も可。
     ただし並列対局時の相互干渉下で要再校正
   - **検出力に注意**: 1000局の 95% 信頼区間は約 ±20 Elo。FT 縮小の期待効果
     (NPS +25% ≒ +15〜22 Elo − 蒸留による精度損) はノイズに埋もれうる。
     固定局数ではなく SPRT (例: H0: -5 Elo, H1: +15 Elo, α=β=0.05) を第一の
     判定基準とし、1000局は打ち切り上限として扱う。有意になるまで再走を繰り返す
     p-hacking はしない
   - **統計は先後ペア単位 (pentanomial) で扱う**: 1000局は独立ではなく 500 ペア
     (同一開始局面の先後入替) であり、ペア内の結果は開始局面の偏りで正相関する
     (互角の判定は水匠10の2億ノード基準であり、2Mノードの対局では決着がつく
     局面も多い)。trinomial + 独立仮定で SPRT/CI を計算すると分散を過小評価し
     α/β が名目値を割るため、fishtest と同様にペア単位の pentanomial モデルで
     SPRT と誤差を計算する
   - ハーネス: cshogi (要 pip install) または やねうら王付属 `script/engine_invoker5.py`
   - Elo 差と95%信頼区間を `calc_rating.py` 相当で算出

## 環境メモ

- CPU: 16 cores, AVX2/BMI2 (AVX-512 なし)
- RAM: 251 GB / GPU: RTX PRO 5000 (48GB) ×3
- Python: python3 のみ (venv + numpy/torch/cshogi を導入予定)
- コンパイラ: g++ (clang++ なし)
- やねうら王 Makefile は `PYTHON=python3` の指定が必要
