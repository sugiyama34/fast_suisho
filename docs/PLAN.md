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
2. **自己対局**: オリジナル vs 高速化版
   - 同一思考時間 (例: 1手1秒 or fish クロック) で最大1000局
   - **検出力に注意**: 1000局の 95% 信頼区間は約 ±20 Elo。FT 縮小の期待効果
     (NPS +25% ≒ +15〜22 Elo − 蒸留による精度損) はノイズに埋もれうる。
     固定局数ではなく SPRT (例: H0: -5 Elo, H1: +15 Elo, α=β=0.05) を第一の
     判定基準とし、1000局は打ち切り上限として扱う。有意になるまで再走を繰り返す
     p-hacking はしない
   - 開幕局面をばらすため互角局面集 or 定跡でスタート、先後入れ替え
   - ハーネス: cshogi (要 pip install) または やねうら王付属 `script/engine_invoker5.py`
   - Elo 差と95%信頼区間を `calc_rating.py` 相当で算出
   - 16コアなので同時対局数は 8 (各エンジン1スレッド) を基本とする

## 環境メモ

- CPU: 16 cores, AVX2/BMI2 (AVX-512 なし)
- RAM: 251 GB / GPU: RTX PRO 5000 (48GB) ×3
- Python: python3 のみ (venv + numpy/torch/cshogi を導入予定)
- コンパイラ: g++ (clang++ なし)
- やねうら王 Makefile は `PYTHON=python3` の指定が必要
