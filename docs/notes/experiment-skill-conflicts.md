# /experiment スキルと本リポジトリの規約の相違点

日付: 2026-07-13

`/experiment` スキルは別プロジェクト (深層学習実験一般) 向けに書かれており、
本リポジトリの規約と食い違う点がある。**相違がある場合は本リポジトリの規約を優先する**
(スキル自身も「リポジトリに既存の実験規約があればそちらに従う」と明記している)。

| 項目 | スキルの指定 | 本リポジトリの規約 (優先) |
|---|---|---|
| フォルダ名 | `experiments/<YYYY-MM-DD>_<slug>/` | `experiments/NNN-name/` (連番。PLAN 記載) |
| ファイル構成 | `hypothesis.md` / `metadata.json` / `config/` / `command.md` / `logs/` / `metrics/` / `figures/` / `report.md` に分割 | `report.md` 1 本に仮説・セットアップ・再現手順・結果・結論を集約し、スクリプトと成果物 (png / npy / txt) を同フォルダに直置き (experiment-000 / experiment-001 の既存形式) |
| 言語 | (英語想定) | ドキュメントは日本語 (CLAUDE.md) |
| 実験の型 | 訓練/評価ラン (seed 分散、複数 seed 前提) | フェーズ2 前半は NPS 計測・プロファイル・等価変換が中心。seed ではなく run 間分散 (±1.5%) と統計プロトコル (フェーズ3 の pentanomial SPRT) で扱う |
| メタデータ | `metadata.json` (commit, dirty flag, HW) | sha256 fingerprint 方式 (000-baseline 参照): エンジン/nn.bin の sha256 + ビルド条件 + governor/負荷条件を report.md とスクリプトのヘッダ出力に記録 |

運用: スキルの本質 (仮説 → 最小の反証実験 → 公平な比較 → 正直なレポート) は採用し、
書式・置き場所は本リポジトリの既存形式に合わせる。
