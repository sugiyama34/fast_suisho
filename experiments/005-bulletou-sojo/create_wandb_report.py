"""experiment 005 の W&B Report を suisho/suisho-test に作成する (本文は日本語)。

uv run python experiments/005-bulletou-sojo/create_wandb_report.py
"""

from __future__ import annotations

import os

import wandb_workspaces.reports.v2 as wr

ENTITY = os.environ.get("WANDB_ENTITY", "suisho")
PROJECT = os.environ.get("WANDB_PROJECT", "suisho-test")

# 最終 run (floodgate テストセットでの再スタート後)
RUN_IDS = {"main": "yl3vfcow", "var-b": "bqby8v4q", "var-c": "27qylllu"}

# 既存レポートを更新する場合はその URL (新規作成なら None)
REPORT_URL = (
    "https://wandb.ai/suisho/suisho-test/reports/"
    "005-bulletou-sojo-BulletOu-%E5%A5%8F%E4%B9%97%E6%95%99%E5%B8%AB%E3%83%87%E3%83%BC%E3%82%BF"
    "%E3%81%A7-WCSC36-%E6%B0%B4%E5%8C%A0%E7%B4%9A%E3%81%AF%E5%86%8D%E7%8F%BE%E3%81%A7%E3%81%8D"
    "%E3%82%8B%E3%81%8B--VmlldzoxNzYwMDI3MA"
)

runset = wr.Runset(
    entity=ENTITY,
    project=PROJECT,
    name="005 final arms (main / var-b / var-c)",
    filters=f"ID in {list(RUN_IDS.values())}",
)

panels = [
    wr.LinePlot(
        title="floodgate test accuracy",
        x="positions",
        y=["test_value_accuracy"],
        smoothing_factor=0,
    ),
    wr.LinePlot(
        title="floodgate test loss",
        x="positions",
        y=["test_value_loss"],
        smoothing_factor=0,
    ),
    wr.LinePlot(
        title="train loss (per superbatch)",
        x="positions",
        y=["train_value_loss"],
        smoothing_factor=0,
    ),
    wr.LinePlot(title="learning rate", x="positions", y=["lr_start"], smoothing_factor=0),
]

blocks = [
    wr.TableOfContents(),
    wr.H1("判定"),
    wr.MarkdownBlock(
        "**再現失敗 (共有コマンドの数値のまま、では)** — 学習は健全に収束したが、"
        "得られたネットは Suisho 11 (= WCSC36 水匠, たややん氏発言によりユーザー確認) に対し "
        "**Elo −293 (95% CI −434〜−204)**。pentanomial SPRT (elo0=−30, elo1=0, α=β=0.05) は "
        "H0 を受理 (LLR −8.27)。\n\n"
        "ただし共有コマンドは `--superbatches` / `--max-epochs` に「ここ調整」注記付きの"
        "**テンプレート**であり、その値 (40M×12sb×16epoch = 76.8億局面) は本環境の GPU で "
        "**68 分**で終わる。「2 日くらい学習」という氏の記述と桁で乖離しており、"
        "氏の実走は学習量が 1〜2 桁大きかった可能性が高い。"
        "**「2 日分の計算量での再現可否」は未検証のまま** — floodgate accuracy は "
        "16 epoch 時点でまだ単調改善中で、学習量を増やす追実験 (exp-006 候補) が次のステップ。"
    ),
    wr.H1("学習結果 (3 アーム)"),
    wr.MarkdownBlock(
        "| Arm | floodgate acc | floodgate loss | train loss |\n"
        "| --- | --- | --- | --- |\n"
        "| main (ファイル順 001→016) | **69.72%** | 0.07736 | 0.03348 |\n"
        "| var-b (009 起点回転) | 69.46% | 0.07818 | 0.03221 |\n"
        "| var-c (005 起点回転) | 69.56% | 0.07866 | 0.03217 |\n\n"
        "- 教師: 奏乗 (washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique) 001–016 = 78.2 億局面\n"
        "- 検証: takaoyamaoka/floodgate.hcpe (2017-2018, R3500+, 856,923 局面) から毎 sb 300k サンプル\n"
        "- BulletOu は初期化シード固定・再シャッフルなし → 同一コマンドは決定的。"
        "ファイル順回転がデータ順感度の代理で、アーム間差は accuracy ±0.13pp と誤差レベル\n"
        "- スループット 2.32M pos/s (RTX PRO 5000 Blackwell ×1/run)、1 run = 68 分・16 epoch 完走 (early stop なし)"
    ),
    wr.PanelGrid(runsets=[runset], panels=panels),
    wr.H1("強さ測定 (フェーズ3 自己対局 vs Suisho 11)"),
    wr.MarkdownBlock(
        "| 項目 | 値 |\n| --- | --- |\n"
        "| 対局 | 32 ペア / 64 局 (互角局面集2025 ストライド抽出、先後入替、直列) |\n"
        "| 得点 (学習ネット側) | **10 / 64 (15.6%)** — 先手 5/32, 後手 5/32 |\n"
        "| pentanomial | [22, 0, 10, 0, 0] |\n"
        "| Elo 差 | **−293** (95% CI −434 〜 −204) |\n"
        "| SPRT | LLR −8.27 ≤ −2.94 → H0 受理 (「−30 Elo より弱い」) |\n"
        "| 終局理由 | 詰み 63, 入玉宣言 1 (千日手なし) |\n\n"
        "補足: 初回測定は千日手判定バグ (同一局面 2 回目で成立 + 優等/劣等局面の誤終局) の"
        "影響下で 14/64 (Elo −221)。ルール修正後の再測定が上記で、正しいルールでは"
        "むしろ差が広がった。判定はどちらも H0 受理で不変。"
    ),
    wr.H1("設定の完全な記録"),
    wr.H2("学習設定 (全パラメータ)"),
    wr.MarkdownBlock(
        "明示指定はやねうら氏共有コマンドと同値。(default) は共有コマンドに現れず "
        "BulletOu のデフォルトを継承した値。\n\n"
        "| パラメータ | 値 | 備考 |\n| --- | --- | --- |\n"
        "| BulletOu | commit 9577f08 | 氏の実行版は不明。build.rs に sm_120 SASS 追加 (数値経路不変) |\n"
        "| --backend | cuda-cpp | 同値 |\n"
        "| --arch | SFNN_halfka2_1024_7_64_k3k3 | 同値 |\n"
        "| --teacher | 奏乗 001–016 明示カンマ列 (arm ごと回転順) | 氏はディレクトリ指定 (中身不明) |\n"
        "| --test-teacher | floodgate.hcpe (856,923 局面) | **氏と異なる** (私有 fg2021 系 hcpe) |\n"
        "| --test-positions | 300000 | 同値 |\n"
        "| --positions-per-superbatch | 40000000 (実効 39,976,960) | 同値 |\n"
        "| --superbatches / --max-epochs | 12 / 16 | 同値 (氏の注記「ここ調整」= テンプレート) |\n"
        "| --lr / --lr-min / --lr-schedule | 0.000875 / 0.000030 / step | 同値 |\n"
        "| --optimizer / weight-decay | ranger (beta1=0.99 default) / 0.0 | 同値 |\n"
        "| --save-rate / --validation-rate | 9999 / 1 (epoch 末 save 有効) | 同値 |\n"
        "| batch size / lambda / scale | 65536 / 1.0 / 290 (default) | 教師 eval 100% |\n"
        "| score_drop_abs / factorizer / loss | 32000 / shared / sigmoid-mse (default) | |\n"
        "| GPU | RTX PRO 5000 Blackwell ×1/run | 学習は固定シードで決定的のためハード非依存 |"
    ),
    wr.H2("強さ測定設定 (全条件)"),
    wr.MarkdownBlock(
        "| 項目 | 値 |\n| --- | --- |\n"
        "| エンジン | やねうら王 YaneuraOu-by-gcc sha256 87857ca8… (両陣営同一バイナリ) |\n"
        "| 評価関数 | baseline: Suisho 11 nn.bin (a78b7f88…) / candidate: main 最終 nn.bin (57f59777…) |\n"
        "| 持ち時間 | go movetime 240 (ms)。校正実測 median 1,974,173 nodes/move |\n"
        "| エンジン設定 | Threads 16, USI_Hash 1024MB, ponder off, MultiPV 1, NetworkDelay=0/0, 定跡なし, MaxMovesToDraw 0 |\n"
        "| 開始局面 | 互角局面集2025 (24手目) の NR%60==1 ストライド 500 局面 |\n"
        "| 終局判定 | 詰み / resign / 入玉宣言 / 同一局面 4 回 (連続王手は cshogi 分類で勝敗) / 320 手引き分け |\n"
        "| 統計 | pentanomial GSPRT (elo0=−30, elo1=0, α=β=0.05, 最小 32 ペア, 上限 500 ペア) |"
    ),
    wr.H2("原設定 (やねうら氏) との差分まとめ"),
    wr.MarkdownBlock(
        "| 項目 | 既知/不明 | 差分と影響 |\n| --- | --- | --- |\n"
        "| テスト教師 | **既知の差** | 私有 fg2021 系 → 公開 floodgate.hcpe。accuracy 絶対値は比較不能。"
        "early stop 判定に影響しうるが今回は発火せず学習重みへの影響なし |\n"
        "| 教師の集合・順序 | 不明 | 本実験 001–016。3 アーム回転で感度検証済み (誤差レベル) |\n"
        "| superbatches / max-epochs | **不明 (最重要)** | 共有値はテンプレート。76.8 億局面 = 68 分で"
        "「2 日」と桁不一致。氏の実走は 1〜2 桁大きい可能性が高く、本結果はテンプレート値への否定 |\n"
        "| BulletOu の版 | 不明 | 本実験 9577f08 (2026-07-27 clone) |\n"
        "| 強さの測定方法 | 不明 | 「水匠と同じくらい」の測定条件不明。本実験はフェーズ3 プロトコル |\n"
        "| GPU / wall-clock | 不明 | 学習は決定的なので同一スケジュールなら結果は不変 |"
    ),
    wr.H1("再現方法・詳細"),
    wr.MarkdownBlock(
        "レポート全文 (図・脅威分析・再現コマンド含む) はリポジトリ "
        "`experiments/005-bulletou-sojo/report.md` を参照。"
        "教師データ取得スクリプト・学習ランチャ・wandb 同期・対局ハーネス (pentanomial SPRT) も同フォルダ。"
    ),
]

DESCRIPTION = (
    "やねうら氏共有レシピの検証実験 (2026-07-27/28)。"
    "結論: 共有された数値そのままでは再現失敗 (Elo −293)。"
)

if REPORT_URL:
    report = wr.Report.from_url(REPORT_URL)
    report.blocks = blocks
    report.description = DESCRIPTION
else:
    report = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="005-bulletou-sojo: BulletOu + 奏乗教師データで WCSC36 水匠級は再現できるか",
        description=DESCRIPTION,
        blocks=blocks,
    )
report.save()
print("report url:", getattr(report, "url", REPORT_URL))
