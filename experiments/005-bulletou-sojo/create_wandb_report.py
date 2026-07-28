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

report = wr.Report(
    entity=ENTITY,
    project=PROJECT,
    title="005-bulletou-sojo: BulletOu + 奏乗教師データで WCSC36 水匠級は再現できるか",
    description="やねうら氏共有レシピの検証実験 (2026-07-27/28)。結論: 共有された数値そのままでは再現失敗 (Elo −221)。",
    blocks=[
        wr.TableOfContents(),
        wr.H1("判定"),
        wr.MarkdownBlock(
            "**再現失敗 (共有コマンドの数値のまま、では)** — 学習は健全に収束したが、"
            "得られたネットは Suisho 11 (= WCSC36 水匠, たややん氏発言によりユーザー確認) に対し "
            "**Elo −221 (95% CI −365〜−124)**。pentanomial SPRT (elo0=−30, elo1=0, α=β=0.05) は "
            "H0 を受理 (LLR −3.56)。\n\n"
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
            "| 対局 | 32 ペア / 64 局 (互角局面集2025 ストライド抽出、先後入替) |\n"
            "| 得点 (学習ネット側) | **14 / 64 (21.9%)** — 先手 10/32, 後手 4/32 |\n"
            "| pentanomial | [20, 2, 6, 2, 2] |\n"
            "| Elo 差 | **−221** (95% CI −365 〜 −124) |\n"
            "| SPRT | LLR −3.56 ≤ −2.94 → H0 受理 (「−30 Elo より弱い」) |\n\n"
            "条件: 同一エンジンバイナリで EvalDir のみ差し替え、`go movetime 240`・16 threads・"
            "hash 1024MB・ponder/定跡なし・直列。校正 median 1.97M nodes/move。"
        ),
        wr.H1("再現方法・詳細"),
        wr.MarkdownBlock(
            "レポート全文 (図・脅威分析・再現コマンド含む) はリポジトリ "
            "`experiments/005-bulletou-sojo/report.md` を参照。"
            "教師データ取得スクリプト・学習ランチャ・wandb 同期・対局ハーネス (pentanomial SPRT) も同フォルダ。"
        ),
    ],
)
url = report.save()
print("report url:", url)
