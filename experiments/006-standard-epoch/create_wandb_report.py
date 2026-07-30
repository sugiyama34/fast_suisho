"""experiment 006 の W&B Report を 20260728_BulletOu_with_Sojo_data に作成する (日本語)。

uv run python experiments/006-standard-epoch/create_wandb_report.py
"""

from __future__ import annotations

import os

import wandb_workspaces.reports.v2 as wr

ENTITY = os.environ.get("WANDB_ENTITY", "suisho")
PROJECT = os.environ.get("WANDB_PROJECT", "20260728_BulletOu_with_Sojo_data")

RUN_IDS = ["8fdedf87", "4p1ut2q0", "rs5o4efk", "njg75f9a"]

runset = wr.Runset(
    entity=ENTITY,
    project=PROJECT,
    name="006 training runs",
    filters=f"ID in {RUN_IDS}",
)

panels = [
    wr.LinePlot(
        title="floodgate test accuracy",
        x="positions",
        y=["test_value_accuracy"],
        smoothing_factor=0,
    ),
    wr.LinePlot(
        title="floodgate test loss", x="positions", y=["test_value_loss"], smoothing_factor=0
    ),
    wr.LinePlot(title="train loss", x="positions", y=["train_value_loss"], smoothing_factor=0),
    wr.LinePlot(title="learning rate", x="positions", y=["lr_start"], smoothing_factor=0),
]

blocks = [
    wr.TableOfContents(),
    wr.H1("判定"),
    wr.MarkdownBlock(
        "1. **やねうらお氏の訂正後の主張を再現**: 氏の実設定 (sb=108 × 16 BulletOu-epoch"
        " = 691 億局面) の net は Suisho 11 (= WCSC36 水匠, 氏の比較対象と同一) に対し"
        " **Elo −104 (95% CI −171〜−43)** — 「R100 ぐらい低い」とほぼ一致\n"
        "2. **「sb を上げればもっと強くなる」は未確認**: sb=367 のベスト (−135 @ 教師 4 周)"
        " は sb=108 のベスト (−104 @ 4.7 周) を上回らず (CI 重なる)\n"
        "3. **新知見 — 強さは教師 4〜5 周でピーク、以降は悪化**: sb=108 を 32 epoch"
        " (9.4 周) まで延長すると **−263** (ピーク比 −159, CI 非重複)。floodgate accuracy"
        " は 70.39% で不変のまま — **accuracy は過学習による強さ低下を検出できない**"
    ),
    wr.H1("用語"),
    wr.MarkdownBlock(
        "- **BulletOu-epoch**: `--superbatches N`×40M 局面の LR サイクル (データ量と無関係)\n"
        "- **standard-epoch (周)**: 教師全体 1 周 = 奏乗全 30 ファイル = 14,668,949,437 局面"
    ),
    wr.H1("学習量 → Elo カーブ (全 17 点, 各 50 ペア固定)"),
    wr.MarkdownBlock(
        "**sb=108 (やねうらお氏実設定系列)**\n\n"
        "| ep | 局面 (周) | Elo | 95% CI |\n| --- | --- | --- | --- |\n"
        "| 1 | 43億 (0.3) | −424 | −609〜−329 |\n"
        "| 2 | 86億 (0.6) | −295 | −423〜−210 |\n"
        "| 4 | 173億 (1.2) | −200 | −298〜−125 |\n"
        "| 8 | 346億 (2.4) | −244 | −353〜−166 |\n"
        "| **16** | **691億 (4.7)** | **−104** | **−171〜−43** ← 氏の実設定 |\n"
        "| 20 | 864億 (5.9) | −151 | −237〜−80 |\n"
        "| 32 | 1382億 (9.4) | −263 | −374〜−185 ← 過学習 |\n\n"
        "**sb=367**: −205 (1周) / −200 (2周) / **−135 (4周)** / −164 (8周) / −205 (16周)\n\n"
        "**sb=12 (005 テンプレート)**: −636 → −449 → −494 → −301 → −372 (0.5〜77億局面)\n\n"
        "カーブ図・詳細はリポジトリ `experiments/006-standard-epoch/report.md` を参照。"
        "測定条件は experiment-005 フェーズ3 と同一 (movetime 240ms ≒ 2M nodes/move,"
        " 互角局面集ストライド 500, pentanomial)。"
    ),
    wr.PanelGrid(runsets=[runset], panels=panels),
    wr.H1("解釈"),
    wr.MarkdownBlock(
        "- LR 周期は「小さすぎると悪い」(sb=12→108 で +190 Elo 級) が、108→367 の追加効果なし\n"
        "- このレシピの限界は LR ではなく**データ再利用**が決めている可能性が高い"
        " (どの sb でも 4〜5 周でピーク)\n"
        "- モデル選択を floodgate accuracy でやってはいけない (ep16→32 で accuracy 不変・"
        "Elo −159) — 氏の当初の誤認もこの乖離が原因とみられる\n"
        "- 到達点: 奏乗 + BulletOu + `SFNN_halfka2_1024_7_64_k3k3` の現状ベストは"
        " **水匠11 比 −100 Elo 前後**。先へ進むには教師拡充 / レシピ工夫 / アーキ変更が必要"
    ),
]

report = wr.Report(
    entity=ENTITY,
    project=PROJECT,
    title="006-standard-epoch: 学習量とLR周期を伸ばせば奏乗レシピはどこまで強くなるか",
    description="結論: やね氏実設定で −104 (訂正主張を再現)。強さは教師4〜5周でピーク、以降は accuracy 不変のまま悪化。",
    blocks=blocks,
)
report.save()
print("report url:", getattr(report, "url", "(saved)"))
