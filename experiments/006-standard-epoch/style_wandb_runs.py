"""W&B run の表示名とワークスペース色を整える (どっちの arm か一目で分かるように)。

- run 表示名: arm 名を先頭に (例: "yane20 sb108x20ep — 006-...")
- 色: yane20 = 青 / std = 橙 の saved view を作成 (dataviz パレット準拠)

uv run python experiments/006-standard-epoch/style_wandb_runs.py
"""

from __future__ import annotations

import os

import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "suisho")
PROJECT = os.environ.get("WANDB_PROJECT", "20260728_BulletOu_with_Sojo_data")

RUNS = {
    "8fdedf87": ("yane20 (sb=108 x 20ep)", "#2a78d6"),
    "rs5o4efk": ("std (sb=367 x 16ep)", "#eb6834"),
}

api = wandb.Api()
for run_id, (name, _color) in RUNS.items():
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    old = run.name
    run.name = name
    run.update()
    print(f"renamed {run_id}: {old!r} -> {name!r}")

try:
    import wandb_workspaces.workspaces as ws

    workspace = ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name="arm-colored view",
        runset_settings=ws.RunsetSettings(
            run_settings={rid: ws.RunSettings(color=color) for rid, (_n, color) in RUNS.items()}
        ),
    )
    workspace.save()
    print("saved view:", workspace.url)
except Exception as err:  # saved view は補助なので失敗しても続行
    print(f"workspace view failed ({err}); run colors can be set in UI via the run color dot")
