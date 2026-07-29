"""W&B run の表示名を整える (どっちの arm か一目で分かるように)。

色はユーザーがオリジナルの workspace view で手動設定済み (2026-07-29)。
saved view の自動作成はパネルを持たず空に見えるためやめた — view には触らない。

uv run python experiments/006-standard-epoch/style_wandb_runs.py
"""

from __future__ import annotations

import os

import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "suisho")
PROJECT = os.environ.get("WANDB_PROJECT", "20260728_BulletOu_with_Sojo_data")

RUNS = {
    "8fdedf87": ("yane20 (sb=108 x 20ep)", None),
    "rs5o4efk": ("std (sb=367 x 16ep)", None),
    "4p1ut2q0": ("yane32 (resume +12ep, cum 32)", None),
}

api = wandb.Api()
for run_id, (name, _color) in RUNS.items():
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    old = run.name
    run.name = name
    run.update()
    print(f"renamed {run_id}: {old!r} -> {name!r}")

# (view/色の操作はしない — ユーザーがオリジナル view で管理)
