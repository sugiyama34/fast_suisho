"""作成済み W&B Report の URL を表示する。"""

import os

import wandb

api = wandb.Api()
entity = os.environ.get("WANDB_ENTITY", "suisho")
project = os.environ.get("WANDB_PROJECT", "suisho-test")
for r in api.reports(f"{entity}/{project}"):
    print(r.url)
