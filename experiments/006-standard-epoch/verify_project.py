"""新 W&B プロジェクトへの書き込み疎通確認 (SA キーの team スコープ検証)。

uv run python experiments/006-standard-epoch/verify_project.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.wandb_utils import init_run  # noqa: E402

print("entity :", os.environ.get("WANDB_ENTITY"))
print("project:", os.environ.get("WANDB_PROJECT"))
with init_run(
    "006-standard-epoch",
    config={"purpose": "project connectivity check"},
    job_type="bench",
    tags=["verify"],
) as run:
    run.log({"ok": 1}, step=1)
    print("run url:", getattr(run.run, "url", run.run.name))
print("write OK")
