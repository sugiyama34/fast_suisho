"""alert 配信の再検証 (検証 E の追試)。

wandb.alert() トグルが Email で ON であることを UI で確認済みの状態で、
service account run からの alert が届くかを最終確認する。

使い方: uv run python experiments/004-wandb-verify/alert_retest.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.wandb_utils import init_run  # noqa: E402

os.environ["WANDB_USER_EMAIL"] = "sugiyama.satoshi1990@gmail.com"

with init_run("004-wandb-verify", {"purpose": "alert-retest"}, job_type="verify") as r:
    r.alert("WandB alert 再検証", "トグル ON 確認後の再送テストです。届いていれば E は成功。")
    print(f"alert sent from run: {r.run.url}")
