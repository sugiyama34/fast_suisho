"""masking 有効時の疎通確認: sentinel しか見えない環境から実 run が作れるか。

sandbox の TLS 終端 proxy が api.wandb.ai 向けリクエストで sentinel を
実キーに置換できていれば、認証が通って run が作成される。

使い方: uv run python experiments/004-wandb-verify/masking_roundtrip.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.wandb_utils import init_run  # noqa: E402

key = os.environ.get("WANDB_API_KEY", "")
print(f"key seen by process: prefix={key[:9]} len={len(key)} (sentinel expected)")

with init_run("004-wandb-verify", {"purpose": "masking-roundtrip"}, job_type="verify") as r:
    r.log({"ok": 1}, step=0)
    print(f"run created: {r.run.url}")
