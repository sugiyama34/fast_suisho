"""検証 D の単発再実行: verify.py が残した捨て run を SA キーで削除できるか実測する。

使い方: uv run python experiments/004-wandb-verify/delete_probe.py
"""

from __future__ import annotations

import wandb

SCRAP = "suisho/suisho-test/7imnb0b8"  # verify.py の delete-probe run

try:
    wandb.Api().run(SCRAP).delete()
    print(f"OK D_sa_delete: SA CAN delete its own run ({SCRAP})")
except Exception as e:  # noqa: BLE001
    print(f"NG D_sa_delete: denied — {type(e).__name__}: {e}")
