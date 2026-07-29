"""yane20 + yane32 を 1 本の連続チャートで見るための combined run。

両者は同じ summary-learn.log (006-yane20/) に追記しており positions も累積なので、
全行をリプレイ + 追従する 1 つの W&B run を作れば epoch 1〜32 が単一の線になる。
32 epoch 完了 (= 108/4 × 32 = 864 行) で自動的にクリーン終了する。

uv run python experiments/006-standard-epoch/sync_combined.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "005-bulletou-sojo"))

from wandb_sync import read_rows  # noqa: E402

from tools.wandb_utils import init_run  # noqa: E402

LOG_PATH = REPO / "data" / "bulletou" / "checkpoints" / "006-yane20" / "summary-learn.log"
EXPECT_ROWS = 32 * (108 // 4)  # 864: 32 epoch × 27 検証行/epoch
DISPLAY_NAME = "yane combined (sb=108, ep1-32 continuous)"

CONFIG = {
    "purpose": "yane20+yane32 連続チャート (同一 summary-learn.log の全行リプレイ+追従)",
    "superbatches": 108,
    "max_epochs_cumulative": 32,
    "source_runs": "yane20=8fdedf87, yane32=4p1ut2q0",
}


def main() -> None:
    with init_run(
        "006-standard-epoch", config=CONFIG, tags=["yane-combined"], job_type="train"
    ) as run:
        try:
            run.run.name = DISPLAY_NAME
        except Exception:
            pass
        print(f"combined run: {getattr(run.run, 'url', run.run.name)}", flush=True)
        n_logged = 0
        while True:
            rows = read_rows(LOG_PATH) if LOG_PATH.exists() else []
            for step, m in enumerate(rows[n_logged:], start=n_logged + 1):
                if any(isinstance(v, float) and math.isnan(v) for v in m.values()):
                    run.alert("NaN detected", f"combined step={step}: {m}")
                run.log(m, step=step)
            if len(rows) > n_logged:
                n_logged = len(rows)
                print(f"synced {n_logged}/{EXPECT_ROWS} rows", flush=True)
            if n_logged >= EXPECT_ROWS:
                print("all rows synced; finishing", flush=True)
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
