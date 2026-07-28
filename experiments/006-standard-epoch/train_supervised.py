"""exp-006 standard-epoch 学習の supervisor (設計は 005/train_supervised.py と同一)。

トレーナと W&B run を単一ライフサイクルで所有: トレーナ exit 0 → Finished /
非 0 (またはキャンセル) → Failed。

使い方 (GPU が見える環境で):
    uv run python experiments/006-standard-epoch/train_supervised.py --gpu 0
"""

from __future__ import annotations

import argparse
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "005-bulletou-sojo"))

from wandb_sync import read_rows  # noqa: E402  (005 の CSV パーサを再利用)

from tools.wandb_utils import init_run  # noqa: E402

EXPERIMENT = "006-standard-epoch"

CONFIG = {
    "trainer": "BulletOu 9577f08 (cuda-cpp, sm_120 patch)",
    "arch": "SFNN_halfka2_1024_7_64_k3k3",
    "teacher": "sojo full 30 files = 14.669B positions (standard-epoch)",
    "test_teacher": "takaoyamaoka/floodgate.hcpe (300k/validation, rate 4sb)",
    "positions_per_superbatch": 40_000_000,
    "superbatches": 367,
    "max_epochs": 16,
    "epoch_definition": "1 BulletOu-epoch = 367sb = 14.67B pos ≒ 1 standard-epoch",
    "lr": 0.000875,
    "lr_min": 0.000030,
    "lr_schedule": "step (1 standard-epoch 周期)",
    "optimizer": "ranger",
    "batch_size": 65536,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--stall-min", type=float, default=45.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    log_path = REPO / "data" / "bulletou" / "checkpoints" / "006-std" / "summary-learn.log"
    start_rows = len(read_rows(log_path)) if log_path.exists() else 0

    cmd = ["bash", str(HERE / "run_training.sh"), "std", str(args.gpu)]
    config = {**CONFIG, "gpu": args.gpu, "log_path": str(log_path)}

    with init_run(EXPERIMENT, config=config, tags=["std", "supervised"], smoke=args.smoke) as run:
        print(f"wandb run: {getattr(run.run, 'url', None) or run.run.name}", flush=True)
        proc = subprocess.Popen(cmd, start_new_session=True)

        cancelled = False

        def forward_signal(_signum, _frame) -> None:
            nonlocal cancelled
            cancelled = True
            proc.terminate()

        signal.signal(signal.SIGTERM, forward_signal)
        signal.signal(signal.SIGINT, forward_signal)

        n_logged = start_rows
        last_new = time.time()
        stall_alerted = False

        def drain() -> None:
            nonlocal n_logged, last_new, stall_alerted
            rows = read_rows(log_path) if log_path.exists() else []
            for step, m in enumerate(rows[n_logged:], start=n_logged + 1 - start_rows):
                if any(isinstance(v, float) and math.isnan(v) for v in m.values()):
                    run.alert("NaN detected", f"006-std step={step}: {m}")
                run.log(m, step=step)
            if len(rows) > n_logged:
                n_logged = len(rows)
                last_new = time.time()
                stall_alerted = False

        while True:
            try:
                rc = proc.wait(timeout=args.interval)
                break
            except subprocess.TimeoutExpired:
                drain()
                if time.time() - last_new > args.stall_min * 60 and not stall_alerted:
                    run.alert(
                        "training stalled?",
                        f"006-std: {args.stall_min:.0f} 分以上新行なし "
                        f"(同期済み {n_logged - start_rows} 行)",
                    )
                    stall_alerted = True

        drain()
        run.log({"trainer_exit_code": rc})
        if cancelled:
            raise RuntimeError(f"training cancelled by signal (trainer rc={rc})")
        if rc != 0:
            raise RuntimeError(f"trainer failed with exit code {rc}")
        print(f"trainer finished (rc=0); synced {n_logged - start_rows} rows", flush=True)


if __name__ == "__main__":
    main()
