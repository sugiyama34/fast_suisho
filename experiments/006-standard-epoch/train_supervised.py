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

ARMS = {
    # やねうらお氏の実設定の再現 (2026-07-28 本人コメントで判明)
    "yane": {
        "superbatches": 108,
        "max_epochs": 16,
        "total_positions": "69.1B ≒ 4.7 standard-epoch",
    },
    # 氏の「20 epoch ぐらい回さないといかん」指針
    "yane20": {
        "superbatches": 108,
        "max_epochs": 20,
        "total_positions": "86.4B ≒ 5.9 standard-epoch",
    },
    # yane20 完了後の追加学習 (+12 epoch = 累積 32)。test accuracy がまだ伸びていたため
    "yane32": {
        "superbatches": 108,
        "max_epochs": "12 (resume; cumulative 32)",
        "total_positions": "138.2B ≒ 9.4 standard-epoch (cumulative)",
    },
    # 「sb 上げればもっと強い」の検証 (LR 周期 ≒ 教師 1 周)
    "std": {"superbatches": 367, "max_epochs": 16, "total_positions": "234.7B ≒ 16 standard-epoch"},
}

CONFIG = {
    "trainer": "BulletOu 9577f08 (cuda-cpp, sm_120 patch)",
    "arch": "SFNN_halfka2_1024_7_64_k3k3",
    "teacher": "sojo full 30 files = 14.669B positions",
    "test_teacher": "takaoyamaoka/floodgate.hcpe (300k/validation, rate 4sb)",
    "positions_per_superbatch": 40_000_000,
    "max_epochs": 16,
    "lr": 0.000875,
    "lr_min": 0.000030,
    "lr_schedule": "step (1 BulletOu-epoch = superbatches サイクル)",
    "optimizer": "ranger",
    "batch_size": 65536,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--stall-min", type=float, default=45.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # yane32 は yane20 と同じ出力ディレクトリに追記する (resume 継続学習)
    out_arm = "yane20" if args.arm == "yane32" else args.arm
    log_path = REPO / "data" / "bulletou" / "checkpoints" / f"006-{out_arm}" / "summary-learn.log"
    start_rows = len(read_rows(log_path)) if log_path.exists() else 0

    cmd = ["bash", str(HERE / "run_training.sh"), args.arm, str(args.gpu)]
    config = {
        **CONFIG,
        **ARMS[args.arm],
        "arm": args.arm,
        "gpu": args.gpu,
        "log_path": str(log_path),
    }

    with init_run(
        EXPERIMENT, config=config, tags=[args.arm, "supervised"], smoke=args.smoke
    ) as run:
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
                    run.alert("NaN detected", f"006-{args.arm} step={step}: {m}")
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
                        f"006-{args.arm}: {args.stall_min:.0f} 分以上新行なし "
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
