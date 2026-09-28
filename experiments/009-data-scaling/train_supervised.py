"""experiment-009 学習の supervisor (設計は experiment-006 の train_supervised.py と同一)。

トレーナと W&B run を単一ライフサイクルで所有: トレーナ exit 0 → Finished /
非 0 (またはキャンセル) → Failed。summary-learn.log の新しい行を W&B に流す。

W&B: project は settings.json の env (`data_ablation_study`)、group 相当として tag に
実験名と arm を付ける。API キーがこのプロセスの環境に無い場合は offline モードで記録し、
後でユーザーが自分の端末から `wandb sync wandb/offline-run-*` で送る
(キーはリポジトリ・settings.json に置かない方針, 2026-09-28 ユーザー決定)。

使い方:
    uv run python experiments/009-data-scaling/train_supervised.py --arm p10 \
        --gpu 0
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "005-bulletou-sojo"))
sys.path.insert(0, str(HERE))

from subsets import ARMS, POS_PER_FILE  # noqa: E402
from wandb_sync import read_rows  # noqa: E402  (005 の CSV パーサを再利用)

from tools.wandb_utils import init_run  # noqa: E402

EXPERIMENT = "009-data-scaling"
SB = 108
POS_PER_SB = 39_976_960  # 40M を batch 65536 の倍数に切り下げた実効値

CONFIG = {
    "trainer": "BulletOu 2a8e5ed (cuda-cpp, sm_120 patch)",
    "arch": "SFNN_halfka2_1024_7_64_k3k3",
    "loss": "sigmoid-MSE (WRM なし, lambda 1.0, scale 290)",
    "test_teacher": "takaoyamaoka/floodgate.hcpe (300k/validation, rate 4sb, test-seed 20260928)",
    "positions_per_superbatch": 40_000_000,
    "superbatches": SB,
    "lr": 0.000875,
    "lr_min": 0.000030,
    "lr_schedule": "step (1 BulletOu-epoch = 108 sb サイクル, epoch 境界で warm restart)",
    "optimizer": "ranger",
    "batch_size": 65536,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--gpu", required=True, help="CUDA_VISIBLE_DEVICES の値 (番号 or UUID)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--stall-min", type=float, default=45.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")

    files = ARMS[args.arm]
    teacher_pos = len(files) * POS_PER_FILE
    total_pos = args.epochs * SB * POS_PER_SB
    # --smoke は W&B を無効にし、トレーナも SMOKE=1 (sb=2, 出力 009-<arm>-smoke/) で動かす
    out_name = f"009-{args.arm}-smoke" if args.smoke else f"009-{args.arm}"
    log_path = REPO / "data" / "bulletou" / "checkpoints" / out_name / "summary-learn.log"
    # smoke は出力を消してから始まるので既存行を数えない
    start_rows = len(read_rows(log_path)) if log_path.exists() and not args.smoke else 0

    cmd = ["bash", str(HERE / "run_training.sh"), args.arm, args.gpu, str(args.epochs)]
    config = {
        **CONFIG,
        "arm": args.arm,
        "teacher_files": " ".join(f"{n:03d}" for n in files),
        "teacher_n_files": len(files),
        "teacher_positions": teacher_pos,
        "teacher_fraction": len(files) / 30,
        "max_epochs": args.epochs,
        "total_positions": total_pos,
        "passes": total_pos / teacher_pos,
        "gpu": args.gpu,
        "log_path": str(log_path),
    }

    with init_run(
        EXPERIMENT, config=config, tags=[args.arm, "supervised"], smoke=args.smoke
    ) as run:
        print(f"wandb run: {getattr(run.run, 'url', None) or run.run.name}", flush=True)
        env = {**os.environ, "SMOKE": "1"} if args.smoke else None
        proc = subprocess.Popen(cmd, start_new_session=True, env=env)
        cancelled = False

        def forward_signal(_signum, _frame) -> None:
            nonlocal cancelled
            cancelled = True
            os.killpg(proc.pid, signal.SIGTERM)  # bulletou と tee も止める

        signal.signal(signal.SIGTERM, forward_signal)
        signal.signal(signal.SIGINT, forward_signal)

        n_logged = start_rows
        last_new = time.time()
        stall_alerted = False

        def drain() -> None:
            nonlocal n_logged, last_new, stall_alerted
            rows = read_rows(log_path) if log_path.exists() else []
            for step, m in enumerate(rows[n_logged:], start=n_logged + 1):
                if any(isinstance(v, float) and math.isnan(v) for v in m.values()):
                    run.alert("NaN detected", f"009-{args.arm} step={step}: {m}")
                if "positions" in m:
                    m["passes"] = m["positions"] / teacher_pos
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
                        f"009-{args.arm}: {args.stall_min:.0f} 分以上新行なし",
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
