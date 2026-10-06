"""experiment-011 学習の supervisor (experiment-009 の train_supervised.py と同じ構造)。

トレーナと W&B run を単一ライフサイクルで所有: トレーナ exit 0 → Finished / 非 0 (またはキャンセル) → Failed。
summary-learn.log の新しい行を W&B に流す。

W&B: entity `suisho`、project `pcalm_vs_backprop`、group `011-pcalm` (環境変数で上書き可)。
API キーがこのプロセスの環境に無い場合は offline で記録し、後で `uv run wandb sync wandb/offline-run-*` で送る
(キーはリポジトリ・settings.json に置かない, 2026-09-28 ユーザー決定)。

使い方:
    uv run python experiments/011-pcalm/train_supervised.py --arm s-bp-lr1 --gpu <MIG UUID>
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "005-bulletou-sojo"))
sys.path.insert(0, str(HERE))

from arms import POS_PER_FILE, parse  # noqa: E402
from wandb_sync import read_rows  # noqa: E402  (005 の CSV パーサを再利用)

from tools.wandb_utils import init_run  # noqa: E402

EXPERIMENT = "011-pcalm"
SB = 108
POS_PER_SB = 39_976_960  # 40M を batch 65536 の倍数に切り下げた実効値

CONFIG = {
    "trainer": "BulletOu 2a8e5ed (cuda-cpp, sm_120)。pcalm / pc は experiment-011 の改造版 (--credit pcalm)",
    "arch": "SFNN_halfka2_1024_7_64_k3k3",
    "loss": "sigmoid-MSE (WRM なし, lambda 1.0, scale 290)",
    "test_teacher": "takaoyamaoka/floodgate.hcpe (300k/validation, rate 4sb, test-seed 20260928)",
    "positions_per_superbatch": 40_000_000,
    "superbatches": SB,
    "lr_schedule": "step (1 BulletOu-epoch = 108 sb サイクル, epoch 境界で warm restart)",
    "optimizer": "ranger",
    "batch_size": 65536,
    "server": "kajiki (MIG 1g.24gb)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, help="arms.py の文法 (例 s-bp-lr1)")
    ap.add_argument("--gpu", required=True, help="CUDA_VISIBLE_DEVICES の値 (MIG の UUID)")
    ap.add_argument(
        "--epochs", type=int, default=None, help="省略時は arm の既定 (小規模 1, 本番 20)"
    )
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--stall-min", type=float, default=45.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("WANDB_RUN_GROUP", EXPERIMENT)
    # settings.json の既定 project (data_ablation_study) より 011 の project を優先する
    os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT_011", "pcalm_vs_backprop")
    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")

    arm = parse(args.arm)
    epochs = args.epochs or arm.epochs
    teacher_pos = len(arm.files) * POS_PER_FILE
    total_pos = epochs * SB * POS_PER_SB
    out_name = f"011-{args.arm}-smoke" if args.smoke else f"011-{args.arm}"
    log_path = REPO / "data" / "bulletou" / "checkpoints" / out_name / "summary-learn.log"
    start_rows = len(read_rows(log_path)) if log_path.exists() and not args.smoke else 0

    cmd = ["bash", str(HERE / "run_training.sh"), args.arm, args.gpu, str(epochs)]
    config = {
        **CONFIG,
        **{k: v for k, v in asdict(arm).items() if k != "files"},
        "teacher_files": " ".join(f"{n:03d}" for n in arm.files),
        "teacher_n_files": len(arm.files),
        "teacher_positions": teacher_pos,
        "max_epochs": epochs,
        "total_positions": total_pos,
        "passes": total_pos / teacher_pos,
        "gpu": args.gpu,
        "log_path": str(log_path),
    }

    with init_run(
        EXPERIMENT, config=config, tags=[args.arm, arm.method, arm.scale], smoke=args.smoke
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
                    run.alert("NaN detected", f"011-{args.arm} step={step}: {m}")
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
                        "training stalled?", f"011-{args.arm}: {args.stall_min:.0f} 分以上新行なし"
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
