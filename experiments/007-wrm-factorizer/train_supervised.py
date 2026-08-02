"""exp-007 wrm-factorizer 学習の supervisor (設計は 006/train_supervised.py と同一)。

トレーナと W&B run を単一ライフサイクルで所有: トレーナ exit 0 → Finished /
非 0 (またはキャンセル) → Failed。

W&B 規約 (2026-07-31 決定): project は既存の `20260728_BulletOu_with_Sojo_data`
を継続使用し、1 実験 = 1 group とする (本実験は group=007-wrm-factorizer、
arm は job_type / tag で区別)。

使い方 (GPU が見える環境で):
    uv run python experiments/007-wrm-factorizer/train_supervised.py --arm ctrl --gpu 0
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

EXPERIMENT = "007-wrm-factorizer"

ARMS = {
    # 006 レシピの BulletOu 2a8e5ed 再現 (コミット更新の交絡を分離するコントロール)
    "ctrl": {"superbatches": 108, "max_epochs": 20, "extra_flags": "none"},
    # 提案1: tatara 系 win-rate-model loss (pow 2.5)
    "wrm": {
        "superbatches": 108,
        "max_epochs": 20,
        "extra_flags": "--win-rate-model --loss-pow-exp 2.5",
    },
    # 提案3: L1/L2/L3 factorizer 無効化
    "nofact": {"superbatches": 108, "max_epochs": 20, "extra_flags": "--sfnn-factorizer none"},
    # 提案1+3 併用
    "both": {
        "superbatches": 108,
        "max_epochs": 20,
        "extra_flags": "--win-rate-model --loss-pow-exp 2.5 --sfnn-factorizer none",
    },
    # ユーザー提案: warm restart なしの単発アニール (総局面数は他 arm と同一)
    "lr1shot": {"superbatches": 2160, "max_epochs": 1, "extra_flags": "none (save-rate 108)"},
    # フェーズ3: 勝者 2 lever の併用 (WRM loss × 単発アニール)
    "wrmshot": {
        "superbatches": 2160,
        "max_epochs": 1,
        "extra_flags": "--win-rate-model --loss-pow-exp 2.5 (save-rate 108)",
    },
}

# W&B 上の表示名 (rename_wandb_runs.py と同一マップ)。既定の
# `007-wrm-factorizer-<ts>-<id>` では arm が判別できないため起動時に設定する
DISPLAY_NAMES = {
    "ctrl": "007-ctrl (sb=108 x 20ep, 006再現@2a8e5ed)",
    "wrm": "007-wrm (win-rate-model pow=2.5)",
    "nofact": "007-nofact (factorizer none)",
    "both": "007-both (wrm + nofact)",
    "lr1shot": "007-lr1shot (sb=2160 x 1ep 単発アニール)",
    "wrmshot": "007-wrmshot (WRM pow=2.5 + 単発アニール)",
}

CONFIG = {
    "trainer": "BulletOu 2a8e5ed (cuda-cpp, sm_120 patch)",
    "arch": "SFNN_halfka2_1024_7_64_k3k3",
    "teacher": "sojo full 30 files = 14.669B positions",
    "test_teacher": "takaoyamaoka/floodgate.hcpe (300k/validation, rate 4sb)",
    "positions_per_superbatch": 40_000_000,
    "total_positions": "86.4B ≒ 5.9 standard-epoch (全 arm 共通)",
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

    log_path = REPO / "data" / "bulletou" / "checkpoints" / f"007-{args.arm}" / "summary-learn.log"
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
        EXPERIMENT,
        config=config,
        job_type=args.arm,
        tags=[args.arm, "supervised"],
        group=EXPERIMENT,
        smoke=args.smoke,
    ) as run:
        run.run.name = DISPLAY_NAMES[args.arm]
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
                    run.alert("NaN detected", f"007-{args.arm} step={step}: {m}")
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
                        f"007-{args.arm}: {args.stall_min:.0f} 分以上新行なし "
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
