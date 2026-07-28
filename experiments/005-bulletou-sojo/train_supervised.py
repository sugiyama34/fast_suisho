"""BulletOu 学習の supervisor: トレーナと W&B run を単一ライフサイクルで所有する。

wandb_sync.py (独立ポーラ) の後継で、学習の標準エントリポイント。
設計は Codex 相談 (2026-07-28) で確定した「プロセス寿命 = 完了プロトコル」方式:

- init_run() に入ってから run_training.sh をサブプロセスとして起動
- 生存中は summary-learn.log の新規行を run.log() へ転送 (NaN/停滞 alert 付き)
- トレーナ終了で最終 drain → exit code 0 なら正常脱出 = **Finished**、
  非 0 なら例外 = **Failed**。外部 kill や行数の当て推量に依存しない
- SIGTERM/SIGINT は「本物のキャンセル」: 子プロセスへ転送して待ち、Failed で閉じる
- 起動時に既存 CSV の行数を基準点として記録し、過去 run の行が混入しないようにする

使い方 (GPU が見える環境で):
    uv run python experiments/005-bulletou-sojo/train_supervised.py --arm main --gpu 0
    uv run python experiments/005-bulletou-sojo/train_supervised.py --arm smoke --smoke
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
sys.path.insert(0, str(HERE))

from wandb_sync import EXPERIMENT, RECIPE, read_rows  # noqa: E402

from tools.wandb_utils import init_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["smoke", "main", "var-b", "var-c"])
    ap.add_argument("--gpu", default=None, help="GPU index (省略時は run_training.sh の既定)")
    ap.add_argument("--interval", type=float, default=30.0, help="CSV ポーリング間隔 (秒)")
    ap.add_argument("--stall-min", type=float, default=45.0, help="この分数新行がなければ alert")
    ap.add_argument("--smoke", action="store_true", help="wandb へ送らない動作確認モード")
    args = ap.parse_args()

    log_path = REPO / "data" / "bulletou" / "checkpoints" / f"005-{args.arm}" / "summary-learn.log"
    config = {**RECIPE, "arm": args.arm, "log_path": str(log_path), "entry": "train_supervised"}
    if args.arm == "smoke":
        config.update(
            positions_per_superbatch=2_000_000,
            superbatches=2,
            max_epochs=1,
            test_positions=50_000,
        )
    if args.arm == "var-b":
        config["teacher_order"] = "rotated: 009..016,001..008"
    if args.arm == "var-c":
        config["teacher_order"] = "rotated: 005..016,001..004"

    # 過去 run の行を新 run に混入させないための基準点
    start_rows = len(read_rows(log_path)) if log_path.exists() else 0

    cmd = ["bash", str(HERE / "run_training.sh"), args.arm]
    if args.gpu is not None:
        cmd.append(str(args.gpu))

    with init_run(
        EXPERIMENT, config=config, tags=[args.arm, "supervised"], smoke=args.smoke
    ) as run:
        print(f"wandb run: {getattr(run.run, 'url', None) or run.run.name}")
        proc = subprocess.Popen(cmd, start_new_session=True)

        # キャンセル (SIGTERM/SIGINT) はトレーナへ転送し、下の wait ループに
        # 「子プロセスの終了」として観測させる (正常完了とは exit code で区別)
        cancelled = False

        def forward_signal(signum, _frame) -> None:
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
                    run.alert("NaN detected", f"arm={args.arm} step={step}: {m}")
                run.log(m, step=step)
            if len(rows) > n_logged:
                n_logged = len(rows)
                last_new = time.time()
                stall_alerted = False

        while True:
            try:
                rc = proc.wait(timeout=args.interval)
                break  # トレーナ終了 (正常/異常問わず) → 最終 drain へ
            except subprocess.TimeoutExpired:
                drain()
                if time.time() - last_new > args.stall_min * 60 and not stall_alerted:
                    run.alert(
                        "training stalled?",
                        f"arm={args.arm}: {args.stall_min:.0f} 分以上新行なし "
                        f"(同期済み {n_logged - start_rows} 行)",
                    )
                    stall_alerted = True

        drain()
        run.log({"trainer_exit_code": rc})
        if cancelled:
            raise RuntimeError(f"training cancelled by signal (trainer rc={rc})")
        if rc != 0:
            raise RuntimeError(f"trainer failed with exit code {rc}")
        print(f"trainer finished (rc=0); synced {n_logged - start_rows} rows; run -> Finished")


if __name__ == "__main__":
    main()
