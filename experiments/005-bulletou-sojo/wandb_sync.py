"""BulletOu summary-learn.log を tail して W&B へ同期する。

BulletOu (Rust) は wandb を知らないので、checkpoint ディレクトリの
`summary-learn.log` (superbatch 境界ごとに 1 行の CSV) をポーリングして
`tools.wandb_utils.init_run()` 経由でログする。

- 起動のたびに新規 run を作り、ログ全行をリプレイする (途中再起動しても
  最新 run が常に完全な履歴を持つ。古い部分 run は無視してよい)
- 学習停滞 (--stall-min 以上新行なし) と loss の NaN を run.alert で通知
- 学習プロセスとは独立に動く (GPU 不要なのでサンドボックス内で実行できる)

使い方:
    uv run python experiments/005-bulletou-sojo/wandb_sync.py --arm main
    uv run python experiments/005-bulletou-sojo/wandb_sync.py --arm smoke --smoke --once
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import time
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.wandb_utils import init_run  # noqa: E402

EXPERIMENT = "005-bulletou-sojo"

# やねうら氏共有レシピ (run_training.sh と一致させる。wandb config 記録用)
RECIPE = {
    "trainer": "BulletOu 9577f08 (cuda-cpp)",
    "arch": "SFNN_halfka2_1024_7_64_k3k3",
    "positions_per_superbatch": 40_000_000,
    "superbatches": 12,
    "max_epochs": 16,
    "lr": 0.000875,
    "lr_min": 0.000030,
    "lr_schedule": "step",
    "optimizer": "ranger",
    "optimizer_weight_decay": 0.0,
    "batch_size": 65536,
    "teacher": "sojo (washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique)",
    "test_teacher": "takaoyamaoka/floodgate.hcpe (856,923 局面, floodgate 2017-2018 R3500+)",
    "test_positions": 300_000,
}


def parse_row(row: dict[str, str]) -> dict[str, float] | None:
    """summary-learn.log の 1 行を wandb メトリクス dict へ。

    数値でない行 (書き込み途中で千切れた末尾行を含む) は None。
    千切れ行は次のポーリングで完全な形で読み直されるので捨ててよい。
    """
    metrics: dict[str, float] = {}
    try:
        for key in ("epoch", "superbatch", "positions"):
            if row.get(key) is None:
                return None
            metrics[key] = int(row[key])
        for key in (
            "train_value_loss",
            "lr_start",
            "lr_end",
            "test_value_accuracy",
            "test_value_loss",
        ):
            val = row.get(key)
            if val not in (None, "", "-"):
                metrics[key] = float(val)
    except ValueError:
        return None
    return metrics


def read_rows(log_path: Path) -> list[dict[str, float]]:
    with log_path.open(newline="") as fh:
        return [m for r in csv.DictReader(fh) if (m := parse_row(r)) is not None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["smoke", "main", "var-b", "var-c"])
    ap.add_argument("--interval", type=float, default=60.0, help="ポーリング間隔 (秒)")
    ap.add_argument("--stall-min", type=float, default=45.0, help="この分数新行がなければ alert")
    ap.add_argument("--once", action="store_true", help="現在のログを 1 回同期して終了")
    ap.add_argument("--smoke", action="store_true", help="wandb へ送らない動作確認モード")
    ap.add_argument(
        "--expect-rows",
        type=int,
        default=0,
        help="この行数まで同期したらクリーン終了 (例: 12sb×16epoch なら 192)。"
        "0 = 無期限 (外部から SIGTERM で停止)",
    )
    args = ap.parse_args()

    # TaskStop 等の SIGTERM で run が 'Crashed' 表示にならないよう、
    # ループを抜けて init_run の正常終了パス (run.finish()) を通す
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    log_path = REPO / "data" / "bulletou" / "checkpoints" / f"005-{args.arm}" / "summary-learn.log"
    config = {**RECIPE, "arm": args.arm, "log_path": str(log_path)}
    if args.arm == "smoke":
        config.update(
            positions_per_superbatch=2_000_000,
            superbatches=2,
            max_epochs=1,
            test_positions=50_000,
            teacher="sojo 030 (smoke)",
        )
    if args.arm == "var-b":
        config["teacher_order"] = "rotated: 009..016,001..008"
    if args.arm == "var-c":
        config["teacher_order"] = "rotated: 005..016,001..004"

    with init_run(EXPERIMENT, config=config, tags=[args.arm], smoke=args.smoke) as run:
        print(f"wandb run: {getattr(run.run, 'url', None) or run.run.name}")
        n_logged = 0
        last_new = time.time()
        stall_alerted = False
        while True:
            rows = read_rows(log_path) if log_path.exists() else []
            for step, m in enumerate(rows[n_logged:], start=n_logged + 1):
                if any(isinstance(v, float) and math.isnan(v) for v in m.values()):
                    run.alert("NaN detected", f"arm={args.arm} step={step}: {m}")
                run.log(m, step=step)
            if len(rows) > n_logged:
                n_logged = len(rows)
                last_new = time.time()
                stall_alerted = False
                print(f"synced {n_logged} rows (latest: {rows[-1]})")
            elif time.time() - last_new > args.stall_min * 60 and not stall_alerted:
                run.alert(
                    "training stalled?",
                    f"arm={args.arm}: {args.stall_min:.0f} 分以上 summary-learn.log に新行なし "
                    f"(同期済み {n_logged} 行)",
                )
                stall_alerted = True
            if args.once or stop_requested:
                break
            if args.expect_rows and n_logged >= args.expect_rows:
                print(f"expected {args.expect_rows} rows synced; finishing run cleanly")
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
