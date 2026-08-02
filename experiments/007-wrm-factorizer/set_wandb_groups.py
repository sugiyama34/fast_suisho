"""既存 W&B run に group を後付けする一回きりのメンテスクリプト。

2026-07-31 の規約決定 (1 project = 実験系列 / 1 group = 1 実験) に合わせ、
project `20260728_BulletOu_with_Sojo_data` 内の既存 run (= experiment-006 の
学習 run) に group `006-standard-epoch` を設定する。

使い方:
    uv run python experiments/007-wrm-factorizer/set_wandb_groups.py           # 一覧のみ
    uv run python experiments/007-wrm-factorizer/set_wandb_groups.py --apply   # 反映
"""

from __future__ import annotations

import argparse
import os

import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "suisho")
PROJECT = os.environ.get("WANDB_PROJECT", "20260728_BulletOu_with_Sojo_data")
TARGET_GROUP = "006-standard-epoch"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="指定がなければ一覧表示のみ")
    args = ap.parse_args()

    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    print(f"{ENTITY}/{PROJECT}: {len(runs)} runs")
    for run in runs:
        print(f"  {run.name}  group={run.group!r}  job_type={run.job_type!r}  state={run.state}")
        if not args.apply:
            continue
        if run.group:
            print("    -> 既に group あり、スキップ")
            continue
        run.group = TARGET_GROUP
        run.update()
        print(f"    -> group={TARGET_GROUP} を設定")


if __name__ == "__main__":
    main()
