"""誤起動などで生じたゴミ W&B run を 1 件削除するメンテスクリプト。

使い方:
    uv run python experiments/007-wrm-factorizer/delete_wandb_run.py <run_id>          # 確認表示のみ
    uv run python experiments/007-wrm-factorizer/delete_wandb_run.py <run_id> --apply  # 削除
"""

from __future__ import annotations

import argparse
import os

import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "suisho")
PROJECT = os.environ.get("WANDB_PROJECT", "20260728_BulletOu_with_Sojo_data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id")
    ap.add_argument("--apply", action="store_true", help="指定がなければ表示のみ")
    args = ap.parse_args()

    run = wandb.Api().run(f"{ENTITY}/{PROJECT}/{args.run_id}")
    print(f"run: {run.name}  state={run.state}  group={run.group!r}  url={run.url}")
    if args.apply:
        run.delete()
        print("-> 削除した")
    else:
        print("-> --apply で削除")


if __name__ == "__main__":
    main()
