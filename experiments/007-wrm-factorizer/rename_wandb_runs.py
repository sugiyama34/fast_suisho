"""007 の W&B run 名を arm が判別できる表示名に揃えるメンテスクリプト。

run 名の既定値は `007-wrm-factorizer-<timestamp>-<id>` で全 arm 同じ見た目に
なるため、006 と同様の「arm (要点)」形式に揃える (job_type = arm で対象を特定)。

使い方:
    uv run python experiments/007-wrm-factorizer/rename_wandb_runs.py           # 表示のみ
    uv run python experiments/007-wrm-factorizer/rename_wandb_runs.py --apply   # 反映
"""

from __future__ import annotations

import argparse
import os

import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "suisho")
PROJECT = os.environ.get("WANDB_PROJECT", "20260728_BulletOu_with_Sojo_data")
GROUP = "007-wrm-factorizer"

# arm (= job_type) -> 表示名
NAMES = {
    "ctrl": "007-ctrl (sb=108 x 20ep, 006再現@2a8e5ed)",
    "wrm": "007-wrm (win-rate-model pow=2.5)",
    "nofact": "007-nofact (factorizer none)",
    "both": "007-both (wrm + nofact)",
    "lr1shot": "007-lr1shot (sb=2160 x 1ep 単発アニール)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="指定がなければ表示のみ")
    args = ap.parse_args()

    for run in wandb.Api().runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}):
        new = NAMES.get(run.job_type)
        print(f"{run.name}  job_type={run.job_type!r}  state={run.state}")
        if new is None:
            print("    -> arm 不明、スキップ")
            continue
        if run.name == new:
            print("    -> 既に設定済み")
            continue
        if args.apply:
            run.name = new
            run.update()
            print(f"    -> {new}")
        else:
            print(f"    -> (dry-run) {new}")


if __name__ == "__main__":
    main()
