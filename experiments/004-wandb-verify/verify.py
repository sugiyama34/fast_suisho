"""WandB 統合の検証スクリプト (docs/wandb-guide.md §4)。

service account キーで以下を実測する:
  A. smoke モード (mode="disabled") の no-op 動作
  B. suisho/suisho-test への実ログ (初回 run)
  C. チーム境界 — 他 entity への書き込みが拒否されること
  D. 削除権限 — SA キーで自分の run を削除できるか (ドキュメント未記載の実測)
  E. alert 配信 — email 帰属付きで run.alert() が届くか

使い方: uv run python experiments/004-wandb-verify/verify.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import wandb  # noqa: E402

from tools.wandb_utils import init_run  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'OK' if ok else 'NG'}] {name}: {detail}", flush=True)


def main() -> None:
    # A. smoke モード — クラウドに一切送らない no-op
    try:
        with init_run("004-wandb-verify", {"purpose": "smoke"}, smoke=True) as r:
            r.log({"x": 1})
        record("A_smoke_disabled", True, "no-op completed")
    except Exception as e:  # noqa: BLE001
        record("A_smoke_disabled", False, f"{type(e).__name__}: {e}")

    # B. 実ログ (初回 run)
    run_url = ""
    try:
        with init_run("004-wandb-verify", {"purpose": "first-run"}, job_type="verify") as r:
            for i in range(3):
                r.log({"loss": 1.0 / (i + 1), "nps": 1000 * (i + 1)}, step=i)
            run_url = r.run.url or ""
        record("B_online_run", True, run_url)
    except Exception as e:  # noqa: BLE001
        record("B_online_run", False, f"{type(e).__name__}: {e}")

    # C. チーム境界 — 他 entity (公開 org "wandb") への project 作成は拒否されるはず
    try:
        wandb.Api().create_project("fast-suisho-boundary-probe", "wandb")
        record("C_team_boundary", False, "UNEXPECTED: SA が他 entity に書き込めた")
    except Exception as e:  # noqa: BLE001
        record("C_team_boundary", True, f"correctly denied ({type(e).__name__})")

    # D. 削除権限の実測 — 捨て run を作って SA キーで削除を試す
    scrap_path = ""
    try:
        with init_run("004-wandb-verify", {"purpose": "delete-probe"}) as r:
            r.log({"x": 0})
            # run.path は "entity/project/run_id" 形式の文字列
            scrap_path = str(r.run.path)
        time.sleep(10)  # backend 反映待ち
        wandb.Api().run(scrap_path).delete()
        record("D_sa_delete", True, f"SA CAN delete its own run ({scrap_path})")
    except Exception as e:  # noqa: BLE001
        record("D_sa_delete", False, f"delete denied for {scrap_path}: {type(e).__name__}: {e}")

    # E. alert 配信 — email 帰属を付けて送信 (届いたかは人間が確認)
    try:
        os.environ["WANDB_USER_EMAIL"] = "sugiyama.satoshi1990@gmail.com"
        with init_run("004-wandb-verify", {"purpose": "alert-test"}) as r:
            r.alert(
                "WandB 統合検証",
                "fast_suisho: alert 配信テスト。届いていればこのテストは成功です。",
            )
        record("E_alert_sent", True, "sent — check email/Slack for delivery")
    except Exception as e:  # noqa: BLE001
        record("E_alert_sent", False, f"{type(e).__name__}: {e}")

    print("\n=== summary ===")
    for name, ok, detail in RESULTS:
        print(f"{'OK' if ok else 'NG'} {name} {detail}")


if __name__ == "__main__":
    main()
