"""WandB ロギングヘルパー (方式 B: 専用チーム + team-scoped service account で運用)。

設計・導入手順・運用規約の全体は docs/wandb-guide.md を参照。

実験スクリプト側の規約:
- `wandb.init()` を直接呼ばず、必ず `init_run()` を使う
- entity / project は環境変数 (`WANDB_ENTITY` / `WANDB_PROJECT`) 側で固定し、
  スクリプトからは指定しない
- run id は毎回新規生成する (resume による履歴上書き事故の防止)
- メトリクスはローカル JSONL にも二重記録する (エージェントの分析は
  クラウド API ではなくローカルファイルを読む)
- checkpoint artifact は best / final のみ無期限、中間 checkpoint は TTL 付き

使い方:
    from tools.wandb_utils import init_run

    with init_run("003-quantize", config={"lr": 1e-3}) as run:
        for step in range(1000):
            run.log({"loss": loss}, step=step)
        run.log_checkpoint(Path("ckpt/best.bin"), best=True)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

# チーム entity / project の既定値。settings.json の env ブロックで
# WANDB_ENTITY / WANDB_PROJECT が設定されていればそちらが優先される。
DEFAULT_ENTITY = "suisho"
DEFAULT_PROJECT = "suisho-test"  # 検証完了後に本番 project 名へ切り替える

# ローカル JSONL の出力先を CWD に依存させないためのリポジトリルート
REPO_ROOT = Path(__file__).resolve().parents[1]

# 中間 checkpoint artifact の保持期間 (best/final は無期限)
INTERMEDIATE_CKPT_TTL_DAYS = 14


class RunHandle:
    """wandb run とローカル JSONL の二重ロガー。"""

    def __init__(self, run: Any, jsonl_path: Path) -> None:
        self._run = run
        self._jsonl_path = jsonl_path
        self._jsonl = jsonl_path.open("a", encoding="utf-8")

    @property
    def run(self) -> Any:
        """生の wandb run オブジェクト (URL 取得等に使う)。"""
        return self._run

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        record: dict[str, Any] = {"_time": time.time()}
        if step is not None:
            record["_step"] = step
        record.update(metrics)
        self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        self._run.log(metrics, step=step)

    def log_checkpoint(
        self, path: Path, *, best: bool = False, final: bool = False, name: str | None = None
    ) -> None:
        """checkpoint を artifact として記録する。best / final 以外は TTL 付き。"""
        import wandb

        artifact = wandb.Artifact(
            name=name or f"{self._run.name}-ckpt",
            type="checkpoint",
            metadata={"best": best, "final": final},
        )
        if not (best or final):
            artifact.ttl = timedelta(days=INTERMEDIATE_CKPT_TTL_DAYS)
        artifact.add_file(str(path))
        aliases = (["best"] if best else []) + (["final"] if final else [])
        self._run.log_artifact(artifact, aliases=aliases or None)

    def alert(self, title: str, text: str) -> None:
        """学習発散等の通知。service account run では届かない可能性あり
        (docs/wandb-guide.md の未確定事項参照)。ローカル JSONL には常に残す。"""
        self.log({"_alert": f"{title}: {text}"})
        self._run.alert(title=title, text=text)

    def close(self) -> None:
        self._jsonl.close()


@contextmanager
def init_run(
    experiment: str,
    config: dict[str, Any] | None = None,
    *,
    job_type: str = "train",
    tags: list[str] | None = None,
    smoke: bool = False,
    log_dir: Path | None = None,
) -> Iterator[RunHandle]:
    """実験 run を開始する。

    Args:
        experiment: 実験フォルダ名 (例: "003-quantize")。run 名と tag に使う。
        config: ハイパーパラメータ等。
        job_type: "train" / "eval" / "bench" など。
        smoke: True なら動作確認モード (mode="disabled"、クラウドに一切送らない)。
        log_dir: ローカル JSONL の出力先。省略時は experiments/<experiment>/。
    """
    # 未インストール環境でも tools パッケージ自体は壊さないよう遅延 import
    import wandb

    # resume 上書き事故防止: 外から run id が来ていても使わず毎回新規にする
    os.environ.pop("WANDB_RUN_ID", None)

    mode = "disabled" if smoke else os.environ.get("WANDB_MODE", "online")
    run_name = f"{experiment}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    log_dir = log_dir or REPO_ROOT / "experiments" / experiment
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_dir / f"metrics-{run_name}.jsonl"

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY),
        project=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
        name=run_name,
        job_type=job_type,
        tags=[experiment, *(tags or [])],
        config=config,
        mode=mode,
        resume="never",
        settings=wandb.Settings(quiet=True),
    )
    handle = RunHandle(run, jsonl_path)
    try:
        yield handle
    except BaseException:
        # 本文の例外は W&B 上でも失敗 (crashed) として記録する
        try:
            run.finish(exit_code=1)
        finally:
            handle.close()
        raise
    try:
        run.finish()
    finally:
        handle.close()
