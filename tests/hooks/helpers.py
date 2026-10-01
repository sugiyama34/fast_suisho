"""Claude Code hook (.claude/hooks/*.sh) のテスト用ヘルパー。

hook は一時ディレクトリに作った使い捨ての git リポジトリに対して実行する。
ユーザーのグローバル git 設定 (core.hooksPath, commit 署名など) の影響を受けないよう、
git と hook の両方を GIT_CONFIG_GLOBAL=/dev/null, GIT_CONFIG_NOSYSTEM=1 で動かす。
"""

import json
import os
import subprocess
from pathlib import Path

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"

GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "hook-test",
    "GIT_AUTHOR_EMAIL": "hook-test@example.com",
    "GIT_COMMITTER_NAME": "hook-test",
    "GIT_COMMITTER_EMAIL": "hook-test@example.com",
}


def git(repo: Path, *args: str) -> str:
    """一時リポジトリで git を実行して stdout を返す。"""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        env={**os.environ, **GIT_ENV},
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def run_hook(
    name: str,
    cwd: Path,
    project_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
    event: str = "PostToolUse",
) -> subprocess.CompletedProcess[str]:
    """hook を Claude Code と同じ形 (stdin に JSON、cwd、CLAUDE_PROJECT_DIR) で実行する。"""
    env = {**os.environ, **GIT_ENV, **(extra_env or {})}
    env["CLAUDE_PROJECT_DIR"] = str(project_dir or cwd)
    payload = {"session_id": "test", "cwd": str(cwd), "hook_event_name": event}
    if event == "PostToolUse":
        payload |= {"tool_name": "Bash", "tool_input": {"command": "true"}}
    else:
        payload["source"] = "startup"
    return subprocess.run(
        [str(HOOKS_DIR / name)],
        input=json.dumps(payload),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
