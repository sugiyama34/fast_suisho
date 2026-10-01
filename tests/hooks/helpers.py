"""Claude Code hook (.claude/hooks/*.sh) のテスト用ヘルパー。

ユーザーのグローバル git 設定 (core.hooksPath, commit 署名など) の影響を受けないよう、
git と hook を GIT_CONFIG_GLOBAL=/dev/null, GIT_CONFIG_NOSYSTEM=1 で動かす。
"""

import json
import os
import subprocess
from pathlib import Path

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"

ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "hook-test",
    "GIT_AUTHOR_EMAIL": "hook-test@example.com",
    "GIT_COMMITTER_NAME": "hook-test",
    "GIT_COMMITTER_EMAIL": "hook-test@example.com",
}


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], env=ENV, check=True, capture_output=True)


def run_hook(name: str, repo: Path) -> subprocess.CompletedProcess[str]:
    """hook を Claude Code と同じ形 (stdin に cwd 入りの JSON、CLAUDE_PROJECT_DIR) で実行する。"""
    return subprocess.run(
        [str(HOOKS_DIR / name)],
        input=json.dumps({"cwd": str(repo)}),
        env={**ENV, "CLAUDE_PROJECT_DIR": str(repo)},
        capture_output=True,
        text=True,
        timeout=60,
    )
