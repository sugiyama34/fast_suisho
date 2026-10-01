""".claude/hooks/check-git-hooks.sh (SessionStart: core.hooksPath 未設定の警告) のテスト。"""

import json
from pathlib import Path

import pytest

from tests.hooks.helpers import git, run_hook

HOOK = "check-git-hooks.sh"


@pytest.mark.parametrize("absolute", [False, True])
def test_installed_is_silent(repo: Path, absolute: bool) -> None:
    git(repo, "config", "core.hooksPath", str(repo / ".githooks") if absolute else ".githooks")
    result = run_hook(HOOK, repo)
    assert (result.returncode, result.stdout) == (0, "")


def test_worktree_with_absolute_path_to_main_repo_is_silent(repo: Path, tmp_path: Path) -> None:
    """worktree (.git がファイル) から見て、元リポジトリの .githooks を指す設定は有効。"""
    worktree = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", "-b", "wt", str(worktree))
    git(repo, "config", "core.hooksPath", str(repo / ".githooks"))
    result = run_hook(HOOK, worktree)
    assert (result.returncode, result.stdout) == (0, "")


def test_unset_warns_without_blocking(repo: Path) -> None:
    result = run_hook(HOOK, repo)
    assert result.returncode == 0
    message = json.loads(result.stdout)["systemMessage"]
    assert "core.hooksPath=未設定" in message
    assert "bash scripts/install-hooks.sh" in message
