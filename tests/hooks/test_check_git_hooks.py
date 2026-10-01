""".claude/hooks/check-git-hooks.sh (SessionStart: core.hooksPath 未設定の警告) のテスト。"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.hooks.helpers import git, run_hook


def session_start(cwd: Path, project_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    return run_hook("check-git-hooks.sh", cwd, project_dir=project_dir, event="SessionStart")


def assert_warns(stdout: str, current: str) -> None:
    assert "bash scripts/install-hooks.sh" in stdout
    assert f"core.hooksPath = {current}" in stdout
    if shutil.which("jq"):
        out = json.loads(stdout)
        assert "install-hooks.sh" in out["systemMessage"]
        assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        assert "install-hooks.sh" in out["hookSpecificOutput"]["additionalContext"]


def test_installed_is_silent(repo: Path) -> None:
    git(repo, "config", "core.hooksPath", ".githooks")
    result = session_start(repo)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize("value", ["./.githooks", ".githooks/"])
def test_equivalent_relative_spellings_are_silent(repo: Path, value: str) -> None:
    git(repo, "config", "core.hooksPath", value)
    result = session_start(repo)
    assert result.returncode == 0
    assert result.stdout == ""


def test_absolute_path_to_githooks_is_silent(repo: Path) -> None:
    git(repo, "config", "core.hooksPath", str(repo / ".githooks"))
    result = session_start(repo)
    assert result.returncode == 0
    assert result.stdout == ""


def test_absolute_path_elsewhere_warns(repo: Path, tmp_path: Path) -> None:
    other = tmp_path / "other" / ".githooks"
    other.mkdir(parents=True)
    git(repo, "config", "core.hooksPath", str(other))
    result = session_start(repo)
    assert result.returncode == 0
    assert_warns(result.stdout, str(other))


def test_unset_warns_without_blocking(repo: Path) -> None:
    result = session_start(repo)
    assert result.returncode == 0
    assert_warns(result.stdout, "未設定")


def test_other_value_warns(repo: Path) -> None:
    git(repo, "config", "core.hooksPath", ".git/hooks")
    result = session_start(repo)
    assert result.returncode == 0
    assert_warns(result.stdout, ".git/hooks")


def test_linked_worktree_installed_is_silent(repo: Path, tmp_path: Path) -> None:
    """worktree では .git がファイル。core.hooksPath は元リポジトリの設定を共有する。"""
    worktree = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", "-b", "wt", str(worktree))
    assert (worktree / ".git").is_file()
    git(repo, "config", "core.hooksPath", ".githooks")

    result = session_start(worktree)
    assert result.returncode == 0
    assert result.stdout == ""


def test_linked_worktree_with_absolute_path_to_main_repo_is_silent(
    repo: Path, tmp_path: Path
) -> None:
    """元のリポジトリの .githooks を絶対パスで指す設定は worktree でも有効なので警告しない。"""
    worktree = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", "-b", "wt", str(worktree))
    git(repo, "config", "core.hooksPath", str(repo / ".githooks"))

    result = session_start(worktree, project_dir=repo)
    assert result.returncode == 0
    assert result.stdout == ""


def test_linked_worktree_unset_warns(repo: Path, tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", "-b", "wt", str(worktree))

    result = session_start(worktree, project_dir=repo)
    assert result.returncode == 0
    assert_warns(result.stdout, "未設定")


def test_repo_without_githooks_is_silent(repo: Path) -> None:
    git(repo, "rm", "-q", ".githooks/pre-commit")
    result = session_start(repo)
    assert result.returncode == 0
    assert result.stdout == ""


def test_outside_git_repo_is_silent(tmp_path: Path) -> None:
    result = session_start(tmp_path)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
