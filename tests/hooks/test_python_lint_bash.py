""".claude/hooks/python-lint-bash.sh (Bash 経由の Python 編集の check-only lint) のテスト。"""

import os
import shutil
import time
from pathlib import Path

import pytest

from tests.hooks.helpers import git, run_hook

HOOK = "python-lint-bash.sh"
BADLY_FORMATTED = "x=1\n"
LINT_ERROR = "import os\n"
CLEAN = '"""clean"""\n\nVALUE = 2\n'


def provide_ruff(project: Path) -> None:
    """hook が最優先で使う <project>/.venv/bin/ruff を、テスト実行中の ruff へのリンクにする。"""
    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff が PATH に無い (uv run pytest で実行する)")
    bin_dir = project / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "ruff").symlink_to(ruff)


def write(path: Path, content: str) -> None:
    """書き込んだうえで mtime を現在時刻 (高精度) に揃える。

    カーネルのファイル時刻は粗いクロックなので、直前の hook 実行のスタンプ
    (高精度クロック) より古く見えることがある。テストを決定的にするため明示的に揃える。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    now = time.time_ns()
    os.utime(path, ns=(now, now))


@pytest.fixture
def lint_repo(repo: Path) -> Path:
    provide_ruff(repo)
    return repo


def test_no_candidate_files_is_silent(lint_repo: Path) -> None:
    result = run_hook(HOOK, lint_repo)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    # fast path ではスタンプも書かない
    assert not (lint_repo / ".git" / "claude-python-lint-bash.stamp").exists()


def test_clean_new_file_passes(lint_repo: Path) -> None:
    write(lint_repo / "pkg" / "clean.py", CLEAN)
    result = run_hook(HOOK, lint_repo)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_badly_formatted_file_is_reported_without_modification(lint_repo: Path) -> None:
    target = lint_repo / "pkg" / "bad.py"
    write(target, BADLY_FORMATTED)
    before = target.stat().st_mtime_ns

    result = run_hook(HOOK, lint_repo)

    assert result.returncode == 2
    assert "pkg/bad.py" in result.stderr
    assert "ruff format --check:" in result.stderr
    assert "Would reformat" in result.stderr
    assert "uv run ruff check --fix pkg/bad.py && uv run ruff format pkg/bad.py" in result.stderr
    assert "What to do:" in result.stderr
    # check-only: 内容も mtime も変えない
    assert target.read_text() == BADLY_FORMATTED
    assert target.stat().st_mtime_ns == before


def test_lint_error_is_reported(lint_repo: Path) -> None:
    write(lint_repo / "lint.py", LINT_ERROR)
    result = run_hook(HOOK, lint_repo)
    assert result.returncode == 2
    assert "ruff check:" in result.stderr
    assert "F401" in result.stderr


def test_unchanged_file_is_not_reported_again(lint_repo: Path) -> None:
    write(lint_repo / "bad.py", BADLY_FORMATTED)
    assert run_hook(HOOK, lint_repo).returncode == 2

    again = run_hook(HOOK, lint_repo)
    assert again.returncode == 0
    assert again.stderr == ""


def test_file_changed_again_is_reported_again(lint_repo: Path) -> None:
    target = lint_repo / "bad.py"
    write(target, BADLY_FORMATTED)
    assert run_hook(HOOK, lint_repo).returncode == 2
    assert run_hook(HOOK, lint_repo).returncode == 0

    write(target, "y=2\n")
    result = run_hook(HOOK, lint_repo)
    assert result.returncode == 2
    assert "bad.py" in result.stderr


def test_modified_tracked_file_is_reported(lint_repo: Path) -> None:
    write(lint_repo / "ok.py", BADLY_FORMATTED)
    result = run_hook(HOOK, lint_repo)
    assert result.returncode == 2
    assert "ok.py" in result.stderr


def test_staged_file_is_reported(lint_repo: Path) -> None:
    write(lint_repo / "staged.py", BADLY_FORMATTED)
    git(lint_repo, "add", "staged.py")
    result = run_hook(HOOK, lint_repo)
    assert result.returncode == 2
    assert "staged.py" in result.stderr


def test_renamed_file_and_following_entries_are_reported(lint_repo: Path) -> None:
    """rename は porcelain -z で元パスが別エントリになる。後続のエントリを取りこぼさない。"""
    git(lint_repo, "mv", "ok.py", "a_renamed.py")
    write(lint_repo / "a_renamed.py", BADLY_FORMATTED)
    write(lint_repo / "z_new.py", BADLY_FORMATTED)
    result = run_hook(HOOK, lint_repo)
    assert result.returncode == 2
    assert "  - a_renamed.py\n  - z_new.py\n" in result.stderr


def test_only_changed_files_are_listed(lint_repo: Path) -> None:
    write(lint_repo / "old_bad.py", BADLY_FORMATTED)
    assert run_hook(HOOK, lint_repo).returncode == 2

    write(lint_repo / "new_bad.py", BADLY_FORMATTED)
    result = run_hook(HOOK, lint_repo)
    assert result.returncode == 2
    assert "new_bad.py" in result.stderr
    assert "old_bad.py" not in result.stderr


def test_gitignored_file_is_skipped(lint_repo: Path) -> None:
    write(lint_repo / "ignored" / "bad.py", BADLY_FORMATTED)
    result = run_hook(HOOK, lint_repo)
    assert result.returncode == 0
    assert result.stderr == ""


def test_linked_worktree_is_checked_via_cwd(lint_repo: Path, tmp_path: Path) -> None:
    """worktree で動く subagent: cwd は worktree、CLAUDE_PROJECT_DIR は元のリポジトリ。"""
    worktree = tmp_path / "wt"
    git(lint_repo, "worktree", "add", "-q", "-b", "wt", str(worktree))
    write(worktree / "bad.py", BADLY_FORMATTED)

    # ruff は CLAUDE_PROJECT_DIR 側の .venv から見つける (worktree には .venv が無い)
    result = run_hook(HOOK, worktree, project_dir=lint_repo)
    assert result.returncode == 2
    assert "bad.py" in result.stderr
    assert str(worktree) in result.stderr


def test_repo_without_ruff_config_is_skipped(lint_repo: Path) -> None:
    write(lint_repo / "pyproject.toml", '[project]\nname = "t"\n')
    write(lint_repo / "bad.py", BADLY_FORMATTED)
    result = run_hook(HOOK, lint_repo)
    assert result.returncode == 0
    assert result.stderr == ""


def test_missing_ruff_is_silent(repo: Path) -> None:
    """.venv/bin/ruff も uv も無ければ作業を止めない (exit 0、出力なし)。"""
    path = "/usr/bin:/bin"
    if shutil.which("uv", path=path):
        pytest.skip("uv が /usr/bin か /bin にある")
    write(repo / "bad.py", BADLY_FORMATTED)
    result = run_hook(HOOK, repo, extra_env={"PATH": path})
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_outside_git_repo_is_silent(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    write(plain / "bad.py", BADLY_FORMATTED)
    result = run_hook(HOOK, plain)
    assert result.returncode == 0
    assert result.stderr == ""
