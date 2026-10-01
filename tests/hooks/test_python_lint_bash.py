""".claude/hooks/python-lint-bash.sh (Bash 経由の Python 編集の検査) のテスト。"""

import os
import shutil
import time
from pathlib import Path

import pytest

from tests.hooks.helpers import run_hook

HOOK = "python-lint-bash.sh"
BAD = "import os\nx=1\n"  # F401 + format 違反


@pytest.fixture
def lint_repo(repo: Path) -> Path:
    """hook が使う $CLAUDE_PROJECT_DIR/.venv/bin/ruff を、テスト実行中の ruff へのリンクにする。"""
    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff が PATH に無い (uv run pytest で実行する)")
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "ruff").symlink_to(ruff)
    return repo


def write(path: Path, content: str) -> None:
    """書き込み、mtime を高精度の現在時刻に揃える (カーネルのファイル時刻は粗いため)。"""
    path.write_text(content)
    now = time.time_ns()
    os.utime(path, ns=(now, now))


def test_no_changed_files_is_silent(lint_repo: Path) -> None:
    result = run_hook(HOOK, lint_repo)
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_clean_file_is_silent(lint_repo: Path) -> None:
    write(lint_repo / "clean.py", "VALUE = 1\n")
    result = run_hook(HOOK, lint_repo)
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_bad_file_is_reported_without_modification(lint_repo: Path) -> None:
    target = lint_repo / "bad.py"
    write(target, BAD)
    mtime = target.stat().st_mtime_ns

    result = run_hook(HOOK, lint_repo)

    assert result.returncode == 2
    assert "F401" in result.stderr
    assert "Would reformat: bad.py" in result.stderr
    assert "Fix: uv run ruff check --fix bad.py && uv run ruff format bad.py" in result.stderr
    assert target.read_text() == BAD
    assert target.stat().st_mtime_ns == mtime


def test_unchanged_file_is_not_reported_again(lint_repo: Path) -> None:
    target = lint_repo / "bad.py"
    write(target, BAD)
    assert run_hook(HOOK, lint_repo).returncode == 2
    assert run_hook(HOOK, lint_repo).returncode == 0

    write(target, BAD + "y=2\n")  # 再び変更されたら再度報告する
    assert run_hook(HOOK, lint_repo).returncode == 2
