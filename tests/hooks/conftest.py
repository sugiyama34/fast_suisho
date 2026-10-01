"""Claude Code hook のテスト用 fixture。ヘルパーは tests/hooks/helpers.py。"""

from pathlib import Path

import pytest

from tests.hooks.helpers import git


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """初期コミット済みの使い捨て git リポジトリ (.githooks/ 付き)。"""
    root = tmp_path / "repo"
    (root / ".githooks").mkdir(parents=True)
    (root / ".githooks" / "pre-commit").write_text("#!/bin/sh\n")
    (root / ".gitignore").write_text(".venv/\n")
    (root / "ok.py").write_text('"""ok"""\n')
    git(root, "init", "-q", "-b", "main")
    git(root, "add", ".githooks/pre-commit", ".gitignore", "ok.py")
    git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")
    return root
