"""Claude Code hook のテスト用 fixture。ヘルパーは tests/hooks/helpers.py。"""

from pathlib import Path

import pytest

from tests.hooks.helpers import git


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """初期コミット済みの使い捨て git リポジトリ (ruff 設定と .githooks/pre-commit 付き)。"""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "t"\n\n[tool.ruff]\nline-length = 100\n'
    )
    (root / ".gitignore").write_text(".venv/\nignored/\n")
    hooks = root / ".githooks"
    hooks.mkdir()
    (hooks / "pre-commit").write_text("#!/bin/sh\nexit 0\n")
    (hooks / "pre-commit").chmod(0o755)
    (root / "ok.py").write_text('"""ok"""\n\nVALUE = 1\n')
    git(root, "add", "pyproject.toml", ".gitignore", ".githooks/pre-commit", "ok.py")
    git(root, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")
    return root
