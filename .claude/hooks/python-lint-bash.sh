#!/bin/bash
# PostToolUse(Bash) hook: Bash 経由 (heredoc, sed -i 等) で変更された .py を ruff で検査する。
# python-lint.sh (Edit/Write) と違い、検査のみでファイルは変更しない。
# Exit 2 = 問題あり (stderr が Claude に表示される)、それ以外は exit 0。

RUFF="$CLAUDE_PROJECT_DIR/.venv/bin/ruff"
cd "$(jq -r '.cwd // empty')" 2>/dev/null && TOP=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$TOP" || exit 0
STAMP="$(git rev-parse --git-dir)/python-lint-bash.stamp"

# 変更済み・未追跡 (gitignore 除く) の .py のうち、前回の検査以降に更新されたもの
FILES=()
while read -r f; do
  [[ -f "$f" && "$f" -nt "$STAMP" ]] && FILES+=("$f")
done < <(git --no-optional-locks status --porcelain -uall -- '*.py' | sed 's/^...//; s/.* -> //')
touch "$STAMP"
[ ${#FILES[@]} -eq 0 ] || [ ! -x "$RUFF" ] && exit 0

FAIL=
OUT=$("$RUFF" check --output-format concise "${FILES[@]}" 2>&1) || FAIL=1
OUT+=$'\n'$("$RUFF" format --check "${FILES[@]}" 2>&1) || FAIL=1
[ -z "$FAIL" ] && exit 0

cat >&2 <<EOF
ruff found issues in Python files changed via Bash (this hook did not modify them):
$OUT
Fix: uv run ruff check --fix ${FILES[*]} && uv run ruff format ${FILES[*]}
EOF
exit 2
