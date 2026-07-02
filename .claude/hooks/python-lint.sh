#!/bin/bash
# Claude Code PostToolUse hook: auto-fix Python lint/format issues after file edits.
# Runs ruff check --fix and ruff format on .py files, then reports any remaining issues.
# Exits 0 on success or when not applicable; exits 2 if uv is missing.

set -euo pipefail

INPUT=$(cat)

# Extract file path from tool_input.file_path (Edit/Write tools)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Skip if no file path or not a Python file
if [[ -z "$FILE_PATH" || "$FILE_PATH" != *.py ]]; then
  exit 0
fi

# Skip if file doesn't exist (e.g., deleted)
if [[ ! -f "$FILE_PATH" ]]; then
  exit 0
fi

# Fail if uv is not available (ruff is installed inside the uv-managed venv)
if ! command -v uv >/dev/null 2>&1; then
  echo "BLOCKED: uv is not installed." >&2
  echo "" >&2
  echo "Why:" >&2
  echo "  The python-lint hook runs ruff via 'uv run', which requires uv on PATH." >&2
  echo "" >&2
  echo "What to do:" >&2
  echo "  Claude Code: Ask the user to install uv, then run 'uv sync'." >&2
  echo "  User: Install uv (https://docs.astral.sh/uv/) and run 'uv sync' to install dev dependencies including ruff." >&2
  exit 2
fi

# Fail if ruff is not installed in the uv-managed environment
if ! uv run ruff --version >/dev/null 2>&1; then
  echo "BLOCKED: ruff is not installed in the uv environment." >&2
  echo "" >&2
  echo "Why:" >&2
  echo "  The python-lint hook runs 'uv run ruff', but ruff is not available." >&2
  echo "  Without this check, missing ruff would silently skip linting." >&2
  echo "" >&2
  echo "What to do:" >&2
  echo "  Claude Code: Ask the user to run 'uv sync' to install dev dependencies." >&2
  echo "  User: Run 'uv sync' to install ruff and other dev dependencies." >&2
  exit 2
fi

# Auto-fix: apply safe lint fixes and format
uv run ruff check --fix --quiet "$FILE_PATH" 2>/dev/null || true
uv run ruff format --quiet "$FILE_PATH" 2>/dev/null || true

# Check for remaining unfixable issues
if ! REMAINING=$(uv run ruff check "$FILE_PATH" 2>&1); then
  echo "Ruff found issues that require manual fixes:"
  echo ""
  echo "$REMAINING"
  echo ""
  echo "Please address these issues in your next edit."
fi

exit 0
