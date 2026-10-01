#!/bin/bash
# Claude Code PostToolUse hook (Bash matcher): CHECK-ONLY ruff lint/format check
# for Python files changed by Bash tool calls.
#
# python-lint.sh auto-fixes files touched by Edit/Write, but Python edits made via
# Bash (heredocs, sed -i, rewrite scripts) bypass it and then fail the Lint CI
# (.github/workflows/lint.yml: `ruff check` + `ruff format --check`).
#
# - Candidates: modified / staged / untracked (not ignored) *.py files from
#   `git status`, whose mtime is newer than a per-worktree stamp file
#   (<git-dir>/claude-python-lint-bash.stamp). The stamp is advanced whenever
#   candidates were seen, so an unchanged file is reported at most once.
# - Never modifies files: no --fix, `ruff format --check`, --no-cache. Auto-fixing
#   after every Bash call would reformat files in the middle of multi-step
#   scripted edits.
# - ruff: <repo>/.venv/bin/ruff, then $CLAUDE_PROJECT_DIR/.venv/bin/ruff, then
#   `uv run ruff`. If none is available, exit 0 silently (never block work).
#
# Exit code 0 = clean / not applicable (no output).
# Exit code 2 = issues found (stderr is shown to Claude; the command already ran).

set -uo pipefail
export LC_ALL=C GIT_OPTIONAL_LOCKS=0

# Stamp time, captured before scanning so that edits racing with this hook are
# re-checked on the next call rather than missed.
START=$EPOCHREALTIME

INPUT=""
[ -t 0 ] || INPUT=$(cat)

# The Bash tool's cwd (may be a git worktree), falling back to the project dir.
CWD=""
if [ -n "$INPUT" ] && command -v jq >/dev/null 2>&1; then
  CWD=$(jq -r '.cwd // empty' <<<"$INPUT" 2>/dev/null || true)
fi

REPO=""
ADMIN_DIR=""
for dir in "$CWD" "${CLAUDE_PROJECT_DIR:-}"; do
  [ -n "$dir" ] && [ -d "$dir" ] || continue
  if out=$(git -C "$dir" rev-parse --show-toplevel --absolute-git-dir 2>/dev/null); then
    REPO=${out%%$'\n'*}
    ADMIN_DIR=${out#*$'\n'}
    break
  fi
done
[ -n "$REPO" ] || exit 0

STAMP="$ADMIN_DIR/claude-python-lint-bash.stamp"

# Porcelain v1 -z: "XY path\0", plus "orig\0" after renames/copies.
# `-nt` is also true when the stamp does not exist yet (first run).
FILES=()
while IFS= read -r -d '' rec; do
  xy=${rec:0:2}
  path=${rec:3}
  [[ "$xy" == *[RC]* ]] && { IFS= read -r -d '' _ || true; }
  [[ "$path" == *.py && -f "$REPO/$path" ]] || continue
  [[ "$REPO/$path" -nt "$STAMP" ]] && FILES+=("$path")
done < <(git -C "$REPO" status --porcelain=v1 -z --untracked-files=all \
  --ignore-submodules=all -- '*.py' 2>/dev/null)

# Fast path: nothing changed since the last check.
[ ${#FILES[@]} -eq 0 ] && exit 0

# Only for repos configured for ruff (e.g. not a sibling repo the cwd moved into).
if ! { [ -f "$REPO/ruff.toml" ] || [ -f "$REPO/.ruff.toml" ] ||
  grep -q '^\[tool\.ruff' "$REPO/pyproject.toml" 2>/dev/null; }; then
  exit 0
fi

touch -d "@$START" "$STAMP" 2>/dev/null || true

RUFF=()
for bin in "$REPO/.venv/bin/ruff" "${CLAUDE_PROJECT_DIR:+$CLAUDE_PROJECT_DIR/.venv/bin/ruff}"; do
  if [ -n "$bin" ] && [ -x "$bin" ]; then
    RUFF=("$bin")
    break
  fi
done
cd "$REPO" || exit 0
if [ ${#RUFF[@]} -eq 0 ]; then
  if command -v uv >/dev/null 2>&1 && uv run --quiet ruff --version >/dev/null 2>&1; then
    RUFF=(uv run --quiet ruff)
  else
    exit 0
  fi
fi

# Run both checks concurrently (each ruff start-up costs ~20 ms). The format
# check's exit code is appended as the last line of its output.
exec 3< <(
  "${RUFF[@]}" format --check --no-cache --force-exclude -- "${FILES[@]}" 2>&1
  echo "$?"
)
CHECK_OUT=$("${RUFF[@]}" check --no-cache --force-exclude --output-format concise \
  -- "${FILES[@]}" 2>&1)
CHECK_RC=$?
FORMAT_OUT=$(cat <&3)
exec 3<&-
FORMAT_RC=${FORMAT_OUT##*$'\n'}
if [[ "$FORMAT_OUT" == *$'\n'* ]]; then FORMAT_OUT=${FORMAT_OUT%$'\n'*}; else FORMAT_OUT=""; fi
[[ "$FORMAT_RC" =~ ^[0-9]+$ ]] || FORMAT_RC=1

[ "$CHECK_RC" -eq 0 ] && [ "$FORMAT_RC" -eq 0 ] && exit 0

# Indent and cap tool output so a huge generated file does not flood the context.
show() {
  local max=30
  local -a lines
  mapfile -t lines <<<"$1"
  printf '  %s\n' "${lines[@]:0:max}"
  if [ ${#lines[@]} -gt "$max" ]; then
    echo "  ... ($((${#lines[@]} - max)) more lines)"
  fi
}

QUOTED=$(printf ' %q' "${FILES[@]}")

{
  echo "LINT ISSUES: ruff found problems in Python files changed outside Edit/Write."
  echo "(Check only: this hook did not modify any file.)"
  echo ""
  echo "Files:"
  printf '  - %s\n' "${FILES[@]}"
  if [ "$CHECK_RC" -ne 0 ]; then
    echo ""
    echo "ruff check:"
    show "$CHECK_OUT"
  fi
  if [ "$FORMAT_RC" -ne 0 ]; then
    echo ""
    echo "ruff format --check:"
    show "$FORMAT_OUT"
  fi
  cat <<MSG

Why:
  The Lint CI (.github/workflows/lint.yml) runs \`uv run ruff check\` and
  \`uv run ruff format --check\`. Edits made through Bash (heredocs, sed -i,
  rewrite scripts) bypass the Edit/Write auto-fix hook (python-lint.sh).

What to do:
  Claude Code: Once the scripted edit is complete, run from the repo root
               ($REPO):
                 uv run ruff check --fix${QUOTED} && uv run ruff format${QUOTED}
               then fix any remaining \`ruff check\` errors by hand. A file is
               reported once per change, so do not wait for a repeat warning.
  User: Run the same command in your terminal.
MSG
} >&2
exit 2
