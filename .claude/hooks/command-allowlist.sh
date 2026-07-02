#!/bin/bash
# Claude Code PreToolUse hook: regex-based allowlist for Bash commands needing precise control.
# Splits compound commands (|, ||, &&, ;, &) and validates each segment independently.
# Governed segments must match an allowed pattern; non-governed segments pass through.
# Exit code 0 = allow/pass-through, exit code 2 = block.

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib/shell-parse.sh"

# Require jq for JSON parsing.
if ! command -v jq >/dev/null 2>&1; then
  cat >&2 <<ERRMSG
BLOCKED: jq is not installed.

Why: This hook requires jq to parse tool input JSON. Without it, commands cannot be validated.

What to do:
  Claude Code: Ask the user to install jq.
  User: Install jq (e.g., brew install jq on macOS, sudo apt-get install jq on Linux).
ERRMSG
  exit 2
fi

INPUT=$(cat)
if ! RAW_COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null); then
  cat >&2 <<ERRMSG
BLOCKED: failed to parse tool input JSON.

Why: The hook received invalid JSON input and cannot validate the command.

What to do:
  Claude Code: Report this error to the user — it may indicate a Claude Code bug or misconfigured hook.
  User: Check that .claude/hooks/command-allowlist.sh is correctly registered in settings.json.
ERRMSG
  exit 2
fi

# Normalize a command segment: trim whitespace and strip trailing shell redirections.
normalize_segment() {
  # Two separate s/// commands: a single s/// without /g stops at the first
  # match, so a `^…|…$` alternation only trims one side when both ends are
  # padded. Splitting trims each end independently. See spec-002.
  echo "$1" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/[[:space:]]+(2>&1|2>\/dev\/null|>&2)[[:space:]]*$//'
}

# Only govern specific command prefixes.
# Commands not listed here pass through to the built-in permission system.
GOVERNED_PREFIXES=(
  "bats"
  "gh api"
  "gh issue"
  "gh label"
  "gh pr"
  "gh run"
  "git add"
  "git branch"
  "git check-ignore"
  "git commit"
  "git config"
  "git mv"
  "git pull"
  "git push"
  "node"
  "python3 scripts/extract_api.py"
  "uv run"
)

# Allowed patterns (extended regex, matched against individual pipe segments)
# Add new patterns here to allow specific operations.
ALLOWED_PATTERNS=(
  # BATS test is only allowed in this specific form
  '^bats tests/hooks/$'

  # Read repo contents — directory listing or single file (read-only GET).
  # Path body forbids consecutive dots, so `..` traversal segments are rejected.
  '^gh api repos/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/contents(/\.?[A-Za-z0-9_/-]+(\.[A-Za-z0-9_/-]+)*)?$'

  # Read repo git tree (read-only GET, optional recursive=1).
  # Ref body forbids `..` and slashes — single-segment refs (HEAD, main, SHA) only.
  '^gh api repos/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/git/trees/\.?[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*(\?recursive=1)?$'

  # Read PR comments (with optional --jq filter)
  '^gh api repos/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/pulls/[0-9]+/comments( --paginate)?$'
  "^gh api repos/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/pulls/[0-9]+/comments( --paginate)? --jq '[^']*'$"

  # Read issue comments (with optional --paginate and/or --jq filter)
  '^gh api repos/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/issues/[0-9]+/comments( --paginate)?$'
  "^gh api repos/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/issues/[0-9]+/comments( --paginate)? --jq '[^']*'$"

  # Reply to PR review comments (body must be single-quoted; apostrophes via '\'' escape)
  "^gh api repos/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/pulls/[0-9]+/comments/[0-9]+/replies -f body='[^']*('\\\\''[^']*)*'$"

  # Read PR reviews and review comments (with optional --paginate and/or --jq filter)
  '^gh api repos/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/pulls/[0-9]+/reviews( --paginate)?$'
  "^gh api repos/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/pulls/[0-9]+/reviews( --paginate)? --jq '[^']*'$"
  '^gh api repos/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/pulls/[0-9]+/reviews/[0-9]+/comments( --paginate)?$'
  "^gh api repos/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/pulls/[0-9]+/reviews/[0-9]+/comments( --paginate)? --jq '[^']*'$"

  # Create sub-issues (used by /start-dev skill)
  '^gh api repos/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+ --method POST -f parent_issue_id=[a-zA-Z0-9_=]+ /issues/[0-9]+/sub_issues$'

  # GraphQL queries (read-only, any query allowed but must be single-quoted; apostrophes via '\'' escape)
  "^gh api graphql -f query='(query([[:space:]({{])|[{])[^']*('\\\\''[^']*)*'$"

  # GitHub issue operations — read-only (any flags allowed)
  '^gh issue (list|status|view)( |$)'
  # GitHub issue operations — write (any flags allowed)
  '^gh issue (create|comment|develop|edit|reopen)( |$)'

  # GitHub label operations (used by reporting skills to ensure labels exist)
  '^gh label (list|create)( |$)'

  # GitHub PR operations — read-only (any flags allowed)
  '^gh pr (list|status|checks|diff|view)( |$)'
  # GitHub PR operations — write (allowed with any args, review prompted commands individually)
  '^gh pr (create|checkout|comment|edit|ready)( |$)'

  # GitHub Actions run operations — read-only (any flags allowed)
  '^gh run (list|view|watch)( |$)'

  # Git add — stage files by path (no `.`, `-A`, or `--all`, which are also
  # blocked by permissions.deny in settings.json)
  '^git add [A-Za-z0-9._/-]+( [A-Za-z0-9._/-]+)*$'
  '^git add -- [A-Za-z0-9._/-]+( [A-Za-z0-9._/-]+)*$'

  # Git branch operations
  # Raw git branch
  '^git branch$'
  # Create a new branch: git branch <name>
  '^git branch [a-zA-Z0-9_./-]+$'
  # Delete a merged branch (safe -d only, not force -D): git branch -d <name>
  '^git branch -d [a-zA-Z0-9_./-]+$'
  # List branches merged into a given branch: git branch --merged <name>
  '^git branch --merged [a-zA-Z0-9_./-]+$'

  # Git check-ignore — read-only check whether path(s) would be excluded by .gitignore.
  '^git check-ignore( (-v|--verbose|-q|--quiet|-n|--non-matching|--no-index))*( [A-Za-z0-9._/-]+)+$'

  # Git commit
  "^git commit -m '[^']*('\\\\''[^']*)*'$"

  # Git config — read-only inspection only. Writes (git config <key> <value>),
  # --unset/--add/--replace-all, and --edit match no pattern and fall through to
  # a block. These reads are additionally blocked at the system level by
  # block-dangerous-git.sh (spec-006); these patterns are this hook's own
  # defense-in-depth layer (spec-009).
  '^git config --get [A-Za-z0-9._-]+$'
  '^git config --get-all [A-Za-z0-9._-]+$'
  '^git config (--global|--local|--system) --get [A-Za-z0-9._-]+$'
  '^git config --list$'
  '^git config (--global|--local|--system) --list$'

  # Git mv — rename/move file(s); paths + safe flags (-v/-n) only, not -f/-k.
  '^git mv( (-v|--verbose|-n|--dry-run|--))* [A-Za-z0-9._/][A-Za-z0-9._/-]*( [A-Za-z0-9._/][A-Za-z0-9._/-]*)+$'

  # Git pull
  '^git pull( (--rebase|--ff-only))?( origin [a-zA-Z0-9_./-]+)?$'

  # Git push
  '^git push$'
  '^git push origin$'
  '^git push (-u |--set-upstream )?origin [a-zA-Z0-9_./-]+$'


  # Node.js script for Codex Companion plugin (specific versioned path, any of the allowed subcommands)
  '^node "?/home/satoshisugiyama/\.claude/plugins/cache/openai-codex/codex/[0-9]+\.[0-9]+\.[0-9]+/scripts/codex-companion\.mjs"? (review|adversarial-review|status|result)( '"'"'[^'"'"']*'"'"')?$'
  '^node "?/home/satoshisugiyama/\.claude/plugins/cache/openai-codex/codex/[0-9]+\.[0-9]+\.[0-9]+/scripts/codex-companion\.mjs"? setup --json( --(enable|disable)-review-gate)?$'

  # Ruff version check (no path args; `2>&1` is stripped by normalize_segment)
  '^uv run ruff --version$'
  # Ruff check command for linting Python code (specific path, any subdirectory, optional trailing slash)
  '^uv run ruff (check|format)( --(fix|quiet|check))*( [A-Za-z0-9._-]*[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9._-]*[A-Za-z0-9_-][A-Za-z0-9._-]*)*/?)+$'
  # Pytest command for running tests (specific path, any subdirectory, optional trailing slash)
  '^uv run pytest( (-v|--tb=short|--no-header))*( [A-Za-z0-9._-]*[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9._-]*[A-Za-z0-9_-][A-Za-z0-9._-]*)*/?)+( (-v|--tb=short|--no-header))*$'
  # extract_api.py manifest generator: one or more .py path args. Prefix dot is
  # unescaped so the settings/allowlist sync test (greps the prefix) matches.
  '^python3 scripts/extract_api.py( [A-Za-z0-9._/-]+\.py)+$'
  '^uv run python scripts/extract_api.py( [A-Za-z0-9._/-]+\.py)+$'

)

# Validate each segment of the pipeline independently.
# Governed segments must match an allowed pattern; non-governed segments pass through.
BLOCKED_SEGMENT=""
while IFS= read -r segment; do
  segment=$(normalize_segment "$segment")
  [ -z "$segment" ] && continue

  # Check if this segment is governed
  segment_governed=false
  for prefix in "${GOVERNED_PREFIXES[@]}"; do
    if [[ "$segment" == "$prefix"* ]]; then
      segment_governed=true
      break
    fi
  done

  [ "$segment_governed" = false ] && continue  # Not governed — pass through

  # Governed segment: must match an allowed pattern
  segment_allowed=false
  for pattern in "${ALLOWED_PATTERNS[@]}"; do
    if echo "$segment" | grep -qE -e "$pattern"; then
      segment_allowed=true
      break
    fi
  done

  if [ "$segment_allowed" = false ]; then
    BLOCKED_SEGMENT="$segment"
    break
  fi
done <<< "$(split_command_segments "$RAW_COMMAND")"

# If all segments passed, allow the command
if [ -z "$BLOCKED_SEGMENT" ]; then
  exit 0
fi

# A governed segment was not in the allowlist — block the entire command
cat >&2 <<ERRMSG
BLOCKED: command not in allowlist.

Command: $BLOCKED_SEGMENT

Why:
  This command segment (for example, this gh api endpoint/flag combination) is not on the approved allowlist.

What to do:
  Claude Code: Try a different approach. If this command genuinely should be allowed,
               run /report-hook-block to open a GitHub Issue proposing the policy update.
  User: To allow this command, add a regex pattern to .claude/hooks/command-allowlist.sh
        or run the command manually in your terminal.
ERRMSG

exit 2
