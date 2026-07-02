#!/bin/bash
# Claude Code PreToolUse hook: block destructive / review-bypassing git operations.
#
# Covered:
#   - git push --force / -f / --force-with-lease     (force push)
#   - git push ... +refspec                          (force push via + prefix)
#   - git push to main/master (explicit or implicit via current branch)
#   - gh pr merge --admin                            (bypasses review)
#   - git -c user.name=... / -c user.email=... commit (author impersonation)
#   - git commit --author=... / --author <value>      (author impersonation)
#   - git config <any args>                           (persistent identity / config change)
#
# Exit code 0 = allow, exit code 2 = block.

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib/shell-parse.sh"

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Detect shell-wrapper commands that execute an arbitrary inner string
# (bash -c, sh -c, zsh -c, eval). This hook cannot statically parse the
# inner shell, so we fail closed when the wrapper carries tokens that
# could indicate a dangerous git operation.
wrapper_unsafe() {
  local seg="$1"
  # Match `c` anywhere in a combined short-option cluster so that forms
  # like `bash -xc`, `bash -cx`, `bash -evc` all get caught, not just the
  # trailing-c case.
  if ! printf '%s\n' "$seg" | grep -qE '(^|[[:space:]])((/bin/|/usr/bin/)?(ba|z)?sh[[:space:]]+-[^[:space:]]*c[^[:space:]]*([[:space:]]|$)|eval([[:space:]]|$))'; then
    return 1
  fi
  if printf '%s\n' "$seg" | grep -qE '(git[[:space:]]+push|gh[[:space:]]+pr[[:space:]]+merge|--force|--force-with-lease|[[:space:]]\+?(main|master)([[:space:]]|$)|--admin|--author|user\.(name|email)=|git[[:space:]]+([^[:space:]]+[[:space:]]+)*config([[:space:]]|$))'; then
    return 0
  fi
  return 1
}

# Analyze a `git push` segment. Returns 2 if dangerous, 0 otherwise.
# Parses options robustly: handles `--`, flags with a separate argument
# (-u / --set-upstream / --repo / --receive-pack / --exec), flag=value
# forms, and multiple refspecs.
analyze_git_push() {
  local seg="$1"
  local rest current_branch remote refspec target
  local -a raw_tokens positional refspecs
  local i token

  # --- --force / --force-with-lease ---
  if printf '%s\n' "$seg" | grep -qE '(^|[[:space:]])(-f|--force|--force-with-lease)([[:space:]]|=|$)'; then
    cat >&2 <<ERRMSG
BLOCKED: force-push flag detected.

Segment: $seg

Why: Force push rewrites history and can destroy teammates' commits. Even
     --force-with-lease has a narrow safety window; deny by default.

What to do:
  Claude Code: Push without --force. If history rewrite is genuinely required,
               ask the user to run it manually.
  User: Run 'git push --force' manually if you decide it is needed.
ERRMSG
    return 2
  fi

  # --- Parse positional args ---
  rest=$(printf '%s\n' "$seg" | sed -E 's/^[[:space:]]*git[[:space:]]+push([[:space:]]|$)//; s/[[:space:]]+(2>&1|2>\/dev\/null|>&2)[[:space:]]*$//; s/[[:space:]]+$//')

  raw_tokens=()
  positional=()
  refspecs=()

  IFS=' ' read -r -a raw_tokens <<< "$rest"

  i=0
  while [ "$i" -lt "${#raw_tokens[@]}" ]; do
    token="${raw_tokens[$i]}"
    if [ -z "$token" ]; then
      i=$((i + 1))
      continue
    fi
    case "$token" in
      --)
        i=$((i + 1))
        while [ "$i" -lt "${#raw_tokens[@]}" ]; do
          [ -n "${raw_tokens[$i]}" ] && positional+=("${raw_tokens[$i]}")
          i=$((i + 1))
        done
        break
        ;;
      --repo|--receive-pack|--exec)
        # These flags consume the next token as their argument.
        i=$((i + 2))
        ;;
      -u|--set-upstream)
        # Boolean flags; do not consume the next token.
        i=$((i + 1))
        ;;
      --*=*|-*)
        i=$((i + 1))
        ;;
      *)
        positional+=("$token")
        i=$((i + 1))
        ;;
    esac
  done

  # First positional is remote, rest are refspecs.
  remote=""
  [ "${#positional[@]}" -ge 1 ] && remote="${positional[0]}"
  [ "${#positional[@]}" -ge 2 ] && refspecs=("${positional[@]:1}")

  # --- Implicit target (no refspec): use current branch ---
  if [ "${#refspecs[@]}" -eq 0 ]; then
    current_branch=$(git branch --show-current 2>/dev/null || printf '')
    current_branch="${current_branch#refs/heads/}"
    if [ "$current_branch" = "main" ] || [ "$current_branch" = "master" ]; then
      cat >&2 <<ERRMSG
BLOCKED: push target is '$current_branch'.

Segment: $seg

Why: All changes must go through a pull request. Direct push to
     '$current_branch' bypasses review and may trigger production CI
     unexpectedly.

What to do:
  Claude Code: Switch to a feature branch and open a PR via 'gh pr create'.
  User: If a direct push is truly needed (emergency revert, etc.), run it
        manually in your terminal.
ERRMSG
      return 2
    fi
    return 0
  fi

  # --- Explicit refspecs: check each ---
  for refspec in "${refspecs[@]}"; do
    # +refspec force push
    if [[ "$refspec" == +* ]]; then
      cat >&2 <<ERRMSG
BLOCKED: '+refspec' force push detected.

Segment: $seg

Why: The '+' prefix on a refspec forces a non-fast-forward update,
     equivalent to --force for that refspec.

What to do:
  Claude Code: Push without the '+' prefix.
  User: Run the force push manually if you decide it is needed.
ERRMSG
      return 2
    fi

    target=""
    if [[ "$refspec" == *:* ]]; then
      target="${refspec##*:}"
    elif [ "$refspec" = "HEAD" ]; then
      target=$(git branch --show-current 2>/dev/null || printf '')
    else
      target="$refspec"
    fi
    target="${target#refs/heads/}"

    [ -z "$target" ] && continue

    if [ "$target" = "main" ] || [ "$target" = "master" ]; then
      cat >&2 <<ERRMSG
BLOCKED: push target is '$target'.

Segment: $seg

Why: All changes must go through a pull request. Direct push to '$target'
     bypasses review and may trigger production CI unexpectedly.

What to do:
  Claude Code: Switch to a feature branch and open a PR via 'gh pr create'.
  User: If a direct push is truly needed (emergency revert, etc.), run it
        manually in your terminal.
ERRMSG
      return 2
    fi
  done

  return 0
}

# Analyze a `git ... commit` segment for author/email-override forms.
# Returns 2 if dangerous, 0 otherwise.
#
# Detection strategy (per Spec-006):
#   - Strip single- and double-quote characters from the segment so that
#     quoted forms (-c 'user.email=x', --author="...") cannot bypass the
#     regex. Accepted side-effect: commit messages containing literal
#     `--author=` or `-c user.{name,email}=` text are also blocked (E3).
#   - Only fire when the `commit` token is present in the segment so that
#     non-commit subcommands (git log, git rebase, git format-patch) and
#     git commit-tree / cherry-pick / etc. remain out of scope (E7).
analyze_git_commit_author() {
  local seg="$1"
  local stripped

  stripped=$(printf '%s\n' "$seg" | tr -d "'\"")

  # Scope: require the `commit` subcommand token (not `commit-tree`, not a
  # commit hash). The regex anchors `commit` between whitespace/edges.
  if ! printf '%s\n' "$stripped" | grep -qE '(^|[[:space:]])commit([[:space:]]|$)'; then
    return 0
  fi

  # B1/B2: `-c user.email=...` / `-c user.name=...` config override.
  if printf '%s\n' "$stripped" | grep -qE '(^|[[:space:]])-c[[:space:]]+user\.(name|email)='; then
    cat >&2 <<ERRMSG
BLOCKED: git -c user.{name,email}=... override on a commit segment.

Segment: $seg

Why: Overriding user.name or user.email at the git invocation level rewrites
     the commit's identity and can impersonate other contributors in the
     commit log. This project's convention is to attribute Claude Code's
     work via a 'Co-Authored-By:' trailer (see CLAUDE.md), not by
     overriding the identity.

What to do:
  Claude Code: Drop the '-c user.name=' / '-c user.email=' override and
               add a 'Co-Authored-By:' trailer in the commit message
               instead. If you need to change the committer identity for
               a real reason, ask the user.
  User: Run the override manually in your terminal if you decide it is
        needed.
ERRMSG
    return 2
  fi

  # B3/B4: `--author=value` (equals form) and `--author value` (separated).
  # `(^|[[:space:]])--author([[:space:]=])` covers both forms and rules
  # out substring matches like `--no-author` or `--authored-by`.
  if printf '%s\n' "$stripped" | grep -qE '(^|[[:space:]])--author([[:space:]=])'; then
    cat >&2 <<ERRMSG
BLOCKED: git commit --author=... override.

Segment: $seg

Why: --author rewrites the commit's authorship and can impersonate other
     contributors in the commit log. This project's convention is to
     attribute Claude Code's work via a 'Co-Authored-By:' trailer (see
     CLAUDE.md), not by overriding the author.

What to do:
  Claude Code: Drop the '--author' flag and add a 'Co-Authored-By:' trailer
               in the commit message instead. If author override is
               genuinely required, ask the user.
  User: Run the override manually in your terminal if you decide it is
        needed.
ERRMSG
    return 2
  fi

  return 0
}

# Analyze a segment for the `git config` subcommand (any args).
# Returns 2 if dangerous, 0 otherwise.
#
# Detection strategy (per Spec-006 B8/B9/E8/E9):
#   - Strip single- and double-quote characters from the segment so that
#     quoted args do not bypass the regex.
#   - Match `config` in the SUBCOMMAND SLOT — the first non-flag positional
#     after `git`, tolerating zero or more `-c key=value` and `-C path`
#     git-level options between `git` and `config` (E9).
#   - Commit messages or other args containing the word `config` are NOT
#     over-blocked because the detection is positional, not substring (E8).
analyze_git_config() {
  local seg="$1"
  local stripped

  stripped=$(printf '%s\n' "$seg" | tr -d "'\"")

  if ! printf '%s\n' "$stripped" | grep -qE '^git([[:space:]]+-[cC][[:space:]]+[^[:space:]]+)*[[:space:]]+config([[:space:]]|$)'; then
    return 0
  fi

  cat >&2 <<ERRMSG
BLOCKED: 'git config' invocation.

Segment: $seg

Why: 'git config' can persistently change repository state — user.email /
     user.name (identity), core.hooksPath (hook bypass), and so on. Per
     CLAUDE.md ('NEVER update the git config'), Claude should not invoke
     'git config' from a tool call. Reads can use the underlying
     .git/config file directly via the Read tool.

What to do:
  Claude Code: Drop the 'git config' call. To read config, use Read on
               .git/config. If a write is genuinely required (e.g., a
               one-off repo setup), ask the user to run it manually.
  User: Run 'git config' manually in your terminal if you decide it is
        needed.
ERRMSG
  return 2
}

# Per-segment analysis. Returns 2 if segment is dangerous, 0 otherwise.
analyze_segment() {
  local seg="$1"
  local trimmed
  trimmed=$(printf '%s\n' "$seg" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')
  [ -z "$trimmed" ] && return 0

  # --- shell wrapper carrying destructive tokens (fail closed) ---
  if wrapper_unsafe "$trimmed"; then
    cat >&2 <<ERRMSG
BLOCKED: shell wrapper (bash -c / sh -c / eval) with destructive git tokens.

Segment: $trimmed

Why: This hook cannot statically analyze commands run through a shell wrapper,
     and the segment carries tokens suggesting a git push / gh pr merge /
     commit author override / git config with --force, --admin, main,
     master, --author, -c user.{name,email}=, or 'git ... config'.
     Failing closed.

What to do:
  Claude Code: Rewrite without the shell wrapper so the command is directly
               visible, or ask the user to run it manually.
  User: Run the command manually in your terminal if it is safe.
ERRMSG
    return 2
  fi

  # --- gh pr merge --admin ---
  if printf '%s\n' "$trimmed" | grep -qE '^gh[[:space:]]+pr[[:space:]]+merge([[:space:]]|$)' \
     && printf '%s\n' "$trimmed" | grep -qE '(^|[[:space:]])--admin([[:space:]]|=|$)'; then
    cat >&2 <<ERRMSG
BLOCKED: 'gh pr merge --admin' bypasses required reviews and branch protection.

Segment: $trimmed

Why: Admin merge is reserved for genuine emergencies and should be a human
     decision, not an agent decision.

What to do:
  Claude Code: Merge without --admin. If admin merge is genuinely required,
               stop and ask the user.
  User: Run 'gh pr merge --admin' manually if you decide it is needed.
ERRMSG
    return 2
  fi

  # --- git push analysis ---
  if printf '%s\n' "$trimmed" | grep -qE '^git[[:space:]]+push([[:space:]]|$)'; then
    analyze_git_push "$trimmed"
    return $?
  fi

  # --- git commit author/email override + git config analysis ---
  # Both analyzers fire on any segment starting with `git`. Each internally
  # scopes to its own subcommand context (commit-author needs the `commit`
  # token; config needs `config` in the subcommand slot).
  if printf '%s\n' "$trimmed" | grep -qE '^git([[:space:]]|$)'; then
    analyze_git_commit_author "$trimmed" || return 2
    analyze_git_config "$trimmed" || return 2
  fi

  return 0
}

# Iterate over segments; block on first dangerous match.
while IFS= read -r segment; do
  if ! analyze_segment "$segment"; then
    exit 2
  fi
done <<< "$(split_command_segments "$COMMAND")"

exit 0
