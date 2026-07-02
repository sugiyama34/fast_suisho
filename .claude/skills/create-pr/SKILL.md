---
name: create-pr
description: Commit remaining changes, push the branch, open a pull request, and run Codex review
allowed-tools: Bash, Read, Edit, Write, Glob, Grep
argument-hint: "[issue-number]"
---

Create a pull request for the current branch. Run two rounds of Codex code review before pushing.

The issue number (if any): $ARGUMENTS

**Operating principle:** Only ask the user when a step explicitly requires it, or when it is strongly necessary (e.g., blocking errors, ambiguous requirements). Do not stop just to report progress — keep moving through the steps autonomously.

---

## Step 1 — Review Uncommitted Changes

Run `git status` and `git diff` to understand what has changed.

If there are uncommitted changes, commit them now following the commit discipline in CLAUDE.md — group by semantic unit, one logical change per commit.

If there is nothing to commit and no unpushed commits, stop and notify the user.

---

## Step 2 — Sync with main

Fetch the latest main and check whether it has commits not yet in this branch:

```bash
git fetch origin main
git log HEAD..origin/main --oneline
```

If there are new commits on main, merge them into the current branch:

```bash
git merge origin/main --no-edit
```

If the merge has conflicts, stop and ask the user for help.

If main is already up-to-date, proceed.

### Check ADR and spec numbering

After syncing, verify that any new ADRs or specs added on this branch don't collide with number prefixes already used on `origin/main`. Parallel branches can independently pick the same NNN; whichever one merges first wins, and the rest need renumbering. Run the ADR check first, then the spec check; commit each rename group separately.

#### ADRs

1. Get ADR numbers used on main:
   ```bash
   git ls-tree -r --name-only origin/main docs/adr/ | sed -n 's|.*/\([0-9]\{3\}\)-.*|\1|p' | sort -u
   ```
2. Get new ADRs added on this branch:
   ```bash
   git diff origin/main...HEAD --diff-filter=A --name-only -- 'docs/adr/*.md'
   ```
3. For each new ADR whose 3-digit prefix appears in the main list:
   - Pick the lowest 3-digit number not used on main or on this branch — gaps in main's numbering are fine to fill (e.g., if main has 008 and 010 but no 009, use 009).
   - Rename the file with `git mv docs/adr/<old>.md docs/adr/<new>.md`.
   - Replace the file's own `# ADR-<old-num>:` header with `# ADR-<new-num>:`.
   - Retarget any branch-introduced references to the renamed ADR. List branch-modified files with `git diff origin/main...HEAD --name-only`, then for each file, use `git diff origin/main...HEAD -- <file>` to identify only the added lines (lines starting with `+`). Search those added lines for the old filename or `ADR-<old-num>`, and update matches to the new filename / `ADR-<new-num>`. Do not modify references that existed on main before this branch.
4. If any renames occurred, commit them: `git commit -m "docs: renumber ADR(s) to avoid collision with main"`.

#### Specs

1. Get spec numbers used on main:
   ```bash
   git ls-tree -r --name-only origin/main docs/specs/ | sed -n 's|.*/\([0-9]\{3\}\)-.*|\1|p' | sort -u
   ```
2. Get new specs added on this branch:
   ```bash
   git diff origin/main...HEAD --diff-filter=A --name-only -- 'docs/specs/*.md'
   ```
3. For each new spec whose 3-digit prefix appears in the main list:
   - Pick the lowest 3-digit number not used on main or on this branch — gaps in main's numbering are fine to fill.
   - Rename the file with `git mv docs/specs/<old>.md docs/specs/<new>.md`.
   - Replace the file's own `# Spec-<old-num>:` header with `# Spec-<new-num>:`.
   - Retarget any branch-introduced references to the renamed spec — including code/test docstrings that cite `docs/specs/<old>.md` paths. Use `git diff origin/main...HEAD --name-only` to list branch-modified files, then for each, look at added lines (`+`-prefixed in `git diff origin/main...HEAD -- <file>`) and update matches to the new filename / `Spec-<new-num>`. Do not modify references that existed on main before this branch.
4. If any renames occurred, commit them: `git commit -m "docs: renumber spec(s) to avoid collision with main"`.

If no new ADRs or specs exist on the branch, skip this step.

---

## Step 3 — Run Tests (if relevant files were changed)

### Hook tests

Check whether any hooks or hook tests were modified in this branch:

```bash
git diff main...HEAD --name-only | grep -E '^(\.(claude/hooks/|githooks/)|tests/hooks/)' || true
```

If there are matches and `bats` is available, run the test suite:

```bash
bats tests/hooks/
```

If no hook-related files were changed, or `bats` is not installed, skip this part.

### Bot tests

Check whether any bot code or bot tests were modified in this branch:

```bash
git diff main...HEAD --name-only | grep -E '^(bot/|tests/bot/|pyproject\.toml|uv\.lock)' || true
```

If there are matches, run the test suite:

```bash
uv run pytest tests/bot/
```

If no bot-related files were changed, skip this part.

### Result

If all tests pass, continue. If any fail, stop and fix, and test again. Repeat until all tests pass. Then, proceed to the next step.
If after 5 attempts the tests still fail, stop and notify the user for assistance.

---

## Step 4 — Codex Code Review Round 1 (Free Review)

Run `/run-codex-review review --base main --background` to review all changes on this branch.

Wait for the review to complete (check with `/run-codex-review status`). Once done, read the results with `/run-codex-review result`.

Fix all issues raised by Codex unless it brings unnecessary complications. Keep it simple!
Commit fixes with `fix: address codex code review round 1`.

**If this PR is spec-bearing** (apply the spec-presence gate in `/resolve-review` Bucket Routing — checks branch planning-doc header AND spec file presence on branch or main), classify each Codex finding per the **Bucket Routing** section of [`/resolve-review`](../resolve-review/SKILL.md). Apply Sanitization rules before forwarding reviewer text to the test-writer subagent. **Ambiguous findings halt `/create-pr`** — surface the finding to the user and wait for direction before pushing. For Behavior-change-bucket findings, follow the three-commit sequence per `/resolve-review`'s Step 4d Behavior-change pattern; the `(round 1)` suffix attaches to the impl commit only — the `docs:` spec commit and `test:` regen commit keep their bucket-standard messages without suffix.

If the review found no issues, proceed.

---

## Step 5 — Codex Code Review Round 2 (Critical Only)

Run `/run-codex-review adversarial-review --base main --background` with focus text:

> Only flag issues that meet one of these criteria:
> 1. The issue would likely cause a critical bug if left unresolved
> 2. The issue would be extremely difficult to resolve once a bug occurs in production

Wait for completion, then read the results.

Evaluate each finding against the two criteria above:

- **Meets criteria**: fix the issue and commit with `fix: address codex code review round 2 (critical)`.
- **Does not meet criteria**: skip it.

**Bucket routing applies to round-2 findings as well.** See Step 4's bucket-routing paragraph above; substitute `(critical)` for `(round 1)` in commit messages — same rule applies, the suffix attaches to single-commit buckets and to the impl commit of Behavior-change. Ambiguous critical findings halt `/create-pr` and surface to the user.

---

## Step 6 — Check Current State Before Pushing

Run the following to verify the branch state, then proceed straight to Step 7. Do not pause for user confirmation:

```bash
git status

git log main..HEAD --pretty=format:"%h  %s  (%cd)" --date=short
```

---

## Step 7 — Push the Branch

Run `git push -u origin <current-branch>`.

If the push fails, report the error and stop — do not force push, resolve the cause of the failure.

---

## Step 8 — Determine Issue Linkage

Check whether this PR closes a GitHub issue:

1. Check the branch name for an issue number (e.g. `42-feature-name` → `#42`). If found, use it.
2. Otherwise, if `$ARGUMENTS` is provided, use that.
3. If neither yields a number, create the PR without an issue reference.

**Also: fetch the branch-linked issue body and grep for `^Original request: #[0-9]+$`.** `/start-dev` writes this line into new development issue bodies when the branch-linked issue differs from the originally requested one. If the grep finds a different issue number, add a second `Closes #<N>` for that original issue to the PR body in Step 8. If the fetch fails or no line is found, proceed with just the branch-linked close.

---

## Step 9 — Create the Pull Request

Run `gh pr create` with:

- A short, clear title summarizing the change
- A body that includes:
  - A brief description of what was changed and why
  - One `Closes #N` line per issue identified in Step 7 (one in the common case; two when a separate original request issue was found)
  - A short testing checklist if relevant

Print the PR URL when done.

---

## Step 10 — Wait for CI and Fix Failures

Wait for all CI checks to pass:

```bash
gh pr checks <PR_NUMBER> --watch --fail-fast
```

If any check fails, investigate the failure, fix it, commit, and push. Then re-check CI. You have a maximum of **3 attempts**:

- **Attempt 1**: investigate and fix.
- **Attempt 2**: investigate and fix.
- **Attempt 3**: **stop** and ask the user for help. Summarize what failed, what you tried, and why it's not working.

If no CI checks are configured, proceed immediately.

---

## Step 11 — Check if settings/hook files were changed

Check whether any settings or hook files were changed in this PR:

```bash
git diff main...HEAD --name-only | grep -E '^(\.(claude/(settings\.json|settings\.local\.json|hooks/)|githooks/)|scripts/install-hooks\.sh)'
```

If any matches are found, remind the user to verify that appropriate entries exist in `.claude/settings.json` under both `permissions.deny` (block Edit/Write through Claude Code) and `sandbox.filesystem.denyWrite` (block writes at the OS sandbox level) to protect those files from future Claude Code modification.
