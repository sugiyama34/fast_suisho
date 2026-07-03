---
name: resolve-review
description: Process unhandled PR review comments on the current branch's open PR — answer questions, modify code, or push back
allowed-tools: Agent, Bash, Read, Edit, Write, Glob, Grep
argument-hint: "[--no-codex-review]"
---

Autonomously process unhandled review comments on the current branch's open pull request.

For each unhandled thread: read context, decide the appropriate action, execute it, and reply on GitHub.

**Arguments.** `$ARGUMENTS` may contain the flag `--no-codex-review`. When present, skip Steps 5b and 5c (both Codex review rounds). Intended for documentation-only changes where running Codex review is unnecessary. All other steps (tests, lint, ADR check, push) still run.

---

## Step 1 — Detect the PR

Find the open PR for the current branch:

```bash
gh pr view --json number,url,title,headRefName
```

If no PR exists, stop and notify the user: "No open PR found for this branch."

Extract the PR number, owner, and repo name for subsequent API calls. Get owner/repo from:

```bash
gh repo view --json owner,name --jq '.owner.login + "/" + .name'
```

---

## Step 2 — Fetch review comments

Fetch all review comments on the PR:

```bash
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments --paginate
```

Reconstruct threads from the flat comment list:

1. **Root comments**: Comments where `in_reply_to_id` is `null` or absent. Each root starts a thread, keyed by the root's own `id`.
2. **Replies**: Comments where `in_reply_to_id` is set — attach each to the thread keyed by its `in_reply_to_id` value.
3. **Sort** each thread's comments by `created_at` (chronological).

---

## Step 3 — Filter to unhandled threads

For each thread, inspect the **last comment's `body`**:

- If it contains `<!-- replied-by-claude-code -->` **or** the shell-escaped variant `<\!-- replied-by-claude-code -->` → already handled, **skip**.
- Otherwise → unhandled, **process it**.

This marker is necessary because the human user and Claude Code share the same GitHub account, making author-based filtering impossible. The HTML comment is invisible in rendered markdown. The `<\!--` variant is accepted because the Bash tool escapes `!` to `\!` when commands pass through its shell layer, so a reply posted via `gh api … -f body='… <!-- … -->'` lands on GitHub with the backslash intact.

If no unhandled threads remain, report "No unhandled review comments" and stop.

---

## Bucket Routing

Shared decision logic referenced by Step 4b and Steps 5b/5c. Per [ADR-014 §10](../../../docs/adr/014-spec-driven-tdd.md).

### Spec-presence gate

Feature-level (not branch-level) — follow-up PRs on already-specced code still qualify. Check once, cache:

```bash
spec_bearing=false
plan=$(git diff main...HEAD --diff-filter=A --name-only -- 'docs/plans/[0-9][0-9][0-9][0-9]-*.md' 2>/dev/null | head -1)
[ -n "$plan" ] && head -5 "$plan" | grep -qE '^\*\*Spec-bearing:\*\*\s+yes\b' && spec_bearing=true
git diff main...HEAD --name-only | grep -q '^docs/specs/[0-9][0-9][0-9]-.*\.md$' && spec_bearing=true
git ls-tree -r --name-only origin/main docs/specs/ | grep -qE '/[0-9][0-9][0-9]-.*\.md$' && spec_bearing=true
```

If `spec_bearing=true` → apply Bucket Routing. Else → prose-only flow (Step 4d's "Prose-only" subsection); Step 6's `Bucket` column reads `n/a`.

The third signal (any spec on main) is intentionally permissive — false positives cost one extra classification; false negatives let tautological tests through.

### The four buckets

| # | Bucket | Trigger | Dispatch | Commits |
|---|--------|---------|----------|---------|
| 1 | **Test-only** | Change in test files only | test-writer subagent (paraphrase first) | one `test:` |
| 2 | **Internal impl** | Impl change, provably preserves public contract | Main agent; reply states *"cannot alter the contract of `<callable>` because `<reason>`"*. Tests-green is necessary but not sufficient — test failure auto-escalates to bucket 3 | one `fix:` or `refactor:` |
| 3 | **Behavior change** | Public contract changing | spec → test-writer → main agent | three commits: `docs:` spec → `test:` regen → `<feat\|fix>:` impl |
| 4 | **Ambiguous** | Cannot place in 1/2/3 | Ask human | none |

**Test-writer prompt** (buckets 1 and 3): use `/develop` Phase 2.A step 4 template, substitute "Paraphrased reviewer feedback" for "Paraphrased planner notes." On halt (pass-on-first-run / spec gap / manifest mismatch): surface verbatim, do not commit, stop.

**Ambiguous handling:** post clarifying reply with marker, continue with other threads. Marker provides cross-invocation persistence; next invocation re-surfaces the thread when the human replies.

### Sanitization rules

Before reaching test-writer (buckets 1 and 3):

- **Drop** paths under `bot/`, `.claude/hooks/`, `.githooks/`, `scripts/`; line numbers; identifiers not in the spec.
- **Keep** the behavioral claim (the *what*, not the *where*).
- **If paraphrasing yields nothing**, the spec is incomplete — halt, update the spec, commit `docs:` edit, re-run.

---

## Step 4 — Process each unhandled thread

For each unhandled thread:

### 4a — Read context

1. Read `path`, `line`, `diff_hunk`, and full comment chain.
2. `Read` the source file around `line`.
3. Understand the reviewer's ask.

### 4b — Decide and act

**Spec-bearing PRs:** classify per Bucket Routing, then dispatch (Test-only/Behavior-change → test-writer; Internal-impl → main agent with contract-preservation reply; Ambiguous → clarifying reply, continue).

**Prose-only PRs:** use judgment. (a) Distinguish questions (answer) from directives (act). (b) Reply to every thread.

### 4c — Reply to threads

Post a reply via the REST endpoint, ending with the `<!-- replied-by-claude-code -->` marker on the **same line** as the reply text. The entire `-f body='...'` argument stays on a single physical line — see "Shell Quoting for `gh api`" in CLAUDE.md for line-break (`<br>`) and apostrophe (`'\''`) handling.

```bash
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments/{root_comment_id}/replies \
  -f body='<reply text> <!-- replied-by-claude-code -->'
```

### 4d — Modify, test, commit, then reply (per bucket)

| Bucket | Commits | Notes |
|--------|---------|-------|
| Test-only | `test: <description>` | test-writer edits → run tests → reply with hash |
| Internal impl | `fix:`/`refactor: <description>` | main agent edits → run tests (any failure → `git checkout -- <file>`, restart as Behavior-change) → reply with hash + contract-preservation argument |
| Behavior change | `docs: update spec NNN-<slug>` → `test: regenerate tests from updated spec NNN-<slug>` (verify failing) → `<feat\|fix>: <description>` (verify all-green) | Reply listing all three hashes |
| Ambiguous | none | Clarifying reply with marker; continue |
| Prose-only | `fix: <description>` | modify → test → commit → reply with hash |

**If tests fail or commit fails:** do NOT post the marker. Stop and report.

---

## Step 5 — Post-change tasks and push

After all threads are processed, if any commits were made:

### 5a — Run tests

Check which files were changed and run the appropriate test suites:
- Python code changed: `uv run pytest tests/`

Verify no lint errors remain in changed files.

### 5b — Codex Code Review Round 1 (Free Review)

**Skip this step (and 5c) if `$ARGUMENTS` contains `--no-codex-review`.** Proceed directly to 5d.

Run `/run-codex-review review --base main --background` to review all changes on this branch.

Wait for the review to complete (check with `/run-codex-review status`). Once done, read the results with `/run-codex-review result`.

Fix all issues unless it brings unnecessary complications.

Classify each finding per Bucket Routing (spec-bearing PRs only). Sanitization applies to Codex output too. Commit conventions: single-commit buckets use `fix: address codex code review round 1`; Behavior-change uses the three-commit sequence (`docs:` spec → `test:` regen → `<feat|fix>: <description> (round 1)`). Ambiguous findings: halt and surface to user.

If the review found no issues, proceed.

### 5c — Codex Code Review Round 2 (Critical Only)

**Skip this step if `$ARGUMENTS` contains `--no-codex-review`.**

Run `/run-codex-review adversarial-review --base main --background` with focus text:

> Only flag issues that meet one of these criteria:
> 1. The issue would likely cause a critical bug if left unresolved
> 2. The issue would be extremely difficult to resolve once a bug occurs in production

Evaluate each finding against the two criteria above:

- **Meets criteria**: fix the issue and commit with `fix: address codex code review round 2 (critical)`. Bucket routing applies — single-commit buckets get the message above; Behavior-change uses the three-commit sequence with `(critical)` on the impl commit only.
- **Does not meet criteria**: skip it.

### 5d — Check ADR and spec numbering

If any ADRs or specs were added on this branch, verify their 3-digit prefixes don't collide with numbers on `origin/main`. Parallel branches may have taken the same NNN while this PR was open.

```bash
git fetch origin main
```

Then run the ADR check, then the spec check. Commit each rename group separately.

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
4. If any renames occurred, commit them: `git commit -m 'docs: renumber ADR(s) to avoid collision with main'`.

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
4. If any renames occurred, commit them: `git commit -m 'docs: renumber spec(s) to avoid collision with main'`.

If no new ADRs or specs exist on the branch, skip this step.

### 5e — Push

```bash
git push
```

If push fails, report the error to the user.

---

## Step 6 — Summary

Print a summary table of all processed threads.

Bucket values: `Test-only`, `Internal impl`, `Behavior change`, `Ambiguous`, or `n/a` (prose-only PRs or non-code threads like questions).

| # | File:Line | Bucket | Type | Action | Status |
|---|-----------|--------|------|--------|--------|
| 1 | `bot/cogs/foo.py:42` | n/a | Question | Answered | Replied |
| 2 | `bot/config.py:15` | Internal impl | Suggestion | Accepted | Code modified, committed, replied |
| 3 | `bot/main.py:8` | Ambiguous | Suggestion | Clarification asked | Replied (waiting on human) |
| 4 | `tests/bot/test_foo.py:12` | Test-only | Suggestion | Test regenerated | test-writer commit + reply |
| 5 | `bot/handler.py:30` | Behavior change | Directive | Spec + tests + impl updated | 3 commits + reply |
