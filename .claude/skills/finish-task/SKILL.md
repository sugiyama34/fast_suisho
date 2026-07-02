---
name: finish-task
description: Post-merge cleanup — update main, prune the remote, and print one paste-ready block of follow-up commands
allowed-tools: Bash
---

Clean up the local repository after a task branch has been merged.

Only `git fetch` and `git checkout main` run automatically. Everything else — force-deleting the merged branch, discarding stale worktree diffs, deleting other already-merged locals — is collected into a **single paste-ready command block** at the end, because the command allowlist and sandbox intentionally block those forms.

**Sandbox fallback.** If any command in this skill is blocked (allowlist, read-only mount, EBUSY, etc.), ask the user to run it on your behalf and continue from their result. Do **not** bypass the sandbox.

---

## Step 1 — Update main

```bash
git fetch origin main:main
git checkout main
```

Fetching into `main:main` first makes the local ref match `origin/main` before checkout, so the squash-merged worktree usually needs no file writes on switch. If the checkout is blocked by the sandbox, apply the Sandbox fallback above — ask the user to run it on your behalf and continue once they confirm success.

Remember the branch you just left (from the checkout output or `git reflog`) as `$PREV_BRANCH`.

---

## Step 2 — Prune stale remote refs

```bash
git fetch --prune
```

---

## Step 3 — Gather follow-up commands

Build a list `$CMDS` by running each check below and appending the matching command. Do not execute any of them.

1. **Leftover worktree diffs** — `git status --porcelain` lists any `M`/`A`/`D` entries for project files:
   - Append `git restore --staged --worktree -- <file1> <file2> …` (name each file explicitly; no `.`, `-A`, `*`, or globs).
2. **Delete the merged branch** — always:
   - Append `git branch -D $PREV_BRANCH`.
3. **Other already-merged locals** — `git branch --merged main | grep -Ev '^[* ]*main$'` lists any:
   - Append `git branch -d <name1> <name2> …` (one command covering all).

---

## Step 4 — Print the block

Emit the gathered commands as one fenced code block, in the order above:

> Please run this in your terminal:
>
> ```
> <all recorded commands, one per line>
> ```

If `$CMDS` is empty, print "No follow-up actions needed."
