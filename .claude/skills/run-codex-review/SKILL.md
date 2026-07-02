---
name: run-codex-review
description: Run a Codex code review (regular or adversarial) by calling the codex-companion runtime directly, with a Claude Code subagent fallback when Codex is unavailable
allowed-tools: Bash, Read, Glob, Agent
argument-hint: "<review|adversarial-review|status|result> [--base <ref>] [--background] [--wait] [--scope <auto|working-tree|branch>] [focus text]"
---

Run a Codex review by calling the `codex-companion.mjs` script directly via Bash, bypassing plugin `disable-model-invocation` restrictions.

If Codex is unavailable for `review` / `adversarial-review` modes (script missing, runtime error, or sandbox-blocked output per issue #142), fall through to **Step 6 — Claude Code review fallback**, which runs a subagent against the branch diff. The fallback never applies to `status` / `result` modes, since those operate on an already-launched Codex job.

Raw arguments:
`$ARGUMENTS`

---

## Step 1 — Resolve the codex-companion.mjs path

Find the script dynamically (the version directory can change on plugin updates):

```bash
ls -d ~/.claude/plugins/cache/openai-codex/codex/*/scripts/codex-companion.mjs 2>/dev/null | sort -V | tail -1
```

Note the output path — all subsequent steps refer to it as `$CODEX_SCRIPT`. Substitute the actual resolved path when running `node` commands.

If the output is empty:

1. Tell the user: "Codex companion script not found. The OpenAI Codex plugin may not be installed or the cache path has changed."
2. Suggest: "Run `/codex:setup` to install or reconfigure the Codex plugin."
3. Branch on mode:
   - `review` or `adversarial-review`: tell the user "Falling back to Claude Code review subagent (foreground)." and jump to Step 6.
   - `status` or `result`: stop. The fallback does not apply to meta-operations on an already-launched Codex job.

---

## Step 2 — Parse mode and arguments

The first positional word in `$ARGUMENTS` is the mode. Valid modes:

| Mode | Description |
|------|-------------|
| `review` | Standard code review |
| `adversarial-review` | Challenge review questioning design choices and assumptions |
| `status` | Check progress of a background review |
| `result` | Retrieve results of a completed review |

Extract the mode and pass all remaining arguments through unchanged.

For `status` and `result` modes, run the command with any remaining arguments:
```bash
node "$CODEX_SCRIPT" <mode> <remaining-args>
```
Then:
- `status`: return the output verbatim and stop — no further steps needed.
- `result`: continue to Step 5 to scan the output for sandbox-blocked markers (issue #142) before returning. The scan is what catches the failure mode this skill is meant to handle.

---

## Step 3 — Run the review

**Shell safety:** All arguments (flags, focus text, etc.) MUST be passed as a single-quoted string to prevent shell metacharacter injection. Use single quotes around the entire argument string, and escape any embedded single quotes with `'\''`.

### If `--background` is present in the arguments:

Launch the review with Bash in the background:
```bash
node "$CODEX_SCRIPT" <mode> '<remaining-args-single-quoted>'
```
Use `Bash` with `run_in_background: true`.

Tell the caller: "Codex review started in the background. Use `/run-codex-review status` to check progress and `/run-codex-review result` to read findings."

### If `--wait` is present, or no execution flag is given:

Run in the foreground:
```bash
node "$CODEX_SCRIPT" <mode> '<remaining-args-single-quoted>'
```

Return the output verbatim.

---

## Step 4 — Handle failure

If the `node` command exits with a non-zero status:

1. Tell the user what failed: include the exit code and any stderr output.
2. Suggest: "Check that Node.js is available and the Codex plugin is properly configured. Try `/codex:setup`."
3. Branch on mode:
   - `review` or `adversarial-review` (foreground / `--wait` paths only — `--background` exits 0 immediately): tell the user "Falling back to Claude Code review subagent (foreground)." and continue to Step 6.
   - `status` or `result`: stop. Fallback does not apply to meta-operations.

---

## Step 5 — Present results

Follow the `codex:codex-result-handling` conventions:

- Return the command stdout verbatim.
- Present findings ordered by severity.
- Preserve file paths and line numbers exactly as reported.
- Do NOT auto-fix any issues. The calling skill decides what to fix.
- If no findings, say so explicitly.

### Detect sandbox-blocked output

Before returning, scan stdout for any of these markers (issue #142):

- `Sandbox(Denied)`
- `bwrap: execvp`
- `failed to write models cache`
- `failed to renew cache TTL`

If a marker is present **and** the output contains no actual review findings (only error noise), Codex ran but its inner sandbox blocked it. Treat this as Codex unavailable:

1. Tell the user: "Codex output indicates its inner sandbox blocked the review (see issue #142)."
2. Do **not** create a new GitHub issue — the failure mode is already tracked by #142, and creating duplicates each run is noise.
3. Branch on mode:
   - `review` or `adversarial-review` (foreground / `--wait`): tell the user "Falling back to Claude Code review subagent." and continue to Step 6.
   - `result`: do **not** fall back here. The original background job snapshotted code at launch time; rebuilding from the current working tree could review different code than the user asked about. Instead tell the user: "Re-run `/run-codex-review review --base <ref>` (or with `--scope working-tree`) to trigger the fallback against a known target." Then stop.
   - `status`: stop.

Apply this scan in the foreground flow (Step 3 output) and the `result` mode flow (Step 2 output).

---

## Step 6 — Claude Code review fallback

Reached only from Steps 1, 4, or 5 above, and only for `review` / `adversarial-review`. Runs synchronously — `--background` is ignored here; tell the user up front.

### Step 6a — Gather inputs

1. Parse `--base <ref>` from `$ARGUMENTS`. Default to `main` if absent.
2. Parse `--scope <auto|working-tree|branch>` from `$ARGUMENTS`. Default to `auto`.
3. Parse remaining positional text from `$ARGUMENTS` as **focus text** for `adversarial-review` mode. For plain `review`, ignore focus text.
4. Resolve the effective scope:
   - `branch`: review the branch diff vs base.
   - `working-tree`: review uncommitted local changes (staged + unstaged + untracked).
   - `auto`: if `git status --porcelain` reports any uncommitted changes (staged, unstaged, or untracked), use `working-tree`; otherwise `branch`. Tell the user which scope was chosen.
5. Capture the diff and changed-file list for the resolved scope:

   **`branch` scope:**
   ```bash
   git diff <base>...HEAD
   git diff <base>...HEAD --name-only
   ```

   **`working-tree` scope:** combine staged, unstaged, and untracked files:
   ```bash
   git diff HEAD                              # staged + unstaged
   git ls-files --others --exclude-standard   # untracked file paths
   git status --porcelain                     # full changed-file list
   ```
   For untracked files, also dump their contents (e.g., `cat <file>`) so the agent can review them.

If the resolved scope yields an empty diff (no changed files), tell the user "No changes for scope `<scope>`; nothing to review." and stop — do not invoke the Agent.

### Step 6b — Invoke the review subagent

Use the `Agent` tool with `subagent_type: general-purpose` and a self-contained reviewer prompt. The prompt MUST include:

- **Role**: state the scope explicitly. For `branch`: "You are reviewing the diff between `<base>` and the current branch HEAD." For `working-tree`: "You are reviewing the uncommitted changes in the working tree (staged + unstaged + untracked)."
- **Inputs**: the changed-file list and the full diff (paste verbatim from Step 6a). For working-tree scope, include any untracked file contents collected in Step 6a.
- **Project context pointer**: instruct the agent to read root `CLAUDE.md` and any `CLAUDE.md` in changed directories before reviewing.
- **Mode-specific framing**:
  - `review`: "Flag likely bugs, regressions, and clear violations of project conventions. Skip nitpicks."
  - `adversarial-review`: "Adversarially challenge design choices and assumptions. Focus on what could break in production." Append the focus text from Step 6a if present.
- **Output format**: findings ordered by severity, each with `path:line` and a one-line rationale. If none, state "No issues found."
- **Do NOT auto-fix** — review only.

Run the agent in the foreground (no `run_in_background: true`) so the caller (e.g. `/simple-dev`, `/create-pr`) gets findings synchronously.

### Step 6c — Return findings

Return the agent's output verbatim, prefaced with:

> Codex unavailable — Claude Code review fallback results:

Then stop — do not perform any further Codex steps.
