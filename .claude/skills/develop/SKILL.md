---
name: develop
description: Full development cycle — plan a feature, implement it, and open a PR
allowed-tools: Agent, Bash, Read, Edit, Write, Glob, Grep
argument-hint: "[--no-test] <feature description>"
---

You are running the full development cycle: planning, implementation, and PR creation.

The feature to develop: $ARGUMENTS

**Operating principle:** Only ask the user when a step explicitly requires it, or when it is strongly necessary (e.g., blocking errors, ambiguous requirements). Do not stop just to report progress — keep moving through the steps autonomously.

---

## Phase 1 — Plan the Feature

Run `/start-dev $ARGUMENTS` to kick off planning.

Wait for `/start-dev` to complete fully (plan approved, issues created, branch ready, plan committed). Do not proceed to Phase 2 until `/start-dev` has finished all its steps.

Note the planning doc path — you will need it in Phase 2.

---

## Phase 2 — Implement the Feature

Read the planning doc from `docs/plans/` (created in Phase 1).

**Multi-issue scope rule:** If `/start-dev` created multiple sub-issues, only implement tasks belonging to the **current sub-issue** (the one linked to this branch). Do not work on tasks for other sub-issues.

**Flag check.** `--no-test` is recognized **only when it is the first whitespace-delimited token in `$ARGUMENTS`** (e.g., `/develop --no-test add foo handler`). A literal `--no-test` later in the feature description (e.g., `/develop add --no-test option to bar`) is part of the description, not a flag. When the leading-token flag is present, set the no-test mode; skip test-writer invocation and post-impl verification per [ADR-014 §12](../../../docs/adr/014-spec-driven-tdd.md). Spec authoring is NOT skipped. See Phase 2.NT below.

**Read the spec-bearing record.** Look for the header line matching `^\*\*Spec-bearing:\*\*\s+(yes|no)\b` near the top of the planning doc. **Cross-check spec file presence:** if the header says `yes`, the spec file referenced in the header MUST exist at `docs/specs/NNN-<slug>.md`; if the header says `no`, no spec file should exist for this feature. If the header is missing/malformed, OR header and spec-file presence disagree, halt with `plan/spec out of sync — header says <X> but spec file <exists|missing>` and ask the user.

### Sanitization rules (applied before any text reaches the test-writer subagent)

The test-writer reads only the spec and the AST manifest. Any planner notes or reviewer comments forwarded into its prompt MUST be paraphrased first:

- **Drop:** absolute or relative paths under `bot/`, `.claude/hooks/`, `.githooks/`, `scripts/`; line numbers; identifier names that do not appear in the spec.
- **Keep:** the behavioral claim (the *what*, not the *where*). Example: "the handler must reject empty messages with status 400" survives; "see `bot/cogs/foo.py:42`" does not.
- **If paraphrasing reduces a claim to nothing**, the spec is incomplete. Halt, update the spec, commit a `docs:` spec edit, then re-run sanitization.

No regex-based sanitizer is mandated — paraphrasing is done by the orchestrator inline. Verbatim reviewer text is never forwarded.

### Phase 2.A — Spec-bearing path (default)

1. **Resolve target paths.** Read the planning doc's task list to determine the impl module path(s). A feature may touch multiple impl files — collect all. Pick test paths by mirroring each under `tests/`:
   - `bot/cogs/foo.py` → `tests/bot/cogs/test_foo.py`
   - `bot/services/foo_service.py` → `tests/bot/services/test_foo_service.py`
   - `scripts/foo.py` → `tests/scripts/test_foo.py`
   - `.claude/hooks/foo.sh` or `.githooks/foo` → `tests/hooks/test_foo.bats`

   For multi-file features, a **single spec covers the whole unit** and a **single test-writer invocation** generates tests across all target test paths.

2. **Extract the API manifest (Python targets only).** Run:
   ```bash
   uv run python scripts/extract_api.py <target-paths>
   ```
   Capture stdout. Shell-hook targets skip this step.

3. **Paraphrase planner claims into spec vocabulary.** Apply the Sanitization rules above before forwarding any planner/reviewer text to the test-writer.

4. **Invoke the test-writer subagent.** Use the Agent tool with `subagent_type: test-writer`. Build the prompt from this template:

   ```
   ## Spec
   {full contents of docs/specs/NNN-<slug>.md, inlined verbatim}

   ## Target language
   {"Python (pytest)" | "Shell (bats)"}

   ## API manifest
   {for Python: stdout of `scripts/extract_api.py <target-paths>`, inlined verbatim}
   {for shell: omit; instead list target hook scripts and adjacent helpers under tests/hooks/helpers/}

   ## Target test paths
   {test paths chosen in step 1}

   ## Constraints
   - Do NOT Read implementation files under `bot/`, `.claude/hooks/`, `.githooks/`, or `scripts/`.
   - Do NOT Read existing tests for the same target unless explicitly named below as spec-derived.
   - Every new test MUST fail on first run. If any new test passes, halt and report — do not commit.
   - Map tests to spec labels (`test_b1_…`, `test_e1_…`, `test_er1_…`, `test_i1_…`).

   ## Spec-derived existing tests for the same target
   {paths of existing tests explicitly named as spec-derived; empty list if none}

   ## Paraphrased planner notes
   {paraphrased claims from Phase 1 / planning doc — already stripped of impl-leak references}
   ```

5. **Verify failing-test deliverable.** Read the agent's reply.
   - All new tests fail → continue.
   - Agent halts (pass on first run, spec gap, or manifest mismatch) → **stop the skill**, surface the agent's report verbatim, do NOT commit, wait for user direction.

6. **Commit failing tests:** `git commit -m "test: add failing tests from spec NNN-<slug>"`. Stage only the new test files (not the whole `tests/` tree).

7. **Implement.** Now write the implementation against the failing tests. Standard TDD: red → green incrementally. You may Read the test files.

8. **Post-impl verification.**
   - Python: `uv run pytest <test-path> --tb=short` — all-green required.
   - Shell: `bats tests/hooks/` — all-green required.
   - On failure, fix the impl and re-run. Do NOT modify the failing tests to make them pass. If the test itself is wrong, classify as a spec gap → halt, surface to user.

9. **Commit impl:** `git commit -m "<feat|fix>: <description>"`. Use `feat:` for new behavior, `fix:` for bug fixes.

**Commit ordering invariant.** The branch's commit log for this feature MUST show: `docs:` spec → `test:` failing tests → `<feat|fix>:` impl. Three separate commits. If tempted to squash locally, don't — the order is auditable evidence the spec-driven flow was followed.

### Phase 2.B — Prose-only path

Skip the test-writer flow entirely. Implement the prose change. No tests required per [ADR-014 §6 / §8](../../../docs/adr/014-spec-driven-tdd.md). Proceed straight to Phase 3.

### Phase 2.NT — `--no-test` mode (spec-bearing feature without new-test authoring)

When `$ARGUMENTS` starts with `--no-test`:

1. Spec authoring follows Phase 2.A steps 1–3 (paths, paraphrase). The spec is still authored and committed.
2. **Skip** Phase 2.A steps 4–6 (test-writer invocation, failing-test commit). No new tests are written in this PR.
3. Implement directly (Phase 2.A step 7).
4. **Keep** Phase 2.A step 8 (post-impl verification). Existing tests still run — `--no-test` means *don't author new tests*, not *don't run tests*. If the repo has no test suite at all (e.g., downstream DL experiment repo), the test command finds nothing and exits clean — that's fine.
5. Commit impl (Phase 2.A step 9).

Commit order collapses to two: `docs:` spec → `<feat|fix>:` impl.

See [ADR-014 §12](../../../docs/adr/014-spec-driven-tdd.md) for the portability rationale.

Once all in-scope tasks are checked off, proceed to Phase 3.

---

## Phase 3 — Create the Pull Request

Run `/create-pr` to commit any remaining changes, run code review, push, and open the PR.

Wait for `/create-pr` to complete fully.

### Multi-Issue Follow-Up

If `/start-dev` created multiple sub-issues, after the PR is created, print:

1. The remaining sub-issue numbers and titles
2. Instructions: after this PR is merged, run `/finish-task`, then run `/develop` again targeting the next sub-issue
