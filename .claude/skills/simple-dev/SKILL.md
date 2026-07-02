---
name: simple-dev
description: Simple development - clarify requirements, draft a plan, checkout a branch, modify files, commit, push, and open a PR
allowed-tools: Agent, Bash, Read, Write, Glob, TaskCreate, TaskUpdate
argument-hint: "[--no-test] <feature description>"
---

You are facilitating a simple development task autonomously. 

The feature to develop: $ARGUMENTS

**Flag check.** `--no-test` is recognized **only when it is the first whitespace-delimited token in `$ARGUMENTS`** (e.g., `/simple-dev --no-test fix foo`). A literal `--no-test` later in the description is part of the description, not a flag. When the leading-token flag is present, set the no-test mode; spec authoring still happens but the test-writer flow is skipped (see [ADR-014 §12](../../../docs/adr/014-spec-driven-tdd.md)).

---

ToDo List.

Do NOT skip any steps. Follow the checklist below.

Before starting, mirror each top-level checklist item into the task list via `TaskCreate`. As you work, call `TaskUpdate` to set each task to `in_progress` when you begin it and `completed` when it's done.

- [ ] Ensure latest main
  - [ ] Check for uncommitted changes. If the working tree is dirty, stop and ask the user to commit or stash first.
  - [ ] Switch to `main` and pull.
- [ ] Clarify requirements
  - **If `$ARGUMENTS` references a GitHub issue** (bare `#NNN`, `NNN`, or a `github.com/.../issues/NNN` URL), fetch the issue's title, body, **and comments** before asking the user anything
  - Ask the user if anything is ambiguous about the feature description. Keep it focused — aim for 2-3 targeted questions at most.
  - If ambiguity remains, ask follow-up questions until the requirements are clear enough to write a plan. If clear, move forward.
- [ ] Read relevant files and ADRs
  - Understand the codebase and context for the feature.
- [ ] Plan file changes
  - Keep changes simple, small, and clear.
- [ ] Determine if this feature needs a spec — apply the Spec Criterion ([ADR-014 §6](../../../docs/adr/014-spec-driven-tdd.md)). Spec-bearing = adds/modifies executable contract (Python under `bot/`, shell under `.claude/hooks/` / `.githooks/`, runtime scripts under `scripts/`). Prose-only = markdown / non-runtime config / test-files-only. Default on ambiguity: write the spec.
- [ ] (If spec-bearing AND not `--no-test`) Write spec doc — create `docs/specs/NNN-<slug>.md` using `docs/specs/000-template.md`. Numbering per `docs/specs/README.md`. Commit: `docs: add spec NNN-<slug>`.
- [ ] (If spec-bearing AND not `--no-test`) Extract API manifest — `uv run python scripts/extract_api.py <target-paths>`. Capture stdout. Shell hooks skip this (no Python AST).
- [ ] (If spec-bearing AND not `--no-test`) Generate failing tests via test-writer subagent — invoke the Agent tool with `subagent_type: test-writer`. Use the same prompt template as `/develop` Phase 2.A step 4 (spec content, language, manifest, target test paths, no-impl-read constraint, paraphrased planner notes). Verify the agent reports all new tests fail. On halt → stop, surface report, wait for user direction. Commit: `test: add failing tests from spec NNN-<slug>`.
- [ ] (If spec-bearing AND `--no-test`) Write spec doc only — same as above but skip the manifest + test-writer + failing-tests bullets. Spec authoring is NOT skipped under `--no-test` (per ADR-014 §12).
- [ ] Create and Push Branch
- [ ] Modify Files using the plan
  - For spec-bearing features, this is the implementation step. The commit ordering invariant is `docs:` spec → `test:` failing tests → `<feat|fix>:` impl, three separate commits. Under `--no-test`, only `docs:` spec → `<feat|fix>:` impl.
- [ ] Test and Lint
  - [ ] Run tests and linter. Fix failures and repeat until both pass. Under `--no-test`, this still runs — only writing *new* tests is skipped; running existing tests is not.
- [ ] Commit Changes
  - For spec-bearing features, the spec and failing-tests commits already happened earlier. This step commits the impl with `<feat|fix>: <description>`.
- [ ] Sync with main
  - [ ] Fetch `origin main` to ensure the local ref is up to date.
  - [ ] If the branch is behind main, merge main into the branch and resolve any conflicts.
  - [ ] If there are conflicts you can't resolve, stop and ask the user for help.
- [ ] Check ADR and spec numbering and modify if necessary (skip if no new ADRs or specs on this branch)
  - [ ] If our ADR number is already used on main, change our ADR number so the numbers stay unique and contiguous.
  - [ ] If our spec number is already used on main, change our spec number so the numbers stay unique and contiguous.
  - [ ] Update all references / occurrences of the renamed ADR or spec number (including the file's own `# ADR-NNN:` / `# Spec-NNN:` header).
  - [ ] If any renames occurred, commit each rename group separately: `docs: renumber ADR(s) to avoid collision with main` and/or `docs: renumber spec(s) to avoid collision with main`.
- [ ] One review round with /run-codex-review
  - [ ] Run `/run-codex-review review --base main --background`
  - [ ] Wait for the review to complete
  - [ ] Once done, read the results with `/run-codex-review result`
  - [ ] Fix all issues raised by Codex unless it brings unnecessary complications. Keep it simple!
  - [ ] **If this PR is spec-bearing** (apply the spec-presence gate in `/resolve-review` Bucket Routing — checks branch planning-doc header AND spec file presence on branch or main), classify each Codex finding per the **Bucket Routing** section of [`/resolve-review`](../resolve-review/SKILL.md). Apply Sanitization rules before forwarding reviewer text to the test-writer subagent. Ambiguous findings halt `/simple-dev` and surface to the user. For Behavior-change-bucket findings, follow the three-commit sequence per `/resolve-review`'s Step 4d Behavior-change pattern; the `(round 1)` suffix attaches to the impl commit only.
  - [ ] Test and Lint
    - [ ] Run tests and linter. Fix failures and repeat until both pass.
    - [ ] If changes were made, commit.
  - [ ] Commit fixes with `fix: address codex code review round 1` (single-commit buckets only — Test-only and Internal-impl. Behavior-change uses the three-commit sequence with `(round 1)` only on the impl commit).
If the review found no issues, proceed.
- [ ] Open PR

---

## Sanitization rules (applied before any text reaches the test-writer subagent)

The test-writer reads only the spec and the AST manifest. Any planner notes or reviewer comments forwarded into its prompt MUST be paraphrased first:

- **Drop:** absolute or relative paths under `bot/`, `.claude/hooks/`, `.githooks/`, `scripts/`; line numbers; identifier names that do not appear in the spec.
- **Keep:** the behavioral claim (the *what*, not the *where*).
- **If paraphrasing reduces a claim to nothing**, the spec is incomplete. Halt, update the spec, commit a `docs:` spec edit, then re-run sanitization.

No regex sanitizer is mandated — paraphrasing is done by the orchestrator inline. Verbatim reviewer text is never forwarded.
