---
name: start-dev
description: Feature kickoff — clarify requirements, draft a planning doc, and create GitHub issues
allowed-tools: Agent, Bash, Read, Write, Glob
argument-hint: "<feature description>"
---

You are facilitating a structured development kickoff. No code is written in this skill — only planning and issue creation.

The feature to develop: $ARGUMENTS

**Operating principle:** Only ask the user when a step explicitly requires it, or when it is strongly necessary (e.g., blocking errors, ambiguous requirements). Do not stop just to report progress — keep moving through the steps autonomously.

---

## Step 1 — Ensure Latest Main

Unless the user explicitly says to work from the current branch, ensure you are on the latest `main`:

1. Check for uncommitted changes. If the working tree is dirty, stop and ask the user to commit or stash first.
2. Switch to `main` and pull:

```bash
git checkout main
git pull origin main
```

---

## Step 2 — Clarify Requirements

Review the feature description above.

**If `$ARGUMENTS` references a GitHub issue** (bare `#NNN`, `NNN`, or a `github.com/.../issues/NNN` URL), fetch the issue's title, body, **and comments** first: `gh issue view <NNN> --json title,body,comments`. Treat the comments as part of the brief — they often record follow-up scope decisions that the body alone doesn't capture. Forward the merged context (body + comments) into the `/grill-me` interview below so the user doesn't have to repeat themselves.

Use `/grill-me` skill to interview me about this feature until we have a shared understanding of the requirements. Pass the feature description as the skill argument.

---

## Step 2.5 — Determine if this feature needs a spec

Apply the **Spec Criterion** from [ADR-014 §6](../../../docs/adr/014-spec-driven-tdd.md). A feature is **spec-bearing** if its diff adds or modifies an *executable contract*:

- Python under `bot/` (new cog, modified handler, new helper with observable output)
- Shell hooks under `.claude/hooks/` or `.githooks/`
- Runtime scripts under `scripts/` invoked by code, CI, or other skills

A feature is **prose-only** if its diff is limited to:

- Markdown (skill prose, agent prose, ADRs, plans, daily logs, READMEs)
- Settings/config without runtime behavior (`.claude/settings.json`, `pyproject.toml` metadata, `Dockerfile` comments)
- Test files only

**Mixed changes:** apply the criterion to the executable portion only. **Default on ambiguity:** write the spec.

Make the call yourself based on the requirements gathered in Step 2. Do not ask the user. The decision is recorded in the planning doc (Step 3 below) so downstream skills (`/develop` Phase 2, `/resolve-review` bucket routing) can read it.

---

## Step 3 — Draft the Planning Doc

Use the **Plan subagent** (Agent tool with `subagent_type: Plan`) to analyze the codebase and read relevant ADRs (in `docs/adr/`) and draft a plan for the feature. Give the subagent the feature description and any clarifications gathered in Step 2.

Once the subagent returns, write its output to `docs/plans/YYYY-MM-DD-NN-<feature-slug>.md` — where `NN` is a zero-padded 2-digit sequence (`01`, `02`, …) that increments for each plan created on the same date. Check `docs/plans/` for existing files starting with today's date and pick the next unused number (start at `01` if none exist). Use a short kebab-case slug derived from the feature name. Create the `docs/plans/` directory if it does not exist.

**Spec-bearing record.** The planning doc MUST start with a one-line header right after the `# Plan — …` title, matching the regex `^\*\*Spec-bearing:\*\*\s+(yes|no)\b`, in one of these two exact forms based on the Step 2.5 decision:

- Spec-bearing: `**Spec-bearing:** yes — covers `<module-or-script-path>` (spec: `docs/specs/NNN-<slug>.md`)`
- Prose-only:   `**Spec-bearing:** no — prose-only`

`/develop` Phase 2 parses this header to branch on spec-bearing. Cross-check: if the line says `yes`, the spec file at the referenced path MUST exist; if `no`, no spec file should exist for this feature. Verify both conditions before committing.

The plan is **free-form** — let the subagent structure it however best fits the feature. However, the plan **must** include these two elements:

1. **A task breakdown** with checkboxes (`- [ ] Task — description`). Steps 7 and 8 will use this list to create GitHub issues, and the checkboxes are also used to track implementation progress during `/develop`.
2. **ADR identification** — proactively evaluate whether any decisions in this feature warrant an ADR (new tool, pattern, structural change, or significant approach choice). If so, include "Create ADR-NNN" as a task. Do not ask the user whether an ADR is needed — make the judgment yourself.

**Do not commit yet.** The plan file stays uncommitted until a linked branch is created in Step 9.

**If spec-bearing per Step 2.5**, the Plan subagent also produces `docs/specs/NNN-<slug>.md` using `docs/specs/000-template.md` as the starting point. The spec describes the behavioral contract only — inputs, outputs, behaviors (`B1`, `B2`, …), edge cases (`E1`, …), error conditions (`ER1`, …), invariants (`I1`, …). It does NOT restate planning material. Numbering follows `docs/specs/README.md`:

1. Candidate set = union of `git ls-files 'docs/specs/[0-9][0-9][0-9]-*.md'` and `ls docs/specs/[0-9][0-9][0-9]-*.md 2>/dev/null`.
2. Pick `max(NNN) + 1` zero-padded to 3 digits (`001` if empty).
3. On `Write` collision or post-write `git status` showing another untracked file at the same NNN, retry with the next free number.

The slug should match the planning doc slug for cross-readability. The spec file stays uncommitted until Step 9.

**For multi-file features** (e.g., a cog plus its service module), one spec covers the whole coherent unit. Do not split into multiple spec docs for the same feature.

---

## Step 4 — Stress-Test the Plan

Use `/grill-me` skill to stress-test the draft plan with the user. Pass the plan file path and a summary of the plan as the skill argument.

Update the plan doc with any changes that emerge from the discussion.

---

## Step 5 — Codex Plan Review Round 1 (Free Review)

Run `/run-codex-review review` to review the uncommitted planning doc.

Wait for the review to complete (check with `/run-codex-review status`). Once done, read the results with `/run-codex-review result`.

Fix **all** issues raised by Codex — update the plan doc accordingly.

Show the revised plan to the user.

If the review found no issues, proceed.

---

## Step 6 — Codex Plan Review Round 2 (Critical Only)

Run `/run-codex-review adversarial-review` with focus text:

> Only flag issues that meet one of these criteria:
> 1. The issue would likely cause a critical bug if left unresolved
> 2. The issue would be extremely difficult to resolve once a bug occurs in production

Evaluate each finding against the two criteria above:

- **Meets criteria**: fix the issue in the plan doc.
- **Does not meet criteria**: skip it.

Show the final plan to the user and **pause here**. Do not proceed until the user confirms.

---

## Step 7 — Decide on Issue Structure

The default is **one issue for the entire feature**. Only split into multiple issues if there is a compelling reason:

- Tasks span genuinely different technical domains (e.g., bot logic vs. infrastructure)
- Tasks are large enough to benefit from independent review and merging
- Tasks can realistically be worked on in parallel

**If splitting into multiple issues:** create a parent issue describing the overall feature, then attach each task as a native GitHub sub-issue. Do not use a checklist of linked issues — use GitHub's sub-issue feature instead. The parent issue is the single tracker; sub-issues are closed individually as their PRs merge.

---

## Step 8 — Create GitHub Issues

Once the structure is confirmed, create the issue(s) using `gh issue create`.

Each issue should include:
- A clear, action-oriented title
- A short description of what needs to be done and why
- Labels if available (`gh label list` to check)

**If multiple issues:** create the parent issue first, then create each sub-issue and attach it to the parent with:
```bash
gh api repos/{owner}/{repo} --method POST \
  -f parent_issue_id=<parent-node-id> \
  /issues/<sub-issue-number>/sub_issues
```

**If `$ARGUMENTS` is an issue number AND you're creating a new issue for the work** (single-issue: the one new issue; multi-issue: every sub-issue, not the parent), add a line `Original request: #<ARGUMENTS>` at the bottom of each new issue's `--body`. `/create-pr` reads this line and adds `Closes #<ARGUMENTS>` to the PR. In the multi-issue case, `#<ARGUMENTS>` closes on the first sub-issue PR to merge (GitHub ignores subsequent `Closes` directives for already-closed issues).

---

## Step 9 — Create Linked Branch and Commit Plan

Use `gh issue develop` to create a branch linked to the issue you will implement first:

- **Single issue:** use that issue's number.
- **Multiple issues:** use the **first sub-issue** number, not the parent.

```bash
gh issue develop <issue-number> --checkout
```

This creates a branch with the issue number in the name (e.g., `42-add-oauth`), ensuring `/create-pr` will auto-add `Closes #42`.

Now commit the planning doc and any review changes. If the feature is **spec-bearing** per Step 2.5, also stage the spec doc and append a body line referencing it. Stage explicit filenames (not the whole `docs/plans/` directory) per the project's git-add policy:

```bash
git add docs/plans/<plan-file>
# If spec-bearing, also stage the spec:
git add docs/specs/<spec-file>

# Prose-only commit (no spec):
git commit -m "docs: add planning doc for <feature-slug>"

# Spec-bearing commit (with spec body line):
git commit -m "docs: add planning doc for <feature-slug>" \
           -m "Adds spec docs/specs/NNN-<slug>.md"
```

Print each issue number and URL, then outline the implementation cycle:

**Single issue:** implement on this branch, then run `/create-pr`.

**Multiple issues:**
1. Start with the first sub-issue — implement on this branch
2. Run `/create-pr` — the PR closes the sub-issue on merge
3. After merge, run `gh issue develop <next-sub-issue-number>` to create a fresh linked branch
4. Implement the next sub-issue on that branch and repeat
