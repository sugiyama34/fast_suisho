---
name: experiment
description: >-
  Run deep-learning experiments as disciplined hypothesis tests — frame the
  question, set up a self-contained per-experiment folder, launch training/eval
  runs (confirming before heavy GPU jobs), track metrics, analyze results
  against a baseline, and write a human-readable report with tables and plots.
  Use this whenever the user is doing experimental ML work: launching or
  preparing a training/finetuning/sampling/eval run, an ablation, or a
  hyperparameter sweep; saying things like "let's try X and see if it helps,"
  "does this change improve FID/accuracy/loss," "compare these two
  runs/checkpoints," "track this experiment," "analyze the results," or "write up
  what we found." Trigger even when the user doesn't say the word "experiment"
  but is clearly testing whether a change moves a metric, or wants results
  organized, compared, or reported reproducibly.
---

# Deep-learning experiments

An experiment is a **hypothesis test**, not just "run training." The value of
this skill is the discipline around the run: a clear question decided by a
metric, a fair baseline, a result you can trust isn't seed noise, and a writeup
someone else (including the user in three months, or a paper reviewer) can read
and reproduce. The folder structure and templates are scaffolding for that
discipline — never the point.

Work at the **user's altitude**: a quick "does bumping LR help?" gets a light
touch; "I'm running the ablation table for the paper" gets the full rigor. Don't
impose paperwork the user didn't ask for, but do ask the judgment questions
below — they're what separates an experiment from a folder of log files.

**Always lead with a recommendation when you ask.** Every question you put to the
user (compute budget, eval rigor, design trade-offs, anything via
`AskUserQuestion`) must include the option you'd pick and a one-line *why* — put
the recommended option first and mark it `(Recommended)` in its label, not buried
in the description. You've already reasoned through these defaults; presenting
options flat and making the user guess which you'd choose wastes the expertise
that makes the question worth asking. Recommend, then let them override.

## First: find the stage, then jump in

Most requests enter mid-stream. Don't re-derive the whole pipeline — identify
where the user is, do that, and link forward/back to what's missing.

| The user says… | Start at |
| --- | --- |
| "I want to test whether X helps", "is it worth trying Y?" | **1. Frame** |
| "set up / prepare a run for X", "what config should I use?" | **2. Design** → **3. Set up** |
| "launch it", "run the training", "kick off the sweep" | **4. Launch** |
| "how's the run going?", "log this", "record these numbers" | **5. Track** |
| "compare these runs", "did it help?", "is this delta real?" | **6. Analyze** |
| "write it up", "make a report", "summarize the results" | **7. Report** |

When you enter mid-stream, glance back: is there a hypothesis on record? a
baseline to compare against? If a load-bearing piece is missing (e.g. someone
wants a report but never recorded what they were testing), name the gap and
offer to fill it — don't silently invent it.

## The experiment folder

Each experiment is one self-contained, dated folder so it's easy to find,
compare, and cite:

```
experiments/<YYYY-MM-DD>_<slug>/
├── hypothesis.md      # the question, prediction, success criterion, baseline
├── metadata.json      # commit, dirty-tree flag, seed(s), hardware, start/end
├── config/            # the exact config(s) used — copied, not referenced
├── command.md         # the exact command(s) to reproduce the run
├── logs/              # captured stdout/stderr per run
├── metrics/           # metrics as CSV/JSON-lines (the source of truth for plots)
├── figures/           # PNG plots embedded into the report
└── report.md          # human-readable writeup: setup, results, interpretation
```

Use today's date from context for `<YYYY-MM-DD>`; a short kebab `<slug>` names
the variable under test (e.g. `lr3e4-vs-1e3`, `repa-loss-ablation`). If the repo
already has an experiments/runs convention, fit into it rather than imposing
this one. Templates for `hypothesis.md`, `report.md`, and `metadata.json` are in
`references/templates.md` — read it when you create or fill those files.

## 1. Frame the hypothesis

Pin down, in `hypothesis.md`, **before** any run:

- **Question** — what are we actually testing? One variable, stated sharply.
- **Prediction** — what do you expect, and roughly how much? Writing this down
  first guards against rationalizing whatever number comes out (HARKing).
- **Decision metric** — the *single* number that settles it (FID, top-1, val
  loss, GenEval, reward…), and the direction/threshold that counts as "it
  helped." "We'll see what looks good" is not a decision rule.
- **Baseline** — what are we comparing to? A delta with no baseline isn't a
  result. The fairest baseline differs from the treatment by *only* the variable
  under test.

Keep this proportional. For a casual probe, three sentences is enough. If the
user explicitly wants to stress-test the design, or the experiment is expensive
enough that a flawed design wastes real GPU-hours, suggest the `/grill-me` skill
rather than reimplementing a deep interrogation here.

Watch for confounds: if the "treatment" run also changed batch size, data, or
step count, you can't attribute the effect to your variable. Flag it.

## 2. Design the cheapest experiment that could be wrong

Before committing GPU-hours, ask: **what is the smallest, fastest run that could
falsify this hypothesis?** A 200-step smoke run, a single-seed short schedule, or
a subset of eval prompts often kills (or green-lights) an idea for a fraction of
the cost. Spend the big run only once the cheap one survives.

Then nail the design:

- **Hold everything else constant.** Change one thing. If two things must move
  together, say so and accept the weaker attribution.
- **Seeds and variance.** A single-seed delta can be pure noise. For anything
  you'll report or act on, plan ≥2–3 seeds (or note explicitly that it's a
  single-seed probe and the number is soft). See `references/analysis.md`.
- **Match the comparison.** Same eval protocol, same checkpoint-selection rule,
  same number of samples for both arms.

## 3. Set up the folder and capture reproducibility

Create the folder and write `metadata.json`. Capture enough to reproduce the run
exactly:

- **Code state:** `git rev-parse HEAD` **and** whether the tree is dirty —
  research runs are routinely launched with uncommitted edits, so a hash alone
  won't reproduce them. Save `git status --porcelain` output (or the full
  `git diff`) when the tree isn't clean. Note this in `metadata.json`.
- **Config:** copy the actual config file(s) into `config/` (copy, don't just
  link — the source may change). Record any command-line overrides.
- **Environment & hardware:** GPU/accelerator, framework versions, container
  image if relevant, and the seed(s).

Surface result-impacting changes. If the experiment depends on a code change
that alters the math/training objective/eval (not just infra), say so plainly in
`hypothesis.md` and `report.md` — that's exactly what a reproducer or reviewer
needs flagged. If this repo keeps a reproducibility/changelog log, note the
change there too.

## 4. Launch: prepare → show → confirm → run

1. **Prepare** the exact command(s) and write them to `command.md`. Resolve real
   paths, config names, output dirs — no placeholders the user has to fill in.
2. **Show** the command and say what it will do (how long, how much compute,
   where outputs land).
3. **Confirm before heavy/long/expensive jobs.** Multi-GB downloads, large
   preprocessing, and long GPU training need an explicit go-ahead — don't kick
   them off unannounced. Quick smoke tests can run once you've shown the command.
4. **Run** and capture stdout+stderr to `logs/` (e.g. `… 2>&1 | tee logs/run.log`).
   Record start time in `metadata.json`.

If launching is environment-bound (e.g. must run inside a container or on a
specific GPU slice) and you can't reach it, hand the user the ready-to-run
command instead of guessing — a correct command they run beats a broken one you
launched.

## 5. Track the run

While it runs (or from finished logs): pull the decision metric and key
secondary signals into `metrics/` as CSV or JSON-lines — one row per
step/epoch/eval — so plots and tables read from a clean source rather than
re-grepping logs each time. Note anomalies as they appear (loss spikes, NaNs,
diverging val, early plateaus); a one-line note now saves a confused analysis
later. If a run dies, record why before relaunching.

## 6. Analyze and compare — honestly

The goal is a trustworthy answer to the hypothesis, not a flattering one.

- **Compare to the baseline** on the decision metric. Report the delta with its
  sign and magnitude, not just "better."
- **Is the delta real or seed noise?** With multiple seeds, report mean ± spread
  and ask whether the gap clears the noise. With one seed, say the result is
  suggestive, not conclusive. `references/analysis.md` covers variance, when a
  difference is meaningful, and common traps (best-checkpoint cherry-picking,
  unequal sample counts, reading a still-moving curve).
- **Make the comparison legible:** a small table of arms × metrics, and plots of
  the decision metric over training/steps for each arm on shared axes. Save plots
  as PNGs in `figures/`. Plotting guidance (what to plot, how to save and embed)
  is in `references/analysis.md`.
- **Report null and negative results.** "It didn't help" is a real finding and
  saves the next person the run. Don't bury it or keep fishing for a cut of the
  data where it looks good.

## 7. Report

Write `report.md` as something a human reads top-to-bottom and *gets it* —
hypothesis, setup, results (tables + embedded `![](figures/…png)` plots),
interpretation, threats to validity, and next steps. Lead with the answer to the
hypothesis, then support it. Include the reproducibility essentials (commit +
dirty flag, config, command, seeds) so the report stands alone. Use the template
in `references/templates.md`.

A good report states what would change the conclusion: "if the seed variance is
larger than we measured, this 0.3 FID gap is within noise." That honesty is what
makes results citable.

## References

- `references/templates.md` — `hypothesis.md`, `report.md`, `metadata.json`
  templates. Read before creating or filling those files.
- `references/analysis.md` — variance/seeds, when a delta is real, comparison
  traps, and plotting guidance (matplotlib, saving PNGs, embedding in the report).
