# Templates

Copy-paste starting points for the experiment-folder files. They're skeletons,
not forms — drop sections that don't apply, add ones that do. Keep them filled in
*as you go*, not at the end.

## hypothesis.md

```markdown
# <slug>: <one-line question>

**Date:** <YYYY-MM-DD>  ·  **Author:** <name>  ·  **Status:** planned | running | done

## Question
<What single thing are we testing? State the one variable under test.>

## Prediction
<What do you expect to happen, and roughly how much? Write this BEFORE the run.>

## Decision metric
<The single number that settles it (e.g. FID-50k, val top-1, eval loss).
State the direction and the threshold that counts as "it helped":
e.g. "FID drops by ≥0.5 vs baseline, same seed budget.">

## Baseline
<What we compare against. It should differ from the treatment by ONLY the
variable under test. Link the baseline run/folder if it already exists.>

## Setup
- Variable under test: <…>
- Held constant: <data, batch size, steps, schedule, eval protocol, …>
- Seeds: <e.g. 0,1,2 — or "single seed, probe only">
- Cheapest falsifying run tried first: <e.g. 200-step smoke / short schedule>

## Confounds / risks
<Anything that could muddy attribution; result-impacting code changes this
depends on.>
```

## metadata.json

```json
{
  "slug": "<slug>",
  "date": "<YYYY-MM-DD>",
  "git_commit": "<rev-parse HEAD>",
  "git_dirty": true,
  "git_status": "<output of `git status --porcelain`, or path to saved diff>",
  "config_files": ["config/train.yaml"],
  "overrides": ["--lr 3e-4"],
  "seeds": [0, 1, 2],
  "hardware": "<e.g. 1x RTX PRO 6000 MIG 1g.24gb>",
  "framework": "<e.g. torch 2.7, cuda 12.8>",
  "container_image": "<if relevant>",
  "started_at": "<ISO timestamp>",
  "ended_at": "<ISO timestamp>"
}
```

Set `git_dirty` honestly. If `true`, save the actual diff (`git diff > metadata_dirty.patch`)
so the run is reproducible — a commit hash alone won't restore uncommitted edits.

## report.md

Lead with the answer, then support it. A reader should get the conclusion from
the first paragraph and the table.

```markdown
# <slug>: <one-line question>

**Verdict:** <Confirmed / Refuted / Inconclusive> — <one sentence on the result.>

## TL;DR
<2–4 sentences: what we tested, what happened, what to do next.>

## Hypothesis
<Restate the question, prediction, and decision metric from hypothesis.md.>

## Setup
- Variable under test: <…>  ·  Held constant: <…>  ·  Seeds: <…>
- Baseline: <link/ref>  ·  Treatment: <link/ref>
- Code: commit `<hash>` (dirty: yes/no)  ·  Config: `config/…`  ·  Command: see `command.md`

## Results

| Arm | <metric> | <metric2> | seeds | notes |
| --- | --- | --- | --- | --- |
| baseline | … ± … | … | 0,1,2 | |
| treatment | … ± … | … | 0,1,2 | |

![<metric> over steps](figures/<metric>_curve.png)

<Read the numbers: delta with sign and magnitude; is it within seed noise?>

## Interpretation
<What the result means for the hypothesis. Confirmed/refuted/inconclusive, and why.>

## Threats to validity
<What would change the conclusion: seed variance, confounds, eval protocol,
checkpoint selection, sample counts.>

## Next steps
<What to run next, or what this rules in/out.>

## Reproduce
<commit + dirty patch, config path, exact command, seeds, environment.>
```

## command.md

````markdown
# How this run was launched

## Environment
<container / conda env / GPU slice, if it matters>

## Command(s)
```bash
<exact command(s), with real paths and config names — no placeholders>
```

## Outputs
- Checkpoints: <path>
- Logs: logs/run.log
- Metrics: metrics/<file>
````
