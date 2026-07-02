# Analysis and plots

How to turn metrics into a trustworthy answer and a legible report. The
through-line: **don't let a number look more certain than it is.**

## Is the delta real, or seed noise?

A single training run is a sample, not the truth. Reinitializing the seed
(weights, data order, augmentation) moves most metrics by a non-trivial amount —
often larger than the effect you're chasing.

- **With ≥2–3 seeds per arm:** report **mean ± std** (or min–max) for each arm.
  The treatment "helped" only if its mean clears the baseline mean by more than
  the seeds wobble. If the error bars overlap heavily, the honest verdict is
  *inconclusive*, not *better*.
- **With one seed per arm:** you cannot separate effect from noise. Say the
  result is *suggestive* and state the typical run-to-run spread for this metric
  if you know it. A single-seed win is a reason to spend more seeds, not a
  conclusion.
- **Rough rule of thumb:** if |mean_treatment − mean_baseline| is smaller than
  ~1 std of either arm, treat it as noise. This is a sanity heuristic, not a
  significance test; for a real test (e.g. a paired comparison across matched
  seeds) say so explicitly.

## Comparison traps to avoid

- **Best-checkpoint cherry-picking.** Picking each arm's single best eval across
  all of training inflates results and favors whichever arm was evaluated more
  often. Fix the checkpoint-selection rule in advance and apply it identically to
  both arms (e.g. "eval at the final step" or "best on a held-out val").
- **Unequal sample budgets.** FID, GenEval, reward, etc. depend on the number of
  generated samples. Use the same count for both arms.
- **Reading a moving curve.** A metric still trending at the last logged step
  isn't converged — comparing two arms at different points on their curves is
  meaningless. Compare at matched, settled points.
- **Different eval protocols.** Same resolution, same prompts/data, same
  preprocessing for both arms. Any mismatch is a confound.
- **Moved more than one variable.** If the treatment also changed batch size,
  data, or steps, the delta isn't attributable to your stated variable.

## Plots

Plots are for the report, so optimize for *legibility at a glance*. Read metrics
from the clean `metrics/` CSV/JSON-lines files, not by re-grepping logs.

What's usually worth plotting:

- **Decision metric vs step/epoch**, one line per arm on **shared axes** — this
  is the core comparison. Label arms, axes, and units.
- **Seed spread:** if you have multiple seeds, show it — either one faint line
  per seed plus a bold mean, or a mean line with a shaded ±std band. A reader
  should see the noise, not just the means.
- **Secondary signals** (train vs val loss, gradient norm) only if they explain
  the result.

Mechanics:

- Save every figure as a **PNG** into `figures/` and embed it in `report.md` with
  `![caption](figures/name.png)` so the report is self-contained.
- Use whatever plotting library the project already uses; **matplotlib** is the
  safe default. Keep it simple: titled, labeled axes, a legend naming the arms.
- A tiny script that reads `metrics/*.csv` and emits the curve PNGs is fine to
  write inline per experiment. If you find yourself writing essentially the same
  plotting script across several experiments, that's the signal to factor it into
  a reusable helper — mention it so it can be added to this skill.

## Tables

For the report, a compact **arms × metrics** table is often clearer than prose:
one row per arm, columns for the decision metric (with ± spread) and key
secondary metrics, plus seed count. Put the better number in context (delta vs
baseline), and don't bold a winner that's within noise.
