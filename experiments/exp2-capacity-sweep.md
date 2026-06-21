# Exp 2: Capacity Sweep

Source: `PROPOSAL.md` section 7.

## Purpose

Measure how latent channel capacity affects delegation quality by sweeping `latent_steps`.

## Research Question

RQ2: How does latent channel capacity limit delegation quality?

## Hypothesis

If latent delegation is capacity-limited, task success should show a measurable cliff or improvement across `latent_steps` values, and that pattern should correlate with probe recovery from Exp 3a.

## Fixed Inputs

- Same Deep Agents graph from Exp 0.
- Same latent delegation implementation from Exp 1.
- Same local model family and RecursiveMAS checkpoints.
- Same task prompts, tools, sandbox, timeouts, and evaluator.
- Medium-difficulty task subset selected before inspecting capacity results.

## Capacity Settings

| Config | `latent_steps` |
|---|---:|
| Latent-DA-16 | 16 |
| Latent-DA-32 | 32 |
| Latent-DA-48 | 48 |

Optional reference:

- Text-DA on the same task subset can be carried forward from Exp 1 for context, but the controlled variable inside Exp 2 is only `latent_steps`.

## Pre-Registration

Before running the sweep, freeze:

- Medium-difficulty task subset.
- Number of repeats or seeds per task.
- Run order or randomization procedure.
- Timeout and token budget.
- Rerun policy for crashes and invalid logs.
- Primary metric: task success rate by `latent_steps`.
- Secondary metrics: test pass rate, efficiency, cosine statistics, and probe recovery when available.

## Execution Plan

### Step 1: Select the Task Subset

Action:

- Choose medium-difficulty tasks from Exp 1.
- Prefer tasks with partial variation in outcomes, not tasks all arms trivially pass or fail.
- Keep the subset fixed before capacity runs begin.

Verification:

- Confirm every selected task has an executable oracle.
- Confirm every selected task has Exp 1 metadata: files, tests, dependency depth, and baseline results if available.
- Check that the subset covers more than one task type.

Success criteria:

- The subset is frozen in a manifest.
- Tasks are neither all trivial nor all impossible based on pilot evidence.
- Every task can be reset deterministically.

If this fails:

- Revise the subset before running any capacity conditions.
- Do not tune the subset after looking at capacity-specific results.

### Step 2: Confirm `latent_steps` Plumbing

Action:

- Add or verify a configuration path for `latent_steps`.
- Ensure the value reaches RecursiveMAS inference and affects the latent rollout.
- Record the configured value in every run log.

Verification:

- Run a smoke call for `latent_steps=16`, `32`, and `48`.
- Confirm output shapes or logged rollout metadata reflect the configured step count.
- Confirm run metadata records the requested and effective `latent_steps`.

Success criteria:

- Each capacity setting produces a valid latent handoff.
- Requested and effective `latent_steps` match.
- The backend does not silently fall back to a default value.

If this fails:

- Stop and fix configuration plumbing.
- Discard any runs where the effective value is unknown.

### Step 3: Validate Runtime Budgets

Action:

- Estimate runtime and GPU memory for each capacity setting.
- Set timeouts that allow larger capacities to complete without giving them unbounded advantage.
- Decide whether timeout is fixed across capacities or scales by a pre-registered rule.

Verification:

- Run one short task at each capacity.
- Record wall-clock, memory use if available, and GPU seconds.
- Confirm the largest setting does not routinely hit timeout.

Success criteria:

- All three capacity settings can complete at least one smoke run.
- Timeout policy is documented.
- Resource usage is measurable or explicitly marked missing.

If this fails:

- Reduce task count, adjust batch size, or revise timeout policy before the full sweep.
- Do not compare capacities if one setting is mostly infrastructure timeouts.

### Step 4: Run the Capacity Sweep

Action:

- For each task and repeat, run Latent-DA-16, Latent-DA-32, and Latent-DA-48.
- Reset fixtures before every run.
- Use paired ordering or randomized ordering from the pre-registration.
- Capture handoff tensors, text fallback if present, cosine values, tokens, runtime, GPU seconds, final diffs, and test results.

Verification:

- Validate each run record after completion.
- Confirm every task-repeat has all three capacity settings or an invalid-run marker.
- Confirm fixture hashes match before each paired run.

Success criteria:

- Required number of valid paired capacity runs is collected.
- Missing or invalid runs are tracked separately from model failures.
- Handoff logs are complete for every valid run.

If this fails:

- Rerun invalid runs only under the pre-registered policy.
- Discard runs with contaminated fixtures or unknown effective `latent_steps`.

### Step 5: Evaluate Outcomes

Action:

- Run the shared evaluator on all capacity outputs.
- Compute task success, test pass rate, import/symbol consistency, and failure category.
- Join outcomes to run metadata and handoff logs.

Verification:

- Check that the evaluator command is identical across capacity settings.
- Confirm every valid run has exactly one evaluation record.
- Spot-check failures at each capacity.

Success criteria:

- Evaluation records are complete and uniformly computed.
- Outcomes can be grouped by `latent_steps`.
- Failure categories are comparable across settings.

If this fails:

- Fix evaluator joining and recompute from stored artifacts.
- Do not rerun agents unless raw outputs are missing.

### Step 6: Analyze Capacity Effects

Action:

- Compute success and test pass rate by `latent_steps`.
- Compute paired deltas where the same task-repeat differs by capacity.
- Identify tasks solved by higher capacity but not lower capacity.
- Plot success versus `latent_steps`.

Verification:

- Check confidence intervals or bootstrap intervals.
- Inspect whether capacity wins are concentrated in one task.
- Compare results with runtime to separate quality gains from compute scaling.

Success criteria:

- Capacity curve is reported with uncertainty.
- Pairwise task-level changes are visible.
- Efficiency cost per capacity is reported.

If this fails:

- Report insufficient power or flat/noisy results.
- Avoid claiming a capacity cliff without paired evidence.

### Step 7: Analyze Channel Health

Action:

- Aggregate `cos(in,out)` mean, variance, and trajectory across handoffs.
- Compare cosine behavior across `latent_steps`.
- Flag collapse-like behavior such as rising mean toward 1.0 or dropping variance.

Verification:

- Confirm cosine fields are finite.
- Plot cosine distributions by capacity and handoff index.
- Check for missing cosine logs by setting.

Success criteria:

- Channel-health metrics are available for each capacity.
- No capacity setting shows unreported collapse-like behavior.
- Any suspicious trend is documented in the analysis note.

If this fails:

- Fix latent logging if possible.
- Mark channel-health analysis partial if raw tensor metadata cannot be recovered.

### Step 8: Connect to Exp 3a When Available

Action:

- Once probes exist, compute probe accuracy or recoverable fraction by `latent_steps`.
- Compare probe recovery with task success and capacity.

Verification:

- Confirm probe train/validation splits do not leak task repeats.
- Confirm the same label schema is used across capacity settings.
- Report correlation descriptively, not as causal proof.

Success criteria:

- Probe recovery is reported by capacity where data supports it.
- Any correlation with task success is clearly labeled correlational.

If this fails:

- Keep Exp 2 results standalone.
- Do not delay capacity reporting on optional probe analysis.

## Metrics

- Task success rate per capacity setting.
- Test pass rate per capacity setting.
- Paired success deltas by task-repeat.
- Tokens per task.
- Wall-clock time per task.
- GPU seconds per successful task.
- `cos(in,out)` mean and variance across handoffs.
- Crash, timeout, and invalid-run rates.
- Optional downstream comparison: probe accuracy by `latent_steps`.

## Experiment-Level Success Criteria

Capacity bottleneck success:

- A visible performance difference appears across 16, 32, and 48 latent steps.
- Lower-capacity settings fail on tasks solved by higher-capacity settings.

Operational success:

- Effective `latent_steps` is logged and verified for every valid run.
- The same tasks, graph, tools, and evaluator are used across all settings.
- Efficiency cost is reported alongside task quality.

Mechanism-supporting success:

- Capacity trend aligns with probe recovery or channel-health metrics, while remaining explicitly correlational.

## Negative Result Interpretation

If performance is flat across capacity settings, report that the tested tasks do not expose a latent capacity bottleneck. Follow-up analysis should check whether the task set is too easy, the latent channel is unused, or fixed released links are the limiting factor.

If higher capacity improves quality but sharply increases GPU seconds, report the tradeoff rather than treating capacity as a free improvement.

## Deliverables

- Frozen medium-task manifest.
- Capacity sweep run logs.
- Success versus `latent_steps` plot.
- Efficiency table by capacity setting.
- Cosine trajectory plots.
- Optional probe recovery comparison after Exp 3a.

## Dependencies

- Exp 0 complete.
- Pilot Exp 1 latent arm running reliably.
- Medium-difficulty task subset selected.
