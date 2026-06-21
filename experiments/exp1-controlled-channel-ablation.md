# Exp 1: Controlled Channel Ablation

Source: `PROPOSAL.md` section 7.

## Purpose

Measure whether latent delegation improves multi-step coding quality and efficiency compared with text delegation when the agent topology and tools are held fixed.

## Research Question

RQ1: On identical Deep Agents topology, does latent delegation beat text delegation on multi-step coding tasks as complexity scales?

## Hypothesis

Latent-DA should preserve more useful task state across delegation boundaries than text-only handoff, producing higher task success on harder multi-file coding tasks.

## Controlled Variable

Only the delegation edge changes.

| Config | Delegation channel | Implementation |
|---|---|---|
| Text-DA | Subagent returns natural-language state | Deep Agents default |
| Latent-DA | Subagent returns RecursiveMAS latent bundle | Sidecar plus slot injection or embed path |

All other factors should remain fixed: roles, tools, sandbox, filesystem, shell access, task prompts, evaluation harness, model family where possible, and stopping budget.

## Scope

In scope:

- Controlled A/B evaluation over the same coding task suite.
- Complexity scaling by files touched, tests affected, and dependency depth.
- Per-run handoff logs for downstream probing and observability analysis.
- Efficiency accounting with tokens, wall-clock, and GPU seconds.

Out of scope:

- Changing the agent graph per condition.
- Tuning prompts separately for one arm.
- Training or fine-tuning RecursiveMAS links.
- Explaining mechanisms from probes alone.

## Task Set Design

Task tiers should be fixed before running the full experiment:

| Tier | Target shape | Minimum oracle |
|---|---|---|
| Small | MBPP+ style or small repo edits touching about 1-2 files | Unit test or exact output check |
| Medium | Multi-file edits touching about 3-5 files | Unit/integration tests |
| Large | Multi-file edits touching about 6-10 files | Unit/integration tests plus import checks |

Each task record should include:

- `task_id`.
- Prompt.
- Starting repository or fixture snapshot.
- Expected test command.
- Expected touched files when known.
- Complexity labels: files, tests, dependency depth, and task type.
- Timeout and budget.
- Any known nondeterminism.

## Pre-Registration

Before running the full sweep, freeze:

- Task list and tier labels.
- Random seeds or sampling procedure.
- Maximum retries per task.
- Timeouts and token budgets.
- Primary metric and tie-breakers.
- Criteria for invalid runs.
- Aggregation procedure.

## Execution Plan

### Step 1: Confirm Exp 0 Readiness

Action:

- Review the Exp 0 run note.
- Confirm both arms completed the golden task without harness crash.
- Confirm handoff logs passed schema validation.

Verification:

- Run the Exp 0 log validator on sample Text-DA and Latent-DA logs.
- Confirm the graph config diff still shows only the delegation backend changing.

Success criteria:

- Exp 0 is marked pass.
- No missing required log fields.
- Shared graph builder is available for both arms.

If this fails:

- Stop and repair Exp 0.
- Do not begin the ablation with incomplete logging or condition-specific graph drift.

### Step 2: Build and Freeze the Task Suite

Action:

- Select tasks across small, medium, and large tiers.
- Prefer tasks with executable tests and objective success oracles.
- Add metadata for files, tests, dependency depth, and expected runtime.

Verification:

- Run each task oracle on the initial fixture and verify the expected starting state.
- Run the oracle on a known-good solution where available.
- Validate that task metadata is complete.

Success criteria:

- Every task has a test command or objective oracle.
- Every task has complexity metadata.
- Tier assignments are reproducible from recorded metadata.

If this fails:

- Remove or replace tasks with ambiguous or missing oracles.
- Do not manually relabel tiers after seeing model results.

### Step 3: Validate Fixture Reset and Isolation

Action:

- Implement or verify a clean reset path for each task.
- Ensure Text-DA and Latent-DA start from identical snapshots.
- Store run outputs outside the source fixture when possible.

Verification:

- Hash or diff the fixture before each condition.
- Run two no-op resets and confirm identical state.
- Confirm logs and artifacts do not pollute later runs.

Success criteria:

- Reset is deterministic.
- Each condition starts from the same file state.
- Failed runs cannot leak edited files into later runs.

If this fails:

- Fix isolation before running any expensive jobs.
- Discard affected runs.

### Step 4: Run a Pilot A/B Batch

Action:

- Run a small pilot batch across at least one task per tier.
- Run Text-DA and Latent-DA on the same tasks with identical budgets.
- Capture logs, diffs, test results, runtime, and token counts.

Verification:

- Check that both arms produce parseable result records.
- Compare task ids and ensure paired runs exist.
- Confirm test commands executed and returned captured exit codes.

Success criteria:

- At least one valid paired run exists for each pilot tier.
- Result schema validation passes.
- Runtime and token accounting are present.

If this fails:

- Fix result collection or task harness.
- Repeat the pilot before scaling.

### Step 5: Run the Full A/B Sweep

Action:

- Run every frozen task under Text-DA.
- Reset fixtures.
- Run every frozen task under Latent-DA.
- Use the same task order or a pre-registered randomized paired order.

Verification:

- Validate every run record immediately after completion.
- Track invalid, timeout, and crash counts separately from task failures.
- Confirm paired coverage: each task has both conditions unless pre-registered otherwise.

Success criteria:

- At least the pre-registered minimum number of valid paired runs is collected per tier.
- Invalid run rate is low enough to support analysis.
- No condition has systematic missing logs.

If this fails:

- Diagnose whether failures are harness errors or model task failures.
- Rerun only invalid runs using the pre-registered rerun policy.
- Do not rerun valid failures just because the solution failed tests.

### Step 6: Evaluate Task Success

Action:

- Run the shared evaluator on final outputs.
- Compute task success, test pass rate, import/symbol consistency, and failure category.
- Preserve raw stdout, stderr, and exit codes.

Verification:

- Spot-check a sample of passing and failing runs.
- Confirm evaluator uses the same oracle across conditions.
- Confirm import/symbol consistency checks do not depend on condition-specific logs.

Success criteria:

- Every valid run has a final evaluation record.
- Primary success is computed uniformly.
- Failure categories are assigned without seeing the condition where practical.

If this fails:

- Fix evaluator determinism or oracle mismatch.
- Re-evaluate from raw artifacts rather than rerunning agents.

### Step 7: Compute Efficiency Metrics

Action:

- Aggregate prompt tokens, completion tokens, wall-clock seconds, and GPU seconds.
- Compute cost per task and cost per successful task.
- Report latent overhead even if Latent-DA improves task success.

Verification:

- Check for missing or impossible values such as negative runtime or token counts.
- Compare role-level totals with run-level totals where both exist.
- Confirm GPU seconds are nullable but explicitly marked when unavailable.

Success criteria:

- Efficiency metrics are available for each valid run or clearly marked missing.
- Per-success efficiency can be computed for each condition with at least one success.
- Missingness is reported by condition.

If this fails:

- Do not drop missing cost records silently.
- Report quality metrics and mark efficiency analysis as partial if needed.

### Step 8: Analyze Scaling

Action:

- Aggregate results by task tier, files touched, tests affected, and dependency depth.
- Compute Latent-DA minus Text-DA deltas.
- Plot task success versus complexity.

Verification:

- Confirm tier labels came from frozen metadata.
- Check confidence intervals or bootstrap intervals for each aggregate.
- Inspect whether one or two tasks dominate the large-tier result.

Success criteria:

- Results show task success by condition and complexity tier.
- The analysis distinguishes quality wins from efficiency costs.
- Any widening or narrowing gap is visible in the plot and table.

If this fails:

- Report insufficient power or task imbalance.
- Avoid claiming scaling behavior without enough valid paired runs.

### Step 9: Prepare Downstream Logs

Action:

- Package handoff logs for Exp 3a and Exp 3b.
- Include join keys linking runs, tasks, handoffs, labels, and outcomes.
- Preserve raw text handoffs and latent metadata.

Verification:

- Run schema validation on exported logs.
- Join logs to task metadata and evaluation records.
- Count handoffs by condition and role transition.

Success criteria:

- Logs can be joined to task outcomes.
- Text and latent records are matched at comparable delegation boundaries.
- No raw handoff data needed for Exp 3 is missing.

If this fails:

- Fix log export and rerun only the missing-log tasks if raw artifacts cannot be recovered.

## Metrics

- Primary: task success rate and test pass rate.
- Scaling: success versus files touched, test count, and dependency depth.
- Quality: import and symbol consistency.
- Efficiency: tokens, wall-clock, GPU seconds, and cost per successful task.
- Channel health: latent handoff `cos(in,out)` trajectory.
- Reliability: crash, timeout, and invalid-run rates.

## Experiment-Level Success Criteria

Primary success:

- Latent-DA has higher pass@1 or resolve rate than Text-DA on medium or large tasks.

Scaling success:

- The Latent-DA minus Text-DA gap widens with file count, test count, or dependency depth.

Operational success:

- Both arms are evaluated on the same frozen tasks with the same graph and tools.
- Logs are complete enough for Exp 3a and Exp 3b.
- Efficiency metrics are reported alongside quality metrics.

## Negative Result Interpretation

If Latent-DA does not beat Text-DA, report the full scaling curve. Use Exp 2 and Exp 3 to determine whether the likely issue is channel capacity, missing encoded state, insufficient task difficulty, or operational overhead.

If Latent-DA wins only on quality but loses heavily on GPU seconds, report the tradeoff directly rather than presenting the result as an unconditional improvement.

## Deliverables

- Frozen task manifest.
- Paired run artifacts for both conditions.
- Metrics script for both arms.
- Success versus complexity plot.
- Per-configuration efficiency table.
- Failure taxonomy summary.
- Handoff logs exported for Exp 3a and Exp 3b.

## Dependencies

- Exp 0 harness spike complete.
- Stable handoff logging.
- Executable tests or objective success oracle for each task.
