# Exp 3b: Operational Observability

Source: `PROPOSAL.md` section 7.

## Purpose

Quantify what a human debugger or monitoring system loses when delegation moves from readable text to latent tensors, and measure whether probes partially compensate for that loss.

## Research Question

RQ5: What operational observability is lost when moving from text to latent delegation, and does probing partially compensate?

## Hypothesis

Text-DA should have higher direct observability because handoffs are readable, loggable, diffable, and replayable. Latent-DA should have lower direct observability, but Exp 3a probes may recover some task-relevant state.

## Directly Available Signals

| Channel | Directly available without probing |
|---|---|
| Text-DA | Full handoff string: readable, loggable, diffable, replayable in session logs |
| Latent-DA | Tensor shape, norm, outer-link `cos(in,out)`; not human-readable semantics without probes |

## Observability Definition

Per handoff, the observability score is the fraction of task-relevant state variables that are human-readable without probing.

The denominator is the union of P1-P4 label fields for that handoff:

- File graph.
- Symbol surface.
- Test dependencies.
- Ambiguity state.

A state variable counts as directly observable if it is explicitly present in the handoff text or trivially derivable with pre-registered string or regex rules over logged text.

## Scoring Categories

Each task-relevant state variable should receive one score:

| Score | Meaning | Example |
|---|---|---|
| `1.0` | Directly observable | File path or symbol appears explicitly in handoff text |
| `0.5` | Trivially derivable | Regex or deterministic parser recovers the field from logged text |
| `0.0` | Not observable | Requires model inference, probe, tensor decoding, or external oracle |
| `NA` | Not applicable | Field is not relevant for that handoff under the frozen denominator |

The main observability score excludes `NA` from the denominator and averages the remaining scores.

## Pre-Registration

Before scoring any full-run handoffs, freeze:

- Denominator fields for P1-P4.
- Exact scoring rules for `1.0`, `0.5`, `0.0`, and `NA`.
- Automatic extraction rules for text handoffs.
- Manual annotation instructions if human scoring is used.
- Inter-rater agreement threshold.
- Aggregation procedure by handoff, run, condition, and complexity tier.
- How probe-recoverable fraction will be computed from Exp 3a.

## Execution Plan

### Step 1: Freeze the Denominator

Action:

- Translate P1-P4 labels from Exp 3a into explicit observability fields.
- Define which fields are required for each handoff type.
- Decide how to handle unknown, optional, or not-applicable fields.

Verification:

- Apply the denominator to 5-10 pilot handoffs.
- Confirm every field can be scored as observable, derivable, not observable, or not applicable.
- Confirm the denominator does not change based on condition results.

Success criteria:

- Denominator is documented before scoring the full corpus.
- Fields map cleanly to Exp 3a labels.
- No full-run score has to invent a new field category.

If this fails:

- Revise the denominator during pilot only.
- Restart pilot scoring after revisions.

### Step 2: Build the Scoring Protocol

Action:

- Write a protocol that defines direct observability and trivial derivability.
- Include examples for file paths, symbols, test mappings, and ambiguity state.
- Define what evidence is insufficient, such as vague natural language hints.

Verification:

- Have at least one reviewer apply the protocol to pilot handoffs.
- Compare scores with expected examples.
- Identify confusing rules and revise before full scoring.

Success criteria:

- The protocol is clear enough for another person or script to reproduce scores.
- Edge cases are documented.
- Manual judgment is minimized where deterministic parsing is possible.

If this fails:

- Narrow the scoring rules.
- Mark subjective fields for inter-rater review.

### Step 3: Implement Automatic Text Scoring

Action:

- Implement regex or deterministic parsers for explicitly logged file paths, symbols, test names, and dependency statements.
- Apply these parsers only to logged text handoffs.
- Store parser outputs and raw matched spans for audit.

Verification:

- Run parser tests on synthetic and real pilot handoffs.
- Compare parser output against manual expected fields.
- Confirm false positives and false negatives are logged.

Success criteria:

- Parser recovers straightforward explicit fields.
- Parser behavior is deterministic.
- Raw evidence spans are saved for review.

If this fails:

- Fall back to manual scoring for affected fields or mark parser-based scoring exploratory.
- Do not let hidden model inference count as direct observability.

### Step 4: Prepare Latent-DA Direct Signal Scoring

Action:

- Define which latent metadata fields are directly observable: tensor shape, norm, `cos(in,out)`, handoff index, source role, target role, and runtime metadata.
- Confirm semantic task fields are not counted as directly observable unless they appear in a separate readable log.

Verification:

- Score pilot Latent-DA handoffs using only metadata available without probes.
- Confirm semantic P1-P4 fields are scored `0.0` unless directly logged as text.

Success criteria:

- Latent direct observability is scored from metadata only.
- The scoring does not smuggle in probe output or oracle labels.
- Channel-health signals are tracked separately from semantic observability.

If this fails:

- Separate operational metadata scores from semantic state scores.
- Re-score pilot latent handoffs.

### Step 5: Run Inter-Rater Pilot

Action:

- Select at least 10 handoffs spanning Text-DA, Latent-DA, role transitions, and complexity tiers.
- Have two reviewers or one reviewer plus deterministic parser score the same handoffs.
- Adjudicate disagreements and update the protocol if needed.

Verification:

- Compute agreement by field and overall.
- List disagreement examples.
- Confirm revisions happen before full scoring.

Success criteria:

- Agreement meets the pre-registered threshold or disagreements are resolved with clearer rules.
- Manual scoring burden is understood before scaling.

If this fails:

- Simplify the denominator or scoring categories.
- Do not run full scoring with unstable subjective rules.

### Step 6: Score the Full Corpus

Action:

- Score every included handoff from Exp 1 and Exp 2 where applicable.
- Store per-field scores, evidence spans, scorer id, parser version, and timestamp.
- Keep Text-DA and Latent-DA scoring under the same denominator.

Verification:

- Run schema validation on score files.
- Check missing scores by condition, task, and field.
- Spot-check random scored handoffs.

Success criteria:

- Every included handoff has per-field scores or explicit `NA`.
- Missing scores are below the pre-registered threshold.
- Evidence is auditable.

If this fails:

- Complete missing scoring or exclude affected handoffs under the pre-registered rule.
- Do not aggregate incomplete handoffs without reporting missingness.

### Step 7: Compute Observability Aggregates

Action:

- Compute observability score per handoff.
- Aggregate by run, condition, complexity tier, role transition, and capacity setting.
- Compute confidence intervals or bootstrap intervals where possible.

Verification:

- Recompute a small sample by hand.
- Confirm `NA` handling matches the protocol.
- Confirm aggregation weights are documented.

Success criteria:

- Text-DA and Latent-DA aggregate scores are comparable.
- Results are available by complexity tier.
- Uncertainty is reported.

If this fails:

- Fix aggregation code before plotting.
- Report only levels with enough scored handoffs.

### Step 8: Compute Probe-Recoverable Fraction

Action:

- Import Exp 3a probe results.
- For each latent handoff field, mark whether the corresponding probe recovers it above chance or above the pre-registered threshold.
- Compute the fraction of task-relevant state recoverable by probes.

Verification:

- Confirm probe results use held-out data and leakage-safe splits.
- Confirm mapping from probe properties to observability fields is documented.
- Keep probe recovery separate from direct observability.

Success criteria:

- Probe-recoverable fraction is computed only from valid Exp 3a probe results.
- The metric is reported as partial compensation, not direct human readability.

If this fails:

- Omit probe-recoverable fraction or mark it exploratory.
- Do not infer probe recovery from training accuracy.

### Step 9: Track Channel Health

Action:

- Aggregate latent `cos(in,out)` mean, variance, and trajectory across handoffs.
- Track tensor norm and shape stability.
- Compare channel-health signals with task success and observability.

Verification:

- Check finite numeric values.
- Plot cosine by handoff index and capacity.
- Flag trends toward identity collapse or near-zero variance.

Success criteria:

- Channel-health metrics are reported separately from semantic observability.
- Any collapse warning is visible in the results.

If this fails:

- Mark channel-health analysis partial.
- Do not use missing cosine data to support claims about active latent transformation.

### Step 10: Build Observability-Performance Plot

Action:

- Join observability aggregates with task success from Exp 1 and Exp 2.
- Plot x = aggregate observability score and y = task success rate.
- Include Text-DA and Latent-DA at `latent_steps` values 16, 32, and 48 where available.

Verification:

- Confirm task success and observability are aggregated over compatible task sets.
- Label points with configuration, capacity, and sample count.
- Include uncertainty where possible.

Success criteria:

- Plot makes the performance and observability tradeoff visible.
- Text-DA, Latent-DA, and capacity settings are clearly distinguishable.
- The caption states whether points use the same task subset.

If this fails:

- Produce separate plots for Exp 1 and Exp 2 subsets.
- Do not compare points from incompatible task sets without a caveat.

## Metrics

- Observability score per handoff.
- Aggregate observability score per configuration.
- Fraction of task-relevant state recoverable without probing.
- Probe-recoverable fraction.
- Directly observable, trivially derivable, and not observable field counts.
- Inter-rater agreement.
- `cos(in,out)` mean and variance across handoffs.
- Task success rate from Exp 1 and Exp 2.

## Target Figure

Observability-performance tradeoff:

- x-axis: aggregate observability score.
- y-axis: task success rate.
- Points: Text-DA and Latent-DA at `latent_steps` values 16, 32, and 48.

## Experiment-Level Success Criteria

Measurement success:

- Text-DA shows higher direct observability than Latent-DA.
- Scores are reproducible from a frozen protocol.
- Missingness and uncertainty are reported.

Probe-compensation success:

- Latent probe-recoverable fraction partially closes the gap between latent direct observability and text direct observability.

Communication success:

- The observability-performance plot shows whether any task-success gain justifies observability loss.

## Negative Result Interpretation

If probe recovery does not close the observability gap, report latent delegation as operationally opaque under this harness. If Latent-DA does not improve task success, the result argues against accepting the observability cost for the tested setting.

If Text-DA and Latent-DA have similar observability because the latent harness logs rich text side channels, report that the implementation is no longer a clean latent-vs-text observability comparison and separate side-channel logs from the latent bundle itself.

## Deliverables

- Pre-registered observability scoring protocol.
- Denominator field schema.
- Automated derivable-text-field scorer where possible.
- Inter-rater pilot report on at least 10 handoffs if manual judgment is used.
- Per-field score files with evidence spans.
- Observability score tables by channel and complexity tier.
- Probe-recoverable fraction table.
- Channel-health summary.
- Observability-performance tradeoff plot.

## Dependencies

- Exp 0 harness spike complete.
- Pilot or full Exp 1 logs available.
- Exp 3a labels and probes available for probe-recoverable fraction.
- Human decision resolved for the observability score denominator.
