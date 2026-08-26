# Exp 3a: Delegation Boundary Probes

Source: `PROPOSAL.md` section 7.

## Purpose

Test what task-relevant coding state can be recovered from latent bundles versus text summaries at delegation boundaries.

## Research Question

RQ4: What task-relevant state survives in the latent bundle that text summarization drops at delegation boundaries?

## Hypothesis

Text handoff compresses agent state by dropping uncertainty about file relevance, partial hypotheses, and implicit dependency structure. The latent bundle may preserve some of these properties in a linearly decodable form even when absent from the text summary.

## Required Inputs

At each delegation handoff, extract matched representations:

- Latent bundle: post-outer-link vectors from Latent-DA, loaded from `latent_artifact_path`.
- Text summary: the message Text-DA passed to the next subagent, from `text_handoff`.

Join on `handoff_id`, and restrict comparisons to handoffs sharing the same `binding`, `source_model`, `target_model`, and role transition. Probing across different model pairs measures the models, not the channel.

Use the same label schema and matched train/validation split for latent and text probes.

### The two channels are not observed in the same run

Under the frozen design, `Latent-DA` stubs its `ToolMessage`, so a latent run has no natural-language summary to probe and a text run has no bundle. Latent and text representations therefore come from **different runs of the same task**, not from the same handoff. Two consequences:

- Pair at the task and role-transition level, not the handoff level, and expect the two arms to have different handoff counts on the same task. Record unmatched handoffs on both sides.
- The primary corpus is one-shot (Exp 1/2) handoffs in the worker → orchestrator direction. If Exp 4 logs exist, its orchestrator → worker feedback handoffs (`outer_s2`) are a different distribution: stratify by `outer_link_key` and treat probes on the feedback direction as exploratory, never pooled with the forward direction.
- `Latent+Text-DA` is the one arm that produces both representations for the *same* handoff. If it runs at sufficient volume, it is the cleanest probe corpus even though it is not the headline arm. Decide in pre-registration whether the primary probe analysis uses paired `Latent+Text-DA` handoffs or unpaired `Latent-DA` versus `Text-DA` handoffs, and report which.

## Probe Targets

| Property | Label source |
|---|---|
| P1: File graph | Relevant files plus directed import or call edges among them |
| P2: Symbol surface | Function/class names and signatures tracked by the planner |
| P3: Test dependencies | Test-to-module and test-to-file mapping |
| P4: Ambiguity state | Whether the agent has committed to one symbol interpretation or still has multiple candidates |

## Labeling Decision

Choose and freeze one label source before probe training:

| Option | Use when | Risk |
|---|---|---|
| Sandbox oracle | Task fixtures encode files, imports, tests, and expected symbols | May miss agent-specific ambiguity state |
| Post-hoc annotation | Human labels are feasible for pilot handoffs | Labor cost and reproducibility risk |
| Synthetic task generator | Task structure is generated from known metadata | Less ecological validity |

Record the chosen strategy, label schema, annotator instructions if any, and expected failure modes before training probes.

## Pre-Registration

Before fitting any probes, freeze:

- Handoff inclusion rules.
- Label source for P1-P4.
- Feature extraction procedure for latent bundles and text summaries.
- Train/validation/test split.
- Leakage controls.
- Probe model class and hyperparameters.
- Primary metric per property.
- Significance test or bootstrap procedure.

## Execution Plan

### Step 1: Validate Handoff Corpus

Action:

- Collect handoff logs from Exp 0 and pilot or full Exp 1.
- Include both Text-DA and Latent-DA runs where handoff boundaries are comparable.
- Exclude invalid runs using pre-registered rules.

Verification:

- Count handoffs by task, condition, source role, target role, and complexity tier.
- Confirm each included handoff joins to task metadata and evaluation output.
- Confirm latent records include tensor artifact references or stored features.

Success criteria:

- Corpus has enough matched handoffs to train and evaluate probes.
- Every included handoff has stable join keys.
- Missing logs are quantified by condition.

If this fails:

- Fix log export or collect more pilot runs.
- Do not train probes on unmatched or partially joined handoffs.

### Step 2: Define Handoff Matching

Action:

- Decide how Text-DA and Latent-DA handoffs are paired or compared.
- Match by `task_id`, role transition, handoff index, and complexity tier where possible.
- Record cases where one condition has additional or missing delegation steps.

Verification:

- Produce a matching report with matched, unmatched-text, and unmatched-latent counts.
- Spot-check representative matched handoffs.

Success criteria:

- Matched handoffs represent the same task stage as closely as the harness allows.
- Unmatched handoffs are not silently dropped without counts.
- The analysis can separate matched-only from all-available results if needed.

If this fails:

- Narrow analysis to comparable role transitions.
- Avoid direct latent-vs-text claims for unmatched stages.

### Step 3: Build Label Schema

Action:

- Define exact label fields for P1-P4.
- For structured labels, define serialization and scoring rules.
- For ambiguity state, define allowed classes such as `committed`, `multiple_candidates`, and `unknown`.

Verification:

- Label 5-10 pilot handoffs manually or through the oracle.
- Check labels for consistency, missing fields, and ambiguous cases.
- Verify that every property can be represented for each task tier.

Success criteria:

- Label schema is complete and machine-readable.
- Ambiguity state has clear decision rules.
- Unknown or not-applicable labels are allowed only where pre-registered.

If this fails:

- Revise the schema before large-scale labeling.
- Do not mix incompatible label definitions across runs.

### Step 4: Generate Ground-Truth Labels

Action:

- Create labels for P1-P4 using the chosen label source.
- Store labels separately from features.
- Include provenance fields showing whether each label came from oracle, annotation, or synthetic metadata.

Verification:

- Run schema validation on labels.
- For annotation, run an inter-rater pilot or adjudication pass on a small sample.
- For oracle labels, compare generated labels against known fixtures.

Success criteria:

- Labels are complete for the pre-registered included handoffs.
- Label provenance is recorded.
- Inter-rater agreement or oracle validation is acceptable for the pilot threshold.

If this fails:

- Fix label generation or narrow the property set before training.
- Mark any low-confidence property as exploratory.

### Step 5: Extract Latent Features

Action:

- Load post-outer-link latent bundles.
- Convert tensors into fixed-size feature vectors using a pre-registered pooling or flattening strategy.
- Record shape, dtype, pooling method, and normalization.
- Note that flattening gives `latent_steps * target_hidden` dimensions, which is tens of thousands of features against a few hundred handoffs. Mean-pooling over the step axis keeps dimensionality at `target_hidden` and is the safer default; whichever is chosen, the dimensionality-to-sample ratio needs to be stated alongside the results.
- If Exp 2 data is included, `latent_steps` varies across runs, so a pooling strategy that is invariant to the step count is required for cross-capacity comparison.

Verification:

- Confirm feature dimensions are constant within each model/configuration.
- Check for NaN, inf, all-zero vectors, and duplicated features.
- Confirm features align with the correct handoff ids.

Success criteria:

- Every latent handoff has one feature row.
- Feature extraction is deterministic.
- Numeric sanity checks pass.

If this fails:

- Fix tensor storage or feature extraction.
- Exclude corrupted handoffs under the pre-registered rule.

### Step 6: Extract Text Features

Action:

- Convert text summaries into baseline features.
- Use a fixed text featurization method such as bag-of-words, TF-IDF, frozen embeddings, or another pre-registered representation.
- Preserve raw text for audit.

Verification:

- Confirm text feature dimensions are stable.
- Check empty or truncated handoffs.
- Confirm text features align with the correct handoff ids.

Success criteria:

- Every included text handoff has one feature row.
- Feature extraction does not use labels.
- Raw text remains available for Exp 3b audit.

If this fails:

- Fix text logging or extraction.
- Report missing text feature rate by condition.

### Step 7: Create Leakage-Safe Splits

Action:

- Split data so related handoffs from the same task, run, or generated family do not leak across train and test.
- Prefer task-level or repository-level splits over random handoff-level splits.
- Apply the same split ids to latent and text features.

Verification:

- Run a leakage check showing no task ids or run ids cross forbidden split boundaries.
- Confirm class balance for P1-P4 is acceptable in train and test.

Success criteria:

- Split policy is documented and reproducible.
- Latent and text probes use identical split assignments.
- Test split contains all evaluated property classes where possible.

If this fails:

- Rebuild splits.
- If data is too small, report pilot-only results without strong claims.

### Step 8: Train Linear Probes

Action:

- Train simple linear probes for each property and channel.
- Use the frozen hyperparameters or a nested validation procedure.
- Train separate probes for latent and text features under matched splits.

Verification:

- Confirm each probe uses only train data.
- Save model weights, hyperparameters, and metrics.
- Compare against simple baselines such as majority class or frequency baseline.
- Add a shuffled-label control per property: refit on permuted labels and confirm accuracy falls to baseline. With high-dimensional features and few handoffs, a linear probe can fit noise, and this is the cheapest way to show it did not.

Success criteria:

- Probe training is reproducible from saved configs.
- Each property has latent, text, baseline, and shuffled-label results.
- No probe uses downstream test labels during training.
- Feature dimensionality and sample count are reported next to every accuracy number.

If this fails:

- Fix training code or split handling.
- Do not report probes without baselines.

### Step 9: Evaluate Probes

Action:

- Evaluate on held-out handoffs.
- Compute per-property metrics.
- Report latent versus text deltas.
- Compute uncertainty with bootstrap or the pre-registered statistical test.

Verification:

- Confirm test handoff ids were not used in training.
- Spot-check predictions for obvious label alignment errors.
- Compare results on matched-only and all-available sets if both are reported.

Success criteria:

- Held-out metrics exist for P1-P4.
- Latent-vs-text comparisons use the same split and label schema.
- Statistical uncertainty is reported.

If this fails:

- Recompute from saved features and splits.
- Mark any invalid property result as missing rather than filling with aggregate averages.

### Step 10: Analyze by Complexity

Action:

- Group probe results by task complexity tier.
- Check whether latent advantages grow with files, tests, or dependency depth.
- Compare probe recovery with Exp 1 task success where available.

Verification:

- Confirm complexity labels come from the frozen task manifest.
- Check sample counts per tier.
- Avoid overinterpreting tiers with too few handoffs.

Success criteria:

- Complexity-stratified results are available where sample size supports them.
- Any claimed growth with complexity is backed by visible per-tier metrics.

If this fails:

- Report aggregate probe results only.
- Treat complexity analysis as exploratory.

## Metrics

- Probe accuracy per property.
- F1, exact match, or structured score where appropriate.
- Latent minus text delta per property.
- Baseline-adjusted improvement over majority or frequency baseline.
- Statistical significance or bootstrap confidence intervals.
- Accuracy versus task complexity.
- Missing-label and missing-feature rates.

## Experiment-Level Success Criteria

Interpretability-supporting success:

- Latent probe accuracy beats text probe accuracy on at least 2 of 4 properties.
- Wins are statistically supported on held-out handoffs.
- Results are reported per property, not only as a pooled average.

Operational success:

- Labels, features, splits, and probe configs are reproducible.
- Leakage checks pass.
- Text and latent comparisons use matched splits and comparable handoff stages.

## Negative Result Interpretation

If text probes match or beat latent probes on at least 3 properties, report that text summarization is sufficient for observable coding state at handoff. In that case, do not claim interpretability as a contribution; narrow the latent advantage claim to Exp 1 performance or efficiency only.

If latent wins on 0 or 1 property, interpretability is not a contribution. Keep Exp 3b observability tradeoff as the analysis angle.

## Deliverables

- `probing/delegation_probes.py`.
- Handoff corpus manifest.
- Label schema and label files.
- Per-handoff feature extractors.
- Leakage-safe split files.
- Probe training configs and saved results.
- Per-property probe tables.
- Probe accuracy versus task complexity plots.

## Dependencies

- Exp 0 harness spike complete, including Step 5b. Probing a channel the receiver ignores measures what the source model encoded, not what was communicated, and the write-up must say so if Step 5b failed.
- At least one pilot Exp 1 run complete.
- Handoff logging infrastructure stable, with `handoff_id` and retrievable `latent_artifact_path` on every latent record.
- Human decision resolved for P1-P4 ground-truth label strategy.
- Pre-registration decision on paired (`Latent+Text-DA`) versus unpaired probe corpus.
- `scikit-learn` installed; it is not in `requirements.txt`.
