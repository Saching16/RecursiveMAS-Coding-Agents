# Exp 4: Outer Recursive Delegation

Source: `PROPOSAL.md` section 7. Reference mechanism: `inference_utils/inference_mas_mixture.py:679-776`.

## Purpose

Test the mechanism that makes RecursiveMAS recursive: repeated bidirectional latent feedback across delegation rounds. Exp 0 through Exp 3 deliberately study a one-shot handoff — one bundle, one boundary, one injection — which exercises the inner rollout recursion and a single outer-link mapping but not the closed feedback loop. This experiment closes the loop.

The released code is the design template. Each round, experts produce latents conditioned on the previous round's feedback (spliced via `HIE_FEEDBACK_SLOT`), the summarizer runs its own latent rollout over them, and `run_hie_summarizer_feedback_latent_stage` maps its state back down through the reverse links to become the next round's feedback. Only the final round's output is decoded. Exp 4 reproduces that shape inside the Deep Agents graph with the Binding A pair.

## Research Question

RQ6: Does repeated bidirectional latent feedback gain more from added rounds than matched text feedback?

The primary statistic is the **channel x rounds interaction**, not the main effect of rounds. "More rounds help" is expected and uninteresting; the claim under test is that latent feedback compounds across rounds better than text feedback does.

## Hypothesis

If the latent channel's advantage is iterative refinement — as the RecursiveMAS framing implies — Latent-DA should improve more per added round than Text-DA, and a one-shot comparison (Exp 1) understates the latent channel. If instead the latent advantage (or deficit) is flat in rounds, the channel's value is in the handoff itself and the recursive framing adds nothing in this harness.

## Loop Design

Both arms run the identical loop; only the feedback medium differs.

For round `r` of `R`:

1. Worker drafts, conditioned on round `r-1` feedback (none when `r=1`).
2. Worker-to-orchestrator handoff: latent bundle via `outer_2s` (Latent arm) or natural-language result (Text arm).
3. If `r < R`: orchestrator produces feedback for round `r+1` — a latent bundle via `outer_s2` spliced into a worker-side feedback slot (Latent arm), or a natural-language critique prepended to the worker's revision prompt (Text arm).
4. If `r = R`: the orchestrator finalizes; this output is the deliverable that gets evaluated.

Round count `R` is fixed and pre-registered per condition. No adaptive stopping — adaptive recursion depth is explicitly out of scope for v1 (`PROPOSAL.md` section 11).

The worker-side feedback slot mirrors `build_hie_expert_prompt_with_feedback_slot` and `HIE_FEEDBACK_SLOT` (`prompts.py:17`, `:131-160`): render the prompt with a marker, split with `split_prompt_ids_by_slots`, splice the feedback bundle between prefix and suffix embeddings.

## Conditions

### Exp 4a — Loop spike

One golden task, `R=2`. Validates that the reverse edge works end to end before any sweep:

- `outer_s2` loads, its output width equals the worker's embedding width, and the feedback bundle has shape `[latent_steps, worker_hidden]`.
- The round-2 worker output responds to the round-1 feedback: shuffled-feedback control (inject feedback computed for a different task) must change the round-2 output. This is Step 5b of Exp 0 applied to the reverse direction.

### Exp 4b — Round sweep

| Config | `R` | `latent_steps` per handoff |
|---|---:|---:|
| Text-DA-R1 / Latent-DA-R1 | 1 | 32 |
| Text-DA-R2 / Latent-DA-R2 | 2 | 32 |
| Text-DA-R3 / Latent-DA-R3 | 3 | 32 |

`R=1` is definitionally the one-shot design and must reproduce Exp 1's numbers on the shared task subset. If it does not, the harness drifted between experiments and the sweep is invalid.

Text-arm feedback budget: cap the critique at the same token budget the latent feedback occupies (`latent_steps` tokens), reusing the truncation path from `controls.py`. Without this the text arm gets unbounded critique length and the interaction is confounded with budget.

### Exp 4c — Budget-controlled recursion

Fixed total communication budget of 48 per direction, distributed across rounds:

| Config | Split |
|---|---|
| 1x48 | one round, 48 steps/tokens |
| 2x24 | two rounds, 24 each |
| 3x16 | three rounds, 16 each |

Run on both channels. This separates "recursion helps" from "more total capacity helps": if 1x48 matches or beats 3x16 for the latent arm, iterating adds nothing beyond budget in this harness.

## Pre-Registration

Before running 4b or 4c, freeze:

- Task subset (reuse the Exp 2 medium-tier manifest unless re-registered).
- Round counts and budget splits exactly as tabled above.
- Feedback-slot prompt template and the text-arm critique prompt, held symmetric.
- Repeats and seeds per cell.
- Primary statistic: channel x rounds interaction, bootstrap clustered by `task_id`.
- Rerun and invalid-run policy.

## Execution Plan

### Step 1: Confirm Upstream Readiness

Action:

- Confirm Exp 0 passed, including Step 5b (forward-direction receiver consumption).
- Confirm a pilot Exp 1 ran and RQ0 held: shuffled one-shot bundles degrade Latent-DA.
- Confirm the Exp 0 backend already loads both link directions and logs `round_index` (Exp 0 future-proofing).

Verification:

- Run the Exp 0 log validator; check `outer_link_key` appears for `outer_2s` records.
- Dry-load `outer_s2` and assert its output width equals the worker's embedding width.

Success criteria:

- One-shot channel is demonstrably load-bearing.
- Reverse link resolves and dimension-checks.

If this fails:

- If RQ0 failed upstream, do not run Exp 4 as a channel study; the recursive sweep of an inert channel is uninterpretable.
- If `outer_s2` fails to load or mismatches, record it — the loop cannot be built from released links and this experiment is blocked, which is itself a reportable constraint.

### Step 2: Implement the Reverse Edge and Feedback Slot

Action:

- Extend `latent_backend.py` with the orchestrator-side rollout + `outer_s2` mapping.
- Add the worker-side feedback slot, mirroring `HIE_FEEDBACK_SLOT` handling in `run_hie_expert_latent_stage` (`inference_mas_mixture.py:290-333`).
- Add the text-arm critique path with the matched token cap in `controls.py`.
- Extend the handoff log: direction is already captured by `source_role`/`target_role` and `outer_link_key`; assert both directions appear.

Verification:

- Unit-smoke both directions on one prompt each; assert shape, dtype, finite `cos(in,out)` per direction.
- Assert the round-2 worker prompt actually contains the spliced feedback rows (embedding-length arithmetic, not logs).

Success criteria:

- Both directions produce valid bundles from persistent models.
- Feedback splice is verified at the embedding level.

If this fails:

- Separate failures into reverse-link loading, slot rendering, and splice arithmetic before touching the graph.

### Step 3: Run Exp 4a (Loop Spike)

Action:

- Run the golden task at `R=2` on both arms from a reset fixture.
- Run the shuffled-feedback control on the latent arm.

Verification:

- Every round boundary has a handoff record with the correct `round_index` and link key.
- Round-2 output differs between own-feedback and shuffled-feedback runs.

Success criteria:

- Both arms complete `R=2` without crash.
- The feedback edge is demonstrably load-bearing.

If this fails:

- If outputs are identical under shuffled feedback, the reverse edge is inert: diagnose as in Exp 0 Step 5b, and if unfixable, report Exp 4 as blocked with the forward-only results standing.

### Step 4: Run Exp 4b (Round Sweep)

Action:

- Run the frozen task subset at `R` in {1, 2, 3}, both channels, with repeats.
- Reset fixtures between runs; validate records on completion.

Verification:

- `R=1` cells reproduce Exp 1 results within noise on the shared tasks.
- Paired coverage across all six cells.

Success criteria:

- Enough valid paired runs per cell for the pre-registered interaction test.

If this fails:

- If `R=1` does not reproduce Exp 1, stop and find the drift before interpreting anything.

### Step 5: Run Exp 4c (Budget-Controlled Splits)

Action:

- Run 1x48, 2x24, 3x16 on both channels over the same subset.

Verification:

- Effective per-round budgets logged and matching the split.
- Total budget identical across splits within each channel.

Success criteria:

- Valid paired runs across all splits.

### Step 6: Analyze

Action:

- Estimate the channel x rounds interaction (4b) with bootstrap intervals clustered by task.
- Plot success vs `R` per channel; success vs split at fixed budget (4c).
- Aggregate `cos(in,out)` separately per direction; the feedback edge can collapse independently of the forward edge.
- Report per-round efficiency: tokens, wall-clock, GPU seconds — recursion multiplies cost, and the honest accounting is cost per successful task at each `R`.

Verification:

- Interaction estimate is computed from paired cells only.
- Direction-stratified cosine plots exist.

Success criteria:

- The interaction is reported with uncertainty, whatever its sign.

## Metrics

- Task success by (channel, `R`) — Exp 4b.
- Channel x rounds interaction estimate with interval — primary.
- Task success by (channel, budget split) at fixed total budget — Exp 4c.
- Delta from `R=1` per channel — how much iteration buys each medium.
- `cos(in,out)` per direction (`outer_2s` vs `outer_s2`) and per round index.
- Round-2 output divergence under shuffled feedback — feedback-edge liveness.
- Tokens, wall-clock, GPU seconds per round and per successful task.

## Experiment-Level Success Criteria

Validity preconditions:

- `R=1` reproduces Exp 1.
- Shuffled feedback changes round-2 output (feedback edge is load-bearing).

Primary success:

- Positive channel x rounds interaction: Latent-DA gains more from added rounds than Text-DA.

Mechanism success:

- In 4c, multi-round splits beat the single-round split at fixed budget for the latent arm — recursion contributes beyond capacity.

Operational success:

- Both directions' channel health is logged; efficiency is reported per round.

## Negative Result Interpretation

If Text-DA gains from rounds and Latent-DA does not, text critique loops beat latent feedback in this harness — report it directly; it is a substantive finding about zero-shot link transfer, not a failure of the study.

If neither channel gains from rounds, the task suite does not reward iteration; revisit task design before concluding anything about the channel.

If 4c shows 1x48 matching 3x16 on the latent arm, the recursive framing adds nothing beyond budget here: the honest summary is "latent delegation may help, latent *recursion* does not," and RQ6 is answered negatively while RQ1/RQ2 results stand.

If the feedback edge is inert (Step 3 fails), report Exp 4 as blocked by the released links and keep the one-shot study as the paper's scope.

## Deliverables

- Reverse-edge + feedback-slot implementation in `integrations/deepagents_latent/`.
- Exp 4a spike run note with shuffled-feedback control results.
- Round-sweep and budget-split run logs, schema-valid, joined to the Exp 1/2 manifests.
- Success-vs-rounds plot per channel; fixed-budget split plot; direction-stratified cosine plots.
- Interaction estimate table.

## Dependencies

- Exp 0 complete, including Step 5b and the future-proofing items (both links loaded, `round_index` logged, persistent models).
- Pilot Exp 1 complete with RQ0 confirmed.
- Exp 2 medium-tier task manifest frozen (reused here).
- `controls.py` token-cap path (shared with Text-DA-cap).

## Explicitly Not This Experiment

- Adaptive round counts or stopping rules.
- The full Mixture ensemble (math + code + science experts + summarizer). That is a possible Exp 5 / robustness study requiring ~27 GB of checkpoints and a three-way aggregation design; this experiment's loop is deliberately the minimal two-agent cycle.
