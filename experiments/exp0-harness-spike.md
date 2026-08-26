# Exp 0: Harness Spike

Source: `PROPOSAL.md` section 7. Engineering prerequisites: `PLAN.md` Gate 0.

## Purpose

Validate that the Deep Agents integration can run both delegation channels end to end on one golden multi-file coding task before any scaling sweeps.

This is the implementation gate for the rest of the study. It does not answer a headline research question directly, with one exception: Step 5b is the first real evidence for RQ0 (is the latent channel load-bearing?). Otherwise it proves that the controlled A/B harness is viable and that handoff logging is available for Exp 1, Exp 2, Exp 3a, and Exp 3b.

## Topology

The role graph is not a free choice. Every latent edge requires an outer link trained for that ordered pair of checkpoints, and the release ships a fixed set (`system_loader.py:78-100`). There is no link between two instances of the same role, so the `plan -> implement -> review` graph in earlier drafts of this document is not runnable.

Default binding (Binding A, `PLAN.md` section 2):

| Deep Agents role | Checkpoint | Outgoing latent edge | Link key |
|---|---|---|---|
| Main agent (orchestrator) | `Mixture-Summarizer-Qwen3.5-2B` | to worker | `outer_s2` |
| Subagent (worker) | `Mixture-Code-Qwen2.5-Coder-3B` | to orchestrator | `outer_2s` |

Multiple delegation boundaries come from multiple `task` calls within a run, not from adding roles. If Gate 0.2 selects Binding B instead, substitute the sequential planner/critic/solver trio and its `outer_12` / `outer_23` / `outer_31` links throughout, and record the substitution in the run note.

The pair is bidirectional, which means it also supports a closed refinement loop (worker → orchestrator → feedback → worker). Exp 0 stays strictly one-shot, but the backend built here must load and smoke **both** directions and log `round_index` from the start — Exp 4 (`exp4-outer-recursive-delegation.md`) reuses this backend for the loop.

## Scope

In scope:

- One shared Deep Agents graph, orchestrator plus one worker subagent.
- One Text-DA baseline using normal natural-language handoff.
- One Latent-DA path using the released RecursiveMAS inference primitives.
- Both arms backed by the **same local checkpoints**.
- One golden multi-file task with executable tests.
- Handoff logging at every delegation boundary.
- A receiver-consumption test (Step 5b).

Out of scope:

- Large task sweeps.
- Probe training.
- Observability scoring.
- Fine-tuning RecursiveLink weights.
- Adaptive recursion depth.

## Conditions

| Condition | `ToolMessage` content | Latent bundle | Expected behavior |
|---|---|---|---|
| Text-DA | Full natural-language handoff | none | Deep Agents default `task` flow completes the task |
| Latent-DA | Minimal stub | yes | Sidecar produces a latent handoff and the receiver consumes injected latent state |
| Shuffled-DA (Step 5b only) | Minimal stub | bundle computed for a different task | Receiver output must change relative to Latent-DA |

The stub-versus-full-text decision is load-bearing: if the latent arm also returns the full natural-language result, the study measures text-plus-latent against text, not latent against text. `PLAN.md` section 6 records the decision; assert on the stub content in the latent runner so it cannot regress silently.

Keep roles, tools, filesystem access, shell access, sandbox, prompts, task fixture, model checkpoints, and evaluation command identical across conditions.

## Required Artifacts

- `integrations/deepagents_latent/` sidecar package.
- Shared graph builder for both arms.
- Text baseline runner.
- Latent runner.
- Golden task fixture.
- Handoff log schema.
- Sample run logs for Text-DA and Latent-DA.
- Run note summarizing failures, missing instrumentation, and go/no-go decision for Exp 1.

## Handoff Log Schema

This schema is the join surface for Exp 1, Exp 2, Exp 3a, and Exp 3b. Fields marked as required by a downstream experiment are not optional here, because adding them later means rerunning every agent.

Each handoff record should include:

| Field | Required | Notes |
|---|---|---|
| `handoff_id` | yes | Globally unique; primary join key for Exp 3a/3b |
| `run_id` | yes | Unique run identifier |
| `task_id` | yes | Golden task id |
| `condition` | yes | `text_da`, `text_da_cap`, `latent_da`, `latent_text_da`, or `shuffled_da` |
| `arm_config` | yes | Frozen dict: `latent_steps`, `text_budget_tokens`, `toolmessage_mode`, seed |
| `binding` | yes | `mixture_pair` or `sequential_trio`; which link set is in use |
| `source_role` / `target_role` | yes | Roles either side of the boundary |
| `source_model` / `target_model` | yes | Resolved checkpoint ids, so Exp 3a never compares across model pairs by accident |
| `outer_link_key` | latent only | e.g. `outer_2s`; must match the role pair |
| `handoff_index` | yes | Monotonic within run |
| `round_index` | yes | Which delegation round, when a role is revisited |
| `text_handoff` | yes | Full untruncated string. Empty only if stubbed by design; record which |
| `text_handoff_seen_by_receiver` | yes | What the receiver actually got, after any cap or stub. Exp 3b scores this, not `text_handoff` |
| `latent_shape` | latent only | Tensor shape after outer link; expect `[latent_steps, target_hidden]` |
| `latent_norm` | latent only | Norm after outer link |
| `latent_artifact_path` | latent only | Where the tensor is persisted; Exp 3a needs the vectors, not just the metadata |
| `cos_in_out` | latent only | Cosine between pre- and post-outer-link states. Null with a reason if widths differ |
| `effective_latent_steps` | latent only | As executed, not as requested. Exp 2 discards runs where these disagree |
| `receiver_consumed_latent` | latent only | Boolean assertion from the injection path, not an inference |
| `fixture_hash` | yes | Hash of the task fixture at run start; Exp 1 Step 3 uses this to prove isolation |
| `prompt_tokens` | yes | Input token count when available |
| `completion_tokens` | yes | Output token count when available |
| `wall_clock_seconds` | yes | Boundary or role-level runtime |
| `gpu_seconds` | latent preferred | Nullable if unavailable |
| `error` | yes | Null on success, structured error on failure |

`text_handoff` and `text_handoff_seen_by_receiver` are separate on purpose. In `text_da_cap` they differ by truncation, and in `latent_da` the first may be populated for analysis while the receiver only saw a stub. Collapsing them makes the Exp 3b observability score unauditable.

## Execution Plan

### Step 1: Pin the Runtime

This step is `PLAN.md` Gate 0 in full. Do not restate it here; run it there and copy the resulting run note.

Action:

- Complete `PLAN.md` Gate 0.1 through 0.5: populated venv, topology binding, compute target, smoke reproduced with the receiver loaded, golden task frozen.
- Record Python version, package versions, CUDA or MPS availability, resolved checkpoint ids, outer link keys with their source and target widths, and the RecursiveMAS commit hash.
- Snapshot repos individually. `resolve_mas_paths("mixture")` pulls roughly 27 GB including a 7B science model that this study never uses.

Verification:

- Run an import smoke test for Deep Agents and RecursiveMAS inference utilities.
- Print the resolved checkpoint paths or Hugging Face identifiers.
- For each planned latent edge, print the link key, its output width, and the target model's embedding width, and assert they match.
- Save environment metadata in the run note.

Success criteria:

- All required packages import cleanly.
- Every planned latent edge has a resolvable link whose output width equals the receiver's embedding width.
- No manual notebook-only setup is required for the harness runner.

If this fails:

- Stop before graph work.
- Record the missing dependency, checkpoint, or link.
- If a planned edge has no link, change the graph rather than improvising one.

### Step 2: Define the Golden Task

Action:

- Choose one multi-file coding task with a small but real dependency graph.
- Include a task prompt, starting repository state, expected files touched, and an executable test command.
- Ensure the task is nontrivial enough to require delegation but small enough to debug quickly.

Verification:

- Run tests on the untouched fixture and confirm the expected baseline failure.
- Apply or inspect the known solution and confirm tests pass.
- Record expected changed files and expected symbols.

Success criteria:

- Baseline fails for the intended reason.
- Known solution passes.
- The task requires at least two files or one file plus one test file.

If this fails:

- Replace the fixture before continuing.
- Do not use a task without an executable oracle unless the limitation is explicitly accepted.

### Step 3: Build the Shared Graph

Action:

- Implement a graph builder that accepts a delegation backend parameter.
- Define the roles from the chosen binding: orchestrator (main agent) and code worker (subagent) under Binding A.
- Back both arms with the same local checkpoints through the same `BaseChatModel`.
- Give each role the same tools in both conditions.
- Keep prompts identical except for the minimal instruction needed to route through the configured handoff channel.

Verification:

- Run a dry graph construction test for both backends.
- Dump the graph configuration for Text-DA and Latent-DA.
- Diff the configurations and confirm only the delegation backend differs.

Success criteria:

- Both graph configurations instantiate successfully.
- Role names, tool lists, sandbox paths, prompt templates, task inputs, and backing checkpoints match.
- The only intended difference is the delegation channel.

If this fails:

- Fix the graph abstraction before implementing results collection.
- Do not proceed with condition-specific graph forks that change tools or role behavior.

### Step 4: Implement Text-DA Baseline

Action:

- Wire the default Deep Agents Task or subagent message handoff.
- Log every natural-language handoff.
- Preserve the full handoff text without truncation unless a separate raw artifact is stored.

Verification:

- Run Text-DA on a minimal toy task.
- Confirm every plan -> implement and implement -> review boundary has a log record.
- Confirm the logged text matches what the receiving role sees.

Success criteria:

- Text-DA completes the toy task.
- Handoff count matches the expected graph transitions.
- No required log fields are missing.

If this fails:

- Fix text logging first.
- Exp 3b cannot be run if text handoffs are not faithful to the receiving role context.

### Step 5: Implement Latent-DA Backend

Action:

- Add `LatentRecursiveBackend` under `integrations/deepagents_latent/`.
- Build it on the primitives — `load_agent_model_and_tokenizer`, `autoregressive_latent_rollout`, `run_inner_adapter`, `run_outer_adapter`, `split_prompt_ids_by_slots`, `pad_left_embeds` — holding the models persistently.
- Mirror `run_hie_expert_latent_stage` (`inference_mas_mixture.py:233-349`) rather than calling it. It loads a model on entry and releases it on exit, and it is batch-oriented over a question list; calling it per delegation boundary reloads a 3B model at every handoff. `PLAN.md` section 4 has the primitive map.
- Emit the post-outer-link latent bundle and metadata, persisting the tensor so Exp 3a can featurize it later.
- Transport the bundle on the subagent's returned graph state, not in the `ToolMessage`. `_return_command_with_state_update` (`subagents.py:484`) forwards any state key outside `{messages, todos, structured_response}`, while `ToolMessage` content is strings only.
- Bridge state to the model call in a `wrap_model_call` middleware, then inject via `generate(inputs_embeds=...)` following `run_solver_latent_stage:1490-1507`.
- **Load both link directions** (`outer_2s` and `outer_s2`), even though Exp 0 only exercises the forward (worker → orchestrator) direction. Exp 4 iterates the reverse edge as a feedback loop; loading and dimension-checking it now costs one smoke call and avoids rewriting the backend later.

Verification:

- Run a latent-only smoke call on a short coding prompt.
- Assert that latent output shape, dtype, device, norm, and `cos(in,out)` are logged.
- Assert the bundle shape is `[latent_steps, target_hidden]` and that `target_hidden` equals the receiver's embedding width.
- Smoke the reverse direction once: `outer_s2` on an orchestrator-side rollout produces a finite bundle whose width equals the worker's embedding width. Not injected anywhere in Exp 0; the assertion is load + shape + finiteness.
- Assert the `ToolMessage` in the latent arm contains the stub and not the model's natural-language result.
- Pin the `generate(inputs_embeds=...)` return convention with an explicit assertion. The released code guesses via `sequences.size(1) > max_new_tokens` (`inference_mas.py:1511-1517`); inherit the behavior deliberately, not accidentally.

Success criteria:

- Latent backend runs without crash on the smoke prompt.
- Models are loaded once and reused across handoffs.
- `cos(in,out)` is finite and not near identity collapse by default.
- Latent metadata and a retrievable tensor artifact are present for every latent handoff.

If this fails:

- Separate failures into model loading, outer-link execution, state transport, and receiver injection.
- Keep Text-DA runnable, but mark Exp 0 as blocked until latent injection or sidecar fallback is explicit.

### Step 5b: Prove the Receiver Actually Uses the Bundle

This is the RQ0 gate. Everything Exp 1 and Exp 2 claim about the channel is void if the receiver ignores what is injected, and a harness can look completely healthy while doing exactly that: shapes are right, cosines are fine, logs are complete, and the model quietly conditions on the prompt text alone.

Action:

- Take one prompt and produce three receiver outputs: with its own bundle, with a bundle computed for an unrelated task, and with `latent_steps=0`.
- Hold sampling seed, prompt text, and every other input fixed.
- Implement these variants in `controls.py` so Exp 1 and Exp 2 reuse them rather than reimplementing them.

Verification:

- Compare the three outputs token by token.
- Repeat across at least a handful of prompts, since a single collision proves nothing.
- Record output divergence rate.

Success criteria:

- Own-bundle output differs from shuffled-bundle output on most prompts.
- Own-bundle output differs from the `latent_steps=0` output.

If this fails:

- The channel is inert. Do not proceed to Exp 1 as a channel study.
- Diagnose in order: is the bundle reaching the model call, is the slot in the rendered prompt where the splice assumes, are the embeddings scaled compatibly with the receiver's embedding distribution, is attention masking correct for the spliced rows.
- If it cannot be fixed, this is itself a reportable finding about injecting released links into an unmodified harness.

### Step 6: Run Both Arms on the Golden Task

Action:

- Run Text-DA once on the golden task.
- Reset the fixture to the same starting state.
- Run Latent-DA once on the golden task.
- Use the exact same task prompt and test command.

Verification:

- Confirm both runs produce run directories with logs, final outputs, patches or file diffs, and test results.
- Confirm the golden task fixture was reset between runs.
- Confirm every expected delegation boundary has a handoff log.

Success criteria:

- Both arms complete without harness crash.
- Test results are captured even if the task solution fails.
- Text and latent runs are comparable from the same initial state.

If this fails:

- If one arm crashes, record the crash class and minimal reproduction.
- If the fixture reset is suspect, discard the run and repeat.
- If tests are missing, mark the run invalid.

### Step 7: Validate Logs for Downstream Experiments

Action:

- Run a schema validation pass over Text-DA and Latent-DA logs.
- Check that latent records include tensor metadata and cosine values.
- Check that text records include readable handoff strings.
- Confirm logs contain enough identifiers to join later with task labels.

Verification:

- Produce a validation report listing record counts, missing fields, and invalid values.
- Spot-check at least one handoff per role transition.

Success criteria:

- Zero missing required fields.
- Handoff counts match expected graph transitions.
- Latent numeric fields are finite.
- Text handoffs are nonempty for Text-DA.

If this fails:

- Fix logging and rerun Exp 0.
- Do not proceed to Exp 1 or Exp 3 with incomplete handoff logs.

### Step 8: Make the Go/No-Go Decision

Action:

- Summarize Text-DA result, Latent-DA result, harness failures, log completeness, and runtime.
- Decide whether Exp 1 can begin.

Verification:

- Review the run note against the success checklist below.
- Confirm every required artifact exists.

Success criteria:

- Shared graph works for both arms.
- The golden task run is reproducible.
- Handoff logs are complete and schema-valid.
- Latent channel health does not show immediate collapse.

If this fails:

- Mark Exp 1 blocked.
- Create a short fix list with owners or next actions.

## Metrics

- End-to-end completion without harness crash.
- Golden task test pass/fail.
- Handoff log completeness.
- Runtime sanity: wall-clock, tokens, and GPU seconds where available.
- Channel-health sanity: outer-link `cos(in,out)` per latent handoff.
- Reproducibility: same command can rerun both conditions from a clean fixture.

## Experiment-Level Success Criteria

Exp 0 passes only if:

- Text-DA and Latent-DA both run on the same graph, same task, and same checkpoints.
- Every latent edge uses a released link whose target width matches the receiver.
- Logs include human-readable text and latent handoff metadata at every required delegation boundary, with retrievable tensor artifacts.
- Latent outer-link cosine does not indicate identity collapse.
- **Step 5b passes**: the receiver's output responds to which bundle it is given.
- Both arms capture final test results.
- The run note documents exact commands and environment metadata.

## Follow-On Experiments

- Exp 1 uses this harness for the controlled channel ablation, and reuses `controls.py` for the budget-matched text arm and the shuffled control.
- Exp 2 reuses the latent backend with different `latent_steps` and the matched text budgets.
- Exp 3a and Exp 3b require the handoff logs and tensor artifacts created here.
- Exp 4 reuses the backend's reverse edge (`outer_s2`) and the `round_index` field to iterate the handoff as a feedback loop; its Step 3 (loop spike) is Step 5b applied to the reverse direction.
