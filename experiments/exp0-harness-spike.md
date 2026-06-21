# Exp 0: Harness Spike

Source: `PROPOSAL.md` section 7.

## Purpose

Validate that the Deep Agents integration can run both delegation channels end to end on one golden multi-file coding task before any scaling sweeps.

This is the implementation gate for the rest of the study. It does not answer a headline research question directly; it proves that the controlled A/B harness is viable and that handoff logging is available for Exp 1, Exp 2, Exp 3a, and Exp 3b.

## Scope

In scope:

- One shared Deep Agents graph with 3 roles: plan -> implement -> review.
- One Text-DA baseline using normal natural-language handoff.
- One Latent-DA path using the released RecursiveMAS inference code.
- One golden multi-file task with executable tests.
- Handoff logging at every delegation boundary.

Out of scope:

- Large task sweeps.
- Probe training.
- Observability scoring.
- Fine-tuning RecursiveLink weights.
- Adaptive recursion depth.

## Conditions

| Condition | Delegation channel | Expected behavior |
|---|---|---|
| Text-DA | Natural-language subagent handoff | Deep Agents default Task/message flow completes the task |
| Latent-DA | RecursiveMAS latent bundle | Sidecar produces a latent handoff and the next agent consumes injected latent state |

Keep roles, tools, filesystem access, shell access, sandbox, prompts, task fixture, and evaluation command identical across conditions.

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

Each handoff record should include:

| Field | Required | Notes |
|---|---|---|
| `run_id` | yes | Unique run identifier |
| `task_id` | yes | Golden task id |
| `condition` | yes | `text_da` or `latent_da` |
| `source_role` | yes | Role emitting the handoff |
| `target_role` | yes | Role receiving the handoff |
| `handoff_index` | yes | Monotonic within run |
| `text_handoff` | yes | Empty only if unavailable by design; record why |
| `latent_shape` | latent only | Tensor shape after outer link |
| `latent_norm` | latent only | Norm after outer link |
| `cos_in_out` | latent only | Cosine between pre- and post-outer-link states |
| `prompt_tokens` | yes | Input token count when available |
| `completion_tokens` | yes | Output token count when available |
| `wall_clock_seconds` | yes | Boundary or role-level runtime |
| `gpu_seconds` | latent preferred | Nullable if unavailable |
| `error` | yes | Null on success, structured error on failure |

## Execution Plan

### Step 1: Pin the Runtime

Action:

- Install Deep Agents in an isolated environment.
- Record Python version, package versions, CUDA or MPS availability, model checkpoint names, and RecursiveMAS commit hash.
- Confirm the released RecursiveMAS inference path can import without changing model code.

Verification:

- Run an import smoke test for Deep Agents and RecursiveMAS inference utilities.
- Print the resolved checkpoint paths or Hugging Face identifiers.
- Save environment metadata in the run note.

Success criteria:

- All required packages import cleanly.
- The selected local model and outer links are resolvable.
- No manual notebook-only setup is required for the harness runner.

If this fails:

- Stop before graph work.
- Record the missing dependency or checkpoint.
- Decide whether to pin a compatible version, add a resolver, or change the local backend.

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
- Define the roles as plan, implement, and review.
- Give each role the same tools in both conditions.
- Keep prompts identical except for the minimal instruction needed to route through the configured handoff channel.

Verification:

- Run a dry graph construction test for both backends.
- Dump the graph configuration for Text-DA and Latent-DA.
- Diff the configurations and confirm only the delegation backend differs.

Success criteria:

- Both graph configurations instantiate successfully.
- Role names, tool lists, sandbox paths, prompt templates, and task inputs match.
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
- Call `run_hie_expert_latent_stage` or the closest released sequential latent chain.
- Emit the post-outer-link latent bundle and metadata.
- Inject or otherwise provide the latent state to the receiving role through the selected local backend path.

Verification:

- Run a latent-only smoke call on a short coding prompt.
- Assert that latent output shape, dtype, device, norm, and `cos(in,out)` are logged.
- Confirm the receiver path consumes the latent bundle rather than silently falling back to text-only behavior.

Success criteria:

- Latent backend runs without crash on the smoke prompt.
- `cos(in,out)` is finite and not near identity collapse by default.
- Latent metadata is present for every latent handoff.

If this fails:

- Separate failures into model loading, outer-link execution, and receiver injection.
- Keep Text-DA runnable, but mark Exp 0 as blocked until latent injection or sidecar fallback is explicit.

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

- Text-DA and Latent-DA both run on the same graph and same task.
- Logs include human-readable text and latent handoff metadata at every required delegation boundary.
- Latent outer-link cosine does not indicate identity collapse.
- Both arms capture final test results.
- The run note documents exact commands and environment metadata.

## Follow-On Experiments

- Exp 1 uses this harness for the controlled channel ablation.
- Exp 2 reuses the latent backend with different `latent_steps`.
- Exp 3a and Exp 3b require the handoff logs created here.
