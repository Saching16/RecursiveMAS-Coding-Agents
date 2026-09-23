# PLAN.md — Pre-Implementation Plan for Deep Agents Integration

> **Scope:** What must be true *before* writing `integrations/deepagents_latent/`.
> **Relationship to other docs:**
> - [`PROPOSAL.md`](PROPOSAL.md) — the research plan (RQs, experiments, paper).
> - [`experiments/exp0-harness-spike.md`](experiments/exp0-harness-spike.md) — the implementation gate, expanded.
> - `experiments/exp1..exp4` — the **study**, not prerequisites. Do not start them first.
>
> This file is the engineering checklist. It exists because the experiment docs
> describe *what to measure*, not *what has to work first*.

**Last verified:** 2026-09-23 · RunPod RTX 4090 (EU-RO-1) · RecursiveMAS `4958be6`
on `deepagents-latent-integration`. Gate 0.1 GREEN; see §1.2 for the live
environment and §1 for the superseded local-Mac snapshot.

---

## 1. Verified environment snapshot

> **The table below describes the LOCAL DEV MAC, not the compute target.** It is
> kept because it explains why Gate 0.3 went remote (§1.1). The environment the
> project actually runs on is the RunPod box in **§1.2**, where Gate 0.1 is
> **GREEN**. Do not read the "Blocker" rows below as open work.

Everything below was checked on this machine, not assumed.

| Item | Verified value | Status |
|---|---|---|
| Python (system) | 3.13.2 (`/Library/Frameworks/Python.framework`) | OK — Deep Agents needs `>=3.11,<4.0` |
| Python (`.venv`) | 3.13.3 | OK |
| Platform | macOS 26.4, arm64 | OK |
| torch | **2.10.0** system-wide; `requirements.txt` pins **2.9.0**; **absent from `.venv`** | Mismatch |
| CUDA | **Not available** | Constraint |
| MPS | Available | Only local accelerator |
| numpy | 2.2.3 system-wide; pinned 2.2.6; **absent from `.venv`** | Minor drift |
| transformers | **Missing** (pinned 5.3.0) | **Blocker** |
| datasets | **Missing** | **Blocker** |
| huggingface_hub | **Missing** | **Blocker** |
| accelerate | **Missing** | **Blocker** |
| safetensors | **Missing** | **Blocker** |
| langchain | **Missing** | **Blocker** |
| deepagents | **Missing** | **Blocker** |
| scikit-learn | **Missing**; not in `requirements.txt` | Needed by Exp 3a probes |
| `.venv` | **Exists**, but contains only `ipykernel` + deps | **Blocker** |
| pip | 26.2.1 (in `.venv`) | OK |
| Free disk | **~2.2 GiB** of 228 GiB (was ~16 GiB on 2026-08-25) | **Blocker** — see §1.1 |

`.venv` was created since the last revision of this file but never populated from
`requirements.txt`. Gate 0.1 is therefore still red, for a different reason than before.

### 1.1 Disk has collapsed since the last snapshot — this decides Gate 0.3

Free space went from ~16 GiB to **~2.2 GiB**. Binding A needs ~11 GB of checkpoints
(§2). **Local MPS is no longer an option for Gate 0.4**, and the Gate 0.3 "compute
target" decision is now forced rather than open: it has to be remote, or ~15 GB has to
be freed first.

Two measurement traps found while working on this machine, both worth knowing before
trusting a `df` number:

- **macOS swap is carved from the same APFS volume.** Free space swung by >1 GB during
  model runs and recovered on process exit. Alarming `df` readings mid-run are usually
  swap, not a runaway write.
- **`du` over-reports reclaimable space.** `uv cache clean` freed ~200 MB against a
  claimed 3.4 GB, because APFS copy-on-write shares blocks between the cache and
  installed venvs. Budget from `df` deltas, not `du` totals.

Also relevant to Gate 0.3: this machine has ~8 GB RAM and was observed at 113 MB
unused with ordinary apps running. A 0.5B model running *inside* an agent graph
thrashed and never completed; single direct model calls were fine. Interactive agent
runs against a 3B checkpoint are not viable here regardless of disk.

### 1.2 Compute target: RunPod RTX 4090 (NVIDIA) — Gate 0.1 GREEN 2026-09-23

AMD Developer Cloud credits were applied for but have an unknown lead time, so
the project did not wait on them. Compute is a RunPod pod; the ROCm notes are
retained at the end of this section for if those credits land, since the code is
device-agnostic and would run unmodified.

**Live pod:** `latent-delegation-gate0` / `xr8mzs5tq7zjm6`, RTX 4090 24 GB,
SECURE tier **$0.74/hr**, EU-RO-1, network volume `mnfmb0n80m` at `/workspace`.
Currently **stopped** (`EXITED`) — GPU released, resumable with `start`.
Community-tier 4090s were $0.34/hr but had no stock.

**Verified environment** (run note: `integrations/deepagents_latent/runs/gate0-20260923T002604Z.md`):

| Item | Value |
|---|---|
| torch | 2.9.0+cu128 (`cuda=12.8`, `hip=None`) |
| transformers | 5.3.0 |
| Python | 3.13.8 |
| Host driver / CUDA | 580.126.20 / 13.0 |
| Device | RTX 4090, 23.5 GiB, bf16 matmul verified on-device |
| **`attn_implementation`** | **`sdpa`** — flash-attn absent. Pin this in BOTH arms (§11 numerics) |

The `torch==2.9.0` pin resolved to `2.9.0+cu128` on a CUDA-13.0 host driver
without an explicit wheel tag: a newer driver runs an older runtime, so no
`--cuda` override was needed.

**Checkpoints already on the volume — 9.7 GB, do not re-download:**

| Repo | Size |
|---|---|
| `Mixture-Code-Qwen2.5-Coder-3B` | 5.8 G |
| `Mixture-Summarizer-Qwen3.5-2B` | 3.6 G |
| `Mixture-Outerlinks` | 322 M |

Fetched with three targeted `snapshot_repo` calls (`hf_resolver.py:11`), not
`resolve_mas_paths` — see §2. Set `HF_HOME=/workspace/hf-cache` or they
re-download to container disk.

#### Two traps found the hard way on 2026-09-23

- **Never put the venv on the network volume.** `/workspace` is MooseFS:
  387 MB/s sequential (fine for weights — 5.8 GB landed in 15 s) but
  metadata-bound work crawls, and `pip install --upgrade pip` had not finished
  after several minutes. Local overlay measures **14.1 GB/s**, ~36× faster.
  Layout that works: **code + `.venv` on local disk (`/root`), weights on
  `/workspace`.** Code is a git clone, so losing it on terminate costs seconds.
  `gate0_setup.sh` defaults the venv to `$REPO_ROOT/.venv`, which is a trap if
  the repo is cloned onto the volume — clone to `/root`, or give the script a
  `--venv-path` flag.
- **Register SSH keys BEFORE creating the pod.** `startSsh` injects
  `PUBLIC_KEY` at create time only. A pod created with no keys registered gets
  `PUBLIC_KEY: "null"` permanently; registering afterwards does not reach a
  running pod ("Keys take effect for pods created afterwards"), so it costs a
  terminate/recreate. Also: a passphrase-protected key must be loaded into
  `ssh-agent` from a real terminal (`ssh-add --apple-use-keychain`) — the
  prompt cannot be satisfied non-interactively, and the symptom is a confusing
  `Server accepts key` immediately followed by `Permission denied`.

#### If the AMD credits land

The code is device-agnostic; `gate0_setup.sh --backend rocm` handles it. Four
differences:

- **`requirements.txt` will install the wrong torch.** `torch==2.9.0` resolves
  to the default PyPI wheel, which has no ROCm support. Install from the ROCm
  index *first* so it satisfies the pin, then install the rest. Do not edit the
  pin to paper over this.
- **Device strings do not change.** PyTorch's ROCm build keeps the `cuda`
  namespace via HIP; `inference_utils/` needs no edit.
- **`release_resources`'s cache-emptying works there.** §4 flags that it only
  empties the CUDA cache, which frees nothing on MPS. On ROCm
  `torch.cuda.empty_cache()` maps to HIP.
- **Attention kernels are the real risk.** `flash-attn`, `xformers` and
  `bitsandbytes` are CUDA-first. Record the fallback and keep it identical
  across arms — on this NVIDIA box it is already `sdpa`, so switching vendors
  mid-study would change numerics unless that is held fixed.

### Import checks

**On the RunPod box (§1.2): both PASS as of 2026-09-23.**

```text
import modeling; import deepagents                                      → ok
from inference_utils.inference_mas import (
    autoregressive_latent_rollout, run_outer_adapter)                   → ok
```

Historical, on the local Mac — the failures that made Gate 0.1 red before the
project moved to remote compute:

```text
import modeling
  → ModuleNotFoundError: No module named 'huggingface_hub'   (modeling.py:7)

from inference_utils.inference_mas import autoregressive_latent_rollout
  → ModuleNotFoundError: No module named 'datasets'          (inference_mas.py:29)
```

### Repositories

| Repo | Commit | Notes |
|---|---|---|
| RecursiveMAS | `4e2934d` on `main` | `origin` = `Saching16/RecursiveMAS-Coding-Agents` (fork), `upstream` = `RecursiveMAS/RecursiveMAS` |
| Deep Agents | `b650b41`, lib version **0.7.8** | Clone at `../RecursiveMAS-HarnessCompare/vendors/deepagents`, symlinked to `../deepagents` |

Deep Agents runtime deps: `langchain>=1.3.16`, `langchain-core>=1.6.0`,
`langchain-anthropic`, `langchain-google-genai`, `langsmith`, `packaging`, `wcmatch`.

### Checkpoints

Both repos named in `PROPOSAL.md` §6 resolve (HTTP 200, metadata only — not downloaded):

- `RecursiveMAS/Mixture-Code-Qwen2.5-Coder-3B`
- `RecursiveMAS/Mixture-Outerlinks`

These two are **not sufficient**. A latent handoff needs a *receiver* whose embedding
width the outer link was trained to emit. See §2.

---

## 2. The released links constrain the topology

This is the finding that most changes the plan, and it is not reflected in
`PROPOSAL.md` §7 or the Exp 0 doc.

### What edges actually exist

`system_loader.py:78-100` enumerates every outer link in the release. Each one is
bound to an ordered **pair of specific checkpoints**, and `load_mas_system` hard-fails
if the link's output width does not equal the target model's embedding width
(`system_loader.py:213-217`).

| Family | Link key | Source → Target | Inner-adapter scope |
|---|---|---|---|
| `mixture` | `outer_2s` | Code `Qwen2.5-Coder-3B` → Summarizer `Qwen3.5-2B` | task-agnostic (`task=None`) |
| `mixture` | `outer_s2` | Summarizer `Qwen3.5-2B` → Code `Qwen2.5-Coder-3B` | task-agnostic |
| `sequential_*` | `outer_12` | Planner → Critic | task-scoped; `code` variant exists |
| `sequential_*` | `outer_23` | Critic → Solver | task-scoped; `code` variant exists |
| `sequential_*` | `outer_31` | Solver → Planner | task-scoped; `code` variant exists |

`hf_resolver.infer_adapter_task_for_dataset` maps `mbppplus` → `"code"`, so the
sequential links do have code-trained variants; the mixture links are task-agnostic.

**There is no code-expert → code-expert link.** A `plan → implement → review` graph
staffed by three copies of the same coder model — what `PROPOSAL.md` §7 and `experiments/exp0`
Step 3 originally assumed — **has no released link to run on**.

### Two admissible bindings

| | **Binding A — mixture pair** | **Binding B — sequential trio** |
|---|---|---|
| Roles | orchestrator = Summarizer 2B; worker = Code expert 3B | plan = Planner 1.7B; review = Critic 1B; implement = Solver 1.5B |
| Latent edges | worker→orch `outer_2s`; orch→worker `outer_s2` | `outer_12`, `outer_23`, `outer_31` (a cycle) |
| Fit to Deep Agents | **Native** — main agent + `task` subagent is exactly this shape | Awkward — the cycle is `plan → review → implement`, not `plan → implement → review` |
| Coding competence | Code expert is a real coder; summarizer is not | Solver is `Qwen2.5-Math-1.5B`, math-specialised |
| Links trained for code | No (task-agnostic) | Yes (`task="code"`) |
| Download (bf16) | ~6 GB + ~4 GB + links ≈ **11 GB** | ~3.4 + 2 + 3 GB + links ≈ **9 GB** |
| Week-1 smoke used it | Yes (`outer_2s`) | Partially (`a100_extended` tier, `task="math"`) |

**Recommended default: Binding A**, because the main-agent/subagent shape is the thing
the paper claims to be studying and the code expert is the only released coder. Binding B
is the robustness check if reviewers push on "your links were never trained for code."
This is a decision for §9, not something to leave implicit.

Note that Binding A is **bidirectional**: `outer_2s` and `outer_s2` together form a
closed worker ↔ orchestrator loop — the same shape as the release's own
`num_recursive_rounds` feedback cycle (`inference_mas_mixture.py:754-776`). Exp 0–3 use
only the forward direction (one-shot handoff); Exp 4
([`experiments/exp4-outer-recursive-delegation.md`](experiments/exp4-outer-recursive-delegation.md))
iterates the loop. Gate 1 must load and smoke **both** directions so Exp 4 reuses the
backend instead of rewriting it.

### Do not call `resolve_mas_paths` / `load_mas_system`

`resolve_mas_paths` snapshots **every** repo in a style spec (`system_loader.py:128-129`).
For `mixture` that includes `Mixture-Science-BioMistral-7B` (~14 GB) and
`Mixture-Math-DeepSeek-R1-Distill-Qwen-1.5B`, totalling roughly **27 GB** — more than the
~16 GiB free on this machine. `load_mas_system` then loads all four onto the device.

The Week-1 notebook avoids this by calling `snapshot_repo` per repo. The integration must
do the same: snapshot only the two role checkpoints plus `Mixture-Outerlinks`, and use
`resolve_outer_paths(outer_dir, task=None)` to pick out `outer_2s` / `outer_s2`.

### Disk verdict

Binding A needs ~11 GB against ~16 GiB free, before any HF cache duplication, task
fixtures, or run artifacts. That is workable but leaves little headroom, and it removes
the option of holding a third role model. It is a real input to the Gate 0.3 compute
decision rather than a footnote.

---

## 3. Documentation gaps found during verification

These are claims in the docs that this tree does not currently support.

| Claim | Reality | Action |
|---|---|---|
| `plan → implement → review` graph, same model per role (`PROPOSAL.md` §7, `exp0` Step 3) | **No released link supports it** — see §2 | Rebind the topology to Binding A or B; propagate to all five experiment docs |
| `PROPOSAL.md` §6 "Week-1 Gate — COMPLETE", artifact `notebooks/smoke_results.json` | **File does not exist** in the repo | Re-run smoke and commit the JSON, or downgrade the claim to "run on Colab, artifact not archived" |
| `notebooks/latent_channel_smoke.ipynb` | 10 cells, **zero saved outputs** | Commit an executed copy so the cosine numbers are reproducible from the repo |
| Notebook links `../PROPOSAL_LONGFORM.md` | **File does not exist** (superseded) | Fix the link |
| Notebook targets Colab (`google.colab`, `cuda`, A100) | No CUDA locally | Decide local-MPS vs remote GPU before Exp 0 Step 1 |
| `integrations/` | Does not exist | Created in Gate 1 |
| `cos(in, out)` reported as the channel-health metric | Computed as `F.cosine_similarity(self_latent, mapped, dim=-1)` (notebook, `latent_cosine_stats`). Source and target embedding widths must be **equal** or this is undefined | Verify code-expert hidden == summarizer hidden in Gate 0.4; if unequal, redefine the metric before quoting it |
| Week-1 smoke "validated the outer link" | It never downloaded the summarizer — only the source side was exercised | Gate 0.4 must load the receiver too |

There is also a placeholder at
`../RecursiveMAS-HarnessCompare/integrations/deepagents_latent/__init__.py`.
That is scratch. The real package belongs in **this** repo at
`integrations/deepagents_latent/`, matching `PROPOSAL.md` §5 and Exp 0.

---

## 4. Key code finding that changes the design

Earlier drafts of `PROPOSAL.md` and Exp 0 both said to call `run_hie_expert_latent_stage`.
Verified signature and lifecycle:

```python
# inference_utils/inference_mas_mixture.py:233-349
def run_hie_expert_latent_stage(
    stage_name: str,
    model_name_or_path: str,
    questions: Sequence[str],
    ...
) -> List[torch.Tensor]:
    ...
    model, tokenizer = base.load_agent_model_and_tokenizer(...)   # line 259
    ...
    base.release_resources(model, tokenizer, inner, outer)        # line 348
    return outputs
```

It **loads the model on entry and releases it on exit**, and it is batch-oriented
(`questions: Sequence[str]`). That is correct for offline eval sweeps and wrong for
an interactive harness: calling it per delegation boundary would reload a 3B model
at every handoff.

**Design consequence:** the sidecar must hold the model **persistently** and call the
lower-level primitives directly:

| Primitive | Location |
|---|---|
| `load_agent_model_and_tokenizer` | `inference_utils/inference_mas.py:641` |
| `release_resources` | `inference_utils/inference_mas.py:665` |
| `run_inner_adapter` | `inference_utils/inference_mas.py:813` |
| `run_outer_adapter` | `inference_utils/inference_mas.py:829` |
| `autoregressive_latent_rollout` | `inference_utils/inference_mas.py:846` |

Treat `run_hie_expert_latent_stage` as the **reference implementation to mirror**,
not the function to call. `experiments/exp0-harness-spike.md` Step 5 has been updated to
match; if it drifts back, this section wins.

### What the latent bundle actually is

`autoregressive_latent_rollout` (`inference_mas.py:846-886`) returns
`[batch, latent_steps, source_hidden]` — one hidden vector per rollout step. After
`run_inner_adapter` and `run_outer_adapter` the per-sample bundle is
`[latent_steps, target_hidden]`, i.e. **`latent_steps` synthetic embedding rows**.

Two consequences worth carrying into the experiment design:

1. `latent_steps` is literally the width of the channel in tokens. That makes a
   **token-matched text baseline** possible and cheap: cap the text handoff at the same
   `k` tokens the latent arm gets. Without it, Exp 1 confounds channel type with
   channel budget, and Exp 2's capacity curve has no comparable text axis.
2. `latent_steps=0` is a legal, already-supported setting: every latent stage returns an
   empty `[0, out_dim]` tensor. That is a free **null-channel control**.

### How the receiver consumes it

`run_refiner_latent_stage:1140-1141` and `run_solver_latent_stage:1490-1507` show the
whole injection pattern: render the prompt with a slot marker, split it with
`split_prompt_ids_by_slots`, embed prefix and suffix, `torch.cat([prefix, latent, suffix])`,
then `model.generate(inputs_embeds=..., attention_mask=...)`. `hf_chat_model.py` should
reproduce this and nothing more.

Rough edges to expect:

- **One slot per prompt.** `run_hie_expert_latent_stage` accepts a single
  `feedback_latents` bundle. If the orchestrator ever fans out to two subagents, the
  aggregation rule (concatenate along the sequence axis? drop all but the last?) is
  undefined by the release and must be a recorded design choice.
- **`generate(inputs_embeds=...)` return format.** `inference_mas.py:1511-1517` uses
  `sequences.size(1) > max_new_tokens` to guess whether the prompt was echoed back.
  That heuristic is fragile; assert on it in Gate 1 rather than inheriting it silently.
- **`release_resources` only empties the CUDA cache** (`inference_mas.py:665-671`).
  On MPS nothing is freed, which matters if the sidecar reloads models.

---

## 5. Gate 0 — must be green before any integration code

Nothing here is Exp 1/2/3. Five tasks.

### 0.1 Create an isolated environment and make imports pass

```bash
cd /Users/sachinganpule/projects/RecursiveMAS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../deepagents/libs/deepagents
```

Expect friction: `requirements.txt` pins `torch==2.9.0` while 2.10.0 is installed
system-wide, and `transformers==5.3.0` is a major pin. Resolve in the venv, do not
edit `requirements.txt` to match a broken local state.

**Done when:**

```bash
python -c "import modeling; import deepagents; print('ok')"
python -c "from inference_utils.inference_mas import autoregressive_latent_rollout, run_outer_adapter; print('ok')"
```

both print `ok`, and the exact versions are written into a run note. Add `scikit-learn`
while you are here — Exp 3a needs it and `requirements.txt` does not list it.

### 0.2 Bind the topology to a released link set

Pick Binding A or Binding B from §2 and write it down. Everything downstream — role
names, which checkpoints get downloaded, which link keys appear in the handoff log, how
Exp 3a matches handoffs — depends on this choice.

**Done when:** the chosen binding is recorded, and for every delegation edge in the
planned graph there is a named link file that exists in the snapshot.

If no such link exists for an edge, that edge cannot be a latent handoff. Redesign the
graph rather than improvising a link.

### 0.3 Decide the compute target

CUDA is unavailable here. Pick one and write it down:

| Option | Cost | Note |
|---|---|---|
| Local MPS | Free | Untested for these checkpoints; bf16 on MPS is a known rough edge; ~16 GiB disk is the binding constraint (§2) |
| Colab / remote A100 | Cheap | Matches the Week-1 smoke setup |
| Other remote GPU | Varies | Needed eventually for Exp 1 sweeps |

The A/B only has to be *internally consistent*; both arms must run on the same device.
Note that Exp 1 and Exp 2 together are a few hundred agent runs against a 3B model — a
decision that is survivable for Exp 0 on MPS may not be for the sweeps.

### 0.4 Reproduce the latent channel smoke, receiver included

Re-run the Week-1 probe on the chosen device with the **full** Binding A pair —
`Mixture-Code-Qwen2.5-Coder-3B`, `Mixture-Summarizer-Qwen3.5-2B`, `Mixture-Outerlinks`,
`latent_steps=32`. The Week-1 run only loaded the source model, so the receiving half of
the channel has never been exercised.

**Done when:**

- `notebooks/smoke_results.json` exists and is committed.
- `cos(in, out)` is finite and far from 1.0 (Week-1 reported mean ≈ −0.004).
- Code-expert hidden width and summarizer hidden width are recorded, and either match
  (so the cosine is well-defined) or the metric is redefined and the change noted.
- A `[latent_steps, target_hidden]` bundle has been injected into the summarizer via
  `inputs_embeds` and produced non-degenerate text.

If this fails, **stop**. The harness cannot inject what the model stack cannot produce.

### 0.5 Freeze the golden task fixture

One small multi-file repo with executable tests (Exp 0 Step 2).

**Done when:** baseline `pytest` fails for the intended reason, the known-good patch
makes it pass, and it touches at least two files. Record expected changed files and
symbols.

---

## 6. Design decisions to freeze before coding

Locking these prevents thrash during Exp 0.

| Decision | Default | Rationale |
|---|---|---|
| Topology | **Binding A**: orchestrator (Summarizer 2B) ↔ worker (Code expert 3B) | Only shape with released links that matches Deep Agents' main-agent/subagent structure (§2) |
| Both arms' weights | **Same local checkpoints in both arms** | "Same weights, different channel" is the whole claim. A Claude/Gemini text baseline measures model quality, not channel |
| Text arm | stock `SubAgent` + `task` tool, backed by the local `BaseChatModel` | Deep Agents default routing is the honest baseline; only the backing model is pinned |
| Latent arm | `CompiledSubAgent` (`subagents.py:167`) | Only documented way to supply an arbitrary runnable |
| **`ToolMessage` content in the latent arm** | **Minimal stub** (`"result delivered via latent channel"`), with a `Latent+Text` arm as a secondary condition | See below — this is the difference between measuring *latent vs text* and *latent+text vs text* |
| Latent transport | extra state key on the subagent's returned state; **not** the `ToolMessage` | `_return_command_with_state_update` (`subagents.py:484`) forwards any key not in `{messages, todos, structured_response}` and not agent-private, so a tensor rides through graph state. `ToolMessage` content is strings only |
| Injection point | `wrap_model_call` middleware reads the state key and binds the bundle to the model; the model does HF `generate(..., inputs_embeds=...)` | **Verified**: `ModelRequest.state` is a populated dict — built-in middleware (`summarization.py:1423`, `skills.py:912`) already reads arbitrary keys off it the same way. No fork of Deep Agents needed |
| Model lifetime | load once, reuse across handoffs | See §4 |
| Checkpointer | none / in-memory for Exp 0 | Tensors are not checkpoint-safe |
| `latent_steps` | 32 for Exp 0 | Sweep is Exp 2, not now |
| Reverse edge readiness | Load + smoke `outer_s2` and log `round_index` from Exp 0, even though Exp 0 itself is one-shot | Exp 4 (outer recursion) reuses this backend; bolting the direction on later means rewriting `latent_backend.py` and rerunning spikes |
| Recursion rounds | `R=1` (one-shot) for Exp 0–3; `R` sweep is Exp 4b/4c | Staged design: prove the single handoff is load-bearing before paying for loops |
| Snapshotting | `snapshot_repo` per repo | `resolve_mas_paths` pulls ~27 GB for `mixture` (§2) |

### Why the `ToolMessage` decision is load-bearing

Deep Agents always returns a string from `task`. If the latent arm returns its normal
natural-language result *and* a latent bundle, then Latent-DA is text-plus-latent and
Text-DA is text — so RQ1 measures whether adding a latent side channel helps, not whether
latent delegation can replace text. Both are publishable questions; they are not the same
question, and the current `PROPOSAL.md` abstract claims the second.

Freeze it as three named arms so the ambiguity cannot survive into the results:

| Arm | `ToolMessage` | Latent bundle |
|---|---|---|
| `Text-DA` | full natural-language handoff | none |
| `Latent-DA` | minimal stub | yes |
| `Latent+Text-DA` | full natural-language handoff | yes |

`Latent-DA` is the headline comparison against `Text-DA`. `Latent+Text-DA` is the
additive-value arm and is cheap once the other two run.

---

## 7. Gate 1 — Exp 0 implementation

Only start when Gate 0 is fully green.

```text
integrations/deepagents_latent/
├── __init__.py
├── graph.py            # one builder; only the delegation backend differs
├── text_runner.py      # stock SubAgent; log every task string
├── latent_backend.py   # persistent model + rollout + outer link
├── hf_chat_model.py    # BaseChatModel wrapping local HF + inputs_embeds
├── latent_middleware.py # wrap_model_call bridge: graph state -> model call
├── controls.py         # null channel, shuffled bundle, token-capped text
├── handoff_log.py      # Exp 0 schema
└── tasks/golden/       # the fixture from 0.5
```

Order of work, matching Exp 0 Steps 3–8:

1. Shared graph builder; dry-construct both arms and diff the configs.
2. Text-DA end to end on a toy task, with complete handoff logs.
3. Latent backend smoke: one prompt, assert shape/dtype/device/norm/`cos(in,out)`.
4. `hf_chat_model.py` + `latent_middleware.py` injection; prove the receiver actually
   consumes the bundle and does not silently fall back to text (Exp 0 Step 5b).
5. Both arms on the golden task from an identical reset fixture.
6. Schema-validate all logs.
7. Write the go/no-go run note.

`controls.py` exists from the start because the shuffled-bundle control in step 4 is the
only real evidence that the latent channel is load-bearing, and Exp 2 needs the null and
token-capped variants anyway. Building them later means rerunning Exp 1.

**Handoff log schema** is already specified in
[`exp0-harness-spike.md`](experiments/exp0-harness-spike.md) §Handoff Log Schema —
use it verbatim so Exp 3a/3b can join against it later.

---

## 8. Explicitly not now

- Exp 1 scaling sweep, Exp 2 capacity sweep
- Exp 3a probes, Exp 3b observability scoring (both require Exp 0 logs plus a pilot Exp 1)
- Exp 4 outer recursion (4a spike requires Exp 0 + pilot Exp 1; 4b/4c sweeps require 4a) —
  but Gate 1 **does** load both link directions and log `round_index` now, so Exp 4 is a
  reuse, not a rewrite
- Exp 5 full Mixture ensemble (~27 GB; robustness study, main-track at most)
- Probe label protocol and observability denominator (`PROPOSAL.md` §16 decisions)
- Fine-tuning RecursiveLink weights
- Adaptive recursion depth (Exp 4 uses fixed, pre-registered round counts)
- Prime Agent / DeepSeek Harness arms

---

## 9. Decisions — RESOLVED 2026-09-21

All six are closed. Experiment docs that contradict these are stale and should be
propagated to, not followed.

| # | Decision | Resolution |
|---|---|---|
| 1 | **Topology binding** (§2, Gate 0.2) | **Binding A** — mixture pair. Orchestrator = Summarizer `Qwen3.5-2B`, worker = Code expert `Qwen2.5-Coder-3B`, joined by `outer_2s` / `outer_s2`. Binding B is not being built. |
| 2 | **`ToolMessage` in the latent arm** (§6) | **Stub.** Three arms as specced: `Text-DA` / `Latent-DA` (stub, headline) / `Latent+Text-DA` (secondary). Headline claim stays "latent replaces text". |
| 3 | **Compute target** (Gate 0.3) | **RunPod RTX 4090**, $0.74/hr SECURE. Forced off local by §1.1. AMD credits applied for but lead time unknown, so the project did not wait; code is device-agnostic either way (§1.2). |
| 4 | **Week-1 evidence** | **Soften `PROPOSAL.md` §6** to "run on Colab, artifact not archived." Do not re-run the old probe; Gate 0.4 supersedes it by loading the receiver the Week-1 run never loaded. |
| 5 | **Golden task source** (Gate 0.5) | **Hand-written fixture.** Chosen so specific facts can be planted and checked for transit — Exp 3a asks *what* crosses the boundary, which a borrowed repo doesn't let you control. |
| 6 | **Disk** | **Free space regardless of #3.** Local room is still needed for venvs even with remote compute. ~7.6 GB sits in package caches (Homebrew 2.3G, pip 1.6G, ms-playwright 1.3G) plus ~5 GB in stale app updaters. |

Consequences worth carrying forward:

- **Binding A's links are task-agnostic, not code-trained.** Expect "your links were
  never trained for code" in review. Binding B is the answer if it lands, but as a
  post-hoc robustness check, not a second main line.
- **A genuine stub means `Latent-DA` will not degrade gracefully.** The orchestrator
  receives almost nothing in text, so a weak channel shows up as a crater, not a dip.
  That is the test working correctly. It also promotes the **token-capped text arm**
  from nice-to-have to the informative middle ground between the two extremes.

---

## 10. Status board

| Gate | Item | Status |
|---|---|---|
| 0.1 | Isolated env, imports pass | **GREEN 2026-09-23** — on pod `xr8mzs5tq7zjm6`, run note in `integrations/deepagents_latent/runs/` (§1.2) |
| 0.2 | Topology bound to a released link set | **Done** — Binding A (§9 #1) |
| 0.3 | Compute target chosen | **Done** — RunPod RTX 4090, provisioned and verified (§1.2). AMD credits still pending, not blocking |
| 0.4 | Smoke reproduced with receiver + artifact committed | **NEXT** — checkpoints already downloaded (§1.2). Needs the Colab notebook adapted; receiver has still never been loaded |
| 0.5 | Golden task frozen | **Done 2026-09-21** — `integrations/deepagents_latent/tasks/golden/`, 50 verifier checks green |
| — | Design decisions frozen (§6, §9) | **Signed off 2026-09-21** |
| — | Salvaged `instrumentation.py` + `tool_calling.py` | **Landed**, 23 tests passing (§11) |
| 1 | `integrations/deepagents_latent/` (incl. both link directions) | Blocked on Gate 0 |
| 2+ | Exp 1 / 2 / 3a / 3b | Blocked on Gate 1 |
| 3+ | Exp 4a loop spike → 4b/4c round sweeps | Blocked on Exp 0 + pilot Exp 1 |


---

## 11. Verified API facts for Gate 1 (from a scratch prototype)

A throwaway prototype of the Deep Agents side was built and discarded
(`~/projects/latentmas-deepagents/latent-delegation`, now marked scratch). Its
design was wrong for this repo — it assumed **same weights on both sides of the
handoff**, which no released outer link supports (§2) — but it hit real API traps
that `hf_chat_model.py`, `latent_middleware.py` and `latent_backend.py` will hit
too. Recorded here so Gate 1 doesn't rediscover them.

Two salvaged modules are already in `integrations/deepagents_latent/`:
`instrumentation.py` (§8 metrics; 10 tests) and `tool_calling.py` (the
`bind_tools` + Hermes parsing seed for `hf_chat_model.py`; 13 tests). Both pass.

### Traps that apply to code we write

| Trap | Detail |
|---|---|
| **`bind_tools` is mandatory** | `BaseChatModel.bind_tools` raises `NotImplementedError` in the base class, and `langchain/agents/factory.py:993` calls it unconditionally whenever tools are configured. `hf_chat_model.py` dies on its first model call without it. |
| **Tool calls arrive as text** | Qwen templates render a schema block via `apply_chat_template(..., tools=[...])` and emit Hermes-style `<tool_call>{"name":…,"arguments":{…}}</tool_call>`, which must be parsed back into LangChain `ToolCall` dicts. Verified against a real Qwen2.5-0.5B-Instruct tokenizer: template output matches `tool_calling.extract_tool_calls`. |
| **`PrivateStateAttr` silently drops keys on `.invoke()`** | It is `OmitFromSchema(input=True, output=True)`; omitting from the *input* schema means LangGraph's compiled input schema strips the key from whatever is passed to `.invoke()`, with no error. Use `OmitFromOutput` for anything the caller must be able to set. Directly relevant to §6's state-key transport if a bundle ever needs to travel **into** a subagent rather than out of one. |
| **`DynamicCache` is not subscriptable** | transformers 5.x returns a `Cache` object, so `past[0][0].shape[-2]` raises `TypeError`. Use `past.get_seq_length()` with a tuple fallback. |
| **`generate()` rejects explicit `cache_position`** | Passing it alongside a primed `past_key_values` raises `ValueError: model_kwargs not used by the model`. `generate()` derives it correctly from `past_key_values` + `attention_mask`. |
| **`Cache.crop()` positive values use legacy semantics** | A **positive** argument is the cache's desired *final absolute size*, not the number of tokens to remove — so `crop(5)` leaves 5 tokens. It still produces correct text (cropping from the end keeps correct leading tokens; `generate()` just recomputes more), so an output-equality test passes while the optimization silently does nothing. Use the negative convention and **assert on cache length**, not just output. |

### Traps that do *not* apply to this repo

Checked and clear, so don't spend Gate 0.1 time on them: `inference_utils/` and
`modeling.py` use neither legacy tuple-cache indexing (`past[0][0]`) nor explicit
`cache_position`, so the two transformers-5.x cache traps above are confined to new
code. `requirements.txt` pins `transformers==5.3.0`; 5.17.0 also works for the
non-RecursiveMAS pieces.

### Measurements that support §2.11 ("no training in v1")

Measured on Qwen2.5-0.5B-Instruct, raw last-layer hidden states fed back as
`inputs_embeds` with **no** link:

- **~666×** the mean input-embedding norm (0.453 vs 301.6). The final RMSNorm plateaus
  it rather than letting it diverge, so it is a large *constant* mismatch.
- Nearest-embedding cosine **0.195**, against **0.130** for a random direction and
  1.0 for a real token embedding — barely above the random floor. Over successive
  raw-feedback steps it decays to **0.096**, *below* random.

Rescaling to the mean embedding norm fixes magnitude but cannot fix direction (cosine
is scale-invariant). This is quantitative support for using the **released trained
links** rather than any hand-rolled projection: the distributional gap is exactly what
the RecursiveLink residual is trained to close.

Related: LatentMAS's least-squares realignment matrix
`M = (WᵒᵘᵗᵀWᵒᵘᵗ)⁻¹WᵒᵘᵗᵀWⁱⁿ` is **exactly identity** for tied-embedding models
(`max |M − I| = 0.0048` measured on Qwen2.5-0.5B). If a tied checkpoint is ever used
as a baseline, that baseline is a no-op by construction.

### Reproducibility warning for the cosine metrics (§3, §8)

The latent rollout is chaotically sensitive: a **~1e-6** perturbation in an injected
vector compounds to cosine **0.761 by step 3**. Latent trajectories are therefore not
bitwise reproducible across dtype, hardware or attention kernel. Pin dtype, record
seeds, and expect run-to-run variance beyond sampling noise when quoting
`cos(in, out)` or probe numbers.
