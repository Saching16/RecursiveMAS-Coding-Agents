# PLAN.md — Pre-Implementation Plan for Deep Agents Integration

> **Scope:** What must be true *before* writing `integrations/deepagents_latent/`.
> **Relationship to other docs:**
> - [`PROPOSAL.md`](PROPOSAL.md) — the research plan (RQs, experiments, paper).
> - [`experiments/exp0-harness-spike.md`](experiments/exp0-harness-spike.md) — the implementation gate, expanded.
> - `experiments/exp1..exp3b` — the **study**, not prerequisites. Do not start them first.
>
> This file is the engineering checklist. It exists because the experiment docs
> describe *what to measure*, not *what has to work first*.

**Last verified:** 2026-08-24 · macOS 26.4 arm64 · RecursiveMAS `4e2934d`

---

## 1. Verified environment snapshot

Everything below was checked on this machine, not assumed.

| Item | Verified value | Status |
|---|---|---|
| Python | 3.13.2 (`/Library/Frameworks/Python.framework`) | OK — Deep Agents needs `>=3.11,<4.0` |
| Platform | macOS 26.4, arm64 | OK |
| torch | **2.10.0** installed; `requirements.txt` pins **2.9.0** | Mismatch |
| CUDA | **Not available** | Constraint |
| MPS | Available | Only local accelerator |
| numpy | 2.2.3 installed; pinned 2.2.6 | Minor drift |
| transformers | **Missing** (pinned 5.3.0) | **Blocker** |
| datasets | **Missing** | **Blocker** |
| huggingface_hub | **Missing** | **Blocker** |
| accelerate | **Missing** | **Blocker** |
| safetensors | **Missing** | **Blocker** |
| langchain | **Missing** | **Blocker** |
| deepagents | **Missing** | **Blocker** |
| conda / venv | **None** — bare system Python | **Blocker** |
| pip | 25.3 | OK |
| Free disk | ~15 GiB of 228 GiB (93% used) | **Tight** |

### Import checks (actual results)

```text
import modeling
  → ModuleNotFoundError: No module named 'huggingface_hub'   (modeling.py:7)

from inference_utils.inference_mas import autoregressive_latent_rollout
  → ModuleNotFoundError: No module named 'datasets'          (inference_mas.py:29)
```

**Exp 0 Step 1 currently fails.** Nothing latent can be built until this is fixed.

### Repositories

| Repo | Commit | Notes |
|---|---|---|
| RecursiveMAS | `4e2934d` on `main` | `origin` = `Saching16/RecursiveMAS-Coding-Agents` (fork), `upstream` = `RecursiveMAS/RecursiveMAS` |
| Deep Agents | `b650b41`, lib version **0.7.8** | Clone at `../RecursiveMAS-HarnessCompare/vendors/deepagents`, symlinked to `../deepagents` |

Deep Agents runtime deps: `langchain>=1.3.16`, `langchain-core>=1.6.0`,
`langchain-anthropic`, `langchain-google-genai`, `langsmith`, `packaging`, `wcmatch`.

### Checkpoints

Both required repos resolve (HTTP 200, metadata only — not downloaded):

- `RecursiveMAS/Mixture-Code-Qwen2.5-Coder-3B`
- `RecursiveMAS/Mixture-Outerlinks`

A 3B model in bf16 is roughly 6 GB. With ~15 GiB free, one model plus outer links
fits; two concurrent role models likely do not.

---

## 2. Documentation gaps found during verification

These are claims in the docs that this tree does not currently support.

| Claim | Reality | Action |
|---|---|---|
| `PROPOSAL.md` §6 "Week-1 Gate — COMPLETE", artifact `notebooks/smoke_results.json` | **File does not exist** in the repo | Re-run smoke and commit the JSON, or downgrade the claim to "run on Colab, artifact not archived" |
| `notebooks/latent_channel_smoke.ipynb` | 10 cells, **zero saved outputs** | Commit an executed copy so the cosine numbers are reproducible from the repo |
| Notebook links `../PROPOSAL_LONGFORM.md` | **File does not exist** (superseded) | Fix the link |
| Notebook targets Colab (`google.colab`, `cuda`, A100) | No CUDA locally | Decide local-MPS vs remote GPU before Exp 0 Step 1 |
| `integrations/` | Does not exist | Created in Gate 1 |

There is also a placeholder at
`../RecursiveMAS-HarnessCompare/integrations/deepagents_latent/__init__.py`.
That is scratch. The real package belongs in **this** repo at
`integrations/deepagents_latent/`, matching `PROPOSAL.md` §5 and Exp 0.

---

## 3. Key code finding that changes the design

`PROPOSAL.md` and Exp 0 both say to call `run_hie_expert_latent_stage`. Verified
signature and lifecycle:

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
not the function to call. Record this deviation from Exp 0 Step 5 in the run note.

---

## 4. Gate 0 — must be green before any integration code

Nothing here is Exp 1/2/3. Four tasks.

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

both print `ok`, and the exact versions are written into a run note.

### 0.2 Decide the compute target

CUDA is unavailable here. Pick one and write it down:

| Option | Cost | Note |
|---|---|---|
| Local MPS | Free | Untested for these checkpoints; bf16 on MPS is a known rough edge |
| Colab / remote A100 | Cheap | Matches the Week-1 smoke setup |
| Other remote GPU | Varies | Needed eventually for Exp 1 sweeps |

The A/B only has to be *internally consistent*; both arms must run on the same device.

### 0.3 Reproduce the latent channel smoke locally

Re-run the Week-1 probe on the chosen device with
`Mixture-Code-Qwen2.5-Coder-3B` + `Mixture-Outerlinks`, `latent_steps=32`.

**Done when:** `notebooks/smoke_results.json` exists, is committed, and
`cos(in, out)` is finite and far from 1.0 (Week-1 reported mean ≈ −0.004).

If this fails, **stop**. The harness cannot inject what the model stack cannot produce.

### 0.4 Freeze the golden task fixture

One small multi-file repo with executable tests (Exp 0 Step 2).

**Done when:** baseline `pytest` fails for the intended reason, the known-good patch
makes it pass, and it touches at least two files. Record expected changed files and
symbols.

---

## 5. Design decisions to freeze before coding

Locking these prevents thrash during Exp 0.

| Decision | Default | Rationale |
|---|---|---|
| Topology | plan → implement → review | Exp 0 scope; identical tools both arms |
| Text arm | stock `SubAgent` + `task` tool | Deep Agents default is the honest baseline |
| Latent arm | `CompiledSubAgent` (`subagents.py:167`) | Only documented way to supply an arbitrary runnable |
| Latent transport | sidecar store keyed by run/handoff; **not** the `ToolMessage` | `task` returns `structured_response` JSON or last `AIMessage` text — strings only |
| Injection point | custom `BaseChatModel` doing HF `forward(..., inputs_embeds=...)` | No native embeds path in Deep Agents |
| Model lifetime | load once, reuse across handoffs | See §3 |
| Checkpointer | none / in-memory for Exp 0 | Tensors are not checkpoint-safe |
| `latent_steps` | 32 for Exp 0 | Sweep is Exp 2, not now |

---

## 6. Gate 1 — Exp 0 implementation

Only start when Gate 0 is fully green.

```text
integrations/deepagents_latent/
├── __init__.py
├── graph.py            # one builder; only the delegation backend differs
├── text_runner.py      # stock SubAgent; log every task string
├── latent_backend.py   # persistent model + rollout + outer link
├── hf_chat_model.py    # BaseChatModel wrapping local HF + inputs_embeds
├── handoff_log.py      # Exp 0 schema
└── tasks/golden/       # the fixture from 0.4
```

Order of work, matching Exp 0 Steps 3–8:

1. Shared graph builder; dry-construct both arms and diff the configs.
2. Text-DA end to end on a toy task, with complete handoff logs.
3. Latent backend smoke: one prompt, assert shape/dtype/device/norm/`cos(in,out)`.
4. `hf_chat_model.py` injection; prove the receiver actually consumes the bundle and
   does not silently fall back to text.
5. Both arms on the golden task from an identical reset fixture.
6. Schema-validate all logs.
7. Write the go/no-go run note.

**Handoff log schema** is already specified in
[`exp0-harness-spike.md`](experiments/exp0-harness-spike.md) §Handoff Log Schema —
use it verbatim so Exp 3a/3b can join against it later.

---

## 7. Explicitly not now

- Exp 1 scaling sweep, Exp 2 capacity sweep
- Exp 3a probes, Exp 3b observability scoring (both require Exp 0 logs plus a pilot Exp 1)
- Probe label protocol and observability denominator (`PROPOSAL.md` §16 decisions)
- Fine-tuning RecursiveLink weights
- Adaptive recursion depth
- Prime Agent / DeepSeek Harness arms

---

## 8. Open decisions for you

1. **Compute target** for Exp 0 (§0.2) — local MPS or remote GPU.
2. **Week-1 evidence** — re-run and commit `smoke_results.json`, or soften the
   "COMPLETE" claim in `PROPOSAL.md` §6 to reflect that the artifact is not in-repo.
3. **Golden task source** — hand-written fixture, a trimmed real repo, or an MBPP+
   multi-file variant.
4. **Disk** — ~15 GiB free is enough for one 3B role model. If the topology needs
   distinct models per role, free space first.

---

## 9. Status board

| Gate | Item | Status |
|---|---|---|
| 0.1 | Isolated env, imports pass | Not started — **currently failing** |
| 0.2 | Compute target chosen | Not started |
| 0.3 | Smoke reproduced + artifact committed | Not started (claimed done, artifact missing) |
| 0.4 | Golden task frozen | Not started |
| — | Design decisions frozen (§5) | Drafted, needs sign-off |
| 1 | `integrations/deepagents_latent/` | Blocked on Gate 0 |
| 2+ | Exp 1 / 2 / 3a / 3b | Blocked on Gate 1 |
