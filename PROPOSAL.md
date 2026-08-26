# Research Proposal: Latent Delegation in Coding Agent Harnesses

> **Purpose:** Living plan for collaborators and Cursor. Maps claims to this repo +
> the planned [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) integration.
>
> **Engineering prerequisites:** see [`PLAN.md`](PLAN.md) for the verified environment
> state and the Gate 0 checklist that must pass before `integrations/deepagents_latent/`
> is written. As of 2026-08-25 the RecursiveMAS imports do **not** resolve locally.
>
> **Topology is constrained by the released links.** The role graph below is not free:
> every latent edge needs an outer link trained for that ordered pair of checkpoints, and
> the release ships a fixed set. See [`PLAN.md`](PLAN.md) §2 before changing any role name
> in this document.
>
> **Supersedes:** the adaptive-halting framing in the May 2026 draft of this file and
> `PROPOSAL_LONGFORM.md` (long-form text / AgentWrite / narrative domains).

---

## 1. One-Sentence Thesis

> *Production coding harnesses delegate work between agents via **text** (Task results,
> subagent messages, re-fed context). We integrate RecursiveMAS **latent inter-agent
> communication** into [Deep Agents](https://github.com/langchain-ai/deepagents) and
> measure whether latent delegation preserves **multi-step coding quality and efficiency**
> better than text delegation when the **same agent topology, tools, model weights, and
> handoff budget** are held fixed.*

This is an **analysis + systems integration** paper: the harness is Deep Agents; the
scientific question is whether the **communication channel** matters, not a new halting rule.

"Same weights" and "same budget" are load-bearing. A text baseline on a frontier model
measures model quality; an uncapped text baseline against a `latent_steps`-sized bundle
measures context budget. Both are easy mistakes to make and both would invalidate the
claim above (§2.3, §2.6, Exp 1).

---

## 2. Suggested Improvements (Agreed Direction)

These refinements make the project sharper and more defensible than the earlier drafts:

1. **Single controlled variable:** Same Deep Agents graph (roles, tools, sandbox). Only
   swap the **delegation edge**: text handoff vs RecursiveMAS latent handoff. Avoid
   confounding with a different scaffold (AgentWrite, custom DAG, etc.).

2. **Code-native scaling axis:** Evaluate on **task complexity** (files touched, test
   count, dependency depth), not word-count targets. Deep Agents already has filesystem,
   shell, and subagents — use that instead of synthetic long-form prose.

3. **Same weights, both arms:** Both arms run the **same local checkpoints**
   (`Mixture-Code-Qwen2.5-Coder-3B` worker + `Mixture-Summarizer-Qwen3.5-2B`
   orchestrator, joined by `Mixture-Outerlinks`). Using Deep Agents' frontier-model
   defaults for the text baseline would measure model quality, not channel. A frontier
   text run may be reported separately as an uncontrolled reference point.

4. **Week-1 gate is provisional:** the smoke test reported PASS on Colab but the artifact
   is not in the repo and the receiving model was never loaded (see §6). Re-running it
   with the receiver is `PLAN.md` Gate 0.4.

5. **Capacity as mechanism (RQ2):** Sweep `latent_steps` ∈ {0, 16, 32, 48}. Because the
   bundle is literally `latent_steps` embedding rows, this axis is directly comparable to
   a **token-capped text handoff** at the same budget — so Exp 2 plots one capacity curve
   per channel rather than a latent-only curve. `latent_steps=0` is the null channel.

6. **The latent arm must not also ship the text.** Deep Agents' `task` tool always
   returns a string. If the latent arm returns its normal natural-language result *and*
   a latent bundle, the comparison is *text+latent vs text* — a different claim from the
   one in §1. Headline `Latent-DA` therefore stubs the `ToolMessage`; `Latent+Text-DA`
   is a separate, secondary arm. See `PLAN.md` §6.

7. **A channel that carries nothing must be detectable.** Two controls, both cheap:
   **null** (`latent_steps=0`) and **shuffled** (inject a bundle computed for a different
   task). If task success is unchanged under shuffled bundles, the receiver is ignoring
   the channel and every RQ1/RQ2 number is about prompt scaffolding, not delegation.

8. **Cosine + probe diagnostics:** Log `cos(in, out)` on every outer link at runtime;
   required linear probing suite (Exp 3a) comparing latent bundle vs text summary at
   delegation boundaries (file graph, symbols, test deps, ambiguity state). Note the
   cosine is only well-defined when source and target embedding widths match
   (`PLAN.md` §3).

9. **Report compute honestly:** Tokens, wall-clock, and GPU seconds per successful task.
   Latent adds forward passes; winning on quality but losing on FLOPs is still a result.

10. **Defer adaptive recursion depth:** RecursiveMAS authors flagged this as future work;
    single-model ACT-style halting is a scoop/novelty risk. Not in abstract.

11. **No training in v1:** Released RecursiveMAS has inference only. v1 is zero-shot /
    released-link integration; fine-tuning links for Deep Agents tasks is main-track.

12. **Stage the recursion.** The one-shot handoff study (Exp 0–3) runs first and stands on
    its own; the outer recursive loop (Exp 4) builds on it. Binding A already ships both
    directions (`outer_2s` worker→orchestrator, `outer_s2` orchestrator→worker), so the
    closed loop needs no new checkpoints — but it is a separate claim with its own
    experiment, not something to fold silently into RQ1.

---

## 3. Research Questions

| ID | Question | Success signal |
|---|---|---|
| **RQ0** | Is the latent channel **load-bearing** at all in this harness? | Success drops when bundles are shuffled across tasks or zeroed (`latent_steps=0`). A null result here invalidates RQ1–RQ2 as channel claims |
| **RQ1** | On identical Deep Agents topology and identical weights, does **latent delegation** beat **text delegation** on multi-step coding tasks as complexity scales? | Higher pass@1 / resolve rate on harder tasks; gap widens with #files or test count |
| **RQ2** | How does **channel capacity** limit delegation quality, and does latent buy more per unit of budget than text? | Success rises with `latent_steps` ∈ {0,16,32,48}; latent curve sits above the token-capped text curve at matched budget |
| **RQ3** | Is the outer link **active** under real harness load (not collapsed)? | `cos(in,out)` ≪ 1 at handoffs; stable under longer prompts (smoke test extended to full tasks) |
| **RQ4** | What task-relevant state survives in the **latent bundle** that **text summarization drops** at delegation boundaries? | Latent linear probe beats text probe on ≥2 of 4 coding properties (Exp 3a); advantage grows with task complexity |
| **RQ5** | **How much** task-relevant state does a text handoff actually expose to a human debugger, and how much of the latent arm's state can probing recover? | Text observability score is measured and is **well below 1.0**; probe-recoverable fraction quantifies how much of the gap probing closes; tradeoff visible on observability–performance plot (Exp 3b) |
| **RQ6** | Does **repeated bidirectional latent feedback** (outer recursion, the mechanism that makes RecursiveMAS recursive) gain more from added rounds than matched text feedback? | Positive **channel × rounds interaction** in Exp 4b; at fixed total budget, distributing capacity across rounds helps latent more than text (Exp 4c) |

RQ5 is deliberately *not* phrased as "is latent less observable than text" — under the
scoring rules in Exp 3b that is 0 vs. something-positive by construction and cannot fail.
The empirical content is the two numbers: what fraction of state text actually exposes
(nobody has measured this for a coding harness), and what fraction probes recover.

RQ0–RQ5 study a **one-shot** handoff: one bundle, one boundary, one injection. That
exercises RecursiveMAS's inner rollout recursion and a single outer-link mapping, but not
the closed feedback loop (`expert → summarizer → feedback → expert`, iterated over
`num_recursive_rounds`) that is the paper's distinguishing mechanism —
`inference_mas_mixture.py:754-776` is the reference implementation. RQ6 / Exp 4 tests that
loop directly, staged **after** the one-shot study on purpose: if a single handoff is
inert (RQ0 fails), an expensive recursive sweep is uninterpretable.

---

## 4. What RecursiveMAS Provides (This Repo)

Verified against the **released inference codebase** (not the paper’s full training stack).

| Capability | Where it lives |
|---|---|
| Inner RecursiveLink (`ln_res_adapter`) | `modeling.py` → `Adapter` |
| Outer RecursiveLink (`outer_ln_res_adapter`) | `modeling.py` → `CrossModelAdapter` |
| Latent rollout + outer mapping | `inference_utils/inference_mas.py` → `autoregressive_latent_rollout`, `run_outer_adapter` |
| Slot injection into a receiver | `inference_mas.py` → `split_prompt_ids_by_slots`, `pad_left_embeds`, `generate(inputs_embeds=...)` |
| Link ↔ role-pair map | `system_loader.py:78-100` → `_OUTER_LAYOUTS` |
| Mixture code expert + summarizer + links | `load_from_repo.py` → `Mixture-Code-Qwen2.5-Coder-3B`, `Mixture-Summarizer-Qwen3.5-2B`, `Mixture-Outerlinks` |
| Fixed recursion rounds | `run.py` / `inference_utils/*` → `num_recursive_rounds` scalar loop |
| Smoke test notebook | `notebooks/latent_channel_smoke.ipynb` |

**Shape of a handoff:** `autoregressive_latent_rollout` returns
`[latent_steps, target_hidden]` after the outer link — `latent_steps` synthetic embedding
rows spliced into the receiver's prompt at a slot marker. This is why `latent_steps` is
directly comparable to a token budget on the text side (§7, Exp 2).

**Does not ship (v1 constraints):**

- Training / inner-outer loop trainer (README: unreleased)
- API-model latent injection (needs local HF forward pass with `inputs_embeds`)
- **Any link between two instances of the same role.** Links are per ordered checkpoint
  pair; there is no code→code link, so a graph of identical coder roles is not runnable
  (`PLAN.md` §2)
- Multi-source aggregation: the released stages inject exactly **one** bundle per prompt

---

## 5. What Deep Agents Provides

From [Deep Agents](https://github.com/langchain-ai/deepagents):

- **Sub-agents** with isolated context — natural match for RecursiveMAS roles
- **Filesystem, shell, tools, LangGraph** — real coding eval without building a harness from scratch
- **Python / extensible** — custom sub-agent backend can call this repo’s inference code
- **Model-agnostic** — latent arm requires a **local** backend compatible with released links

**Role binding (Binding A — see `PLAN.md` §2):**

| Deep Agents role | Checkpoint | Latent edge out | Link |
|---|---|---|---|
| Main agent (orchestrator) | `Mixture-Summarizer-Qwen3.5-2B` | → worker | `outer_s2` |
| Subagent (worker) | `Mixture-Code-Qwen2.5-Coder-3B` | → orchestrator | `outer_2s` |

This is the only released link set whose shape matches Deep Agents' native
main-agent/`task`-subagent structure. Multiple delegation boundaries come from multiple
`task` calls, not from adding roles.

Because both directions ship, the pair also supports a **closed refinement loop**
(worker → orchestrator via `outer_2s`, orchestrator → worker via `outer_s2`, iterated) —
the same shape as the release's own `num_recursive_rounds` loop with expert feedback.
Exp 0–3 use only the one-shot forward direction; Exp 4 exercises the loop.

**Integration pattern (planned):**

```text
Deep Agents graph (unchanged tools + sandbox)
        │
        ├─ Text arm:  subagent Task → ToolMessage text → main agent
        │
        └─ Latent arm: subagent Task → RecursiveMAS sidecar
                         (rollout + outer link) → bundle on graph state
                         → wrap_model_call middleware → inputs_embeds at receiver
                         (ToolMessage carries a stub, not the result)
```

Target code location: `integrations/deepagents_latent/` (to be added).

---

## 6. Week-1 Gate — PROVISIONAL PASS (not reproducible from this repo)

**Probe:** Do released RecursiveLink adapters collapse to identity on OOD coding prompts?

**Setup:** Colab A100, `Mixture-Code-Qwen2.5-Coder-3B`, `outer_2s`, `latent_steps=32`.

| Condition | Prompt tokens | cos(in, out) mean |
|---|---|---|
| Short code spec | 79 | **−0.0078** |
| Long code spec | 1,599 | **+0.0004** |
| Synthetic (adapter only) | 8–128 seq | **≈ 0** |

**Verdict: PASS, with caveats.** Mean cos ≈ **−0.004** — active transformation, not
collapse. Long prompts did not drift toward 1.0.

**Why provisional:**

- `notebooks/smoke_results.json` is **not in the repo**, and the notebook has no saved
  outputs — the numbers above cannot currently be reproduced from this tree.
- The run only loaded the **source** model. The summarizer that `outer_2s` targets was
  never instantiated, so the receiving half of the channel is unverified, and the
  cosine's well-definedness (equal embedding widths) was assumed rather than checked.

**Next gate:** `PLAN.md` Gate 0.4 re-runs this with the receiver loaded and commits the
artifact. Then Exp 0: text vs latent delegation on one **multi-file** Deep Agents task
(same graph, N ≥ 20 prompts). Proceed to sweeps only if implementation works end-to-end.

Artifacts: `notebooks/smoke_results.json` (Colab run, 2026-06-05) — **missing**.

---

## 7. Experimental Plan

### Exp 0 — Harness spike

- Install Deep Agents; build the Binding A graph (orchestrator ↔ code worker) with `task`
  delegation. Both arms share one graph builder.
- Implement `LatentRecursiveBackend` on the **primitives** (`load_agent_model_and_tokenizer`,
  `autoregressive_latent_rollout`, `run_inner_adapter`, `run_outer_adapter`), mirroring
  `run_hie_expert_latent_stage` rather than calling it — it loads and releases a model per
  call, which is wrong for an interactive harness (`PLAN.md` §4).
- One golden multi-file task; confirm both arms complete without crash.
- Prove the receiver consumes the bundle: shuffled bundles must change its output.
- **Future-proof for Exp 4:** load and smoke **both** link directions (`outer_2s` and
  `outer_s2`), keep models persistent, and log `round_index` from day one — so the outer
  recursion experiment reuses this backend instead of rewriting it.

### Exp 1 — Controlled channel ablation (headline)

Same topology, same tools, **same checkpoints in every arm**:

| Config | `ToolMessage` | Latent bundle | Role |
|---|---|---|---|
| **Text-DA** | full natural-language handoff | — | Baseline |
| **Text-DA-cap(k)** | truncated to `k` tokens | — | Budget-matched baseline, `k = latent_steps` |
| **Latent-DA** | stub | yes | Headline treatment |
| **Latent+Text-DA** | full natural-language handoff | yes | Additive-value arm |
| **Shuffled-DA** | stub | bundle from a different task | Control (RQ0) |
| *(reference)* Frontier-Text | full handoff | — | Uncontrolled context; different weights, reported separately |

**Tasks (code-scaled):**

- Start: MBPP+ / small repo edits (repo already has `mbppplus` path). MBPP+ ships
  single-function problems, so the multi-file tiers must be constructed.
- Scale by files touched: **1–2 → 3–5 → 6–10** (small / medium / large), each with an
  executable oracle.

**Metrics:** test pass rate, import/symbol consistency, task success rate, tokens, wall-clock.

### Exp 2 — Capacity sweep (RQ2)

Fix task set at medium difficulty; sweep `latent_steps` ∈ {0, 16, 32, 48} and the matched
text budget `k` ∈ {0, 16, 32, 48}. Plot **both** curves on one success-vs-budget axis.

`latent_steps=0` is the null channel and is already supported by the released code (every
latent stage returns an empty `[0, out_dim]` tensor), so it costs nothing to include and
anchors the bottom of both curves.

### Exp 3 — Delegation boundary analysis (required; downstream of Exp 0 + Exp 1)

**Dependency:** Run Exp 3a/3b only after Exp 0 harness spike and at least a pilot Exp 1
run (handoff logging infrastructure must exist). Full sweeps can overlap Exp 2 once
logging is stable.

**Hypothesis (RQ4):** Text verbalization at handoff compresses agent state — dropping
uncertainty about file relevance, uncommitted partial hypotheses, and implicit dependency
structure. The latent bundle may preserve these properties in a linearly decodable form
even when they are absent from the text summary.

#### Exp 3a — Systematic delegation boundary probes (RQ4)

At **each delegation handoff**, extract:

- **Latent bundle** (post–outer-link vectors from Latent-DA)
- **Text summary** (the message Text-DA would pass to the next subagent)

Train **linear probes** (same label schema, matched train/val split per property) to recover:

| Property | Label source (coding context) |
|---|---|
| **P1 — File graph** | Which files are relevant; directed import/call edges among them |
| **P2 — Symbol surface** | Function/class names and signatures the planner has tracked |
| **P3 — Test dependencies** | Which tests assert which modules; test→file mapping |
| **P4 — Ambiguity state** | Whether the agent has committed to a symbol interpretation vs. multiple candidates remain |

**Success criterion:** Latent probe accuracy **beats text probe on ≥2 of 4** properties
(statistically significant on held-out handoffs; report per-property tables, not pooled average).

**Negative result (still publishable):** If text probes match or beat latent on ≥3 properties,
report as **“text summarization is sufficient for observable coding state at handoff”** —
narrows the latent advantage claim to RQ1 efficiency/performance only, not interpretability.
If latent wins on 0–1 properties, interpretability is **not** a contribution; keep RQ5
observability tradeoff as the analysis angle.

**Deliverables:** `probing/delegation_probes.py`, per-handoff feature extractors, probe
accuracy vs task complexity plots.

#### Exp 3b — Operational observability characterization (RQ5)

Compare what a **human debugger or monitoring system** can recover from each channel at
each handoff:

| Channel | Directly available without probing |
|---|---|
| **Text-DA** | Full handoff string — readable, loggable, diffable across steps, replayable in session logs |
| **Latent-DA** | Tensor shape, norm, outer-link `cos(in,out)`; **not** human-readable semantics without probes |

**Observability score (per handoff):** Fraction of **task-relevant state variables**
(union of P1–P4 label fields for that handoff) that are **human-readable without probing**
— i.e., explicitly present in the text handoff or trivially derivable by string/regex over
logged text (pre-registered coding scheme in appendix).

Also report:

- **Probe-recoverable fraction:** proportion of the same state variables recoverable via
  Exp 3a probes above chance (latent arm only; text arm upper-bounded by text probe accuracy)
- **Cosine trajectory variance** across handoffs (RQ3 extended): rising mean `cos(in,out)`
  or dropping variance → early collapse warning

Measure observability score and recoverable fraction **across both arms** and **across
task complexity levels** (same task tiers as Exp 1).

**Target figure (not yet a result):** **Observability–Performance tradeoff** — 2-axis plot
with **x = observability score** (aggregate per configuration) and **y = task success rate**;
one point per configuration: **Text-DA** (uncapped), **Text-DA-cap** at 16 / 32 / 48
tokens, and **Latent-DA** at each `latent_steps` ∈ {0, 16, 32, 48}. The capped text points
are what make this a tradeoff curve rather than two isolated clusters — they sweep
observability continuously between the latent floor and full text. Expect Text-DA
upper-right on observability; the question is whether the latent series sits above the
text series at matched observability.

### Exp 4 — Outer recursive delegation (RQ6; downstream of Exp 0 + pilot Exp 1)

Exp 0–3 test a one-shot handoff. Exp 4 tests the loop that gives RecursiveMAS its name:
repeated bidirectional feedback across delegation rounds, mirroring the release's own
round structure (`inference_mas_mixture.py:679-776`) inside the Deep Agents graph.

**Loop shape (both arms, identical):** for round *r* of *R*: worker drafts (conditioned on
round *r−1* feedback) → orchestrator receives → if *r < R*, orchestrator produces feedback
for round *r+1*; the final round's output is the deliverable. Only the feedback medium
differs: a latent bundle via `outer_s2` spliced into a worker-side feedback slot
(mirroring `HIE_FEEDBACK_SLOT`), or a natural-language critique.

| Sub-exp | Question | Design |
|---|---|---|
| **4a — Loop spike** | Does the reverse edge work end to end? | One golden task, `R=2`; verify `outer_s2` injection and that round-2 worker output responds to round-1 feedback (shuffled-feedback control) |
| **4b — Round sweep** | Does latent gain more from rounds than text? | `R` ∈ {1, 2, 3}, fixed `latent_steps=32`, both channels; `R=1` must reproduce Exp 1's one-shot numbers |
| **4c — Budget-controlled recursion** | Is it recursion or just more compute? | Fixed total budget 48: 1×48 vs 2×24 vs 3×16, both channels |

**Primary statistic:** the **channel × rounds interaction**, not the main effect of
rounds. "More rounds help" is unsurprising; the claim under test is that latent feedback
compounds across rounds better than text feedback. 4c separates that from a raw compute
effect: if 1×48 matches 3×16 for the latent arm, recursion adds nothing beyond budget.

**Scope caveat:** a code↔summarizer loop exercises outer recursive feedback, but not the
full released Mixture topology (three experts aggregated by the summarizer, ~27 GB of
checkpoints). The full ensemble is a possible **Exp 5 / robustness study**, out of scope
for v1 (§11).

---

## 8. Metrics

| Metric | Role |
|---|---|
| Task success / test pass rate | Primary (RQ1) |
| Δ success under shuffled / null bundles | **Channel is load-bearing (RQ0) — read this before RQ1** |
| Success vs #files / test count | Scaling curve |
| Tokens + wall-clock + GPU-seconds | Efficiency (honest) |
| Success vs channel budget, latent and text on one axis | RQ2 |
| **Channel × rounds interaction**; success vs `R` per channel | Exp 4b / RQ6 |
| Success at fixed total budget across round splits (1×48 / 2×24 / 3×16) | Exp 4c — recursion vs compute |
| `cos(in, out)` per link **direction** (`outer_2s` vs `outer_s2`) | Exp 4 channel health; feedback edge may collapse independently |
| Probe accuracy (latent vs text) **per property type** (P1–P4) | Exp 3a / RQ4 |
| Observability score per handoff | Exp 3b / RQ5 |
| Fraction of task-relevant state recoverable **without probing** | Exp 3b / RQ5 |
| Probe-recoverable fraction (latent arm; text upper-bounded by text probes) | Exp 3b — partial observability compensation |
| `cos(in, out)` per handoff; **variance across handoffs** | Collapse monitor (RQ3); channel-health trajectory |

---

## 9. Pre-emptive Reviews

| Objection | Response |
|---|---|
| “Just RecursiveMAS + another harness” | Contribution is **controlled comparison** of communication channel inside a standard coding agent stack — not re-claiming RecursiveMAS training. |
| “Latent only works with tiny local models” | Acknowledged limitation; report same-model text vs latent before any API-model claims. |
| “Orthogonal cos ≠ useful” | Pair geometry (§6) with task success (Exp 1) and probes (Exp 3). |
| “Deep Agents already summarizes context” | That is exactly the **text baseline** we beat or lose to fairly. |
| “Your latent arm just has a bigger context budget” | `Text-DA-cap(k)` matches the text handoff to the same `k`-token budget the latent bundle occupies (Exp 1, Exp 2). |
| “The receiver probably ignores the injected embeddings” | `Shuffled-DA` and `latent_steps=0` are pre-registered controls (RQ0). If they match `Latent-DA`, we report that the channel is inert rather than reporting RQ1. |
| “Your topology isn’t the one you claim” | The role graph is bound to the released link set (`PLAN.md` §2) and the binding is stated in the method section; we do not invent links. |
| “The observability loss is obvious and not a contribution” | Prior work **asserts** latent efficiency; we **measure** the observability cost on a fixed harness with a pre-registered score and show where probing closes the gap (Exp 3b + tradeoff figure). The contribution is empirical quantification, not the claim that latents are opaque. |
| “Linear probes don’t prove the latent channel is causally responsible for task success” | We do **not** claim probes imply causation. RQ1 (Exp 1) addresses task outcomes; RQ4 probes test **what information is encoded** at handoffs. Causation would require interventional ablations (out of v1 scope; see Limitations). |

---

## 10. Limitations (Honest)

- **Probing shows correlation, not causation.** Higher probe accuracy on latents does not prove
  that decoded properties *drive* downstream success. Exp 1 task metrics and Exp 3a probe
  metrics must be reported separately; do not infer mechanism from probe wins alone.
- **Observability characterization is harness-specific.** The observability score is defined
  for Deep Agents handoff logs (text messages vs latent tensors). Generalization to other
  coding harnesses is **unverified** until replicated.
- **Released checkpoints only.** All probe and observability results use shipped RecursiveLink
  weights without fine-tuning. Results may change if links are trained on harness trajectories
  (main-track scope).
- **The links were not trained for this.** Under Binding A the mixture links are
  task-agnostic and were trained for a math/code/science expert ensemble feeding a
  summarizer — not for agentic coding handoffs carrying file graphs and test state. A
  weak latent arm is therefore consistent with "links are out of distribution" as well as
  with "latent delegation does not help"; Exp 2 and Exp 3a are what separate these.
- **Topology is not free.** Roles are chosen from the released link set rather than from
  what a coding harness would ideally use. Conclusions are about *this* pair of models on
  *this* graph.
- **Small models.** A 2B orchestrator and 3B worker may fail multi-file tasks for reasons
  unrelated to the channel. Floor and ceiling effects are a real threat to RQ1; the
  budget-matched and null arms are what keep the comparison interpretable when absolute
  success rates are low.

---

## 11. Out of Scope (v1)

- OpenCode or other TS harness forks
- AgentWrite / DeepWriter / long-form narrative generation
- Adaptive halting / ACT / per-node recursion depth — Exp 4 uses **fixed, pre-registered**
  round counts, never an adaptive stopping rule
- Full Mixture ensemble (math + code + science experts + summarizer, ~27 GB) — possible
  **Exp 5 / robustness study**, main-track
- Training RecursiveLink on new domains
- API-only frontier models for the **latent** arm (no hidden-state access)
- SWE-bench Verified at full scale (stretch → main track)

---

## 12. Risk Register

| Risk | Mitigation |
|---|---|
| **No released link for the intended topology** | Bind roles to `_OUTER_LAYOUTS` before writing code (`PLAN.md` Gate 0.2); Binding B (sequential trio) is the fallback |
| **Latent arm silently also passes full text** | Freeze `ToolMessage` content as a named arm; assert stub content in the latent runner |
| **Receiver ignores the bundle** | Shuffled + null controls in `controls.py` from Exp 0, not bolted on later |
| Deep Agents ↔ local HF embed injection is awkward | Bundle rides on graph state; `wrap_model_call` bridges to the model; spike Exp 0 early |
| Latent arm loses to text on easy tasks | Report full curve; RQ2 capacity may still publish |
| Link OOD on harness tasks | Cosine logging + probe; smoke test passed at 1.6k tokens on the source side only |
| **Disk / download footprint** | Snapshot only the two role repos + links (~11 GB); never call `resolve_mas_paths("mixture")` (~27 GB) |
| No training pipeline | Scope to released links; honest limitation section |
| Integration scope creep | One graph, one channel variable, one eval harness |
| **Probes trivially solved by text (latent offers no RQ4 advantage)** | Pre-register P1–P4; report negative as narrowing claim; lean on RQ5 tradeoff + RQ1 performance |
| **Observability score too subjective / irreproducible** | Pre-register label fields and “human-readable without probing” rules in appendix; automate derivable text fields; pilot inter-rater on 10 handoffs |
| **Exp 3a/3b delay Exp 1/2** | Hard gate: no Exp 3 full sweep until Exp 0 + pilot Exp 1 logging land; 3a/3b reuse same logged handoffs |

---

## 13. Paper Structure (Draft)

| Section | Content |
|---|---|
| Intro | Coding agents delegate via text; we test latent delegation in Deep Agents |
| Background | RecursiveMAS links; Deep Agents subagents |
| Method | Controlled topology; latent sidecar; text baseline; probe + observability protocols |
| Results | Exp 0 channel-liveness controls (RQ0); Exp 1 ablation + scaling; Exp 2 joint capacity curves; Exp 4 round sweep + channel × rounds interaction (RQ6) |
| Analysis | Exp 3a probes (RQ4); Exp 3b observability (RQ5); Exp 4c recursion-vs-compute decomposition; efficiency |
| **Target figure** | **Observability–Performance tradeoff:** x = observability score, y = task success rate; two series — Text-DA capped at 16/32/48 tokens plus uncapped, and Latent-DA at `latent_steps` {0, 16, 32, 48} *(design target — not yet measured)* |
| Limitations | §10: causation vs correlation; harness-specific observability; released checkpoints only |

**Target venue:** NeurIPS 2026 workshop / Algoverse Summer 2026 (tentative).

---

## 14. Repo Map (Correct Paths)

| What | Where |
|---|---|
| Latent rollout, inner/outer link application, slot injection | `inference_utils/inference_mas.py:813-886`, `:1041-1158`, `:1404-1522` |
| Reference latent stage to mirror (do **not** call per handoff) | `inference_utils/inference_mas_mixture.py:233-349` |
| Reference **round loop + feedback edge** for Exp 4 | `inference_mas_mixture.py:679-776` (loop), `:453-580` (`run_hie_summarizer_feedback_latent_stage`), `prompts.py` → `HIE_FEEDBACK_SLOT` |
| Link ↔ role-pair map and target-width assertion | `system_loader.py:78-100`, `:213-217` |
| Checkpoint ids per style | `load_from_repo.py` → `STYLE_SPECS` |
| Deep Agents subagent return path (what can carry a tensor) | `deepagents/middleware/subagents.py:167-244`, `:474-512` |

```bash
# Fixed-depth recursion (background — not our headline)
grep -rn "num_recursive_rounds" --include="*.py" inference_utils/ run.py

# No adaptive halting in inference path
grep -rn "halt\|early_exit\|adaptive_depth\|ponder" --include="*.py" .
```

`PLAN.md` §4 has the authoritative primitive map with line numbers.

---

## 15. Implementation Checklist

- [~] Week-1 latent channel smoke test (`notebooks/latent_channel_smoke.ipynb`) — ran on
      Colab, artifact missing, receiver never loaded; redone as `PLAN.md` Gate 0.4
- [ ] **Gate 0** (`PLAN.md` §5): populated venv, topology binding, compute target,
      smoke + artifact, golden task fixture
- [ ] `integrations/deepagents_latent/` — sidecar + Deep Agents sub-agent wrapper
- [ ] Text baseline graph on the **same local checkpoints**
- [ ] `controls.py` — null channel, shuffled bundle, token-capped text
- [ ] Shared eval: multi-file task suite + metrics script
- [ ] **Exp 0** harness spike + handoff logging (latent tensor + text message per boundary)
- [ ] **Exp 0** receiver-consumption test (shuffled bundle changes output) — **RQ0**
- [ ] **Exp 1** scaling sweep (RQ1) incl. budget-matched text arm
- [ ] **Exp 2** joint capacity sweep: `latent_steps` and text budget on one axis (RQ2)
- [ ] **Exp 3a** (after Exp 0 + pilot Exp 1): P1–P4 label extraction + linear probes + per-property tables
- [ ] **Exp 3b** (after Exp 0 + pilot Exp 1): observability score protocol + cross-complexity aggregates
- [ ] **Exp 4a** (after Exp 0 + pilot Exp 1): reverse-edge (`outer_s2`) + feedback-slot loop spike, shuffled-feedback control
- [ ] **Exp 4b** round sweep `R` ∈ {1, 2, 3}, both channels; `R=1` reproduces Exp 1
- [ ] **Exp 4c** fixed-budget round splits (1×48 / 2×24 / 3×16), both channels
- [ ] Paper figures: success vs complexity; capacity curve; **success vs rounds per channel (RQ6)**; **observability–performance tradeoff (target)**

---

## 16. Human Decisions / Dependencies (review before sprint)

> **⚠️ Decision needed (blocks everything):** **Topology binding** — Binding A (mixture
> pair: summarizer orchestrator ↔ code worker) or Binding B (sequential trio, links
> trained on `task="code"` but a math-specialised solver). See `PLAN.md` §2. Every role
> name, checkpoint download, log field, and handoff-matching rule downstream depends on
> this. Recommendation: **A**.

> **⚠️ Decision needed:** **What the latent arm's `ToolMessage` contains** — a stub
> (headline claim: latent *replaces* text) or the full handoff (claim: latent *augments*
> text). Both are defensible; only one matches §1 as currently written.

> **⚠️ Decision needed:** Exp 3a requires **ground-truth labels** for P1–P4 at each handoff
> (file graph, symbols, test deps, ambiguity). Options: (a) oracle from sandbox filesystem +
> test manifest, (b) post-hoc annotation, (c) synthetic multi-file tasks with known structure.
> Choice affects reproducibility and labor cost — pick one before building `probing/`.

> **⚠️ Decision needed:** Observability score denominator (“task-relevant state variables”) must
> be frozen in a protocol doc before Exp 3b pilot; otherwise cross-run comparisons are invalid.

> **No conflict with Exp 1/2:** Exp 3 reuses handoff logs from the same runs; it extends rather
> than replaces RQ1/RQ2. **Scheduling conflict:** if 3a/3b run before handoff logging exists,
> work is wasted — checklist order is intentional.

---

*Last updated: 2026-08-26 · Primary integration: [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)*
