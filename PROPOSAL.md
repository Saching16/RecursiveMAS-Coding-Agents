# Research Proposal: Latent Delegation in Coding Agent Harnesses

> **Purpose:** Living plan for collaborators and Cursor. Maps claims to this repo +
> the planned [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) integration.
>
> **Supersedes:** the adaptive-halting framing in the May 2026 draft of this file and
> `PROPOSAL_LONGFORM.md` (long-form text / AgentWrite / narrative domains).

---

## 1. One-Sentence Thesis

> *Production coding harnesses delegate work between agents via **text** (Task results,
> subagent messages, re-fed context). We integrate RecursiveMAS **latent inter-agent
> communication** into [Deep Agents](https://github.com/langchain-ai/deepagents) and
> measure whether latent delegation preserves **multi-step coding quality and efficiency**
> better than text delegation when the **same agent topology and tools** are held fixed.*

This is an **analysis + systems integration** paper: the harness is Deep Agents; the
scientific question is whether the **communication channel** matters, not a new halting rule.

---

## 2. Suggested Improvements (Agreed Direction)

These refinements make the project sharper and more defensible than the earlier drafts:

1. **Single controlled variable:** Same Deep Agents graph (roles, tools, sandbox). Only
   swap the **delegation edge**: text handoff vs RecursiveMAS latent handoff. Avoid
   confounding with a different scaffold (AgentWrite, custom DAG, etc.).

2. **Code-native scaling axis:** Evaluate on **task complexity** (files touched, test
   count, dependency depth), not word-count targets. Deep Agents already has filesystem,
   shell, and subagents — use that instead of synthetic long-form prose.

3. **Local model + released links:** Latent arm uses RecursiveMAS checkpoints that match
   released links (e.g. `Mixture-Code-Qwen2.5-Coder-3B` + `Mixture-Outerlinks`). Text
   baseline can use the **same** local model with normal chat delegation, or Deep Agents
   defaults — but the cleanest comparison is **same weights, different channel**.

4. **Week-1 gate is done:** Smoke test passed (see §6). Next gate is **harness A/B**
   on a small multi-file task before scaling sweeps.

5. **Capacity as mechanism (RQ2):** Sweep `latent_steps` ∈ {16, 32, 48}. Relate
   breakdown in task success to bottleneck size — publishable even if latent does not win.

6. **Cosine + probe diagnostics:** Log `cos(in, out)` on every outer link at runtime;
   required linear probing suite (Exp 3a) comparing latent bundle vs text summary at
   delegation boundaries (file graph, symbols, test deps, ambiguity state).

7. **Report compute honestly:** Tokens, wall-clock, and GPU seconds per successful task.
   Latent adds forward passes; winning on quality but losing on FLOPs is still a result.

8. **Defer adaptive recursion depth:** RecursiveMAS authors flagged this as future work;
   single-model ACT-style halting is a scoop/novelty risk. Not in abstract.

9. **No training in v1:** Released RecursiveMAS has inference only. v1 is zero-shot /
   released-link integration; fine-tuning links for Deep Agents tasks is main-track.

---

## 3. Research Questions

| ID | Question | Success signal |
|---|---|---|
| **RQ1** | On identical Deep Agents topology, does **latent delegation** beat **text delegation** on multi-step coding tasks as complexity scales? | Higher pass@1 / resolve rate on harder tasks; gap widens with #files or test count |
| **RQ2** | How does **latent channel capacity** (`latent_steps`) limit delegation quality? | Performance cliff between 16 / 32 / 48; correlates with probe recovery |
| **RQ3** | Is the outer link **active** under real harness load (not collapsed)? | `cos(in,out)` ≪ 1 at handoffs; stable under longer prompts (smoke test extended to full tasks) |
| **RQ4** | What task-relevant state survives in the **latent bundle** that **text summarization drops** at delegation boundaries? | Latent linear probe beats text probe on ≥2 of 4 coding properties (Exp 3a); advantage grows with task complexity |
| **RQ5** | What **operational observability** is lost when moving from text to latent delegation, and does probing partially compensate? | Latent arm shows lower observability score than text; probe-recoverable fraction partially closes gap; tradeoff visible on observability–performance plot (Exp 3b) |

---

## 4. What RecursiveMAS Provides (This Repo)

Verified against the **released inference codebase** (not the paper’s full training stack).

| Capability | Where it lives |
|---|---|
| Inner RecursiveLink (`ln_res_adapter`) | `modeling.py` → `Adapter` |
| Outer RecursiveLink (`outer_ln_res_adapter`) | `modeling.py` → `CrossModelAdapter` |
| Latent rollout + outer mapping | `inference_utils/inference_mas.py` → `autoregressive_latent_rollout`, `run_outer_adapter` |
| Mixture code expert + links | `load_from_repo.py` → `Mixture-Code-Qwen2.5-Coder-3B`, `Mixture-Outerlinks` |
| Fixed recursion rounds | `run.py` / `inference_utils/*` → `num_recursive_rounds` scalar loop |
| Smoke test notebook | `notebooks/latent_channel_smoke.ipynb` |

**Does not ship (v1 constraints):**

- Training / inner-outer loop trainer (README: unreleased)
- API-model latent injection (needs local HF forward pass with `inputs_embeds`)

---

## 5. What Deep Agents Provides

From [Deep Agents](https://github.com/langchain-ai/deepagents):

- **Sub-agents** with isolated context — natural match for RecursiveMAS roles
- **Filesystem, shell, tools, LangGraph** — real coding eval without building a harness from scratch
- **Python / extensible** — custom sub-agent backend can call this repo’s inference code
- **Model-agnostic** — latent arm requires a **local** backend compatible with released links

**Integration pattern (planned):**

```text
Deep Agents graph (unchanged tools + sandbox)
        │
        ├─ Text arm:  subagent Task → message history (baseline)
        │
        └─ Latent arm: subagent Task → RecursiveMAS sidecar
                         (latent rollout + outer link → inject at next agent)
```

Target code location: `integrations/deepagents_latent/` (to be added).

---

## 6. Week-1 Gate — COMPLETE

**Probe:** Do released RecursiveLink adapters collapse to identity on OOD coding prompts?

**Setup:** Colab A100, `Mixture-Code-Qwen2.5-Coder-3B`, `outer_2s`, `latent_steps=32`.

| Condition | Prompt tokens | cos(in, out) mean |
|---|---|---|
| Short code spec | 79 | **−0.0078** |
| Long code spec | 1,599 | **+0.0004** |
| Synthetic (adapter only) | 8–128 seq | **≈ 0** |

**Verdict: PASS.** Mean cos ≈ **−0.004** — active transformation, not collapse. Long
prompts did not drift toward 1.0.

**Next gate (Week 2):** Text vs latent delegation on one **multi-file** Deep Agents task
(same graph, N ≥ 20 prompts). Proceed to sweeps only if implementation works end-to-end.

Artifacts: `notebooks/smoke_results.json` (Colab run, 2026-06-05).

---

## 7. Experimental Plan

### Exp 0 — Harness spike (Week 2)

- Install Deep Agents; define 3-role graph (plan → implement → review) with Task delegation.
- Implement `LatentRecursiveBackend` calling `run_hie_expert_latent_stage` (or sequential chain).
- One golden multi-file task; confirm both arms complete without crash.

### Exp 1 — Controlled channel ablation (headline)

Same topology, same tools, same local model family where possible:

| Config | Delegation channel | Implementation |
|---|---|---|
| **Text-DA** | Subagent returns natural-language state | Deep Agents default |
| **Latent-DA** | Subagent returns RecursiveMAS latent bundle | Sidecar + slot injection / embed path |

**Tasks (code-scaled):**

- Start: MBPP+ / small repo edits (repo already has `mbppplus` path)
- Scale: multi-file tasks (2 → 5 → 10 files) generated or curated with test harness

**Metrics:** test pass rate, import/symbol consistency, task success rate, tokens, wall-clock.

### Exp 2 — Capacity sweep (RQ2)

Fix task set at medium difficulty; sweep `latent_steps` ∈ {16, 32, 48}. Plot success vs capacity.

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
one point per configuration: **Text-DA**, **Latent-DA** at each `latent_steps` ∈ {16, 32, 48}.
Expect Text-DA upper-right on observability; Latent-DA points trace whether success gains
justify observability loss.

---

## 8. Metrics

| Metric | Role |
|---|---|
| Task success / test pass rate | Primary (RQ1) |
| Success vs #files / test count | Scaling curve |
| Tokens + wall-clock + GPU-seconds | Efficiency (honest) |
| `latent_steps` vs success | RQ2 |
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

---

## 11. Out of Scope (v1)

- OpenCode or other TS harness forks
- AgentWrite / DeepWriter / long-form narrative generation
- Adaptive halting / ACT / per-node recursion depth
- Training RecursiveLink on new domains
- API-only frontier models for the **latent** arm (no hidden-state access)
- SWE-bench Verified at full scale (stretch → main track)

---

## 12. Risk Register

| Risk | Mitigation |
|---|---|
| Deep Agents ↔ local HF embed injection is awkward | Sidecar returns structured state; spike Exp 0 early |
| Latent arm loses to text on easy tasks | Report full curve; RQ2 capacity may still publish |
| Link OOD on harness tasks | Cosine logging + probe; smoke test already passed at 1.6k tokens |
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
| Results | Exp 1 ablation + scaling; Exp 2 capacity |
| Analysis | Exp 3a probes (RQ4); Exp 3b observability (RQ5); efficiency |
| **Target figure** | **Observability–Performance tradeoff:** x = observability score, y = task success rate; points for Text-DA and Latent-DA at each `latent_steps` {16, 32, 48} *(design target — not yet measured)* |
| Limitations | §10: causation vs correlation; harness-specific observability; released checkpoints only |

**Target venue:** NeurIPS 2026 workshop / Algoverse Summer 2026 (tentative).

---

## 14. Repo Map (Correct Paths)

```bash
# Fixed-depth recursion (background — not our headline)
grep -rn "num_recursive_rounds" --include="*.py" inference_utils/ run.py

# No adaptive halting in inference path
grep -rn "halt\|early_exit\|adaptive_depth\|ponder" --include="*.py" .
```

---

## 15. Implementation Checklist

- [x] Week-1 latent channel smoke test (`notebooks/latent_channel_smoke.ipynb`)
- [ ] `integrations/deepagents_latent/` — sidecar + Deep Agents sub-agent wrapper
- [ ] Text baseline graph (Deep Agents only)
- [ ] Shared eval: multi-file task suite + metrics script
- [ ] **Exp 0** harness spike + handoff logging (latent tensor + text message per boundary)
- [ ] **Exp 1** scaling sweep (RQ1)
- [ ] **Exp 2** `latent_steps` sweep (RQ2)
- [ ] **Exp 3a** (after Exp 0 + pilot Exp 1): P1–P4 label extraction + linear probes + per-property tables
- [ ] **Exp 3b** (after Exp 0 + pilot Exp 1): observability score protocol + cross-complexity aggregates
- [ ] Paper figures: success vs complexity; capacity curve; **observability–performance tradeoff (target)**

---

## 16. Human Decisions / Dependencies (review before sprint)

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

*Last updated: June 2026 · Primary integration: [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)*
