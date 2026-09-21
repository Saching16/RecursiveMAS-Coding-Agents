"""Token / latency / compute accounting for the arm comparison.

Serves `PROPOSAL.md` §2.9 ("report compute honestly: tokens, wall-clock and
GPU seconds per successful task") and feeds the Exp 0 handoff log. Latent
adds forward passes, and winning on quality while losing on FLOPs is still
a result, so the accounting has to make that visible rather than reporting
tokens alone.

`arm` is a free-form string — use the frozen names from `PLAN.md` §6:
`Text-DA`, `Latent-DA`, `Latent+Text-DA`, plus the `controls.py` variants
(null channel, shuffled bundle, token-capped text).

Three numbers matter when comparing text delegation against latent
delegation, and they don't move together:

- **tokens**: what you pay an API for, and what LatentMAS reports. Split
  into prompt (encoded) and generated (decoded) because they cost
  differently and the latent path shifts weight between them.
- **wall seconds**: what a user feels. Not a pure function of tokens once
  latent rollout steps enter the picture.
- **forward passes**: the honest compute proxy. This is the one that makes
  latent steps visible at all -- a latent step is a full forward pass that
  produces zero tokens, so a token-only accounting would make the latent
  path look free when it isn't.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field


@dataclass
class CallRecord:
    """One model call (a decode turn, or a latent-priming turn)."""

    kind: str
    """`generate` (ordinary turn), `capture` (turn + latent rollout), or
    `prime` (a turn seeded from another agent's latent packet)."""

    prompt_tokens: int
    generated_tokens: int
    latent_steps: int
    forward_passes: int
    wall_seconds: float


@dataclass
class RunMetrics:
    """Everything recorded for one task under one arm."""

    arm: str
    task_id: str = ""
    calls: list[CallRecord] = field(default_factory=list)
    wall_seconds: float = 0.0
    correct: bool | None = None
    error: str | None = None
    final_answer: str = ""
    delegated: bool = False

    def add(self, record: CallRecord) -> None:
        self.calls.append(record)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def generated_tokens(self) -> int:
        return sum(c.generated_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens

    @property
    def forward_passes(self) -> int:
        return sum(c.forward_passes for c in self.calls)

    @property
    def latent_steps(self) -> int:
        return sum(c.latent_steps for c in self.calls)

    @property
    def model_calls(self) -> int:
        return len(self.calls)

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "task_id": self.task_id,
            "correct": self.correct,
            "error": self.error,
            "delegated": self.delegated,
            "final_answer": self.final_answer,
            "wall_seconds": round(self.wall_seconds, 3),
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "total_tokens": self.total_tokens,
            "forward_passes": self.forward_passes,
            "latent_steps": self.latent_steps,
            "model_calls": self.model_calls,
            "calls": [asdict(c) for c in self.calls],
        }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize(runs: list[RunMetrics]) -> dict:
    """Per-arm aggregates. Accuracy is computed over runs that actually
    produced a verdict, while token/latency means cover every run that
    completed -- an errored run still consumed real tokens and time, and
    dropping it would flatter whichever arm errors more."""
    by_arm: dict[str, list[RunMetrics]] = {}
    for run in runs:
        by_arm.setdefault(run.arm, []).append(run)

    summary: dict[str, dict] = {}
    for arm, arm_runs in by_arm.items():
        scored = [r for r in arm_runs if r.correct is not None]
        summary[arm] = {
            "n": len(arm_runs),
            "n_scored": len(scored),
            "n_errors": sum(1 for r in arm_runs if r.error),
            "accuracy": (sum(1 for r in scored if r.correct) / len(scored)) if scored else None,
            "delegation_rate": _mean([1.0 if r.delegated else 0.0 for r in arm_runs]),
            "mean_total_tokens": _mean([r.total_tokens for r in arm_runs]),
            "mean_prompt_tokens": _mean([r.prompt_tokens for r in arm_runs]),
            "mean_generated_tokens": _mean([r.generated_tokens for r in arm_runs]),
            "mean_forward_passes": _mean([r.forward_passes for r in arm_runs]),
            "mean_wall_seconds": _mean([r.wall_seconds for r in arm_runs]),
            "mean_model_calls": _mean([r.model_calls for r in arm_runs]),
        }
    return summary


def format_comparison(summary: dict, baseline_arm: str = "text") -> str:
    """Render the summary as a table, with deltas against the baseline arm
    so the headline claim (fewer tokens / less wall time at comparable
    accuracy) is readable without post-processing."""
    rows = [
        ("accuracy", "accuracy", "{:.1%}"),
        ("delegation rate", "delegation_rate", "{:.1%}"),
        ("total tokens", "mean_total_tokens", "{:.1f}"),
        ("  prompt tokens", "mean_prompt_tokens", "{:.1f}"),
        ("  generated tokens", "mean_generated_tokens", "{:.1f}"),
        ("forward passes", "mean_forward_passes", "{:.1f}"),
        ("wall seconds", "mean_wall_seconds", "{:.2f}"),
        ("model calls", "mean_model_calls", "{:.1f}"),
        ("errors", "n_errors", "{:.0f}"),
    ]
    arms = list(summary)
    width = 22
    lines = ["", "metric".ljust(width) + "".join(a.ljust(14) for a in arms) + "delta"]
    lines.append("-" * (width + 14 * len(arms) + 10))
    for label, key, fmt in rows:
        cells = ""
        for arm in arms:
            value = summary[arm].get(key)
            cells += ("n/a" if value is None else fmt.format(value)).ljust(14)
        delta = ""
        base = summary.get(baseline_arm, {}).get(key)
        others = [a for a in arms if a != baseline_arm]
        if base not in (None, 0) and others:
            other = summary[others[0]].get(key)
            if other is not None:
                delta = f"{(other - base) / base:+.1%}"
        lines.append(label.ljust(width) + cells + delta)
    return "\n".join(lines)


def write_results(path: str, runs: list[RunMetrics], summary: dict, config: dict) -> None:
    with open(path, "w") as handle:
        json.dump(
            {"config": config, "summary": summary, "runs": [r.to_dict() for r in runs]},
            handle,
            indent=2,
        )
