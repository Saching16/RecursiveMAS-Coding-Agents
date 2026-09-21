"""Unit tests for the benchmark accounting. Pure stdlib — no torch, no
langchain, no weights — so the numbers the benchmark will report can be
checked before any GPU time is spent on producing them.

Run with: python3 -m unittest discover -s integrations -v
"""

from __future__ import annotations

import unittest

from deepagents_latent.instrumentation import CallRecord, RunMetrics, format_comparison, summarize


def _record(prompt=10, generated=5, latent=0, prefills=1, wall=1.0, kind="generate") -> CallRecord:
    return CallRecord(
        kind=kind,
        prompt_tokens=prompt,
        generated_tokens=generated,
        latent_steps=latent,
        forward_passes=prefills + latent + generated,
        wall_seconds=wall,
    )


class RunMetricsTests(unittest.TestCase):
    def test_empty_run_totals_are_zero(self):
        run = RunMetrics(arm="text")
        self.assertEqual(run.total_tokens, 0)
        self.assertEqual(run.forward_passes, 0)
        self.assertEqual(run.model_calls, 0)

    def test_totals_sum_across_calls(self):
        run = RunMetrics(arm="text")
        run.add(_record(prompt=10, generated=5))
        run.add(_record(prompt=20, generated=7))
        self.assertEqual(run.prompt_tokens, 30)
        self.assertEqual(run.generated_tokens, 12)
        self.assertEqual(run.total_tokens, 42)
        self.assertEqual(run.model_calls, 2)

    def test_latent_steps_counted_as_forward_passes_but_not_tokens(self):
        """The whole reason forward_passes exists: latent steps are real
        compute that produces zero tokens, so a token-only accounting would
        make them invisible."""
        run = RunMetrics(arm="latent")
        run.add(_record(prompt=10, generated=5, latent=4, kind="capture"))
        self.assertEqual(run.total_tokens, 15)
        self.assertEqual(run.latent_steps, 4)
        self.assertEqual(run.forward_passes, 1 + 4 + 5)

    def test_to_dict_roundtrips_call_detail(self):
        run = RunMetrics(arm="latent", task_id="t1")
        run.add(_record(kind="prime", latent=3))
        payload = run.to_dict()
        self.assertEqual(payload["arm"], "latent")
        self.assertEqual(payload["task_id"], "t1")
        self.assertEqual(len(payload["calls"]), 1)
        self.assertEqual(payload["calls"][0]["kind"], "prime")


class SummarizeTests(unittest.TestCase):
    def _run(self, arm, correct, tokens, wall=1.0, error=None, delegated=True):
        run = RunMetrics(arm=arm, correct=correct, wall_seconds=wall, error=error, delegated=delegated)
        run.add(_record(prompt=tokens, generated=0))
        return run

    def test_accuracy_and_means_per_arm(self):
        runs = [
            self._run("text", True, 100),
            self._run("text", False, 200),
            self._run("latent", True, 50),
            self._run("latent", True, 70),
        ]
        summary = summarize(runs)
        self.assertEqual(summary["text"]["accuracy"], 0.5)
        self.assertEqual(summary["latent"]["accuracy"], 1.0)
        self.assertEqual(summary["text"]["mean_total_tokens"], 150)
        self.assertEqual(summary["latent"]["mean_total_tokens"], 60)

    def test_errored_runs_excluded_from_accuracy_but_kept_in_cost(self):
        """An errored run still burned tokens and wall time. Dropping it
        from the cost means would flatter whichever arm fails more."""
        runs = [
            self._run("text", True, 100),
            self._run("text", None, 300, error="boom"),
        ]
        summary = summarize(runs)
        self.assertEqual(summary["text"]["n"], 2)
        self.assertEqual(summary["text"]["n_scored"], 1)
        self.assertEqual(summary["text"]["n_errors"], 1)
        self.assertEqual(summary["text"]["accuracy"], 1.0)
        self.assertEqual(summary["text"]["mean_total_tokens"], 200)

    def test_all_errored_arm_reports_none_accuracy_not_crash(self):
        summary = summarize([self._run("latent", None, 10, error="boom")])
        self.assertIsNone(summary["latent"]["accuracy"])

    def test_delegation_rate_tracks_whether_the_tool_was_actually_called(self):
        runs = [
            self._run("latent", True, 10, delegated=True),
            self._run("latent", True, 10, delegated=False),
        ]
        summary = summarize(runs)
        self.assertEqual(summary["latent"]["delegation_rate"], 0.5)


class FormatComparisonTests(unittest.TestCase):
    def test_renders_delta_against_baseline(self):
        runs = [
            RunMetrics(arm="text", correct=True, wall_seconds=2.0),
            RunMetrics(arm="latent", correct=True, wall_seconds=1.0),
        ]
        for run, tokens in zip(runs, (100, 50)):
            run.add(_record(prompt=tokens, generated=0))
        table = format_comparison(summarize(runs), baseline_arm="text")
        self.assertIn("total tokens", table)
        self.assertIn("-50.0%", table)

    def test_single_arm_does_not_crash(self):
        run = RunMetrics(arm="text", correct=True)
        run.add(_record())
        self.assertIn("total tokens", format_comparison(summarize([run]), baseline_arm="text"))


if __name__ == "__main__":
    unittest.main()
