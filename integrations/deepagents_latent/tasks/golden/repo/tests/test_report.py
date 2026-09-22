"""summarize() must surface throughput, using the same unit basis
test_stats.py pins."""

import unittest

from srcpkg import report


class SummarizeTests(unittest.TestCase):
    def test_summarize_includes_existing_metrics(self):
        out = report.summarize([1048576], 1.0)
        self.assertIn("total bytes: 1048576", out)
        self.assertIn("peak: 1.00 MiB", out)

    def test_summarize_includes_throughput(self):
        out = report.summarize([1048576, 1048576], 2.0)
        self.assertIn("throughput: 1.00 MiB/s", out)
