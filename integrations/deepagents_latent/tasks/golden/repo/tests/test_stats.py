"""The unit basis for throughput is pinned HERE and nowhere else.

srcpkg/units.py offers both to_mebibytes and to_mebibits, and stats.py's
module docstring does not say which a throughput figure should use. This
test is the only place that resolves it: 2 MiB over 2 s is 1.0, so the
answer is mebibytes per second. An implementation that reaches for
to_mebibits gets 8.0 and fails here.
"""

import unittest

from srcpkg import stats


class TotalBytesTests(unittest.TestCase):
    def test_total_bytes(self):
        self.assertEqual(stats.total_bytes([100, 200, 300]), 600)

    def test_peak_mib(self):
        self.assertAlmostEqual(stats.peak_mib([1048576, 524288]), 1.0)


class ThroughputTests(unittest.TestCase):
    def test_throughput_mib_s(self):
        # 2 MiB transferred over 2 seconds -> 1.0 MiB/s
        self.assertAlmostEqual(stats.throughput_mib_s([1048576, 1048576], 2.0), 1.0)

    def test_throughput_zero_seconds_is_zero(self):
        self.assertEqual(stats.throughput_mib_s([1048576], 0.0), 0.0)
