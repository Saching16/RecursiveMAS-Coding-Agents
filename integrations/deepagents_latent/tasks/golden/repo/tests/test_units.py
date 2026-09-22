"""Passes on the untouched fixture. Present so the baseline is not all-red
and the intended failure is attributable."""

import unittest

from srcpkg import units


class UnitTests(unittest.TestCase):
    def test_to_mebibytes(self):
        self.assertAlmostEqual(units.to_mebibytes(1048576), 1.0)

    def test_to_mebibits(self):
        self.assertAlmostEqual(units.to_mebibits(1048576), 8.0)
