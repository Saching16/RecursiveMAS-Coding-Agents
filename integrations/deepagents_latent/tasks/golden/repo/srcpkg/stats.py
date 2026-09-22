"""Aggregations over transfer samples.

A sample is a byte count for one observation window.
"""

from collections.abc import Sequence

from .units import to_mebibytes


def total_bytes(samples: Sequence[int]) -> int:
    """Sum of all sample byte counts."""
    return sum(samples)


def peak_mib(samples: Sequence[int]) -> float:
    """Largest single sample, in mebibytes."""
    if not samples:
        return 0.0
    return to_mebibytes(max(samples))
