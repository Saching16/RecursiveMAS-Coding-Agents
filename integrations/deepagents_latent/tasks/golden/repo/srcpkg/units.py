"""Byte/bit unit conversions.

Two mebi-scale conversions live here on purpose. Callers must pick one;
nothing in this module says which is correct for a given metric.
"""

BITS_PER_BYTE = 8
BYTES_PER_MEBIBYTE = 1024 * 1024
BITS_PER_MEBIBIT = 1024 * 1024


def to_mebibytes(n_bytes: int) -> float:
    """Convert a byte count to mebibytes (MiB)."""
    return n_bytes / BYTES_PER_MEBIBYTE


def to_mebibits(n_bytes: int) -> float:
    """Convert a byte count to mebibits (Mib)."""
    return (n_bytes * BITS_PER_BYTE) / BITS_PER_MEBIBIT
