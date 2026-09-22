"""Human-readable summaries of transfer samples."""

from collections.abc import Sequence

from .stats import peak_mib, total_bytes


def summarize(samples: Sequence[int], seconds: float) -> str:
    """One line per reported metric."""
    lines = [
        f"samples: {len(samples)}",
        f"total bytes: {total_bytes(samples)}",
        f"peak: {peak_mib(samples):.2f} MiB",
    ]
    return "\n".join(lines)
