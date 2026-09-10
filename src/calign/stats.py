"""Small statistics helpers (kept dependency-free so reports are reproducible anywhere)."""

from __future__ import annotations

import math


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (95% by default)."""
    if n <= 0:
        return (0.0, 0.0)
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    # round away floating-point dust (e.g. 1e-17 at k=0) so bounds are clean and reproducible
    return (round(max(0.0, centre - half), 12), round(min(1.0, centre + half), 12))


def rate_summary(k: int, n: int) -> dict[str, float | int]:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else 0.0, "ci95_low": lo, "ci95_high": hi}
