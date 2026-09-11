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


def mean_summary(values: list[float], n_boot: int = 2000, seed: int = 0) -> dict[str, float | int | None]:
    """Mean with a percentile-bootstrap 95% interval (deterministic for a given seed)."""
    import random

    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = sum(values) / n
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[min(n_boot - 1, int(0.975 * n_boot))]
    return {"n": n, "mean": round(mean, 12), "ci95_low": round(lo, 12), "ci95_high": round(hi, 12)}


def rate_summary(k: int, n: int) -> dict[str, float | int]:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else 0.0, "ci95_low": lo, "ci95_high": hi}
