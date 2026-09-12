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


def diff_rate_summary(k1: int, n1: int, k2: int, n2: int) -> dict[str, float | int | None]:
    """Difference of two independent proportions p1 - p2 with Newcombe's hybrid score interval (method 10).

    The interval combines the Wilson limits of both proportions: lower = d - sqrt((p1-l1)^2 + (u2-p2)^2),
    upper = d + sqrt((u1-p1)^2 + (p2-l2)^2). Deterministic and dependency-free.
    """
    if n1 <= 0 or n2 <= 0:
        return {"k1": k1, "n1": n1, "k2": k2, "n2": n2, "diff": None, "ci95_low": None, "ci95_high": None}
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1)
    l2, u2 = wilson_interval(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return {
        "k1": k1,
        "n1": n1,
        "k2": k2,
        "n2": n2,
        "diff": round(d, 12),
        "ci95_low": round(max(-1.0, lo), 12),
        "ci95_high": round(min(1.0, hi), 12),
    }


def diff_mean_summary(
    a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0
) -> dict[str, float | int | None]:
    """mean(a) - mean(b) for independent samples with a percentile-bootstrap 95% interval (deterministic)."""
    import random

    if not a or not b:
        return {"n_a": len(a), "n_b": len(b), "diff": None, "ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    d = sum(a) / len(a) - sum(b) / len(b)
    boots = sorted(
        sum(rng.choices(a, k=len(a))) / len(a) - sum(rng.choices(b, k=len(b))) / len(b) for _ in range(n_boot)
    )
    lo, hi = boots[int(0.025 * n_boot)], boots[min(n_boot - 1, int(0.975 * n_boot))]
    return {"n_a": len(a), "n_b": len(b), "diff": round(d, 12), "ci95_low": round(lo, 12), "ci95_high": round(hi, 12)}


def repetition_ratio(text: str, n: int = 4) -> float:
    """Share of repeated word n-grams (0 = no repetition); a cheap incoherence signal for generated text."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def replacement_char_count(text: str) -> int:
    """Number of U+FFFD replacement characters (garbled byte sequences in decoded model output)."""
    return text.count("�")
