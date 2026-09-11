"""Probe metrics in numpy: AUROC, balanced accuracy, Spearman, and a cluster (scenario) bootstrap for CIs."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based AUROC (ties get average ranks); None if one class is missing."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(int)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rank(scores) + 1  # average ranks, 1-based
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def balanced_accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float | None:
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(int)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    tpr = float((scores[pos] >= threshold).mean())
    tnr = float((scores[neg] < threshold).mean())
    return (tpr + tnr) / 2


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising balanced accuracy (Youden), taken between consecutive sorted scores."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(int)
    uniq = np.unique(scores)
    if len(uniq) < 2:
        return float(uniq[0]) if len(uniq) else 0.0
    cands = (uniq[:-1] + uniq[1:]) / 2
    best, best_t = -1.0, float(cands[0])
    for t in cands:
        b = balanced_accuracy(scores, labels, t)
        if b is not None and b > best:
            best, best_t = b, float(t)
    return best_t


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(v: np.ndarray) -> np.ndarray:
    """0-based average ranks (ties share the mean of their positions), vectorised."""
    _, inverse, counts = np.unique(np.asarray(v, dtype=np.float64), return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)  # exclusive end position of each tie group
    avg = ends - (counts + 1) / 2  # mean of positions start..end-1
    return avg[inverse].astype(np.float64)


def cluster_bootstrap(
    clusters: Sequence[str],
    stat: Callable[[np.ndarray], float | None],
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """95% percentile interval of `stat(index_array)` under resampling of clusters (e.g. scenarios) with replacement.

    `stat` receives the indices of the resampled rows (with repeats) and returns the statistic or None (skipped).
    """
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in picked])
        v = stat(idx)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def classification_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    clusters: Sequence[str],
    threshold: float,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """AUROC and balanced accuracy (at `threshold`) with cluster-bootstrap 95% CIs, plus counts."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(int)
    a = auroc(scores, labels)
    b = balanced_accuracy(scores, labels, threshold)
    a_lo, a_hi = (
        cluster_bootstrap(clusters, lambda i: auroc(scores[i], labels[i]), n_boot, seed)
        if a is not None
        else (None, None)
    )
    b_lo, b_hi = (
        cluster_bootstrap(clusters, lambda i: balanced_accuracy(scores[i], labels[i], threshold), n_boot, seed)
        if b is not None
        else (None, None)
    )
    return {
        "n": int(len(labels)),
        "n_pos": int(labels.sum()),
        "n_neg": int((1 - labels).sum()),
        "n_clusters": int(len(set(clusters))),
        "auroc": a,
        "auroc_ci95": [a_lo, a_hi],
        "balanced_accuracy": b,
        "balanced_accuracy_ci95": [b_lo, b_hi],
        "threshold": float(threshold),
        "mean_score_pos": float(scores[labels == 1].mean()) if labels.sum() else None,
        "mean_score_neg": float(scores[labels == 0].mean()) if (1 - labels).sum() else None,
    }
