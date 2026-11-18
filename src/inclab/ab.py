"""Average treatment effects for a randomized test, done the careful way.

Includes Welch's t-test with a bootstrap interval, CUPED variance reduction
with a pre-period covariate, a rank test for heavy-tailed outcomes,
winsorization, and Benjamini-Hochberg correction across several comparisons.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def welch(y_t: np.ndarray, y_c: np.ndarray, alpha: float = 0.05) -> dict:
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    diff = y_t.mean() - y_c.mean()
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    t_stat, p = stats.ttest_ind(y_t, y_c, equal_var=False)
    z = stats.norm.ppf(1 - alpha / 2)
    base = y_c.mean()
    return {"control_mean": base, "treated_mean": y_t.mean(), "diff": diff, "se": se,
            "ci_low": diff - z * se, "ci_high": diff + z * se, "p_value": float(p),
            "rel_lift": diff / base if base else np.nan, "n_t": len(y_t), "n_c": len(y_c)}


def bootstrap_diff_ci(y_t, y_c, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap for the difference in means; useful when the outcome is skewed."""
    rng = np.random.default_rng(seed)
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(y_t, len(y_t)).mean() - rng.choice(y_c, len(y_c)).mean()
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def cuped(y: np.ndarray, x: np.ndarray, treated: np.ndarray, alpha: float = 0.05) -> dict:
    """CUPED: subtract the part of the outcome a pre-period covariate predicts.

    theta = cov(y, x) / var(x), estimated on the pooled sample. Because x is
    measured before assignment, it is independent of treatment, so the
    adjusted difference stays unbiased while its variance shrinks by roughly
    corr(y, x)^2.
    """
    y, x, treated = np.asarray(y, float), np.asarray(x, float), np.asarray(treated, bool)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    raw = welch(y[treated], y[~treated], alpha)
    adj = welch(y_adj[treated], y_adj[~treated], alpha)
    return {"theta": float(theta), "raw": raw, "adjusted": adj,
            "variance_reduction": 1 - (adj["se"] ** 2) / (raw["se"] ** 2)}


def mann_whitney(y_t, y_c) -> float:
    """Rank test: did the distribution shift? Answers a different question than the mean."""
    return float(stats.mannwhitneyu(y_t, y_c, alternative="two-sided").pvalue)


def winsorize(y: np.ndarray, upper_q: float = 0.99) -> np.ndarray:
    y = np.asarray(y, float)
    cap = np.quantile(y, upper_q)
    return np.minimum(y, cap)


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Which tests stay significant while controlling the false discovery rate."""
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresholds
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    keep = np.zeros(m, bool)
    keep[order[:k]] = True
    return keep.tolist()
