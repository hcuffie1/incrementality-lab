"""Geo holdout measurement when you cannot randomize individuals.

A common real-world setup: a campaign launches in some markets and not
others, and you need its effect on sales. This module simulates a market-week
panel with a known lift, then estimates it with difference-in-differences on
matched control markets, checks pre-trends with an event study, and runs a
placebo (permutation) test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def simulate_geo_panel(n_markets: int = 50, n_treated: int = 12, pre_weeks: int = 12,
                       post_weeks: int = 8, lift: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Weekly sales by market. Treated markets get a `lift` (in log points) after launch."""
    rng = np.random.default_rng(seed)
    weeks = np.arange(pre_weeks + post_weeks)
    season = 0.08 * np.sin(2 * np.pi * weeks / 26)
    rows = []
    treated_ids = set(rng.choice(n_markets, n_treated, replace=False))
    for m in range(n_markets):
        level = rng.normal(10, 0.5)
        trend = rng.normal(0.002, 0.001)
        sens = rng.normal(1.0, 0.25)
        for w in weeks:
            post = w >= pre_weeks
            treated = m in treated_ids
            log_sales = level + trend * w + sens * season[w] + rng.normal(0, 0.03)
            if treated and post:
                log_sales += lift
            rows.append({"market": m, "week": int(w), "treated": treated, "post": bool(post),
                         "log_sales": log_sales})
    return pd.DataFrame(rows)


def match_controls(panel: pd.DataFrame, k: int = 3, method: str = "rmse") -> list[int]:
    """For each treated market, pick the k control markets with the closest pre-period path.

    method="rmse" (default) compares demeaned weekly paths by squared distance,
    so a control must move by the same AMOUNT as the treated market.
    method="corr" ranks by correlation instead. Correlation ignores scale, so
    it favors controls with strong seasonal swings (high signal relative to
    noise) and, when the post period falls in a seasonal dip, biases the lift
    upward. `coverage_study` shows this in simulation.
    """
    pre = panel[~panel["post"]].pivot(index="week", columns="market", values="log_sales")
    treated = panel.loc[panel["treated"], "market"].unique()
    controls = [m for m in pre.columns if m not in set(treated)]
    chosen: set[int] = set()
    if method == "rmse":
        pre = pre - pre.mean()
    for m in treated:
        if method == "corr":
            ranked = pre[controls].corrwith(pre[m]).sort_values(ascending=False)
        elif method == "rmse":
            ranked = (pre[controls].sub(pre[m], axis=0) ** 2).mean().sort_values()
        else:
            raise ValueError(method)
        chosen.update(int(c) for c in ranked.index[:k])
    return sorted(chosen)


def did_estimate(panel: pd.DataFrame, control_markets: list[int] | None = None) -> dict:
    """Two-period difference-in-differences on market means.

    Each market collapses to one pre-to-post change, so the standard error is
    clustered by market by construction. The interval uses a t critical value
    with min(n_treated, n_control) - 1 degrees of freedom because the number of
    markets, not weeks, is the real sample size.
    """
    df = panel.copy()
    if control_markets is not None:
        df = df[df["treated"] | df["market"].isin(control_markets)]
    by = df.groupby(["market", "treated", "post"])["log_sales"].mean().unstack("post")
    by["delta"] = by[True] - by[False]
    d_t = by.xs(True, level="treated")["delta"]
    d_c = by.xs(False, level="treated")["delta"]
    est = d_t.mean() - d_c.mean()
    se = np.sqrt(d_t.var(ddof=1) / len(d_t) + d_c.var(ddof=1) / len(d_c))
    crit = stats.t.ppf(0.975, min(len(d_t), len(d_c)) - 1)
    return {"estimate": float(est), "se": float(se), "ci_low": float(est - crit * se),
            "ci_high": float(est + crit * se), "n_treated": int(len(d_t)), "n_control": int(len(d_c))}


def event_study(panel: pd.DataFrame, control_markets: list[int] | None = None) -> pd.DataFrame:
    """Treated minus control gap by week, normalized to the average pre-period gap.

    Flat gaps before launch support the parallel-trends assumption; the jump
    after launch is the effect.
    """
    df = panel.copy()
    if control_markets is not None:
        df = df[df["treated"] | df["market"].isin(control_markets)]
    gap = df.groupby(["week", "treated"])["log_sales"].mean().unstack("treated")
    gap = (gap[True] - gap[False]).rename("gap").reset_index()
    last_pre = df.loc[~df["post"], "week"].max()
    gap["gap"] -= float(gap.loc[gap["week"] <= last_pre, "gap"].mean())
    gap["relative_week"] = gap["week"] - (last_pre + 1)
    return gap


def placebo_test(panel: pd.DataFrame, n_iter: int = 500, seed: int = 0) -> dict:
    """Reassign 'treated' to random control markets many times; how often is the fake effect as big?"""
    rng = np.random.default_rng(seed)
    real = did_estimate(panel)["estimate"]
    n_t = panel.loc[panel["treated"], "market"].nunique()
    controls = panel.loc[~panel["treated"], "market"].unique()
    base = panel[panel["market"].isin(controls)].copy()
    fakes = []
    for _ in range(n_iter):
        fake_t = set(rng.choice(controls, n_t, replace=False))
        base["treated"] = base["market"].isin(fake_t)
        fakes.append(did_estimate(base)["estimate"])
    fakes = np.array(fakes)
    return {"estimate": real, "placebo_p_value": float((np.abs(fakes) >= abs(real)).mean()),
            "placebo_sd": float(fakes.std())}


def coverage_study(n_sims: int = 150, lift: float = 0.05, method: str | None = "rmse") -> dict:
    """Run many simulated geo tests: is the estimator unbiased, and do 95% intervals cover 95% of the time?

    method=None uses every control market; otherwise controls are matched with `match_controls`.
    """
    est, hits = [], 0
    for seed in range(n_sims):
        panel = simulate_geo_panel(lift=lift, seed=1000 + seed)
        r = did_estimate(panel, match_controls(panel, method=method) if method else None)
        est.append(r["estimate"])
        hits += r["ci_low"] <= lift <= r["ci_high"]
    est = np.array(est)
    return {"mean_estimate": float(est.mean()), "bias": float(est.mean() - lift),
            "bias_se": float(est.std(ddof=1) / np.sqrt(n_sims)), "sd": float(est.std(ddof=1)),
            "coverage": hits / n_sims, "n_sims": n_sims}
