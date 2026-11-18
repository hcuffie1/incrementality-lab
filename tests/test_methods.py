"""Every estimator is checked against simulated data where the right answer is known."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from inclab import ab, checks, did, uplift  # noqa: E402
from inclab.data import feature_matrix, simulate_hillstrom_like  # noqa: E402


def test_srm_flags_a_broken_split_and_passes_a_clean_one():
    assert checks.srm_test({"a": 50_000, "b": 50_000})["p_value"] > 0.5
    assert checks.srm_test({"a": 50_600, "b": 49_400})["p_value"] < 0.001


def test_balance_on_randomized_data():
    df = simulate_hillstrom_like(n=20000, seed=1)
    smd = checks.standardized_mean_differences(df, "arm", "mens", "control", ["recency", "history", "newbie"])
    assert smd["balanced"].all()


def test_cuped_is_unbiased_and_reduces_variance():
    rng = np.random.default_rng(0)
    estimates_raw, estimates_adj = [], []
    for _ in range(200):
        n = 4000
        x = rng.normal(100, 20, n)
        t = rng.random(n) < 0.5
        y = 0.8 * x + rng.normal(0, 10, n) + 2.0 * t
        r = ab.cuped(y, x, t)
        estimates_raw.append(r["raw"]["diff"])
        estimates_adj.append(r["adjusted"]["diff"])
    assert abs(np.mean(estimates_adj) - 2.0) < 0.1
    assert np.std(estimates_adj) < 0.6 * np.std(estimates_raw)


def test_welch_ci_covers_truth_about_95_percent():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(400):
        yt = rng.exponential(1.2, 2000)
        yc = rng.exponential(1.0, 2000)
        w = ab.welch(yt, yc)
        hits += w["ci_low"] <= 0.2 <= w["ci_high"]
    assert 0.92 <= hits / 400 <= 0.98


def test_benjamini_hochberg_known_case():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert ab.benjamini_hochberg(p, 0.05) == [True, True] + [False] * 8


def _uplift_fixture():
    df = simulate_hillstrom_like(n=40000, seed=11)
    sub = df[df["arm"].isin(["mens", "control"])].reset_index(drop=True)
    X = feature_matrix(sub).to_numpy()
    t = (sub["arm"] == "mens").to_numpy()
    y = sub["visit"].to_numpy(float)
    test = np.random.default_rng(0).random(len(sub)) < 0.5
    return sub, X, t, y, test


def test_x_learner_finds_the_true_effect_and_beats_random():
    sub, X, t, y, test = _uplift_fixture()
    model = uplift.XLearner().fit(X[~test], t[~test], y[~test])
    pred = model.predict(X[test])
    truth = sub.loc[test, "true_tau_mens"].to_numpy()
    corr_x = np.corrcoef(pred, truth)[0, 1]
    corr_t = np.corrcoef(uplift.TLearner().fit(X[~test], t[~test], y[~test]).predict(X[test]), truth)[0, 1]
    # Binary outcome with a small effect and ~13k training rows: 0.35+ is a solid recovery.
    assert corr_x > 0.35
    assert corr_x > corr_t
    q_model = uplift.qini_coefficient(y[test], t[test], pred)
    q_random = uplift.qini_coefficient(y[test], t[test], np.random.default_rng(1).random(test.sum()))
    q_oracle = uplift.qini_coefficient(y[test], t[test], truth)
    assert q_model > q_random + 0.25 * (q_oracle - q_random)


def test_qini_of_perfect_ranking_beats_reversed():
    rng = np.random.default_rng(5)
    n = 20000
    tau = rng.uniform(0, 0.2, n)
    t = rng.random(n) < 0.5
    y = rng.binomial(1, 0.1 + tau * t).astype(float)
    assert uplift.qini_coefficient(y, t, tau) > 0 > uplift.qini_coefficient(y, t, -tau)


def test_did_recovers_known_lift_and_placebo_is_small():
    panel = did.simulate_geo_panel(lift=0.05, seed=2)
    est = did.did_estimate(panel, did.match_controls(panel))
    assert est["ci_low"] <= 0.05 <= est["ci_high"]
    assert did.placebo_test(panel, n_iter=200)["placebo_p_value"] < 0.05


def test_did_finds_nothing_when_there_is_nothing():
    panel = did.simulate_geo_panel(lift=0.0, seed=4)
    est = did.did_estimate(panel)
    assert est["ci_low"] <= 0.0 <= est["ci_high"]


def test_event_study_pre_period_is_flat_and_post_jumps():
    gaps_pre, gaps_post = [], []
    for seed in range(20):
        panel = did.simulate_geo_panel(lift=0.05, seed=seed)
        es = did.event_study(panel, did.match_controls(panel))
        gaps_pre.append(es[es["relative_week"] < 0]["gap"].mean())
        gaps_post.append(es[es["relative_week"] >= 0]["gap"].mean())
    assert abs(np.mean(gaps_pre)) < 0.01
    assert abs(np.mean(gaps_post) - 0.05) < 0.01


def test_distance_matching_is_unbiased_where_correlation_matching_is_not():
    rmse = did.coverage_study(n_sims=60, method="rmse")
    corr = did.coverage_study(n_sims=60, method="corr")
    assert abs(rmse["bias"]) < 3 * rmse["bias_se"]
    assert corr["bias"] > abs(rmse["bias"])
    assert rmse["coverage"] >= 0.88
