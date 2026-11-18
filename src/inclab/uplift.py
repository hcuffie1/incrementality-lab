"""Who responds? Conditional average treatment effects with meta-learners.

The X-learner (Kunzel, Sekhon, Bickel and Yu, 2019) imputes an individual
effect for every customer using the OTHER arm's outcome model, then learns
those imputed effects directly. It does well when arms are unbalanced or when
the effect is simpler than the outcome itself, which is common in marketing.
The T-learner is included as a baseline.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


def _default_learner():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=200, l2_regularization=1.0, random_state=0)


class TLearner:
    def __init__(self, learner=None):
        self.learner = learner or _default_learner()

    def fit(self, X, t, y):
        X, t, y = np.asarray(X, float), np.asarray(t, bool), np.asarray(y, float)
        self.mu1 = clone(self.learner).fit(X[t], y[t])
        self.mu0 = clone(self.learner).fit(X[~t], y[~t])
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return self.mu1.predict(X) - self.mu0.predict(X)


class XLearner:
    """X-learner with gradient-boosted base learners.

    Stage 1: outcome models mu0 (control) and mu1 (treated).
    Stage 2: imputed effects D1 = y - mu0(x) on the treated and
             D0 = mu1(x) - y on the controls, each learned by its own model.
    Stage 3: blend tau(x) = g(x) * tau0(x) + (1 - g(x)) * tau1(x), where g is
             the propensity score. In a randomized test g is the design share,
             so it is estimated as a constant unless `fit_propensity=True`.
    """

    def __init__(self, learner=None, effect_learner=None, fit_propensity: bool = False):
        self.learner = learner or _default_learner()
        self.effect_learner = effect_learner or _default_learner()
        self.fit_propensity = fit_propensity

    def fit(self, X, t, y):
        X, t, y = np.asarray(X, float), np.asarray(t, bool), np.asarray(y, float)
        mu0 = clone(self.learner).fit(X[~t], y[~t])
        mu1 = clone(self.learner).fit(X[t], y[t])
        d1 = y[t] - mu0.predict(X[t])
        d0 = mu1.predict(X[~t]) - y[~t]
        self.tau1 = clone(self.effect_learner).fit(X[t], d1)
        self.tau0 = clone(self.effect_learner).fit(X[~t], d0)
        if self.fit_propensity:
            self.g = LogisticRegression(max_iter=1000).fit(X, t)
        else:
            self.g_const = float(t.mean())
        return self

    def propensity(self, X):
        X = np.asarray(X, float)
        if self.fit_propensity:
            return self.g.predict_proba(X)[:, 1]
        return np.full(len(X), self.g_const)

    def predict(self, X):
        X = np.asarray(X, float)
        g = self.propensity(X)
        return g * self.tau0.predict(X) + (1 - g) * self.tau1.predict(X)


def qini_curve(y, t, score, n_points: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Incremental outcomes captured when targeting the top fraction by `score`.

    Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k), the treated outcomes among the
    top k minus the control outcomes scaled to the same group size. Returned
    per customer in the whole sample so curves from different models compare.
    """
    y, t, score = np.asarray(y, float), np.asarray(t, bool), np.asarray(score, float)
    order = np.argsort(-score, kind="stable")
    y, t = y[order], t[order]
    cum_t = np.cumsum(t)
    cum_c = np.cumsum(~t)
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * ~t)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = cum_yt - np.where(cum_c > 0, cum_yc * cum_t / cum_c, 0.0)
    idx = np.linspace(0, len(y) - 1, n_points).astype(int)
    frac = (idx + 1) / len(y)
    return frac, q[idx] / len(y)


def qini_coefficient(y, t, score) -> float:
    """Area between the model's Qini curve and random targeting (higher is better; 0 = random)."""
    frac, q = qini_curve(y, t, score)
    random_line = frac * q[-1]
    return float(np.trapezoid(q - random_line, frac)) if hasattr(np, "trapezoid") else float(np.trapz(q - random_line, frac))


def uplift_by_decile(y, t, score, n_bins: int = 10) -> list[dict]:
    """Observed treated-minus-control difference within each predicted-uplift decile."""
    y, t, score = np.asarray(y, float), np.asarray(t, bool), np.asarray(score, float)
    ranks = np.argsort(np.argsort(-score))
    bins = (ranks * n_bins // len(score)).astype(int)
    out = []
    for b in range(n_bins):
        m = bins == b
        yt, yc = y[m & t], y[m & ~t]
        out.append({"decile": b + 1, "predicted": float(score[m].mean()),
                    "observed": float(yt.mean() - yc.mean()) if len(yt) and len(yc) else np.nan,
                    "n": int(m.sum())})
    return out
