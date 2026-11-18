"""Turn predicted uplift into a targeting decision with money attached."""
from __future__ import annotations

import numpy as np


def targeting_table(y_spend, t, score, cost_per_email: float, margin: float,
                    fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> list[dict]:
    """Net incremental profit per 1,000 customers if we email only the top fraction.

    Within the targeted group, randomization still holds, so the treated minus
    control difference in spend is an unbiased incremental-spend estimate.
    Profit = incremental spend x margin - email cost, for the emailed share.
    """
    y, t, s = np.asarray(y_spend, float), np.asarray(t, bool), np.asarray(score, float)
    order = np.argsort(-s, kind="stable")
    rows = []
    for f in fractions:
        k = max(1, int(round(f * len(y))))
        top = order[:k]
        yt, yc = y[top][t[top]], y[top][~t[top]]
        if len(yt) < 30 or len(yc) < 30:
            continue
        incr = yt.mean() - yc.mean()
        se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
        per_1000_emailed = 1000 * f
        profit = per_1000_emailed * (incr * margin - cost_per_email)
        rows.append({"share_emailed": f, "incremental_spend_per_email": incr, "se": se,
                     "net_profit_per_1000_customers": profit,
                     "profit_low": per_1000_emailed * ((incr - 1.96 * se) * margin - cost_per_email),
                     "profit_high": per_1000_emailed * ((incr + 1.96 * se) * margin - cost_per_email)})
    return rows
