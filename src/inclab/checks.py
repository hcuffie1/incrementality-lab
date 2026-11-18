"""Can we trust the experiment? Sample ratio mismatch and covariate balance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def srm_test(counts: dict[str, int], expected_share: dict[str, float] | None = None) -> dict:
    """Chi-square test of observed arm sizes against the designed split.

    A small p-value (below 0.001 is the usual alarm) means assignment or
    logging is broken, and no downstream estimate should be trusted until it
    is explained.
    """
    arms = list(counts)
    observed = np.array([counts[a] for a in arms], dtype=float)
    if expected_share is None:
        expected_share = {a: 1 / len(arms) for a in arms}
    expected = np.array([expected_share[a] for a in arms]) * observed.sum()
    chi2, p = stats.chisquare(observed, expected)
    return {"arms": arms, "observed": observed.astype(int).tolist(),
            "expected": expected.round(1).tolist(), "chi2": float(chi2), "p_value": float(p)}


def standardized_mean_differences(df: pd.DataFrame, arm_col: str, treated: str, control: str,
                                  covariates: list[str]) -> pd.DataFrame:
    """SMD = difference in means / pooled SD. |SMD| under 0.1 is the usual bar for balance."""
    t = df[df[arm_col] == treated]
    c = df[df[arm_col] == control]
    rows = []
    for col in covariates:
        mt, mc = t[col].mean(), c[col].mean()
        sd = np.sqrt((t[col].var() + c[col].var()) / 2)
        rows.append({"covariate": col, "treated_mean": mt, "control_mean": mc,
                     "smd": (mt - mc) / sd if sd > 0 else 0.0})
    out = pd.DataFrame(rows)
    out["balanced"] = out["smd"].abs() < 0.1
    return out
