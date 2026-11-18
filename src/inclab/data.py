"""Loading the Hillstrom email experiment, plus a simulator with the same shape.

The Hillstrom (MineThatData, 2008) dataset covers 64,000 customers who bought
in the past year. Each was randomly assigned to one of three arms for a
two-week campaign: a men's merchandise email, a women's merchandise email, or
no email. Outcomes are whether they visited the site, whether they bought,
and how much they spent in the following two weeks.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import pandas as pd

HILLSTROM_URLS = [
    # Public mirror used by scikit-uplift's fetch_hillstrom.
    "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz",
    # Original MineThatData file.
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
]
DEFAULT_PATH = os.path.join("data", "hillstrom.csv.gz")

ARMS = {"No E-Mail": "control", "Mens E-Mail": "mens", "Womens E-Mail": "womens"}
OUTCOMES = ["visit", "conversion", "spend"]
PRE_COVARIATES = ["recency", "history", "mens", "womens", "newbie"]


def download_hillstrom(path: str = DEFAULT_PATH) -> str:
    """Download the dataset once and cache it locally."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    last_error = None
    for url in HILLSTROM_URLS:
        try:
            tmp = path + (".tmp.gz" if url.endswith(".gz") else ".tmp.csv")
            urllib.request.urlretrieve(url, tmp)
            df = pd.read_csv(tmp)
            df.to_csv(path, index=False, compression="gzip")
            os.remove(tmp)
            return path
        except Exception as err:  # try the next mirror
            last_error = err
    raise RuntimeError(f"Could not download the Hillstrom data: {last_error}")


def clean_hillstrom(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["arm"] = df["segment"].map(ARMS)
    if df["arm"].isna().any():
        raise ValueError(f"Unexpected segment values: {df.loc[df['arm'].isna(), 'segment'].unique()}")
    for col in ["visit", "conversion"]:
        df[col] = df[col].astype(int)
    df["spend"] = df["spend"].astype(float)
    return df


def load_hillstrom(path: str = DEFAULT_PATH, download: bool = True) -> pd.DataFrame:
    if download:
        download_hillstrom(path)
    return clean_hillstrom(pd.read_csv(path))


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-treatment features only. Nothing here can be affected by the email."""
    X = df[["recency", "history", "mens", "womens", "newbie"]].astype(float).copy()
    X["log_history"] = np.log1p(df["history"].astype(float))
    for col in ["zip_code", "channel"]:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
        X = pd.concat([X, dummies], axis=1)
    return X


def simulate_hillstrom_like(n: int = 30000, seed: int = 0) -> pd.DataFrame:
    """A dataset with the Hillstrom schema and a KNOWN treatment effect.

    Used by the tests and by `run_report.py --synthetic` to check that every
    estimator recovers the truth before it is trusted on real data. The true
    visit uplift of each email is stored in `true_tau_mens` and
    `true_tau_womens`.
    """
    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 13, n)
    history = np.round(rng.lognormal(mean=5.0, sigma=0.9, size=n), 2)
    mens = rng.binomial(1, 0.55, n)
    womens = np.where(mens == 1, rng.binomial(1, 0.45, n), 1)
    newbie = rng.binomial(1, 0.5, n)
    zip_code = rng.choice(["Surburban", "Urban", "Rural"], n, p=[0.45, 0.40, 0.15])
    channel = rng.choice(["Web", "Phone", "Multichannel"], n, p=[0.44, 0.44, 0.12])
    segment = rng.choice(list(ARMS), n)
    hist_seg = pd.cut(history, [0, 100, 200, 350, 500, 750, 1000, np.inf],
                      labels=["1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500",
                              "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +"]).astype(str)

    base = -2.1 + 0.25 * np.log1p(history) / 6 - 0.08 * recency + 0.3 * (channel == "Multichannel")
    # Heterogeneous effects: men's email works for recent men's buyers, women's email for women's buyers.
    tau_m = 0.07 * mens * (recency <= 6) + 0.02 * (1 - newbie)
    tau_w = 0.05 * womens + 0.03 * (zip_code == "Rural")
    p0 = 1 / (1 + np.exp(-base))
    p = p0 + np.where(segment == "Mens E-Mail", tau_m, 0) + np.where(segment == "Womens E-Mail", tau_w, 0)
    visit = rng.binomial(1, np.clip(p, 0, 1))
    conversion = visit * rng.binomial(1, 0.09, n)
    spend = np.where(conversion == 1, np.round(rng.lognormal(4.6, 0.6, n), 2), 0.0)

    raw = pd.DataFrame({
        "recency": recency, "history_segment": hist_seg, "history": history, "mens": mens,
        "womens": womens, "zip_code": zip_code, "newbie": newbie, "channel": channel,
        "segment": segment, "visit": visit, "conversion": conversion, "spend": spend,
    })
    df = clean_hillstrom(raw)
    df["true_tau_mens"] = tau_m
    df["true_tau_womens"] = tau_w
    return df
