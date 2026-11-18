"""Run the full analysis and write reports/RESULTS.md plus figures.

    python scripts/run_report.py              # real Hillstrom data (downloads once)
    python scripts/run_report.py --synthetic  # simulated data with a known answer

The headline block in README.md (between the RESULTS markers) is refreshed
from whichever run happens last, labeled with its data source.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from inclab import ab, checks, did, figures, policy, uplift  # noqa: E402
from inclab.data import PRE_COVARIATES, feature_matrix, load_hillstrom, simulate_hillstrom_like  # noqa: E402

REPORTS = os.path.join(ROOT, "reports")
FIGS = os.path.join(REPORTS, "figures")


def fmt_pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def money(x, d=0):
    return f"{'-' if x < 0 else ''}${abs(x):,.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--arm", default="mens", choices=["mens", "womens"], help="email arm to model uplift for")
    ap.add_argument("--cost", type=float, default=0.10, help="cost per email in dollars")
    ap.add_argument("--margin", type=float, default=0.30, help="gross margin on incremental spend")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)

    if args.synthetic:
        df = simulate_hillstrom_like(n=64000, seed=args.seed)
        source = "SYNTHETIC data with a known treatment effect (`--synthetic`)"
    else:
        df = load_hillstrom()
        source = "Hillstrom MineThatData email experiment (64,000 customers)"
    out = [f"# Results\n\nData: {source}.\n"]

    # 1. Can we trust the randomization?
    counts = df["arm"].value_counts().to_dict()
    srm = checks.srm_test(counts)
    out.append("## 1. Randomization checks\n")
    out.append(f"Arm sizes: {', '.join(f'{a} {c:,}' for a, c in counts.items())}. "
               f"Sample ratio mismatch test against an equal split: chi-square {srm['chi2']:.2f}, "
               f"p = {fmt_p(srm['p_value'])} ({'no SRM' if srm['p_value'] > 0.001 else 'SRM DETECTED, stop and investigate'}).\n")
    bal_rows = []
    for arm in ["mens", "womens"]:
        smd = checks.standardized_mean_differences(df, "arm", arm, "control", PRE_COVARIATES)
        for _, r in smd.iterrows():
            bal_rows.append({"arm": arm, "covariate": r["covariate"], "SMD": f"{r['smd']:+.3f}",
                             "balanced (abs < 0.1)": "yes" if r["balanced"] else "NO"})
    max_smd = max(abs(float(r["SMD"])) for r in bal_rows)
    out.append("Covariate balance against control:\n\n" + md_table(pd.DataFrame(bal_rows)) + "\n")

    # 2. Average effects
    out.append("## 2. Average treatment effects\n")
    rows, pvals = [], []
    ctrl = df[df["arm"] == "control"]
    for arm in ["mens", "womens"]:
        trt = df[df["arm"] == arm]
        sub = pd.concat([trt, ctrl])
        is_t = (sub["arm"] == arm).to_numpy()
        for y in ["visit", "conversion", "spend"]:
            w = ab.welch(trt[y], ctrl[y])
            cu = ab.cuped(sub[y].to_numpy(), sub["history"].to_numpy(), is_t)
            pvals.append(w["p_value"])
            scale = 1 if y == "spend" else 100
            rows.append({"arm": arm, "metric": y,
                         "control": f"{w['control_mean']:.4f}", "treated": f"{w['treated_mean']:.4f}",
                         "lift": (f"{'+' if w['diff'] >= 0 else '-'}${abs(w['diff']):.3f}" if y == "spend"
                                  else f"{w['diff'] * scale:+.2f} pp"),
                         "95% CI": f"[{w['ci_low'] * scale:+.3f}, {w['ci_high'] * scale:+.3f}]",
                         "relative": fmt_pct(w["rel_lift"]), "p": fmt_p(w["p_value"]),
                         "CUPED var. reduction": fmt_pct(cu["variance_reduction"])})
    keep = ab.benjamini_hochberg(pvals)
    for r, k in zip(rows, keep):
        r["significant after BH"] = "yes" if k else "no"
    ate = pd.DataFrame(rows)
    out.append(md_table(ate) + "\n")

    trt = df[df["arm"] == args.arm]
    lo, hi = ab.bootstrap_diff_ci(trt["spend"], ctrl["spend"], n_boot=1000, seed=args.seed)
    w99 = ab.welch(ab.winsorize(trt["spend"]), ab.winsorize(ctrl["spend"]))
    mw = ab.mann_whitney(trt["spend"], ctrl["spend"])
    out.append(f"Spend is zero for most customers and heavy-tailed among buyers, so the {args.arm} spend effect "
               f"gets three extra reads: bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]; winsorized at the 99th "
               f"percentile {w99['diff']:+.3f} (p = {fmt_p(w99['p_value'])}); Mann-Whitney p = {fmt_p(mw)}. "
               "CUPED uses pre-period `history` spend as the covariate; its variance reduction is small here "
               "because last year's spend predicts a two-week window weakly.\n")

    # 3. Heterogeneous effects
    out.append(f"## 3. Who responds: uplift modeling for the {args.arm} email (outcome: visit)\n")
    sub = pd.concat([trt, ctrl]).reset_index(drop=True)
    X = feature_matrix(sub).to_numpy()
    t = (sub["arm"] == args.arm).to_numpy()
    y = sub["visit"].to_numpy(float)
    rng = np.random.default_rng(args.seed)
    test = rng.random(len(sub)) < 0.5
    xl = uplift.XLearner().fit(X[~test], t[~test], y[~test])
    tl = uplift.TLearner().fit(X[~test], t[~test], y[~test])
    s_x, s_t = xl.predict(X[test]), tl.predict(X[test])
    s_r = rng.random(test.sum())
    q = {name: uplift.qini_coefficient(y[test], t[test], s) for name, s in
         [("X-learner", s_x), ("T-learner", s_t), ("Random", s_r)]}
    out.append("Trained on a random half, evaluated on the other half. Qini coefficient (area between the "
               "model's cumulative incremental-visit curve and random targeting; higher is better):\n\n"
               + md_table(pd.DataFrame([{"model": k, "Qini (x1000)": f"{v * 1000:.2f}"} for k, v in q.items()])) + "\n")
    if "true_tau_mens" in sub:
        true_tau = sub.loc[test, f"true_tau_{args.arm}"].to_numpy()
        out.append(f"Because the data are simulated, the true individual effect is known. Correlation of "
                   f"predicted with true uplift: X-learner {np.corrcoef(s_x, true_tau)[0, 1]:.2f}, "
                   f"T-learner {np.corrcoef(s_t, true_tau)[0, 1]:.2f}.\n")
    dec = uplift.uplift_by_decile(y[test], t[test], s_x)
    out.append("Predicted and observed lift by X-learner decile (test half):\n\n" + md_table(pd.DataFrame([
        {"decile": r["decile"], "predicted": f"{100 * r['predicted']:+.2f} pp",
         "observed": f"{100 * r['observed']:+.2f} pp", "n": f"{r['n']:,}"} for r in dec])) + "\n")
    curves = {"X-learner": uplift.qini_curve(y[test], t[test], s_x),
              "T-learner": uplift.qini_curve(y[test], t[test], s_t)}
    figures.qini_plot(curves, os.path.join(FIGS, "qini.png"), f"Qini curve, {args.arm} email, test half")
    figures.decile_plot(dec, os.path.join(FIGS, "uplift_deciles.png"), "Predicted vs observed lift by decile")
    out.append("![Qini curve](figures/qini.png)\n\n![Deciles](figures/uplift_deciles.png)\n")

    # 4. Targeting decision
    out.append(f"## 4. Who should get the email? (cost ${args.cost:.2f}/email, margin {fmt_pct(args.margin, 0)})\n")
    spend = sub.loc[test, "spend"].to_numpy()
    tab = policy.targeting_table(spend, t[test], s_x, args.cost, args.margin)
    figures.policy_plot(tab, os.path.join(FIGS, "targeting_profit.png"), "Net profit by targeting depth")
    out.append(md_table(pd.DataFrame([{
        "share emailed": fmt_pct(r["share_emailed"], 0),
        "incremental spend per emailed customer": f"${r['incremental_spend_per_email']:.3f}",
        "net profit per 1,000 customers": money(r["net_profit_per_1000_customers"]),
        "95% interval": f"[{money(r['profit_low'])}, {money(r['profit_high'])}]"} for r in tab])) + "\n")
    best = max(tab, key=lambda r: r["net_profit_per_1000_customers"])
    full = tab[-1]
    sens = []
    for c in [0.05, 0.10, 0.20, 0.30]:
        tb = policy.targeting_table(spend, t[test], s_x, c, args.margin)
        b = max(tb, key=lambda r: r["net_profit_per_1000_customers"])
        sens.append({"cost per email": f"${c:.2f}", "best share to email": fmt_pct(b["share_emailed"], 0),
                     "profit per 1,000": money(b["net_profit_per_1000_customers"])})
    out.append("Sensitivity of the decision to email cost:\n\n" + md_table(pd.DataFrame(sens)) + "\n")
    out.append("![Targeting profit](figures/targeting_profit.png)\n")

    # 5. Geo holdout
    out.append("## 5. When you cannot randomize people: a simulated geo holdout\n")
    panel = did.simulate_geo_panel(lift=0.05, seed=args.seed)
    controls = did.match_controls(panel)
    naive = did.did_estimate(panel)
    matched = did.did_estimate(panel, controls)
    plac = did.placebo_test(panel, n_iter=300, seed=args.seed)
    es = did.event_study(panel, controls)
    figures.event_study_plot(es, os.path.join(FIGS, "event_study.png"), "Event study, simulated geo test (true lift 5%)")
    out.append("Fifty markets, twelve launch a campaign after week 12, and the true lift is 5.0% of sales. Markets "
               "differ in level, trend and seasonal sensitivity, so any single test is noisy.\n\n"
               f"- DiD with all {naive['n_control']} control markets: {fmt_pct(naive['estimate'], 2)} "
               f"[{fmt_pct(naive['ci_low'], 2)}, {fmt_pct(naive['ci_high'], 2)}]\n"
               f"- DiD with {matched['n_control']} path-matched controls: {fmt_pct(matched['estimate'], 2)} "
               f"[{fmt_pct(matched['ci_low'], 2)}, {fmt_pct(matched['ci_high'], 2)}]\n"
               f"- Placebo test (300 fake launches among control markets): p = {fmt_p(plac['placebo_p_value'])}\n\n"
               "![Event study](figures/event_study.png)\n")
    cov_rows = []
    covs = {}
    for label, method in [("All controls", None), ("Matched on pre-period correlation", "corr"),
                          ("Matched on pre-period distance (default)", "rmse")]:
        c = did.coverage_study(n_sims=150, method=method)
        covs[method] = c
        cov_rows.append({"estimator": label, "mean estimate": fmt_pct(c["mean_estimate"], 2),
                         "bias": f"{100 * c['bias']:+.2f} pp (SE {100 * c['bias_se']:.2f})",
                         "SD across tests": fmt_pct(c["sd"], 2), "95% CI coverage": fmt_pct(c["coverage"], 0)})
    cov = covs["rmse"]
    out.append("Is the estimator trustworthy beyond one draw? 150 independent simulated tests per estimator:\n\n"
               + md_table(pd.DataFrame(cov_rows)) + "\n\n"
               "Matching on correlation looks sensible and is biased. Correlation ignores scale, so it picks "
               "controls with strong seasonal swings; the post period falls in a seasonal dip, those controls dip "
               "harder than the treated markets, and the lift is overstated. Matching on squared distance between "
               "demeaned paths requires controls to move by the same amount, and the bias disappears.\n")

    with open(os.path.join(REPORTS, "RESULTS.md"), "w") as f:
        f.write("\n".join(out))

    # README headline block
    m = ate[(ate["arm"] == args.arm) & (ate["metric"] == "visit")].iloc[0]
    s = ate[(ate["arm"] == args.arm) & (ate["metric"] == "spend")].iloc[0]
    headline = (
        f"_Last run on: {source}._\n\n"
        f"- Randomization holds: SRM p = {fmt_p(srm['p_value'])}; largest pre-period covariate SMD {max_smd:.3f} "
        f"({'all within' if max_smd < 0.1 else 'NOT all within'} the 0.1 bar).\n"
        f"- The {args.arm} email lifted visits {m['lift']} ({m['relative']} relative) and spend {s['lift']} per customer over two weeks.\n"
        f"- X-learner Qini {q['X-learner'] * 1000:.2f} vs T-learner {q['T-learner'] * 1000:.2f} vs random {q['Random'] * 1000:.2f} (x1000, held-out half).\n"
        f"- At ${args.cost:.2f} per email and {fmt_pct(args.margin, 0)} margin, emailing the top "
        f"{fmt_pct(best['share_emailed'], 0)} by predicted uplift earns {money(best['net_profit_per_1000_customers'])} "
        f"per 1,000 customers vs {money(full['net_profit_per_1000_customers'])} for emailing everyone.\n"
        f"- Geo DiD with path-matched controls averages {fmt_pct(cov['mean_estimate'], 2)} against a true 5.0% lift "
        f"over {cov['n_sims']} simulated tests, with {fmt_pct(cov['coverage'], 0)} interval coverage. Correlation "
        f"matching, a common default, overstates it by {100 * covs['corr']['bias']:.2f} pp.\n"
    )
    readme = os.path.join(ROOT, "README.md")
    if os.path.exists(readme):
        text = open(readme).read()
        a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if a in text and b in text:
            text = text.split(a)[0] + a + "\n" + headline + b + text.split(b)[1]
            open(readme, "w").write(text)
    print(headline)


if __name__ == "__main__":
    main()
