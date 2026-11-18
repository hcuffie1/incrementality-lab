# incrementality-lab

Did the email cause the sale, and who should get the next one?

This repo works through the questions a marketing team asks after a campaign, using a public randomized experiment: 64,000 customers split between a men's merchandise email, a women's merchandise email, and no email (the Hillstrom MineThatData dataset). It goes from "can we trust this test" to "which customers should we email, at what cost" and then to the case where randomizing individuals is not possible at all.

Every estimator is first run on simulated data where the true effect is known. If a method cannot recover a planted answer, its output on real data does not get reported.

## Headline results

<!-- RESULTS:START -->
_Last run on: Hillstrom MineThatData email experiment (64,000 customers)._

- Randomization holds: SRM p = 0.904; largest pre-period covariate SMD 0.009 (all within the 0.1 bar).
- The men's email lifted visits +7.66 pp (72.1% relative) and spend +$0.770 per customer over two weeks.
- X-learner Qini 0.87 vs T-learner 0.29 vs random -0.48 (x1000, held-out half).
- At $0.10 per email and 30% margin, emailing everyone is best ($139 per 1,000 customers). Targeting starts to pay at $0.20 per email, where the top 60% by predicted uplift earns $68.
- Geo DiD with path-matched controls averages 5.08% against a true 5.0% lift over 150 simulated tests, with 97% interval coverage. Correlation matching, a common default, overstates it by 0.49 pp.
<!-- RESULTS:END -->

Full tables and figures: [`reports/RESULTS.md`](reports/RESULTS.md).

## The questions, in order

**1. Can the test be trusted?** Before any lift is computed: a sample ratio mismatch test on arm sizes (a broken split invalidates everything downstream) and standardized mean differences on every pre-period covariate.

**2. What did each email do on average?** Lift in visits, conversions and spend for both emails, with Welch intervals, CUPED variance reduction using prior-year spend, and Benjamini-Hochberg correction across the six comparisons. Spend is mostly zeros with a long tail among buyers, so the spend effect also gets a bootstrap interval, a winsorized estimate and a rank test, since each answers a slightly different question.

**3. Who responds?** An X-learner estimates each customer's individual lift from the email, with a T-learner as the baseline. Models train on a random half and are judged on the other half with Qini curves and a predicted-versus-observed lift table by decile. On simulated data the predicted lift is also checked directly against the true individual effect.

**4. Who should get the email?** Predicted lift becomes a decision: net profit per 1,000 customers at each targeting depth, given an email cost and a margin, with intervals, plus how the best depth moves as email cost changes. Both are command-line arguments (`--cost`, `--margin`).

**5. What if you cannot randomize people?** Many campaigns (TV, out-of-home, a regional launch) can only be turned on by market. A simulated 50-market panel with a known 5% lift is analyzed with difference-in-differences, matched control markets, an event study for pre-trends, and a placebo test that reassigns the launch to random control markets. The estimator is then run on 150 independent simulated tests to check bias and interval coverage.

## One finding worth calling out

Matching control markets on the correlation of their pre-period sales looks like a sensible default, and in simulation it overstates the lift. Correlation ignores scale, so it favors markets with strong seasonal swings. When the post period lands in a seasonal dip, those controls fall harder than the treated markets and the campaign gets credit for the difference. Matching on the squared distance between demeaned pre-period paths removes the bias. The coverage table in `RESULTS.md` shows both.

## Run it

```bash
git clone https://github.com/hcuffie1/incrementality-lab.git
cd incrementality-lab
make install            # pip install -e ".[dev]"
make test               # 11 tests, each against a known answer
make report             # downloads the data once to data/, writes reports/
make report-synthetic   # same pipeline on simulated data
```

Python 3.10+. Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib.

## Layout

```
src/inclab/
  data.py      download and clean Hillstrom; simulator with a known effect
  checks.py    sample ratio mismatch, covariate balance
  ab.py        Welch, bootstrap, CUPED, Mann-Whitney, winsorizing, Benjamini-Hochberg
  uplift.py    X-learner, T-learner, Qini curve and coefficient, decile table
  policy.py    targeting depth to net profit
  did.py       geo panel simulator, DiD, matched controls, event study, placebo, coverage study
  figures.py   report charts
scripts/run_report.py   runs everything, writes reports/RESULTS.md and refreshes the block above
tests/test_methods.py   recovery tests on simulated data
```

## Method notes

- **X-learner** (Künzel, Sekhon, Bickel and Yu, 2019). Fit an outcome model per arm, impute each customer's effect using the other arm's model, learn those imputed effects, and blend the two effect models by propensity. In a randomized test the propensity is the design share, so it is a constant here. Base learners are scikit-learn histogram gradient boosting with conservative leaf sizes, because individual effects are small relative to outcome noise.
- **Qini coefficient.** Area between the model's cumulative incremental-outcome curve and the random-targeting line, computed on held-out data. Zero means the ranking is no better than random.
- **CUPED.** The adjustment coefficient is estimated on the pooled sample from a covariate measured before assignment, which keeps the estimate unbiased. How much variance it removes depends on how well the covariate predicts the outcome.
- **DiD intervals** use market-level changes and a t critical value, because the number of markets, not weeks, is the effective sample size.

## Data

Kevin Hillstrom, "MineThatData E-Mail Analytics and Data Mining Challenge," 2008. The loader tries a public mirror first and falls back to the original file. Data is downloaded to `data/` and not committed.

## License

MIT
