# Results

Data: Hillstrom MineThatData email experiment (64,000 customers).

## 1. Randomization checks

Arm sizes: womens 21,387, mens 21,307, control 21,306. Sample ratio mismatch test against an equal split: chi-square 0.20, p = 0.904 (no SRM).

Covariate balance against control:

| arm | covariate | SMD | balanced (abs < 0.1) |
|---|---|---|---|
| mens | recency | +0.007 | yes |
| mens | history | +0.008 | yes |
| mens | mens | -0.005 | yes |
| mens | womens | +0.008 | yes |
| mens | newbie | -0.001 | yes |
| womens | recency | +0.005 | yes |
| womens | history | +0.007 | yes |
| womens | mens | -0.009 | yes |
| womens | womens | +0.005 | yes |
| womens | newbie | +0.003 | yes |

## 2. Average treatment effects

| arm | metric | control | treated | lift | 95% CI | relative | p | CUPED var. reduction | significant after BH |
|---|---|---|---|---|---|---|---|---|---|
| mens | visit | 0.1062 | 0.1828 | +7.66 pp | [+6.995, +8.323] | 72.1% | <0.001 | 0.4% | yes |
| mens | conversion | 0.0057 | 0.0125 | +0.68 pp | [+0.500, +0.861] | 118.8% | <0.001 | 0.1% | yes |
| mens | spend | 0.6528 | 1.4226 | +$0.770 | [+0.485, +1.055] | 117.9% | <0.001 | 0.0% | yes |
| womens | visit | 0.1062 | 0.1514 | +4.52 pp | [+3.889, +5.157] | 42.6% | <0.001 | 0.4% | yes |
| womens | conversion | 0.0057 | 0.0088 | +0.31 pp | [+0.150, +0.472] | 54.3% | <0.001 | 0.1% | yes |
| womens | spend | 0.6528 | 1.0772 | +$0.424 | [+0.169, +0.680] | 65.0% | 0.001 | 0.0% | yes |

Spend is zero for most customers and heavy-tailed among buyers, so the men's spend effect gets three extra reads: bootstrap 95% CI [+0.472, +1.026]; winsorized at the 99th percentile +0.376 (p < 0.001); Mann-Whitney p < 0.001. CUPED uses pre-period `history` spend as the covariate; its variance reduction is small here because last year's spend predicts a two-week window weakly.

## 3. Who responds: uplift modeling for the men's email (outcome: visit)

Trained on a random half, evaluated on the other half. Qini coefficient (area between the model's cumulative incremental-visit curve and random targeting; higher is better):

| model | Qini (x1000) |
|---|---|
| X-learner | 0.87 |
| T-learner | 0.29 |
| Random | -0.48 |

Predicted and observed lift by X-learner decile (test half):

| decile | predicted | observed | n |
|---|---|---|---|
| 1 | +11.05 pp | +9.49 pp | 2,120 |
| 2 | +9.21 pp | +8.14 pp | 2,119 |
| 3 | +8.52 pp | +9.21 pp | 2,120 |
| 4 | +8.13 pp | +3.54 pp | 2,119 |
| 5 | +7.84 pp | +9.38 pp | 2,119 |
| 6 | +7.56 pp | +7.19 pp | 2,120 |
| 7 | +7.32 pp | +5.07 pp | 2,119 |
| 8 | +7.00 pp | +7.35 pp | 2,120 |
| 9 | +6.54 pp | +6.69 pp | 2,119 |
| 10 | +5.69 pp | +6.69 pp | 2,119 |

![Qini curve](figures/qini.png)

![Deciles](figures/uplift_deciles.png)

## 4. Who should get the email? (cost $0.10/email, margin 30%)

| share emailed | incremental spend per emailed customer | net profit per 1,000 customers | 95% interval |
|---|---|---|---|
| 10% | $2.835 | $75 | [$0, $150] |
| 20% | $1.386 | $63 | [-$22, $148] |
| 30% | $1.258 | $83 | [-$16, $183] |
| 40% | $1.155 | $99 | [-$4, $202] |
| 50% | $0.985 | $98 | [-$6, $202] |
| 60% | $1.044 | $128 | [$20, $236] |
| 70% | $0.897 | $118 | [$6, $231] |
| 80% | $0.785 | $108 | [-$7, $223] |
| 90% | $0.809 | $128 | [$9, $248] |
| 100% | $0.797 | $139 | [$18, $260] |

Sensitivity of the decision to email cost:

| cost per email | best share to email | profit per 1,000 |
|---|---|---|
| $0.05 | 100% | $189 |
| $0.10 | 100% | $139 |
| $0.20 | 60% | $68 |
| $0.30 | 10% | $55 |

![Targeting profit](figures/targeting_profit.png)

## 5. When you cannot randomize people: a simulated geo holdout

Fifty markets, twelve launch a campaign after week 12, and the true lift is 5.0% of sales. Markets differ in level, trend and seasonal sensitivity, so any single test is noisy.

- DiD with all 38 control markets: 3.10% [0.91%, 5.29%]
- DiD with 17 path-matched controls: 3.06% [0.64%, 5.47%]
- Placebo test: 0 of 300 fake launches among control markets produced an estimate as large as the real one (p < 0.0033)

![Event study](figures/event_study.png)

Is the estimator trustworthy beyond one draw? 150 independent simulated tests per estimator:

| estimator | mean estimate | bias | SD across tests | 95% CI coverage |
|---|---|---|---|---|
| All controls | 5.10% | +0.10 pp (SE 0.08) | 0.93% | 95% |
| Matched on pre-period correlation | 5.49% | +0.49 pp (SE 0.07) | 0.92% | 95% |
| Matched on pre-period distance (default) | 5.08% | +0.08 pp (SE 0.07) | 0.88% | 97% |

Matching on correlation looks sensible and is biased. Correlation ignores scale, so it picks controls with strong seasonal swings; the post period falls in a seasonal dip, those controls dip harder than the treated markets, and the lift is overstated. Matching on squared distance between demeaned paths requires controls to move by the same amount, and the bias disappears.
