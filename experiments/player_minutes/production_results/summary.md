# Player model versus production series XGBoost

Pre-playoff series: candidate log loss **0.56158**, production **0.59288**; delta **−0.03130**, 95% season-bootstrap CI **[−0.08513, +0.02162]**. Candidate wins in **7/14 seasons**. The interval crosses zero. The player-free game model is slightly better still at **0.55814**. Largest candidate series gain: 2024 (−0.25044); largest loss: 2026 (+0.14671). **HOLD: production stays unchanged.**

**Retrospective diagnostic. No promotion; production and the original 10/14 rule are unchanged.**

Each production fold trains only on earlier seasons and uses its saved training-only nested feature selection. The deployed full-data artifact is never applied to historical training seasons. Models score identical cases, cutoffs and conditional home courts.

Pre-playoff inputs use complete regular-season statistics and the exact live blending code. October/January rows are **sensitivity analyses**, not exact production replays: historical daily BPM is unavailable, so prior BPM/estimated ages are carried forward; NBA-derived team rates may differ from Basketball Reference. Neither early sensitivity results nor previously inspected historical cases qualify as new promotion evidence.

| Checkpoint | Target | Model | N | Log loss | Brier | ECE |
|---|---|---|---:|---:|---:|---:|
| january | game | production_series_xgboost | 9973 | 0.64817 | 0.22798 | 0.04484 |
| january | game | with_players | 9973 | 0.62891 | 0.21956 | 0.02561 |
| january | game | without_players | 9973 | 0.63098 | 0.22043 | 0.02757 |
| october | game | production_series_xgboost | 16116 | 0.66358 | 0.23510 | 0.05933 |
| october | game | with_players | 16116 | 0.64493 | 0.22632 | 0.04584 |
| october | game | without_players | 16116 | 0.65182 | 0.22934 | 0.05077 |
| pre_playoff | game | production_series_xgboost | 1211 | 0.65105 | 0.22930 | 0.05962 |
| pre_playoff | game | with_players | 1211 | 0.63850 | 0.22378 | 0.01899 |
| pre_playoff | game | without_players | 1211 | 0.63756 | 0.22328 | 0.04658 |
| january | series | production_series_xgboost | 210 | 0.67569 | 0.22229 | 0.12862 |
| january | series | with_players | 210 | 0.59354 | 0.20364 | 0.09797 |
| january | series | without_players | 210 | 0.61790 | 0.21338 | 0.11336 |
| october | series | production_series_xgboost | 195 | 0.71993 | 0.24747 | 0.19348 |
| october | series | with_players | 195 | 0.66036 | 0.22943 | 0.14996 |
| october | series | without_players | 195 | 0.69840 | 0.24236 | 0.17678 |
| pre_playoff | series | production_series_xgboost | 210 | 0.59288 | 0.19830 | 0.09094 |
| pre_playoff | series | with_players | 210 | 0.56158 | 0.18992 | 0.05781 |
| pre_playoff | series | without_players | 210 | 0.55814 | 0.18933 | 0.05038 |

## Paired comparison: candidate minus production

- game/january: log-loss delta -0.01925, season-bootstrap 95% CI [-0.02773, -0.01154]; improved in 11/14 seasons.
- game/october: log-loss delta -0.01865, season-bootstrap 95% CI [-0.02639, -0.01119]; improved in 12/13 seasons.
- game/pre_playoff: log-loss delta -0.01254, season-bootstrap 95% CI [-0.02555, +0.00016]; improved in 10/14 seasons.
- series/january: log-loss delta -0.08214, season-bootstrap 95% CI [-0.13724, -0.02715]; improved in 12/14 seasons.
- series/october: log-loss delta -0.05956, season-bootstrap 95% CI [-0.10479, -0.01083]; improved in 9/13 seasons.
- series/pre_playoff: log-loss delta -0.03130, season-bootstrap 95% CI [-0.08513, +0.02162]; improved in 7/14 seasons.

## Pre-playoff per-season differences

| Ending year | Game log-loss delta | Series log-loss delta |
|---|---:|---:|
| 2013 | -0.01773 | +0.05920 |
| 2014 | -0.03028 | -0.13498 |
| 2015 | -0.04414 | -0.08894 |
| 2016 | -0.06383 | -0.10048 |
| 2017 | -0.00998 | +0.02074 |
| 2018 | -0.03084 | +0.02620 |
| 2019 | +0.02493 | +0.04258 |
| 2020 | +0.00336 | -0.04325 |
| 2021 | -0.02520 | -0.06909 |
| 2022 | -0.00087 | +0.07304 |
| 2023 | +0.03638 | +0.01666 |
| 2024 | -0.00610 | -0.25044 |
| 2025 | -0.01772 | -0.13611 |
| 2026 | +0.00333 | +0.14671 |

The immutable prospective policy is `../prospective_promotion_rule.json`: three wholly new complete seasons (2026–27 through 2028–29), archived real as-of inputs and forecasts, all-three-season pre-playoff improvements plus the other fixed gates, no tuning or optional stopping. Missing exact inputs or uncertain evidence means HOLD. Original historical rejection is not waived.

Detailed files: `per_season_differences.csv`, `metrics.csv`, `calibration.csv`, `predictions.csv.gz`, `report.json`.


Source release note: this is an authored summary of a prior local research run. Supporting event-level tables, caches, and fitted artifacts named above are excluded from this package; aggregate findings are retained.
