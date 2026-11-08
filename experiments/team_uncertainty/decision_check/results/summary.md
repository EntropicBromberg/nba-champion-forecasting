# Paired season-level check versus calibrated production

**HOLD. No promotion. Production remains primary; any experimental display is explicitly labeled.**

Each paired effect weights seasons equally, with 2000 paired whole-season bootstrap resamples. Positive deltas favor calibrated production. October/January production inputs are proxy-based and not decisive promotion evidence. Pre-playoff uses completed regular-season statistics.

| Target | Checkpoint | Seasons | Uncertainty − calibrated production log loss | 95% CI | Improved seasons |
|---|---|---:|---:|---|---:|
| game | january | 14 | -0.01868 | [-0.02712, -0.01043] | 11 |
| game | october | 13 | -0.01198 | [-0.02184, -0.00245] | 10 |
| game | pre_playoff | 14 | -0.01419 | [-0.02706, -0.00276] | 11 |
| series | january | 14 | -0.01628 | [-0.05655, +0.02498] | 8 |
| series | october | 13 | +0.02288 | [-0.03455, +0.08194] | 6 |
| series | pre_playoff | 14 | -0.01394 | [-0.06102, +0.02847] | 9 |
| title | january | 14 | +0.19205 | [-0.08553, +0.46884] | 5 |
| title | october | 13 | +0.25275 | [-0.21541, +0.76610] | 6 |
| title | pre_playoff | 14 | +0.12984 | [-0.08832, +0.37165] | 4 |

## Title calibration

Title log loss is multiclass champion log loss; Brier is summed over all teams within a season. Monte Carlo counts receive additive 0.5 smoothing to avoid infinite scores from finite-sample zeros. Reliability bins are descriptive: there are only 13/14 champions per checkpoint, not hundreds of independent team outcomes. Schedule revisions and pandemic formats are conditioned on as realized. This is not a genuinely new validation cohort.

`annual_scores.csv`, `paired_season_differences.csv`, `title_predictions.csv`, `reliability_bins.csv` and `report.json` contain full metrics, coefficients and audits.

## Descriptive title reliability

| Checkpoint | Model | Champions | Title log loss | Summed Brier | Team-row ECE |
|---|---|---:|---:|---:|---:|
| january | production_calibrated | 14 | 1.43032 | 0.66005 | 0.01341 |
| january | uncertainty | 14 | 1.62237 | 0.71838 | 0.01337 |
| october | production_calibrated | 13 | 2.24115 | 0.85449 | 0.01392 |
| october | uncertainty | 13 | 2.49390 | 0.90318 | 0.01144 |
| pre_playoff | production_calibrated | 14 | 1.59962 | 0.75657 | 0.01422 |
| pre_playoff | uncertainty | 14 | 1.72945 | 0.73536 | 0.01557 |

Only 13/14 independent champion outcomes inform each checkpoint. Team-row ECE is descriptive and dominated by low-probability teams; it is not evidence from hundreds of independent champions. Favorite reliability and counts are in `title_calibration_summary.csv`. Production calibration applies to its game/series heads, not a title-odds calibrator.


Source release note: this is an authored summary of a prior local research run. Supporting event-level tables, caches, and fitted artifacts named above are excluded from this package; aggregate findings are retained.
