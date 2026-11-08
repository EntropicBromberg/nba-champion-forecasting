# Pre-playoff uncertainty vs production, with prior-only calibration

**Retrospective diagnostic only. Production, displayed odds and original experiment gates remain unchanged.**

Identical 14-season pre-playoff cases, outcomes, cutoffs and projected home courts, verified before reusing cached out-of-season forecasts. One positive logit slope per model/target is fitted on earlier complete-season pre-playoff OOS predictions only. No intercept. 2013–2015 use identity calibration; fitted slopes begin in 2016. Production receives the same calibration opportunity.

| Target | Variant | N | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| game | fixed_calibrated | 1211 | 0.638263 | 0.223406 | 0.032082 |
| game | fixed_raw | 1211 | 0.637837 | 0.223326 | 0.033074 |
| game | production_calibrated | 1211 | 0.651787 | 0.229630 | 0.066590 |
| game | production_raw | 1211 | 0.651046 | 0.229304 | 0.059624 |
| game | uncertain_calibrated | 1211 | 0.638125 | 0.223449 | 0.032220 |
| game | uncertain_raw | 1211 | 0.637694 | 0.223362 | 0.040383 |
| series | fixed_calibrated | 210 | 0.563631 | 0.191940 | 0.088422 |
| series | fixed_raw | 210 | 0.563263 | 0.191895 | 0.067109 |
| series | production_calibrated | 210 | 0.572813 | 0.193437 | 0.079981 |
| series | production_raw | 210 | 0.592877 | 0.198302 | 0.090938 |
| series | uncertain_calibrated | 210 | 0.562550 | 0.191641 | 0.081305 |
| series | uncertain_raw | 210 | 0.558874 | 0.190629 | 0.070865 |

## Paired season-bootstrap comparisons: negative favors candidate

- uncertain_raw_minus_production_raw, game: delta -0.013352, 95% CI [-0.027932, -0.001274], Brier delta -0.005941, wins 11/14 seasons.
- uncertain_raw_minus_production_raw, series: delta -0.034002, 95% CI [-0.095915, +0.020391], Brier delta -0.007672, wins 8/14 seasons.
- uncertain_raw_minus_production_calibrated, game: delta -0.014093, 95% CI [-0.026719, -0.002844], Brier delta -0.006267, wins 11/14 seasons.
- uncertain_raw_minus_production_calibrated, series: delta -0.013939, 95% CI [-0.061024, +0.028472], Brier delta -0.002807, wins 9/14 seasons.
- uncertain_calibrated_minus_production_raw, game: delta -0.012921, 95% CI [-0.027436, -0.000743], Brier delta -0.005854, wins 9/14 seasons.
- uncertain_calibrated_minus_production_raw, series: delta -0.030327, 95% CI [-0.093434, +0.024862], Brier delta -0.006661, wins 8/14 seasons.
- uncertain_calibrated_minus_production_calibrated, game: delta -0.013662, 95% CI [-0.026436, -0.002127], Brier delta -0.006180, wins 10/14 seasons.
- uncertain_calibrated_minus_production_calibrated, series: delta -0.010263, 95% CI [-0.058134, +0.034725], Brier delta -0.001796, wins 8/14 seasons.
- uncertain_calibrated_minus_fixed_calibrated, game: delta -0.000138, 95% CI [-0.001420, +0.001011], Brier delta +0.000043, wins 6/14 seasons.
- uncertain_calibrated_minus_fixed_calibrated, series: delta -0.001081, 95% CI [-0.006438, +0.003950], Brier delta -0.000299, wins 7/14 seasons.
- uncertain_calibrated_minus_uncertain_raw, game: delta +0.000430, 95% CI [-0.000413, +0.001392], Brier delta +0.000087, wins 4/14 seasons.
- uncertain_calibrated_minus_uncertain_raw, series: delta +0.003675, 95% CI [-0.002432, +0.010625], Brier delta +0.001011, wins 5/14 seasons.
- production_calibrated_minus_production_raw, game: delta +0.000741, 95% CI [-0.001963, +0.004433], Brier delta +0.000326, wins 7/14 seasons.
- production_calibrated_minus_production_raw, series: delta -0.020063, 95% CI [-0.042664, +0.004849], Brier delta -0.004865, wins 8/14 seasons.

Per-season and fitted-calibration-active results are in `metrics.csv`; coefficients and strictly earlier training years are in `report.json`; `predictions.csv.gz`, `calibration_bins.csv` and `home_court_audit.csv` contain the complete audit.

This output calibration is a diagnostic, separately fitted for games and series. It has not been wired into the persistent-strength season simulator and does not establish calibrated title probabilities. These reused historical outcomes do not satisfy a genuinely new prospective promotion cohort.


Source release note: this is an authored summary of a prior local research run. Supporting event-level tables, caches, and fitted artifacts named above are excluded from this package; aggregate findings are retained.
