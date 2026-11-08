# Team-strength uncertainty experiment

**Decision: HOLD — production unchanged.**

14 walk-forward seasons (2012–13 through 2025–26), same October/January/pre-playoff cases. All rating parameters and game-model training use earlier data. October 2020 is omitted because the season had not started. No player data or historical BPM approximations.

| Target | Arm | Forecasts | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| game | fixed | 27300 | 0.642448 | 0.225368 | 0.039241 |
| game | uncertain | 27300 | 0.639050 | 0.224120 | 0.028819 |
| series | fixed | 615 | 0.633352 | 0.217860 | 0.105858 |
| series | uncertain | 615 | 0.605305 | 0.209823 | 0.075555 |

## Uncertain minus fixed: negative is better

- game: log-loss delta -0.003398, 95% whole-season CI [-0.004321, -0.002525]; Brier delta -0.001249; 14/14 seasons improve.
- series: log-loss delta -0.028047, 95% whole-season CI [-0.037989, -0.017283]; Brier delta -0.008037; 12/14 seasons improve.
- game/january: log-loss delta -0.001433, 95% whole-season CI [-0.002699, -0.000301]; Brier delta -0.000532; 9/14 seasons improve.
- game/october: log-loss delta -0.004859, 95% whole-season CI [-0.006114, -0.003599]; Brier delta -0.001789; 13/13 seasons improve.
- game/pre_playoff: log-loss delta -0.000142, 95% whole-season CI [-0.001509, +0.001039]; Brier delta +0.000037; 5/14 seasons improve.
- series/january: log-loss delta -0.019850, 95% whole-season CI [-0.029620, -0.008076]; Brier delta -0.006118; 11/14 seasons improve.
- series/october: log-loss delta -0.062353, 95% whole-season CI [-0.083791, -0.040157]; Brier delta -0.017396; 12/13 seasons improve.
- series/pre_playoff: log-loss delta -0.004389, 95% whole-season CI [-0.012923, +0.002826]; Brier delta -0.001266; 9/14 seasons improve.

## Fixed gates
- game_significant: True
- game_brier_improves: True
- game_consistent: True
- game_pre_not_worse: False
- series_significant: True
- series_brier_improves: True
- series_consistent: True
- series_pre_not_worse: True
- calibration_not_materially_worse: False

## Limits
- New team-only mean estimator in BOTH arms: this isolates uncertainty, not superiority over deployed production.
- XGBoost is trained on posterior means; additional marginalization could double-count uncertainty already learned by the classifier.
- Normal margin errors and fixed drift are simplifying assumptions, not proof of calibrated latent strength intervals.
- Future fixtures and playoff pairs are conditioned on as observed; historical schedule revisions were not archived.
- Future quality is held constant per simulation; future trades, injuries and latent drift are not modeled.
- Historical outcomes have been examined in previous experiments; any passing result still needs genuinely new prospective validation.

`strength_ranges.csv` contains dated mean/SD/95% latent ranges, not claimed empirical coverage. `parameter_audit.csv` records prior-only variance estimation. `per_season_differences.csv`, `calibration.csv` and `predictions.csv.gz` contain detailed results. `title_simulation_demo.csv` demonstrates joint persistent draws; historical future fixtures are conditioned on, not reconstructed archived schedules.

Research inspiration: [Lopez, Matthews & Baumer](https://arxiv.org/abs/1701.05976), which uses betting-market state-space models. This experiment instead uses a small Gaussian point-margin filter plus an unchanged XGBoost game head in both arms; it is not a replication.

## Separate pre-playoff results

| Target | Arm | Log loss | Brier | ECE |
|---|---|---:|---:|---:|
| game | fixed | 0.637837 | 0.223326 | 0.033074 |
| game | uncertain | 0.637694 | 0.223362 | 0.040383 |
| series | fixed | 0.563263 | 0.191895 | 0.067109 |
| series | uncertain | 0.558874 | 0.190629 | 0.070865 |

The pre-playoff game Brier point estimate increases by 0.000037; its paired interval includes zero, so this is not evidence of significant harm, but it fails the prespecified no-regression gate. Pre-playoff game ECE increases from 0.033074 to 0.040383, exceeding the fixed 0.005 tolerance. Neither gate is relaxed after scoring.

Nine tests passed, including exact covariance-transform validation, matching the original forest, zero-uncertainty equivalence, persistent-series integration, one strength draw per complete season path, causal cutoffs, and unchanged production/earlier-experiment hashes.


Source release note: this is an authored summary of a prior local research run. Supporting event-level tables, caches, and fitted artifacts named above are excluded from this package; aggregate findings are retained.
