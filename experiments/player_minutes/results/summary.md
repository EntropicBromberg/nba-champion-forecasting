# Offline player/minutes experiment

**Decision: DO_NOT_PROMOTE**. Production app/model files are unchanged.

24 historical seasons (2002–03 through 2025–26); first ten train-only; 14 walk-forward test seasons (2012–13 through 2025–26). October 2020 is skipped because the 2020–21 season had not begun. Frozen checkpoint forecasts score subsequent games and conditional playoff matchups.

| Target | Model | Forecasts | Log loss | Brier | Calibration ECE |
|---|---|---:|---:|---:|---:|
| game | current_srs_series_baseline | 27300 | 0.66136 | 0.23358 | 0.07986 |
| game | srs_game_logistic | 27300 | 0.64492 | 0.22626 | 0.04363 |
| game | with_players | 27300 | 0.63879 | 0.22374 | 0.03520 |
| game | without_players | 27300 | 0.64357 | 0.22582 | 0.03756 |
| series | current_srs_series_baseline | 615 | 0.63719 | 0.21773 | 0.11073 |
| series | srs_game_logistic | 615 | 0.62234 | 0.21348 | 0.08913 |
| series | with_players | 615 | 0.60382 | 0.20713 | 0.07383 |
| series | without_players | 615 | 0.62302 | 0.21435 | 0.09393 |

Player feature minus otherwise identical model (negative is better):
- game: log-loss delta -0.00478, whole-season 95% CI [-0.00698, -0.00272]; Brier delta -0.00208; log loss improved in 11/14 seasons.
- series: log-loss delta -0.01920, whole-season 95% CI [-0.03556, -0.00442]; Brier delta -0.00722; log loss improved in 9/14 seasons.

## Checkpoint log loss

| Target | Checkpoint | Without players | With players | Delta |
|---|---|---:|---:|---:|
| game | january | 0.63098 | 0.62891 | -0.00207 |
| game | october | 0.65182 | 0.64493 | -0.00689 |
| game | pre_playoff | 0.63756 | 0.63850 | +0.00095 |
| series | january | 0.61790 | 0.59354 | -0.02436 |
| series | october | 0.69840 | 0.66036 | -0.03804 |
| series | pre_playoff | 0.55814 | 0.56158 | +0.00344 |

## Preregistered checks
- game_logloss_significant: True
- game_brier_improves: True
- series_logloss_improves: True
- series_brier_improves: True
- game_season_consistency: True
- series_season_consistency: False
- checkpoint_consistency: True
- calibration_not_materially_worse: True

## Limits
- Prior-season BPM proxy, not EPM; no injury/transaction archives.
- Rookies/trades enter rotations only after observed appearances; prior rosters seed opening games.
- Frozen forecasts are conditional on actual future fixtures/venues and actual playoff matchups; archived schedules are unavailable, especially COVID rescheduling.
- Series home court is projected from snapshot information, not eventual venues; bubble is evaluated conditional on neutral venues.
- Future rest information is not updated after snapshot; possible series games use equal rest.
- Repeated game forecasts across checkpoints are dependent; intervals cluster the whole season.
- Final-season BPM is recomputed by the modern source; prior-season data availability is reconstructed, not a vendor vintage archive.

Detailed results: `metrics.csv`, `calibration.csv`, `predictions.csv.gz`, `projected_minutes_audit.csv.gz`, `report.json`. Fixed setup: `../spec.json`. No tuning or production promotion after seeing test results.

Methodological inspiration: [Dunks & Threes EPM methodology](https://dunksandthrees.com/about/epm). This experiment uses prior-season BPM, not their proprietary EPM or injury-aware minutes forecasts.


Source release note: this is an authored summary of a prior local research run. Supporting event-level tables, caches, and fitted artifacts named above are excluded from this package; aggregate findings are retained.
