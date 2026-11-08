# Development timeline

This timeline describes my development of the forecasting system with Codex assistance. Dates are reconstructed from local file metadata, self-recorded experiment specifications, and forecast timestamps; they are not historical Git commit dates.

| Date | Work |
|---|---|
| October 25, 2026 | Built the historical feature pipeline, symmetric series model, nested chronological evaluation, and season simulation. Local source files record creation on this date. |
| October 25, 2026 | Investigated player/minutes inputs, persistent team-strength uncertainty, and calibration. Experiment specifications record freeze times that evening; results preserved mixed findings and hold decisions. |
| October 25–26, 2026 | Integrated the optional uncertainty forecast into the dashboard. |
| October 26, 2026 | Added dated injury scenarios, player-role and depth penalties, and recovery stress assumptions. |
| November 3, 2026 | Generated the recorded primary and experimental forecast snapshots. |
| November 8, 2026 | Repaired the offline injury regression check, prepared the independent MIT release, added a synthetic demo and portable tests, and published a README probability view. |

## Research decisions

The system evolved from matchup probabilities to an explicit championship path: regular-season features → series model → qualification and play-in → playoff bracket.

Richer player and uncertainty models remained experimental when probability scores, bootstrap intervals, or calibration checks did not establish a reliable overall improvement. Injury scenarios expose assumptions rather than claim causal injury effects. The main lessons were to respect information cutoffs, avoid treating repeated backtests as fresh evidence, and distinguish sampling error from structural uncertainty.

The public history starts at the initial release. It does not fabricate dated implementation commits for work that was not recorded incrementally.
