# Forecasting methodology and operations

[Return to the project overview](../README.md).

This guide describes the full local research system. The public source release excludes real datasets, snapshots, and model artifacts. Start with [setup](SETUP.md); the offline synthetic demo is separate from real NBA forecasts. Paths in commands are relative to the repository root.

## Current forecast: 2026–27

A single XGBoost playoff-series model powers two Streamlit pages: **championship probabilities** and **feature correlations**. The deployed model now trains on all **25 completed seasons, 2001–02 through 2025–26: 375 series / 750 mirrored rows**. Current 2026–27 games update prediction inputs, never training labels.

## Launch

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The probability page automatically fetches current published regular-season statistics on a fresh application launch. While open it checks every 30 minutes; **Refresh now** clears that cache immediately. Source publication can lag completed games. The page shows both the latest game date included and the last successful refresh time, displayed in Vancouver time.

Prepared historical data, a trained model and a saved snapshot were present in the private working project. They are excluded from this source release and must be generated locally if the source terms and access permit. If a refresh fails, the app explicitly warns and shows the saved snapshot with its original timestamp. It never labels a failed refresh as fresh. No account or API key is needed.

To refresh outside Streamlit:

```bash
python live.py
```

## How current title odds work

The playoff field is not known in October. Giving the current top eight all the odds would incorrectly exclude teams that can still qualify. Instead, `live.py`:

1. Fetches current Basketball Reference team/player advanced tables and bulk NBA regular-season game logs. All 30 teams' log counts must match the published standings; inconsistent feeds preserve the last valid forecast.
2. Blends current strength features with 20 prior-season pseudo-games: current weight is `GP / (GP + 20)`. Availability and age stay current. Before ten games, current MOV stands in for unstable early-season SRS before blending. Observed wins/losses are never blended.
3. Uses the XGBoost model to score every potential series in both orderings, enforcing `p(A,B) = 1 − p(B,A)` by averaging. Single-game probabilities for regular-season and play-in simulations invert the iid best-of-seven win curve; this is an approximation, not a separately fitted model.
4. Simulates 50,000 remaining seasons using ESPN's announced schedule, then the 7–10 play-in games and all four playoff rounds. The standard conference bracket is 1–8, 4–5, 2–7, 3–6. All 30 teams can qualify; probabilities sum to 100%.

Unassigned NBA Cup/TBD games (normally two per team before assignment) use a league-average opponent. Large schedule gaps fail the refresh. Tied final records use random tiebreaks rather than the complete NBA hierarchy. The same seed makes unchanged inputs reproduce the same simulation. Current strength is held fixed over simulated future games; this does not forecast future trades or recoveries from injury.

These are **regular-season-based title projections**, not sportsbook prices or live possession odds. The existing evaluation tests series predictions from completed regular seasons. **Early-season blending, season qualification and single-game conversion have not yet been backtested at historical in-season checkpoints.** Current playoff results are not ingested; this version is intended for tracking the regular season. Sampling error is recorded per team as `simulation_se`; displayed 0% may mean no sampled titles, not mathematical impossibility.

## Historical features and model

Season identifiers are ending years: `2027` means 2026–27. For each completed season, all teams enter within-season standardization; each playoff series supplies AB and BA with flipped labels. No playoff performance enters the features.

- **Core:** SRS; minutes-weighted top-three BPM; players 4–8 BPM residualized against top-three BPM within season; defense share `z(-DRtg) − z(ORtg)`; last-25 MOV minus season MOV; actual minus published Pythagorean win percentage; top-eight availability share; top-two star-out proxy; home court; round.
- **Four factors:** offensive minus defensive eFG%; opponent minus own TOV%; ORB% minus `(100 − DRB%)`; own minus opponent FT/FGA.
- **Optional:** minutes-weighted top-eight age. Career playoff experience is omitted because complete pre-2002 career minutes are unavailable.

Use regular-season team-stint player rows, excluding TOT/2TM/3TM aggregates. Eligibility is at least 500 team-stint minutes or 15 mpg; choose up to eight by team-stint minutes. Availability weights are `max(BPM + 2, 0) × minutes`, applied to `player GP / team GP`; fall back to minutes if all talent weights are zero. Top-two stars are selected by eligible team-stint minutes. The star-out flag means either missed at least three of the last five available regular-season games; a team with fewer than three games cannot trigger it. This proxy does not distinguish rest from injury.

Team features are z-scored within season, then differenced A minus B. Home court is 1/0 (0.5 in the 2020 bubble); round is 1–4. Historical game-one venues identify observed series home court. Historical bracket reconstruction discards outcomes after connecting fixed slots and propagates every entrant's probability.

`model.py` fixes depth 2, 600 trees, learning rate .03, strong regularization and monotone directions. No early stopping on training data and no outer-test tuning. Walk-forward evaluation starts after ten training seasons and now scores 2011–12 through 2025–26. Each training window's final three seasons are inner validation: ablate four-factor/age blocks, prefer the smaller set within .003 log loss, then consider at most one core deletion requiring .003 improvement. Apply the same procedure to all 25 seasons before the final fit.

Evaluation-only baselines are higher-SRS wins and logistic regression on SRS difference. `artifacts/evaluation.json` records selected features, nested ablations, series log loss/Brier/accuracy, and 1,000 whole-season bootstrap intervals. `walk_forward.csv` contains 225 held-out series per model; `historical_title_odds.csv` contains historical bracket forecasts. The deployed model remains XGBoost. These historical intervals do not quantify individual teams' live title uncertainty.

## Rebuild and verify

```bash
python scripts/fetch_reference.py
python scripts/fetch_logs.py
python data_pipeline.py
python train.py
python live.py
python -m unittest discover -s tests -v
```

Historical fetchers cache completed seasons indefinitely under gitignored `data/raw/`. Basketball Reference requests are sequential and spaced at least 7.6 seconds apart. Current tables/logs are forcibly refreshed; the season schedule is cached for at most 24 hours. Writes to the final live snapshot are atomic so a failed source/build does not replace the last good forecast.

- `data_pipeline.py`: historical feature and mirrored-label construction.
- `model.py`: XGBoost fitting, symmetric inference and historical exact bracket odds.
- `train.py`: nested validation and the saved 25-season XGBoost model.
- `live.py`: source refresh, strength blending, season/play-in/playoff simulation and snapshot fallback.
- `app.py`: forecast and feature-correlation pages; no model training during launch.
- `data/live_snapshot.json`: current inputs, probabilities, dates and simulation assumptions.

## Public sources

- [Basketball Reference 2026–27 team tables](https://www.basketball-reference.com/leagues/NBA_2027.html) and [player advanced tables](https://www.basketball-reference.com/leagues/NBA_2027_advanced.html).
- [llimllib NBA data](https://github.com/llimllib/nba_data): bulk NBA Stats regular-season game logs, 2010 onward.
- [hoopR ESPN logs](https://github.com/sportsdataverse/hoopR-data): 2002–2009 historical game logs.
- ESPN public NBA team schedule feeds, season 2027.

## Optional experimental forecast

The championship page defaults to **Production (primary)**. Its separate **Experimental — uncertainty-aware** selector shows team-only persistent-strength simulations, clearly labeled as unpromoted. It does not alter production calculations, model files or primary odds. Experimental inputs must match the primary snapshot's date and records; errors leave the primary forecast available. Experimental odds are not established as calibrated title probabilities.

The paired season-level checkpoint and title diagnostics are in `experiments/team_uncertainty/decision_check/results/summary.md`. Early historical production inputs remain proxy-based; these comparisons cannot authorize promotion.


### Live injury scenario

Run `.venv/bin/python live.py`, then `.venv/bin/python injury_live.py`; launch Streamlit with `.venv/bin/python -m streamlit run app.py`. Open `http://127.0.0.1:8501/?forecast=injury` for the opt-in **Experimental — injury-adjusted** view. Production remains the default without that query parameter.

The separate `data/injury_snapshot.json` uses ESPN's timestamped current injury feed, matched to current player/team identities. For established players listed Out, deduct `max(prior-season BPM, 0) × projected MPG / 48` from SRS until the feed's estimated return. MPG blends 20 prior games with current team-game exposure. Neutral BPM replacement is a conservative scenario assumption, not a fitted causal injury effect. Unknown durations affect only the next day; no text-based recovery predictions. Season-out statuses persist through the playoffs. Recalculate matchup probabilities by injury state for dated remaining games and approximate playoff round dates. The source's estimated dates are not confirmed medical timelines. Already-absent players can be double-counted; negative/unknown BPM and unmatched players are reported but not penalized. No promotion or validation claim.

### Uncertainty + manual injury stress

Open `http://127.0.0.1:8501/?forecast=injury-stress`, or run `.venv/bin/python injury_uncertainty.py` to generate this scenario. It keeps the uncertainty model's posterior, covariance, persistent quality draws, game classifier, schedule, home court, rest and RNG seeds unchanged. All 30 no-injury probabilities were checked against the original uncertainty forecast. Production remains the default view when no scenario URL is supplied.

Player value is `max(prior BPM + 2, 0) × projected MPG / 48`. The three largest projected-minute roles get a 1.75 multiplier. Roster fragility is `1 + 0.5 × clip((1 − bench BPM)/4, 0, 1) + 0.5 × top-three share of top-eight positive talent`. The UI provides a global severity (default 2), an additional selected-team multiplier (default 1), and a recovery decay time (default 90 days). Total team penalties are capped at 12 strength points. These constants are manual stress assumptions, not estimates selected from validation.

Absence costs full value; after a long absence, the return-day cost is 65% and decays exponentially over the selected time. Short absences get smaller, faster recovery tails. Thus a return one month before the playoffs does not erase the penalty. Each possible playoff game uses its dated recovery penalty over approximate round dates, retaining the same latent strength draw. NBA Cup/TBD slots use December 15, and play-in games April 14. Prior-season BPM is used instead of noisy current on/off metrics. Unknown historical ratings or roster identities are omitted explicitly.

`data/raw/injury_recovery_history.json` remembers observed episodes and their first observed dates, so removal from the feed does not silently erase recovery costs. Report-cleared dates are inferred recovery starts, not medically verified return dates. Unknown-duration injuries get a one-day absence scenario and no invented recovery tail. Current roster coverage, estimated return dates, double-counting existing absences, and manual penalty severity remain limitations. This scenario has not been historically calibrated; existing experiment reports apply only to the original models. Default source models and prior evaluation results are not promoted or retuned.
