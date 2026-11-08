"""A single deployed XGBoost series model; symmetric inference and exact bracket odds."""
from functools import lru_cache
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from data_pipeline import DIRECTIONS, matchup

# Fixed before evaluating outer test seasons. No random row splits or test-set tuning.
PARAMS = dict(objective='binary:logistic', eval_metric='logloss', max_depth=2,
    learning_rate=0.03, n_estimators=600, min_child_weight=10, subsample=0.7,
    colsample_bytree=0.6, reg_lambda=10, reg_alpha=0.5, gamma=1.0,
    random_state=0, n_jobs=1)

def fit(rows, features):
    model = XGBClassifier(**PARAMS, monotone_constraints=tuple(DIRECTIONS[f] for f in features))
    model.fit(rows[features], rows.y)
    return model

def paired_predictions(model, rows, features):
    """Rows must have AB and BA adjacent; score only AB once per actual series."""
    if not rows.iloc[::2].series_id.reset_index(drop=True).equals(rows.iloc[1::2].series_id.reset_index(drop=True)):
        raise ValueError('Each series needs adjacent, reversed orderings')
    probabilities = model.predict_proba(rows[features])[:,1]
    return 0.5 * (probabilities[::2] + 1 - probabilities[1::2])

def bracket_tree(series):
    """Recover fixed bracket topology, then discard outcomes.

    Winner identities connect source rows to their bracket slots only. Every first-round
    entrant retains its full probability of advancing; observed later-round entrants
    never restrict the calculation. Changing outcomes consistently preserves the tree.
    """
    nodes = {}
    for rnd in range(1,5):
        for s in series.loc[series['round'].eq(rnd)].itertuples():
            a = nodes.pop(s.winner, s.winner)
            b = nodes.pop(s.loser, s.loser)
            nodes[s.winner] = (a,b)
    if len(nodes) != 1:
        raise ValueError('Invalid playoff tree')
    return next(iter(nodes.values()))

def title_odds(model, features, teams, series):
    indexed = teams.set_index('team', drop=False)
    neutral = int(teams.season.iloc[0]) == 2020
    @lru_cache(None)
    def probability(a,b,rnd):
        ab = matchup(indexed,a,b,rnd,neutral=neutral)
        ba = matchup(indexed,b,a,rnd,neutral=neutral)
        x = pd.DataFrame([ab,ba])[features]
        p = model.predict_proba(x)[:,1]
        return float((p[0]+1-p[1])/2)
    def advance(node, rnd):
        if isinstance(node,str):
            return {node:1.0}
        left, right = (advance(child,rnd-1) for child in node)
        result = {}
        for a, pa in left.items():
            for b, pb in right.items():
                p = probability(a,b,rnd)
                result[a] = result.get(a,0)+pa*pb*p
                result[b] = result.get(b,0)+pa*pb*(1-p)
        return result
    odds = advance(bracket_tree(series),4)
    if not np.isclose(sum(odds.values()),1):
        raise ValueError('Bracket probabilities do not sum to one')
    output = teams[['team','name','conference','wins','losses']].copy()
    output['title_probability'] = output.team.map(odds).fillna(0)
    output['in_bracket'] = output.team.isin(odds)
    return output.sort_values('title_probability',ascending=False).reset_index(drop=True)
