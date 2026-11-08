"""Opt-in, unpromoted team-only odds. Never writes primary forecasts or production models."""
from datetime import datetime,timezone
from pathlib import Path
from io import BytesIO
import hashlib
import json
import requests
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from experiments.team_uncertainty.decision_check.common import research,UNC,GameProbabilities,UncertainPaths
from experiments.team_uncertainty.decision_check.paths import season_paths
from ratings import Ratings
from live import get_schedule,atomic_json

ROOT=Path(__file__).resolve().parent
CACHE=UNC/'decision_check/cache'
OUTPUT=ROOT/'data/experimental_snapshot.json'


def full_game_model():
    path=CACHE/'experimental_game_through_2026.json'
    if not path.exists():
        rows=pd.read_parquet(UNC/'cache/pregame.parquet')
        if rows.season.max()!=2026:raise ValueError('Experimental training cache has wrong boundary')
        flipped=rows.copy();flipped.d_strength=-flipped.d_strength;flipped.d_rest=-flipped.d_rest;flipped.home_court=1-flipped.home_court;flipped.y=1-flipped.y
        train=pd.concat([rows,flipped],ignore_index=True)
        model=XGBClassifier(**research.SPEC['game_model'],monotone_constraints=(1,1,0))
        model.fit(train[['d_strength','home_court','d_rest']],train.y)
        model.save_model(path)
    model=XGBClassifier();model.load_model(path)
    return GameProbabilities(model)


def current_games(snapshot):
    year=snapshot['season'];base='https://basketball-data.billmill.org/nba/stats'
    def read(name):
        r=requests.get(f'{base}/{name}/season={year}/data.parquet',timeout=30);r.raise_for_status()
        return pd.read_parquet(BytesIO(r.content))
    games=read('team_game_logs');identities=read('team_seasons')
    codes=identities.set_index('team_id').nba_abbrev.replace(research.ALIASES).to_dict()
    games=games.loc[games.game_id.str.startswith('002')].copy()
    games['team']=games.team_id.map(codes);games['opponent']=games.opp_team_id.map(codes)
    games['date']=pd.to_datetime(games.game_date)
    primary=pd.DataFrame(snapshot['teams']).set_index('team')
    limits=primary.cutoff.map(pd.Timestamp).to_dict()
    games=games.loc[[r.date<=limits.get(r.team,pd.Timestamp.min) for r in games.itertuples()]]
    for code,row in primary.iterrows():
        observed=games.loc[games.team.eq(code)]
        if len(observed)!=row.gp or observed.win.sum()!=row.wins:raise ValueError('Experimental inputs do not match the primary snapshot date/records')
    if not games.groupby('game_id').size().eq(2).all():raise ValueError('Partially aligned game')
    selected=games.home.eq(True)|(games.home.isna()&games.team.lt(games.opponent))
    games=games.loc[selected].rename(columns={'team':'a','opponent':'b','plus_minus':'mov'})
    games['home_court']=np.where(games.home.isna(),.5,1.)
    games['y']=games.mov.gt(0).astype(int);games['phase']='regular';games['season']=year
    return games.sort_values(['date','game_id'])


def live_inputs(primary):
    """Shared unchanged posterior, records and schedule for paired live scenarios."""
    games=current_games(primary);previous=research.HISTORY.loc[research.HISTORY.season.eq(2026)].set_index('team')
    codes=sorted(pd.DataFrame(primary['teams']).team)
    audit=pd.read_csv(UNC/'results/parameter_audit.csv').set_index('season').loc[2026]
    completed=research.load_games(2026);regular=completed.loc[completed.phase.eq('regular')]
    expected=regular.a.map(previous.srs)-regular.b.map(previous.srs)+3*(2*regular.home_court-1)
    variance=(audit.observation_sd**2*audit.past_noise_games+((regular.mov-expected)**2).sum())/(audit.past_noise_games+len(regular))
    rating=Ratings(codes,[previous.srs.get(c,0) for c in codes],research.prior_variance(2027),variance)
    rating.max_date=pd.Timestamp(previous.cutoff.max())
    for g in games.itertuples():rating.update(g.a,g.b,g.mov,g.home_court,g.date)
    cutoff=pd.Timestamp(primary['games_through'])+pd.Timedelta(days=1)
    state=rating.snapshot(cutoff)
    primary_teams=pd.DataFrame(primary['teams']).set_index('team').loc[codes].reset_index()
    played={(g.a,str(g.date.date())) for g in games.itertuples()}|{(g.b,str(g.date.date())) for g in games.itertuples()}
    pending=[];remaining=(82-primary_teams.gp.to_numpy()).astype(int)
    for i,game in enumerate(get_schedule(primary_teams)):
        if (game['home'],game['date']) in played and (game['away'],game['date']) in played:continue
        a,b=state['index'][game['home']],state['index'][game['away']]
        remaining[[a,b]]-=1
        pending.append(dict(game_id=str(i),season=2027,date=pd.Timestamp(game['date']),a=game['home'],b=game['away'],phase='regular',y=0,home_court=1.))
    if remaining.min()<0 or remaining.max()>4:raise ValueError('Incomplete experimental schedule')
    pending=pd.DataFrame(pending)
    fixture=research.features(state,pending)
    return state,primary_teams,fixture,remaining


def forecast(primary):
    if primary['season']!=2027:raise ValueError('Experimental view currently supports 2026–27')
    key={k:primary[k] for k in ['season','games_through','teams']}
    algorithm=[Path(__file__),UNC/'ratings.py',UNC/'probabilities.py',UNC/'spec.json',UNC/'decision_check/paths.py']
    key['algorithm_hash']=hashlib.sha256(b''.join(p.read_bytes() for p in algorithm)).hexdigest()
    signature=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    if OUTPUT.exists():
        saved=json.loads(OUTPUT.read_text())
        if saved.get('input_signature')==signature:return saved
    state,primary_teams,fixture,remaining=live_inputs(primary)
    probabilities=full_game_model();adapter=UncertainPaths(state,probabilities,10000,year=2027)
    result=season_paths(state,fixture,primary_teams.conference.tolist(),adapter.game_probability,adapter.series_probability,10000,year=2027,tbd=remaining)
    output=primary_teams[['team','name','conference','wins','losses','gp']].copy()
    for field,source in [('title_probability','title'),('playoff_probability','playoff'),('projected_wins','projected_wins'),('simulation_se','title_se')]:output[field]=result[source]
    output=output.sort_values('title_probability',ascending=False)
    payload=dict(season=2027,label='Experimental — uncertainty-aware; not promoted',games_through=primary['games_through'],
        computed_at=datetime.now(timezone.utc).isoformat(),input_signature=signature,training_through=2026,training_game_seasons=24,
        simulations=10000,unassigned_team_games=int(remaining.sum()),odds=json.loads(output.to_json(orient='records')))
    atomic_json(OUTPUT,payload)
    return payload

if __name__=='__main__':
    current=json.loads((ROOT/'data/live_snapshot.json').read_text())
    result=forecast(current)
    print(pd.DataFrame(result['odds'])[['name','title_probability']].head(6).to_string(index=False))
