"""Unvalidated, dated injury scenario over the established production XGBoost model."""
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier
from data_pipeline import DATA,TARGET,parse_players,name_key,team_key,standardize
from live import blend_strength,simulate,probability_tables,get_schedule,atomic_json,ARTIFACTS

URL='https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries'
OUTPUT=DATA/'injury_snapshot.json'
REPLACEMENT_BPM=0.0  # Conservative average-player replacement; no fitted injury coefficients.


def fetch_report():
    response=requests.get(URL,timeout=30);response.raise_for_status();source=response.json()
    now=pd.Timestamp.now(tz='UTC')
    stamp=pd.Timestamp(source['timestamp'])
    if stamp.tzinfo is None:stamp=stamp.tz_localize('UTC')
    if abs((now-stamp).total_seconds())>86400:raise ValueError('Injury feed timestamp is older than 24 hours or in the future')
    if source.get('season',{}).get('year')!=TARGET:raise ValueError('Injury report is for the wrong season')
    rows=[]
    for team in source['injuries']:
        for injury in team.get('injuries',[]):
            rows.append(dict(team_name=team['displayName'],player=injury['athlete']['displayName'],
                status=injury['status'],reported_at=injury.get('date'),
                estimated_return=injury.get('details',{}).get('returnDate'),
                reason=injury.get('shortComment',''),context=injury.get('longComment',''),
                season_out=injury.get('details',{}).get('fantasyStatus',{}).get('abbreviation')=='OFS'))
    return dict(source=URL,source_timestamp=source['timestamp'],fetched_at=now.isoformat(),reports=rows)


def build_adjustments(primary,report):
    """Prior-season BPM, projected MPG; never infer recovery duration from injury wording."""
    current=parse_players(TARGET);prior=parse_players(TARGET-1)
    identities={team_key(t['name']):t for t in primary['teams']}
    now=pd.Timestamp(report['fetched_at']).tz_convert('America/New_York').tz_localize(None).normalize()
    rows=[]
    for item in report['reports']:
        item=item.copy();team=identities.get(team_key(item['team_name']))
        if team is None:continue
        item['team']=team['team'];key=name_key(item['player'])
        active=current.loc[current.team.eq(team['team']) & current.player.map(name_key).eq(key)]
        previous=prior.loc[prior.player.map(name_key).eq(key)]
        same=previous.loc[previous.team.eq(team['team'])]
        previous=(same if len(same) else previous).nlargest(1,'mp')
        item.update(applied=False,strength_penalty=0.,projected_mpg=0.,prior_bpm=None)
        # Only a matched current roster player with an established historical rating is adjusted.
        if len(active) and len(previous) and previous.iloc[0].mp>=500 and item['status'].lower()=='out':
            old=previous.iloc[0];new=active.iloc[0]
            mpg=(20*(old.mp/old.gp)+new.mp)/(20+max(team['gp'],1))
            bpm=float(old.bpm);delta=max(bpm-REPLACEMENT_BPM,0)*min(mpg,40)/48
            end=pd.to_datetime(item['estimated_return'],errors='coerce')
            uncertain=False
            if item['season_out']:end=pd.Timestamp(f'{TARGET}-07-01')
            elif pd.isna(end) or end<=now:
                end=now+pd.Timedelta(days=1);uncertain=True
            item.update(applied=bool(delta>0),strength_penalty=float(delta),projected_mpg=float(mpg),prior_bpm=bpm,
                active_until=end.date().isoformat(),unknown_duration=uncertain)
        rows.append(item)
    return rows


class InjuryTables:
    def __init__(self,base,model,features,adjustments):
        self.base=base;self.model=model;self.features=features;self.adjustments=adjustments;self.cache={}
    def __call__(self,date):
        active=tuple(i for i,r in enumerate(self.adjustments) if r['applied'] and date<r['active_until'])
        if active not in self.cache:
            state=self.base.copy()
            for i in active:
                row=self.adjustments[i];mask=state.team.eq(row['team'])
                state.loc[mask,'srs']-=row['strength_penalty']
            # Reuse exactly the native model features; no invented injury-trained coefficients.
            self.cache[active]=probability_tables(self.model,self.features,standardize(state))
        return self.cache[active]


def forecast(primary,force=False):
    signature=hashlib.sha256((json.dumps(primary,sort_keys=True)+__import__('pathlib').Path(__file__).read_text()).encode()).hexdigest()
    if OUTPUT.exists() and not force:
        saved=json.loads(OUTPUT.read_text())
        if saved.get('input_signature')==signature and (datetime.now(timezone.utc)-datetime.fromisoformat(saved['computed_at'])).total_seconds()<1800:return saved
    report=fetch_report();adjustments=build_adjustments(primary,report)
    current=pd.DataFrame(primary['teams']);previous=pd.read_csv(DATA/'teams.csv');previous=previous.loc[previous.season.eq(TARGET-1)]
    strength=blend_strength(current,previous)
    evaluation=json.loads((ARTIFACTS/'evaluation.json').read_text());model=XGBClassifier();model.load_model(ARTIFACTS/'xgboost.json')
    tables=InjuryTables(strength,model,evaluation['features'],adjustments)
    logs=pd.read_parquet(DATA/'raw/logs'/f'{TARGET}.parquet');codes={team_key(t.name):t.team for t in current.itertuples()}
    played={(codes[team_key(g.team)],str(g.date)[:10]) for g in logs.loc[logs.player.eq('')].itertuples()}
    # Identical schedule, seed, model and 50k paths; only dated strength penalties differ.
    odds,unassigned=simulate(strength,model,evaluation['features'],get_schedule(current),played,dated_tables=tables)
    baseline={r['team']:r['title_probability'] for r in primary['odds']}
    odds['production_title_probability']=odds.team.map(baseline)
    odds['injury_change_pp']=100*(odds.title_probability-odds.production_title_probability)
    payload=dict(season=TARGET,label='Experimental — injury-adjusted',computed_at=datetime.now(timezone.utc).isoformat(),
        games_through=primary['games_through'],input_signature=signature,report=report,adjustments=adjustments,
        simulations=50000,unassigned_team_games=unassigned,odds=json.loads(odds.to_json(orient='records')),
        assumptions='Unvalidated scenario: prior-season BPM above 0 × projected MPG / 48 deducted from SRS until ESPN estimated return. Unknown duration: one day only. Estimated returns are not confirmed medical timelines. Approximate playoff-round dates; no recovery ramp. Season-long stats may already reflect absences; penalties can double-count that effect.')
    atomic_json(OUTPUT,payload);return payload

if __name__=='__main__':
    primary=json.loads((DATA/'live_snapshot.json').read_text());result=forecast(primary,force=True)
    print(pd.DataFrame(result['odds'])[['name','title_probability','production_title_probability','injury_change_pp']].head(10).to_string(index=False))
    for row in result['adjustments']:
        if row['team']=='OKC':print(json.dumps(row))
