"""Manual injury stress scenario on unchanged uncertainty-model draws; not calibrated odds."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from data_pipeline import DATA,TARGET,parse_players,name_key,team_key
from experimental_live import live_inputs,full_game_model,UncertainPaths,season_paths
from experiments.team_uncertainty.decision_check.common import best7
from live import atomic_json
from injury_live import fetch_report

OUTPUT=DATA/'injury_uncertainty_snapshot.json'
PATHS=10000


def current_report():
    cached=DATA/'injury_snapshot.json'
    if cached.exists():
        report=json.loads(cached.read_text())['report']
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(report['fetched_at'])).total_seconds()
        if 0<=age<1800:return report
    return fetch_report()


def roster_fragility(depth_bpm,top3_share):
    return float(1+.5*np.clip((1-depth_bpm)/4,0,1)+.5*top3_share)


def track_recovery(report):
    """Remember observed injury episodes after removal from the live report."""
    path=DATA/'raw/injury_recovery_history.json'
    saved=json.loads(path.read_text()) if path.exists() else {'season':TARGET,'episodes':{}}
    episodes=saved['episodes'] if saved.get('season')==TARGET else {}
    today=pd.Timestamp(report['fetched_at']).tz_convert('America/New_York').date().isoformat()
    observed={}
    for raw in report['reports']:
        if raw['status'].lower()!='out':continue
        key=f"{raw['team_name']}|{raw['player']}";old=episodes.get(key)
        first=(old.get('_first_seen') if old and not old.get('_recovery') else None) or (raw.get('reported_at') or today)[:10]
        observed[key]=dict(raw,_first_seen=first,_recovery=False)
    for key,old in episodes.items():
        if key in observed:continue
        # Disappearance means report-cleared, not a medically verified return.
        if not old.get('_recovery'):
            old=dict(old,_recovery=True,_recovery_return=today,status='Recovery (report cleared)')
        end=pd.Timestamp(old['_recovery_return'])
        if pd.Timestamp(today)<=end+pd.Timedelta(days=540):observed[key]=old
    atomic_json(path,{'season':TARGET,'episodes':observed,'checked_at':report['fetched_at']})
    return dict(report,reports=list(observed.values()))


def profiles(primary,report,return_overrides=None):
    current=parse_players(TARGET);previous=parse_players(TARGET-1)
    current['key']=current.player.map(name_key);previous['key']=previous.player.map(name_key)
    teams={team_key(t['name']):t for t in primary['teams']}
    team_profiles={}
    for team in primary['teams']:
        rows=[]
        for player in current.loc[current.team.eq(team['team'])].itertuples():
            prior=previous.loc[previous.key.eq(player.key)]
            same=prior.loc[prior.team.eq(team['team'])]
            prior=(same if len(same) else prior).nlargest(1,'mp')
            known=bool(len(prior) and prior.iloc[0].mp>=500)
            bpm=float(prior.iloc[0].bpm) if known else -2.
            prior_mpg=float(prior.iloc[0].mp/prior.iloc[0].gp) if known else player.mp/max(player.gp,1)
            mpg=min(40.,(20*prior_mpg+player.mp)/(20+max(team['gp'],1)))
            rows.append(dict(player=player.player,key=player.key,mpg=mpg,bpm=bpm,known=known))
        rows=sorted(rows,key=lambda r:r['mpg'],reverse=True)
        top=rows[:8];bench=top[3:]
        depth=float(np.average([r['bpm'] for r in bench],weights=[r['mpg'] for r in bench])) if bench else -2.
        talent=[max(r['bpm']+2,0)*r['mpg'] for r in top]
        concentration=sum(talent[:3])/sum(talent) if sum(talent)>0 else .375
        fragility=roster_fragility(depth,concentration)
        team_profiles[team['team']]=(rows,depth,concentration,float(fragility))
    now=pd.Timestamp(report['fetched_at']).tz_convert('America/New_York').tz_localize(None).normalize()
    output=[]
    for injury in report['reports']:
        team=teams.get(team_key(injury['team_name']))
        if team is None:continue
        rows,depth,concentration,fragility=team_profiles[team['team']]
        matches=[(i,r) for i,r in enumerate(rows) if r['key']==name_key(injury['player'])]
        row=dict(injury,team=team['team'],applied=False,base_penalty=0.,top3=False,depth_bpm=depth,top3_share=concentration,fragility=fragility)
        key=f"{team['team']}|{injury['player']}"
        end=pd.to_datetime((return_overrides or {}).get(key,injury.get('_recovery_return',injury['estimated_return'])),errors='coerce')
        duration_known=not pd.isna(end) and (end>now or injury.get('_recovery',False))
        if injury['season_out']:end=pd.Timestamp(f'{TARGET}-07-01');duration_known=True
        if not duration_known:end=now+pd.Timedelta(days=1)
        reported=pd.to_datetime(injury.get('_first_seen',injury.get('reported_at')),utc=True,errors='coerce')
        start=min(now,pd.Timestamp(injury['_first_seen']).normalize()) if injury.get('_first_seen') else (now if pd.isna(reported) else min(now,reported.tz_convert('America/New_York').tz_localize(None).normalize()))
        duration=max(1.,float((end-start).days))
        row.update(return_date=end.date().isoformat(),duration_days=duration,duration_known=duration_known,
            recovery_fraction=.65*min(duration/30,1.) if duration_known else 0.,manual_return=key in (return_overrides or {}))
        if matches:
            rank,player=matches[0];top3=rank<3
            value=max(player['bpm']+2,0)*player['mpg']/48
            base=value*(1.75 if top3 else 1.)*fragility
            row.update(top3=top3,role_rank=rank+1,projected_mpg=player['mpg'],prior_bpm=player['bpm'],
                base_penalty=float(base),applied=bool((injury['status'].lower()=='out' or injury.get('_recovery',False)) and player['known'] and base>0))
        output.append(row)
    return output


def penalty(row,date,recovery_days):
    """0.65 return-day impairment after a >=30-day absence, with a manual decay time."""
    date=pd.Timestamp(date).normalize();end=pd.Timestamp(row['return_date'])
    if date<end:return row['base_penalty']
    # Short rest absences recover quickly; long absences retain the chosen decay time.
    tau=max(1.,float(recovery_days)*min(row['duration_days']/60,1.))
    return row['base_penalty']*row['recovery_fraction']*np.exp(-(date-end).days/tau)


class InjuryPaths(UncertainPaths):
    def __init__(self,state,model,rows,severity=2.,recovery_days=90,team_multipliers=None,**kwargs):
        super().__init__(state,model,**kwargs)
        self.injuries=rows;self.index=state['index'];self.severity=float(severity);self.recovery_days=recovery_days
        self.multipliers=team_multipliers or {};self.date=pd.Timestamp(state['cutoff']);self.penalties={}
    def set_date(self,date):self.date=pd.Timestamp(date)
    def deductions(self,date):
        key=str(pd.Timestamp(date).date())
        if key not in self.penalties:
            values=np.zeros(len(self.index))
            for row in self.injuries:
                if row['applied']:
                    values[self.index[row['team']]]+=self.severity*self.multipliers.get(row['team'],1.)*penalty(row,date,self.recovery_days)
            self.penalties[key]=np.minimum(values,12.)  # Explicit scenario cap: points per game, not title %.
        return self.penalties[key]
    def dated_difference(self,a,b,date):
        minus=self.deductions(date)
        return super().difference(a,b)-minus[a]+minus[b]
    def game_probability(self,a,b,home,rest=0):
        return self.probabilities.conditional(self.dated_difference(a,b,self.date),home,rest)
    def series_probability(self,a,b,rnd,home_a):
        if self.severity==0:return super().series_probability(a,b,rnd,home_a)
        start=pd.Timestamp({1:f'{self.year}-04-20',2:f'{self.year}-05-06',3:f'{self.year}-05-22',4:f'{self.year}-06-06'}[rnd])
        pattern=[1,1,0,0,1,0,1]
        return best7([self.probabilities.conditional(self.dated_difference(a,b,start+pd.Timedelta(days=2*i)),np.where(home_a,float(h),1.-h)) for i,h in enumerate(pattern)])


def forecast(primary,report=None,severity=2.,recovery_days=90,team_multipliers=None,return_overrides=None):
    if not 0<=severity<=4 or not 1<=recovery_days<=180:raise ValueError('Scenario settings out of range')
    if any(not 0<=v<=3 for v in (team_multipliers or {}).values()):raise ValueError('Team multiplier out of range')
    report=track_recovery(report or current_report())
    settings=dict(severity=float(severity),recovery_days=int(recovery_days),team_multipliers=team_multipliers or {},return_overrides=return_overrides or {})
    engine=Path(__file__).parent/'experiments/team_uncertainty/decision_check/paths.py'
    version=hashlib.sha256(Path(__file__).read_bytes()+engine.read_bytes()+(Path(__file__).parent/'experimental_live.py').read_bytes()).hexdigest()
    signature=hashlib.sha256(json.dumps(dict(primary=primary,report=report,settings=settings,version=version),sort_keys=True).encode()).hexdigest()
    cache=DATA/'raw'/f'injury_uncertainty_{signature}.json'
    if cache.exists():
        payload=json.loads(cache.read_text());atomic_json(OUTPUT,payload);return payload
    state,teams,fixtures,remaining=live_inputs(primary);model=full_game_model()
    rows=profiles(primary,report,return_overrides)
    baseline=UncertainPaths(state,model,PATHS,year=TARGET)
    adjusted=InjuryPaths(state,model,rows,severity,recovery_days,team_multipliers,paths=PATHS,year=TARGET)
    arguments=dict(paths=PATHS,year=TARGET,tbd=remaining)
    original=season_paths(state,fixtures,teams.conference.tolist(),baseline.game_probability,baseline.series_probability,**arguments)
    result=season_paths(state,fixtures,teams.conference.tolist(),adjusted.game_probability,adjusted.series_probability,date_hook=adjusted.set_date,**arguments)
    output=teams[['team','name','conference','wins','losses','gp']].copy()
    for name,source in [('title_probability','title'),('playoff_probability','playoff'),('projected_wins','projected_wins'),('simulation_se','title_se')]:output[name]=result[source]
    output['uncertainty_title_probability']=original['title'];output['injury_change_pp']=100*(result['title']-original['title'])
    output['playoff_strength_penalty']=adjusted.deductions(f'{TARGET}-04-20')
    for row in rows:row['playoff_player_penalty']=float(severity*(team_multipliers or {}).get(row['team'],1.)*penalty(row,f'{TARGET}-04-20',recovery_days)) if row['applied'] else 0.
    payload=dict(season=TARGET,label='Experimental — uncertainty + injury stress',computed_at=datetime.now(timezone.utc).isoformat(),
        games_through=primary['games_through'],simulations=PATHS,settings=settings,report=report,adjustments=rows,input_signature=signature,
        odds=json.loads(output.sort_values('title_probability',ascending=False).to_json(orient='records')),
        assumptions='Manual stress scenario, not a calibrated injury model. Same uncertainty draws, game model, schedule and random seeds. Value = (prior BPM + 2, floored at 0) × projected MPG / 48; top-three role multiplier 1.75; fragility = 1 + 0.5 × weak-depth score + 0.5 × top-three talent share. Severity scales this value. Team deductions capped at 12 points. After return, impairment starts at up to 65% and decays exponentially; default long-absence decay is 90 days. Return estimates are not medical predictions. Role ranking uses projected minutes; unknown ratings/roster matches are omitted and unknown duration gets no recovery tail. Observed injury episodes are retained after report removal; report-cleared dates are approximate recovery starts, not verified appearances. Existing absences may already affect team strength, so double-counting is possible. All historical evaluation results remain for the original models only.')
    atomic_json(cache,payload);atomic_json(OUTPUT,payload);return payload

if __name__=='__main__':
    primary=json.loads((DATA/'live_snapshot.json').read_text());result=forecast(primary)
    print(pd.DataFrame(result['odds'])[['name','title_probability','uncertainty_title_probability','injury_change_pp','playoff_strength_penalty']].head(10).to_string(index=False))
    print(json.dumps(next(r for r in result['adjustments'] if r['player']=='Jalen Williams'),indent=2))
