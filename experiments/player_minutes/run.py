"""Leakage-audited, frozen-checkpoint player/minutes ablation. Production is never imported for writes."""
from collections import defaultdict,deque
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,brier_score_loss
from xgboost import XGBClassifier

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from data_pipeline import parse_players,name_key,team_key
from fetch import canonical
SPEC=json.loads((HERE/'spec.json').read_text())
CACHE=HERE/'cache'
RESULTS=HERE/'results'
RESULTS.mkdir(exist_ok=True)
BASE=SPEC['features_without']
ADDED=SPEC['features_with']
MODELS=['without_players','with_players','srs_game_logistic','current_srs_series_baseline']
HISTORY=pd.read_csv(ROOT/'data/teams.csv')
LABELS=pd.read_csv(ROOT/'data/series.csv')
SERIES_ROWS=pd.read_csv(ROOT/'data/matchups.csv')


def load_games(year):
    logs=pd.read_parquet(CACHE/f'{year}.parquet')
    logs['date']=pd.to_datetime(logs.date)
    teams=logs.loc[logs.player.eq('')].copy()
    if teams[['team','opponent','mov']].isna().any().any():
        raise ValueError(f'{year}: missing game identities, venues or scores')
    sizes=teams.groupby('game_id').size()
    if not sizes.eq(2).all():raise ValueError(f'{year}: incomplete two-team game logs')
    valid=teams.groupby('game_id').home.apply(lambda h: h.isna().all() or (not h.isna().any() and h.sum()==1))
    if not valid.all():raise ValueError(f'{year}: invalid home flags')
    # Source explicitly uses NULL for neutral-site games: choose a canonical orientation.
    neutral=teams.home.isna()
    chosen=teams.home.eq(True) | (neutral & teams.team.lt(teams.opponent))
    games=teams.loc[chosen].copy().rename(columns={'team':'a','opponent':'b'})
    games['y']=games.mov.gt(0).astype(int)
    games['home_court']=np.where(games.home.isna() | ((year==2020)&games.date.ge('2020-07-30')),.5,1.)
    games=games.sort_values(['date','game_id']).reset_index(drop=True)
    players=logs.loc[logs.player.ne('')&logs.minutes.gt(0)].copy()
    players['key']=players.player.map(name_key)
    players=players.groupby(['game_id','team','key'],as_index=False).agg(minutes=('minutes','sum'))
    expected=HISTORY.loc[HISTORY.season.eq(year)].set_index('team').gp
    counts=teams.loc[teams.phase.eq('regular')].groupby('team').size()
    for code,gp in expected.items():
        if counts.get(canonical(code),0)!=gp:raise ValueError(f'{year}: incomplete regular-season logs for {code}')
    return games,players


class State:
    def __init__(self,year):
        self.year=year
        current=HISTORY.loc[HISTORY.season.eq(year)]
        self.codes=sorted(current.team.map(canonical))
        self.index={code:i for i,code in enumerate(self.codes)}
        prior=HISTORY.loc[HISTORY.season.eq(year-1)].copy()
        prior['team']=prior.team.map(canonical)
        self.prior_srs=np.array([prior.set_index('team').srs.get(code,0.) for code in self.codes])
        self.matrix=20*np.eye(len(self.codes))
        self.rhs=20*self.prior_srs.copy()
        self.wins=np.zeros(len(self.codes));self.gp=np.zeros(len(self.codes))
        self.last={code:pd.NaT for code in self.codes}
        self.recent={code:deque(maxlen=5) for code in self.codes}
        self.roster={code:set() for code in self.codes}
        self.assigned={};self.prior_mpg={};self.ratings={}
        p=parse_players(year-1).copy()
        p['key']=p.player.map(name_key);p['team']=p.team.map(canonical)
        p=p.loc[p.mp.gt(0)]
        p['weighted']=p.bpm.fillna(-2)*p.mp
        total=p.groupby('key').agg(mp=('mp','sum'),weighted=('weighted','sum'))
        self.ratings=((total.weighted-1000)/(total.mp+500)).to_dict()
        self.prior_mpg={(r.team,r.key):min(r.mp/r.gp,48.) for r in p.itertuples() if r.gp>0}
        previous=pd.read_parquet(ROOT/'data/raw/logs'/f'{year-1}.parquet')
        previous=previous.loc[previous.player.ne('')&previous.minutes.gt(0)].copy()
        previous['key']=previous.player.map(name_key)
        names={team_key(r.name):canonical(r.team) for r in prior.itertuples()}
        previous['code']=previous.team.map(team_key).map(names)
        previous=previous.sort_values(['date','game_id']).drop_duplicates('key',keep='last')
        for r in previous.itertuples():
            if r.code in self.roster:
                self.roster[r.code].add(r.key);self.assigned[r.key]=r.code
        self.max_date=pd.to_datetime(prior.cutoff.max())

    def snapshot(self,cutoff):
        cutoff=pd.Timestamp(cutoff)
        if not self.max_date<cutoff:raise AssertionError('Snapshot contains same-day/future results')
        strength=np.linalg.solve(self.matrix,self.rhs)
        projected=np.zeros(len(self.codes));coverage=np.zeros(len(self.codes));audit=[]
        for team,i in self.index.items():
            recent=self.recent[team]
            missing=5-len(recent)
            minutes={key:(sum(g.get(key,0.) for g in recent)+missing*self.prior_mpg.get((team,key),0.))/5
                     for key in self.roster[team]}
            total=sum(minutes.values())
            for key,minute in minutes.items():
                if minute<=0:continue
                minute=240*minute/total
                rating=self.ratings.get(key,-2.)
                projected[i]+=minute*rating/48
                coverage[i]+=minute*(key in self.ratings)/240
                audit.append(dict(team=team,player=key,projected_minutes=minute,rating=rating,
                    rating_season=self.year-1,known_rating=key in self.ratings))
        return dict(cutoff=cutoff,max_input_date=self.max_date,codes=self.codes,index=self.index,
            srs=strength,player=projected,coverage=coverage,wins=self.wins.copy(),gp=self.gp.copy(),
            last=self.last.copy(),audit=audit)

    def observe(self,games,players):
        # Caller invokes this only AFTER every feature row/checkpoint on this date is constructed.
        for r in players.itertuples():
            old=self.assigned.get(r.key)
            if old!=r.team:
                if old in self.roster:self.roster[old].discard(r.key)
                self.roster[r.team].add(r.key);self.assigned[r.key]=r.team
        groups={(gid,team):dict(zip(g.key,g.minutes)) for (gid,team),g in players.groupby(['game_id','team'])}
        for g in games.itertuples():
            i,j=self.index[g.a],self.index[g.b]
            if g.phase=='regular':
                x=np.zeros(len(self.codes));x[i]=1;x[j]=-1
                self.matrix+=np.outer(x,x)
                self.rhs+=x*(g.mov-3*(2*g.home_court-1))
                self.gp[[i,j]]+=1;self.wins[i]+=g.y;self.wins[j]+=1-g.y
            for code in [g.a,g.b]:
                self.recent[code].append(groups.get((g.game_id,code),{}))
                self.last[code]=g.date
            self.max_date=max(self.max_date,g.date)


def fixtures(snapshot,games):
    rows=[];std=float(np.std(snapshot['srs'])) or 1.
    for g in games.itertuples():
        i,j=snapshot['index'][g.a],snapshot['index'][g.b]
        def rest(team):
            last=snapshot['last'][team]
            return 7 if pd.isna(last) else float(np.clip((g.date-last).days-1,0,7))
        rows.append(dict(game_id=g.game_id,season=g.season,date=g.date,phase=g.phase,y=g.y,
            d_team_strength=snapshot['srs'][i]-snapshot['srs'][j],
            d_srs_z=(snapshot['srs'][i]-snapshot['srs'][j])/std,
            d_player_strength=snapshot['player'][i]-snapshot['player'][j],
            home_court=g.home_court,d_rest=rest(g.a)-rest(g.b),
            cutoff=snapshot['cutoff'],max_input_date=snapshot['max_input_date']))
    return pd.DataFrame(rows)


def reverse(frame):
    other=frame.copy()
    for col in ['d_team_strength','d_srs_z','d_player_strength','d_rest']:
        if col in other:other[col]=-other[col]
    other['home_court']=1-other.home_court
    if 'y' in other:other['y']=1-other.y
    return other


def build():
    target=CACHE/'pregame.parquet'
    if target.exists():return pd.read_parquet(target)
    all_rows=[];audit=[];snapshots={};counts=[]
    for year in range(2003,2027):
        games,players=load_games(year);games['season']=year
        state=State(year)
        first=games.loc[games.phase.eq('regular'),'date'].min()
        end=games.loc[games.phase.eq('regular'),'date'].max()+pd.Timedelta(days=1)
        checkpoints={'october':pd.Timestamp(year=year-1,month=11,day=1),
                     'january':pd.Timestamp(year=year,month=1,day=16),'pre_playoff':end}
        checkpoints={label:date for label,date in checkpoints.items() if first<date}
        pending=dict(checkpoints)
        by_game={gid:g for gid,g in players.groupby('game_id')}
        for date,day in games.groupby('date',sort=True):
            for label,cutoff in list(pending.items()):
                if cutoff<=date:
                    snap=state.snapshot(cutoff);snapshots[(year,label)]=snap
                    audit.extend(dict(season=year,checkpoint=label,cutoff=str(cutoff.date()),
                        max_input_date=str(snap['max_input_date'].date()),**r) for r in snap['audit'])
                    del pending[label]
            snap=state.snapshot(date)
            all_rows.append(fixtures(snap,day))
            appearances=pd.concat([by_game[gid] for gid in day.game_id if gid in by_game],ignore_index=True)
            state.observe(day,appearances)
        if pending:raise ValueError(f'{year}: missing checkpoints {pending}')
        # Avoid pickle: snapshots consist of auditable numeric arrays and dated identities.
        serial={}
        for (yr,label),snap in snapshots.items():
            if yr!=year:continue
            serial[label]={k:(v.tolist() if isinstance(v,np.ndarray) else
                {t:None if pd.isna(d) else str(d.date()) for t,d in v.items()} if k=='last' else
                str(v.date()) if isinstance(v,pd.Timestamp) else v)
                for k,v in snap.items() if k!='audit'}
        (CACHE/f'snapshots_{year}.json').write_text(json.dumps(serial))
        counts.append(dict(season=year,games=len(games),checkpoints=list(checkpoints)))
        print('Snapshots built',year,len(games),flush=True)
    result=pd.concat(all_rows,ignore_index=True)
    if not pd.to_datetime(result.max_input_date).lt(pd.to_datetime(result.date)).all():raise AssertionError('Leaked pregame data')
    result.to_parquet(target,index=False)
    pd.DataFrame(audit).to_csv(RESULTS/'projected_minutes_audit.csv.gz',index=False)
    (RESULTS/'coverage.json').write_text(json.dumps(counts,indent=2))
    return result


def checkpoint(year,label):
    snap=json.loads((CACHE/f'snapshots_{year}.json').read_text())[label]
    for key in ['srs','player','coverage','wins','gp']:snap[key]=np.asarray(snap[key])
    for key in ['cutoff','max_input_date']:snap[key]=pd.Timestamp(snap[key])
    snap['last']={team:pd.NaT if date is None else pd.Timestamp(date) for team,date in snap['last'].items()}
    return snap


def fit_models(training,year):
    paired=pd.concat([training,reverse(training)],ignore_index=True)
    fitted={}
    for name,features in [('without_players',BASE),('with_players',ADDED)]:
        m=XGBClassifier(**SPEC['xgboost'],monotone_constraints=tuple(0 if f=='d_rest' else 1 for f in features))
        m.fit(paired[features],paired.y);fitted[name]=(m,features)
    m=LogisticRegression(C=1,random_state=0,max_iter=500).fit(paired[['d_team_strength','home_court']],paired.y)
    fitted['srs_game_logistic']=(m,['d_team_strength','home_court'])
    prior=SERIES_ROWS.loc[SERIES_ROWS.season.lt(year)]
    m=LogisticRegression(C=1,random_state=0).fit(prior[['d_srs']],prior.y)
    fitted['current_srs_series_baseline']=(m,['d_srs'])
    return fitted


def predict(fitted,frame):
    result={}
    combined=pd.concat([frame,reverse(frame)],ignore_index=True)
    for name,(model,features) in fitted.items():
        x=combined.copy()
        if name=='current_srs_series_baseline':x['d_srs']=x.d_srs_z
        raw=model.predict_proba(x[features])[:,1]
        probability=.5*(raw[:len(frame)]+1-raw[len(frame):])
        if name=='current_srs_series_baseline':
            grid=np.linspace(0,1,10001)
            probability=np.interp(probability,35*grid**4-84*grid**5+70*grid**6-20*grid**7,grid)
        result[name]=np.clip(probability,1e-6,1-1e-6)
    return result


def series_probability(probabilities):
    dp=np.zeros((5,5));dp[0,0]=1
    for won in range(4):
        for lost in range(4):
            p=probabilities[won+lost]
            dp[won+1,lost]+=dp[won,lost]*p
            dp[won,lost+1]+=dp[won,lost]*(1-p)
    return float(dp[4,:4].sum())


def series_forecasts(year,label,snap,games,fitted):
    result=[]
    for s in LABELS.loc[LABELS.season.eq(year)].itertuples():
        a,b=sorted([canonical(s.winner),canonical(s.loser)])
        actual=games.loc[games.phase.eq('playoff') & games.a.isin([a,b]) & games.b.isin([a,b])]
        if actual.empty:raise ValueError(f'Missing playoff games for {year}: {a}-{b}')
        start=actual.date.min()
        if start<snap['cutoff']:continue
        i,j=snap['index'][a],snap['index'][b]
        if label=='pre_playoff':ranking=snap['wins']/np.maximum(snap['gp'],1)
        else:
            prior=1/(1+np.exp(-snap['srs']/6))
            ranking=(snap['wins']+20*prior)/(snap['gp']+20)
        home=a if (ranking[i],snap['srs'][i],a)>(ranking[j],snap['srs'][j],b) else b
        pattern=[1,1,0,0,0,1,1] if year<=2013 and s.round==4 else [1,1,0,0,1,0,1]
        neutral=year==2020
        std=float(np.std(snap['srs'])) or 1.
        rows=pd.DataFrame([dict(d_team_strength=snap['srs'][i]-snap['srs'][j],
            d_player_strength=snap['player'][i]-snap['player'][j],
            d_srs_z=(snap['srs'][i]-snap['srs'][j])/std,d_rest=0.,
            home_court=.5 if neutral else float((bool(h) and home==a) or (not h and home==b))) for h in pattern])
        for name,p in predict(fitted,rows).items():
            result.append(dict(season=year,checkpoint=label,kind='series',phase='playoff',
                event=f'{year}-{s.round}-{a}-{b}',model=name,y=int(a==canonical(s.winner)),
                probability=series_probability(p),cutoff=str(snap['cutoff'].date())))
    return result


def calibration(frame):
    rows=[]
    bins=np.minimum((frame.probability.to_numpy()*10).astype(int),9)
    for i in range(10):
        group=frame.loc[bins==i]
        rows.append(dict(lower=i/10,upper=(i+1)/10,n=len(group),
            mean_prediction=None if group.empty else float(group.probability.mean()),
            win_rate=None if group.empty else float(group.y.mean())))
    return rows


def score(frame):
    bins=calibration(frame)
    ece=sum(r['n']*abs(r['mean_prediction']-r['win_rate']) for r in bins if r['n'])/len(frame)
    return dict(n=len(frame),log_loss=float(frame.loss.mean()),brier=float(frame.brier.mean()),ece=ece)


def paired_bootstrap(frame):
    a=frame.loc[frame.model.eq('without_players')]
    b=frame.loc[frame.model.eq('with_players')]
    pair=a.merge(b,on=['season','checkpoint','kind','event'],suffixes=('_base','_player'),validate='one_to_one')
    pair['loss_delta']=pair.loss_player-pair.loss_base
    pair['brier_delta']=pair.brier_player-pair.brier_base
    annual=pair.groupby('season').agg(n=('loss_delta','size'),loss=('loss_delta','sum'),brier=('brier_delta','sum'))
    rng=np.random.default_rng(0)
    sampled=rng.integers(0,len(annual),(2000,len(annual)))
    n=annual.n.to_numpy()[sampled].sum(axis=1)
    losses=annual.loss.to_numpy()[sampled].sum(axis=1)/n
    briers=annual.brier.to_numpy()[sampled].sum(axis=1)/n
    return dict(log_loss_delta=float(pair.loss_delta.mean()),brier_delta=float(pair.brier_delta.mean()),
        log_loss_delta_95=np.quantile(losses,[.025,.975]).tolist(),
        brier_delta_95=np.quantile(briers,[.025,.975]).tolist(),
        seasons_improved=int((annual.loss<0).sum()),seasons=int(len(annual)))


def run():
    protected=[ROOT/p for p in ['app.py','live.py','model.py','train.py','data_pipeline.py','artifacts/xgboost.json','artifacts/evaluation.json']]
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    rows=build();forecasts=[];folds=[];coverage=[]
    for year in range(2013,2027):
        training=rows.loc[rows.season.lt(year)]
        fitted=fit_models(training,year)
        games,_=load_games(year);games['season']=year
        labels=json.loads((CACHE/f'snapshots_{year}.json').read_text())
        for label in labels:
            snap=checkpoint(year,label)
            future=games.loc[games.date.ge(snap['cutoff'])]
            frame=fixtures(snap,future)
            for name,p in predict(fitted,frame).items():
                for g,prob in zip(frame.itertuples(),p):
                    forecasts.append(dict(season=year,checkpoint=label,kind='game',phase=g.phase,
                        event=g.game_id,model=name,y=g.y,probability=float(prob),cutoff=str(snap['cutoff'].date())))
            forecasts.extend(series_forecasts(year,label,snap,games,fitted))
            coverage.append(dict(season=year,checkpoint=label,cutoff=str(snap['cutoff'].date()),
                max_input_date=str(snap['max_input_date'].date()),mean_known_rating_minutes=float(snap['coverage'].mean())))
        folds.append(dict(test_season=year,train_first=int(training.season.min()),train_last=int(training.season.max()),train_games=len(training)))
        print('Evaluated',year,'training games',len(training),'checkpoints',list(labels),flush=True)
    frame=pd.DataFrame(forecasts)
    frame['probability']=frame.probability.clip(1e-6,1-1e-6)
    frame['loss']=-frame.y*np.log(frame.probability)-(1-frame.y)*np.log(1-frame.probability)
    frame['brier']=(frame.probability-frame.y)**2
    frame.to_csv(RESULTS/'predictions.csv.gz',index=False)
    tables=[]
    for columns,scope in [(['kind','model'],'overall'),(['kind','checkpoint','model'],'checkpoint'),(['kind','season','model'],'season')]:
        for values,g in frame.groupby(columns):
            tables.append(dict(scope=scope,**dict(zip(columns,values)),**score(g)))
    metrics=pd.DataFrame(tables);metrics.to_csv(RESULTS/'metrics.csv',index=False)
    calibration_rows=[]
    for (kind,point,model),g in frame.groupby(['kind','checkpoint','model']):
        calibration_rows.extend(dict(kind=kind,checkpoint=point,model=model,**b) for b in calibration(g))
    pd.DataFrame(calibration_rows).to_csv(RESULTS/'calibration.csv',index=False)
    comparisons={kind:paired_bootstrap(g) for kind,g in frame.groupby('kind')}
    game=metrics.loc[metrics.scope.eq('overall')&metrics.kind.eq('game')].set_index('model')
    point=metrics.loc[metrics.scope.eq('checkpoint')&metrics.kind.eq('game')]
    point=point.pivot(index='checkpoint',columns='model',values='log_loss')
    improved=int((point.with_players<point.without_players).sum())
    rule=dict(game_logloss_significant=comparisons['game']['log_loss_delta_95'][1]<0,
        game_brier_improves=comparisons['game']['brier_delta']<0,
        series_logloss_improves=comparisons['series']['log_loss_delta']<0,
        series_brier_improves=comparisons['series']['brier_delta']<0,
        game_season_consistency=comparisons['game']['seasons_improved']>=math.ceil(2*comparisons['game']['seasons']/3),
        series_season_consistency=comparisons['series']['seasons_improved']>=math.ceil(2*comparisons['series']['seasons']/3),
        checkpoint_consistency=improved>=2,
        calibration_not_materially_worse=float(game.loc['with_players','ece'])<=float(game.loc['without_players','ece'])+.005)
    after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    if before!=after:raise AssertionError('Production files changed')
    report=dict(spec_sha256=hashlib.sha256((HERE/'spec.json').read_bytes()).hexdigest(),folds=folds,
        decision='KEEP_CANDIDATE' if all(rule.values()) else 'DO_NOT_PROMOTE',promotion_checks=rule,
        comparisons=comparisons,checkpoint_types_improved=improved,coverage=coverage,
        production_unchanged=True,production_hashes=after,
        limitations=['Prior-season BPM proxy, not EPM; no injury/transaction archives.',
            'Rookies/trades enter rotations only after observed appearances; prior rosters seed opening games.',
            'Frozen forecasts are conditional on actual future fixtures/venues and actual playoff matchups; archived schedules are unavailable, especially COVID rescheduling.',
            'Series home court is projected from snapshot information, not eventual venues; bubble is evaluated conditional on neutral venues.',
            'Future rest information is not updated after snapshot; possible series games use equal rest.',
            'Repeated game forecasts across checkpoints are dependent; intervals cluster the whole season.',
            'Final-season BPM is recomputed by the modern source; prior-season data availability is reconstructed, not a vendor vintage archive.'])
    (RESULTS/'report.json').write_text(json.dumps(report,indent=2))
    (RESULTS/'summary.md').write_text(summary(metrics,report))
    print(json.dumps(dict(decision=report['decision'],comparisons=comparisons,promotion_checks=rule),indent=2),flush=True)


def summary(metrics,report):
    lines=['# Offline player/minutes experiment','',
        '**Decision: '+report['decision']+'**. Production app/model files are unchanged.','',
        '24 historical seasons (2002–03 through 2025–26); first ten train-only; 14 walk-forward test seasons (2012–13 through 2025–26). October 2020 is skipped because the 2020–21 season had not begun. Frozen checkpoint forecasts score subsequent games and conditional playoff matchups.','',
        '| Target | Model | Forecasts | Log loss | Brier | Calibration ECE |','|---|---|---:|---:|---:|---:|']
    for r in metrics.loc[metrics.scope.eq('overall')].itertuples():
        lines.append(f'| {r.kind} | {r.model} | {r.n} | {r.log_loss:.5f} | {r.brier:.5f} | {r.ece:.5f} |')
    lines+=['','Player feature minus otherwise identical model (negative is better):']
    for kind,r in report['comparisons'].items():
        lo,hi=r['log_loss_delta_95']
        lines.append(f"- {kind}: log-loss delta {r['log_loss_delta']:+.5f}, whole-season 95% CI [{lo:+.5f}, {hi:+.5f}]; Brier delta {r['brier_delta']:+.5f}; log loss improved in {r['seasons_improved']}/{r['seasons']} seasons.")
    lines+=['','## Checkpoint log loss','', '| Target | Checkpoint | Without players | With players | Delta |', '|---|---|---:|---:|---:|']
    points=metrics.loc[metrics.scope.eq('checkpoint')].pivot(index=['kind','checkpoint'],columns='model',values='log_loss')
    for (kind,point),row in points.iterrows():
        lines.append(f"| {kind} | {point} | {row.without_players:.5f} | {row.with_players:.5f} | {row.with_players-row.without_players:+.5f} |")
    lines+=['','## Preregistered checks']+[f'- {name}: {passed}' for name,passed in report['promotion_checks'].items()]
    lines+=['','## Limits']+['- '+text for text in report['limitations']]
    lines+=['','Detailed results: `metrics.csv`, `calibration.csv`, `predictions.csv.gz`, `projected_minutes_audit.csv.gz`, `report.json`. Fixed setup: `../spec.json`. No tuning or production promotion after seeing test results.','',
        'Methodological inspiration: [Dunks & Threes EPM methodology](https://dunksandthrees.com/about/epm). This experiment uses prior-season BPM, not their proprietary EPM or injury-aware minutes forecasts.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':run()
