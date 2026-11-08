"""Team-only, season-walk-forward uncertainty ablation. Writes only this experiment's files."""
from pathlib import Path
import hashlib
from functools import lru_cache
import json
import math
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from ratings import Ratings
from probabilities import GameProbabilities,FEATURES
from simulator import simulate

SPEC=json.loads((HERE/'spec.json').read_text())
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
CACHE=HERE/'cache';CACHE.mkdir(exist_ok=True)
TEAM_SOURCE=ROOT/'experiments/player_minutes/cache'
ALIASES={'NJN':'BRK','NOH':'NOP','NOK':'NOP','SEA':'OKC','CHA':'CHO','BKN':'BRK','PHX':'PHO'}


@lru_cache(maxsize=1)
def load_history():
    path=ROOT/'data/teams.csv'
    if not path.exists():
        raise FileNotFoundError('Local historical data is absent; see docs/SETUP.md. No NBA data is bundled.')
    frame=pd.read_csv(path,usecols=['season','team','name','conference','srs','gp','cutoff'])
    frame['team']=frame.team.replace(ALIASES)
    return frame

@lru_cache(maxsize=1)
def load_labels():
    return pd.read_csv(ROOT/'data/series.csv')

def __getattr__(name):
    if name=='HISTORY':return load_history()
    if name=='LABELS':return load_labels()
    raise AttributeError(name)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_games(year):
    columns=['game_id','date','team','opponent','home','mov','phase']
    rows=pd.read_parquet(TEAM_SOURCE/f'{year}.parquet',columns=columns,filters=[('player','=','')])
    rows['date']=pd.to_datetime(rows.date)
    if not rows.groupby('game_id').size().eq(2).all():raise ValueError('Incomplete games')
    valid=rows.groupby('game_id').home.apply(lambda h:h.isna().all() or (not h.isna().any() and h.sum()==1))
    if not valid.all():raise ValueError('Invalid home flags')
    chosen=rows.home.eq(True)|(rows.home.isna()&rows.team.lt(rows.opponent))
    games=rows.loc[chosen].rename(columns={'team':'a','opponent':'b'}).copy()
    games['home_court']=np.where(games.home.isna()|((year==2020)&games.date.ge('2020-07-30')),.5,1.)
    games['y']=games.mov.gt(0).astype(int);games['season']=year
    counts=rows.loc[rows.phase.eq('regular')].groupby('team').size()
    expected=load_history().loc[HISTORY.season.eq(year)].set_index('team').gp
    if not counts.reindex(expected.index).eq(expected).all():raise ValueError('Incomplete regular season')
    return games.sort_values(['date','game_id']).reset_index(drop=True)


def prior_variance(year):
    earlier=load_history().loc[HISTORY.season.lt(year),['season','team','srs']].sort_values(['team','season'])
    earlier['change']=earlier.groupby('team').srs.diff()
    values=earlier.change.dropna().to_numpy()
    return max(float(np.mean(values**2)),1.) if len(values) else 16.


def features(snapshot,games):
    i=np.array([snapshot['index'][c] for c in games.a]);j=np.array([snapshot['index'][c] for c in games.b])
    frame=games[['game_id','season','date','a','b','phase','y','home_court']].copy()
    frame['d_strength']=snapshot['mean'][i]-snapshot['mean'][j]
    p=snapshot['covariance']
    frame['d_variance']=np.maximum(p[i,i]+p[j,j]-2*p[i,j],0)
    def rest(team,date):
        last=snapshot['last'][team]
        return 7 if pd.isna(last) else int(np.clip((date-last).days-1,0,7))
    frame['d_rest']=[rest(a,d)-rest(b,d) for a,b,d in zip(games.a,games.b,games.date)]
    frame['cutoff']=snapshot['cutoff'];frame['max_input_date']=snapshot['max_input_date']
    return frame


def build():
    snapshots={};all_rows=[];parameter_audit=[];range_rows=[];noise_sse=0.;noise_n=0
    for year in range(2003,2027):
        games=load_games(year)
        previous=load_history().loc[HISTORY.season.eq(year-1)].set_index('team')
        current=load_history().loc[HISTORY.season.eq(year)].set_index('team')
        codes=sorted(current.index)
        mean=np.array([previous.srs.get(code,0.) for code in codes])
        variance=prior_variance(year);observation=max(noise_sse/noise_n,1.) if noise_n else 144.
        state=Ratings(codes,mean,variance,observation)
        state.max_date=pd.Timestamp(previous.cutoff.max())
        start=games.loc[games.phase.eq('regular'),'date'].min()
        end=games.loc[games.phase.eq('regular'),'date'].max()+pd.Timedelta(days=1)
        points={'october':pd.Timestamp(year=year-1,month=11,day=1),'january':pd.Timestamp(year=year,month=1,day=16),'pre_playoff':end}
        pending={point:date for point,date in points.items() if start<date}
        parameter_audit.append(dict(season=year,learned_through=year-1,prior_sd=math.sqrt(variance),observation_sd=math.sqrt(observation),past_noise_games=noise_n))
        for date,day in games.groupby('date',sort=True):
            for point,cutoff in list(pending.items()):
                if cutoff<=date:
                    snap=state.snapshot(cutoff);snapshots[(year,point)]=snap
                    sd=np.sqrt(np.maximum(np.diag(snap['covariance']),0))
                    range_rows.extend(dict(season=year,checkpoint=point,team=c,mean=float(snap['mean'][i]),sd=float(sd[i]),
                        lower95=float(snap['mean'][i]-1.96*sd[i]),upper95=float(snap['mean'][i]+1.96*sd[i]),
                        cutoff=str(cutoff.date()),max_input_date=str(snap['max_input_date'].date())) for i,c in enumerate(codes))
                    del pending[point]
            all_rows.append(features(state.snapshot(date),day))
            for g in day.itertuples():state.update(g.a,g.b,g.mov,g.home_court,g.date,g.phase=='regular')
        if pending:raise AssertionError('Unbuilt checkpoints')
        # Fit NEXT season's observation noise only after this season is complete.
        regular=games.loc[games.phase.eq('regular')]
        expected=regular.a.map(current.srs)-regular.b.map(current.srs)+3*(2*regular.home_court-1)
        noise_sse+=float(((regular.mov-expected)**2).sum());noise_n+=len(regular)
    frame=pd.concat(all_rows,ignore_index=True)
    if not pd.to_datetime(frame.max_input_date).lt(frame.date).all():raise AssertionError('Leaked input date')
    frame.to_parquet(CACHE/'pregame.parquet',index=False)
    pd.DataFrame(parameter_audit).to_csv(OUT/'parameter_audit.csv',index=False)
    pd.DataFrame(range_rows).to_csv(OUT/'strength_ranges.csv',index=False)
    # Retain numeric covariance snapshots for repeatable invariant tests and independent audit.
    for (year,point),s in snapshots.items():
        np.savez_compressed(CACHE/f'{year}_{point}.npz',mean=s['mean'],covariance=s['covariance'])
    return frame,snapshots


def series_rows(year,point,snapshot,probabilities):
    records=[];games=load_games(year)
    for row in load_labels().loc[LABELS.season.eq(year)].itertuples():
        a,b=sorted([ALIASES.get(row.winner,row.winner),ALIASES.get(row.loser,row.loser)])
        actual=games.loc[games.phase.eq('playoff')&games.a.isin([a,b])&games.b.isin([a,b])]
        if actual.empty:raise ValueError('Missing playoff series')
        if actual.date.min()<snapshot['cutoff']:continue
        i,j=snapshot['index'][a],snapshot['index'][b]
        ranking=snapshot['wins']/np.maximum(snapshot['gp'],1) if point=='pre_playoff' else (snapshot['wins']+20/(1+np.exp(-snapshot['mean']/6)))/(snapshot['gp']+20)
        home_a=(ranking[i],snapshot['mean'][i],a)>(ranking[j],snapshot['mean'][j],b)
        p=snapshot['covariance'];variance=max(float(p[i,i]+p[j,j]-2*p[i,j]),0)
        for arm in ['fixed','uncertain']:
            value=probabilities.series(snapshot['mean'][i]-snapshot['mean'][j],variance,home_a,
                neutral=year==2020,old_finals=year<=2013 and row.round==4,uncertain=arm=='uncertain')
            records.append(dict(season=year,checkpoint=point,kind='series',phase='playoff',event=f'{year}-{row.round}-{a}-{b}',
                arm=arm,y=int(a==ALIASES.get(row.winner,row.winner)),probability=value,cutoff=str(snapshot['cutoff'].date())))
    return records


def calibration(frame):
    group=np.minimum((frame.probability.to_numpy()*10).astype(int),9);bins=[]
    for i in range(10):
        g=frame.loc[group==i]
        bins.append(dict(bin=i,n=len(g),mean_probability=float(g.probability.mean()) if len(g) else None,
            win_rate=float(g.y.mean()) if len(g) else None))
    return bins


def score(frame):
    ece=sum(r['n']*abs(r['mean_probability']-r['win_rate']) for r in calibration(frame) if r['n'])/len(frame)
    return dict(n=len(frame),log_loss=float(frame.loss.mean()),brier=float(frame.brier.mean()),ece=ece)


def paired(frame):
    keys=['season','checkpoint','kind','event']
    result=frame.loc[frame.arm.eq('fixed')].merge(frame.loc[frame.arm.eq('uncertain')],on=keys,suffixes=('_fixed','_uncertain'),validate='one_to_one')
    if len(result)*2!=len(frame) or not result.y_fixed.eq(result.y_uncertain).all():raise AssertionError('Different forecast cases')
    result['loss_delta']=result.loss_uncertain-result.loss_fixed
    result['brier_delta']=result.brier_uncertain-result.brier_fixed
    return result


def bootstrap(frame):
    annual=frame.groupby('season').agg(n=('loss_delta','size'),loss=('loss_delta','sum'),brier=('brier_delta','sum'))
    rng=np.random.default_rng(0);sample=rng.integers(0,len(annual),(2000,len(annual)))
    size=annual.n.to_numpy()[sample].sum(axis=1)
    return dict(log_loss_delta=float(frame.loss_delta.mean()),brier_delta=float(frame.brier_delta.mean()),
        log_loss_delta_95=np.quantile(annual.loss.to_numpy()[sample].sum(axis=1)/size,[.025,.975]).tolist(),
        brier_delta_95=np.quantile(annual.brier.to_numpy()[sample].sum(axis=1)/size,[.025,.975]).tolist(),
        seasons_improved=int((annual.loss<0).sum()),seasons=len(annual),n=len(frame))


def protected_files():
    files=[ROOT/p for p in ['app.py','live.py','model.py','train.py','data_pipeline.py','artifacts/xgboost.json','artifacts/evaluation.json']]
    files += [p for p in (ROOT/'experiments/player_minutes').rglob('*') if p.is_file() and 'cache' not in p.parts and '__pycache__' not in p.parts]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def run():
    protected=protected_files();rows,snapshots=build();records=[];folds=[];demo=[]
    for year in range(2013,2027):
        training=rows.loc[rows.season.lt(year)]
        reverse=training.copy();reverse.d_strength=-reverse.d_strength;reverse.d_rest=-reverse.d_rest;reverse.home_court=1-reverse.home_court;reverse.y=1-reverse.y
        train=pd.concat([training,reverse],ignore_index=True)
        model=XGBClassifier(**SPEC['game_model'],monotone_constraints=(1,1,0))
        model.fit(train[FEATURES],train.y)
        model.save_model(CACHE/f'game_{year}.json')
        probabilities=GameProbabilities(model)
        folds.append(dict(season=year,train_through=int(training.season.max()),training_games=len(training),
            same_model_both_arms=True,model_sha256=digest(CACHE/f'game_{year}.json'),integration_split_count=len(probabilities.cuts)))
        games=load_games(year)
        for (yr,point),snapshot in snapshots.items():
            if yr!=year:continue
            future=features(snapshot,games.loc[games.date.ge(snapshot['cutoff'])])
            for arm in ['fixed','uncertain']:
                p=probabilities.games(future.d_strength.to_numpy(),future.d_variance.to_numpy(),future.home_court.to_numpy(),future.d_rest.to_numpy(),arm=='uncertain')
                records.extend(dict(season=year,checkpoint=point,kind='game',phase=g.phase,event=g.game_id,arm=arm,y=g.y,probability=float(value),cutoff=str(snapshot['cutoff'].date())) for g,value in zip(future.itertuples(),p))
            records.extend(series_rows(year,point,snapshot,probabilities))
            if year==2026:
                conferences=load_history().loc[HISTORY.season.eq(year)].set_index('team').loc[snapshot['codes'],'conference'].tolist()
                for arm in ['fixed','uncertain']:
                    simulated=simulate(snapshot,future.loc[future.phase.eq('regular')],conferences,probabilities,arm=='uncertain')
                    demo.extend(dict(season=year,checkpoint=point,arm=arm,team=team,
                        title_probability=float(simulated['title'][i]),playoff_probability=float(simulated['playoff'][i]),
                        projected_wins=float(simulated['projected_wins'][i]),title_sampling_se=float(simulated['title_se'][i])) for i,team in enumerate(snapshot['codes']))
        print('Evaluated',year,flush=True)
    frame=pd.DataFrame(records);frame.probability=frame.probability.clip(1e-6,1-1e-6)
    frame['loss']=-frame.y*np.log(frame.probability)-(1-frame.y)*np.log(1-frame.probability)
    frame['brier']=(frame.probability-frame.y)**2
    frame.to_csv(OUT/'predictions.csv.gz',index=False)
    pd.DataFrame(demo).to_csv(OUT/'title_simulation_demo.csv',index=False)
    metrics=[];bins=[]
    for columns,scope in [(['kind','arm'],'overall'),(['kind','checkpoint','arm'],'checkpoint')]:
        for values,g in frame.groupby(columns):metrics.append(dict(scope=scope,**dict(zip(columns,values)),**score(g)))
    metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'metrics.csv',index=False)
    for (kind,point,arm),g in frame.groupby(['kind','checkpoint','arm']):bins.extend(dict(kind=kind,checkpoint=point,arm=arm,**r) for r in calibration(g))
    pd.DataFrame(bins).to_csv(OUT/'calibration.csv',index=False)
    pair=paired(frame)
    annual=pair.groupby(['season','checkpoint','kind']).agg(n=('event','size'),fixed_loss=('loss_fixed','mean'),uncertain_loss=('loss_uncertain','mean'),log_loss_delta=('loss_delta','mean'),fixed_brier=('brier_fixed','mean'),uncertain_brier=('brier_uncertain','mean'),brier_delta=('brier_delta','mean')).reset_index()
    annual.to_csv(OUT/'per_season_differences.csv',index=False)
    comparisons={kind:bootstrap(g) for kind,g in pair.groupby('kind')}
    checkpoints={f'{kind}/{point}':bootstrap(g) for (kind,point),g in pair.groupby(['kind','checkpoint'])}
    rules={}
    for kind,r in comparisons.items():
        rules[kind+'_significant']=r['log_loss_delta_95'][1]<0
        rules[kind+'_brier_improves']=r['brier_delta']<0
        rules[kind+'_consistent']=r['seasons_improved']>=10
        pre=checkpoints[kind+'/pre_playoff']
        rules[kind+'_pre_not_worse']=pre['log_loss_delta']<=0 and pre['brier_delta']<=0
    cal=metrics.loc[metrics.scope.eq('checkpoint')].pivot(index=['kind','checkpoint'],columns='arm',values='ece')
    rules['calibration_not_materially_worse']=bool((cal.uncertain<=cal.fixed+.005).all())
    if protected_files()!=protected:raise AssertionError('Production or earlier experiments changed')
    report=dict(decision='PROMISING_OFFLINE_ONLY' if all(rules.values()) else 'HOLD',promotion_authorized=False,
        spec_sha256=digest(HERE/'spec.json'),code_hashes={p.name:digest(p) for p in HERE.glob('*.py')},
        folds=folds,comparisons=comparisons,checkpoints=checkpoints,gate_checks=rules,
        protected_hashes=protected,production_unchanged=True,
        integration='Exact Gaussian integration of forest split intervals; no simulation noise in game/series probabilities.',
        simulation='2026 demos use 10000 persistent joint strength paths and common outcome draws. Title probabilities are not validated by this demonstration.',
        caveats=SPEC['limitations'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    lines=['# Team-strength uncertainty experiment','',f"**Decision: {report['decision']} — production unchanged.**",'',
        '14 walk-forward seasons (2012–13 through 2025–26), same October/January/pre-playoff cases. All rating parameters and game-model training use earlier data. October 2020 is omitted because the season had not started. No player data or historical BPM approximations.','',
        '| Target | Arm | Forecasts | Log loss | Brier | ECE |','|---|---|---:|---:|---:|---:|']
    for r in metrics.loc[metrics.scope.eq('overall')].itertuples():lines.append(f'| {r.kind} | {r.arm} | {r.n} | {r.log_loss:.6f} | {r.brier:.6f} | {r.ece:.6f} |')
    lines+=['','## Uncertain minus fixed: negative is better','']
    for key,r in {**comparisons,**checkpoints}.items():
        lo,hi=r['log_loss_delta_95'];lines.append(f"- {key}: log-loss delta {r['log_loss_delta']:+.6f}, 95% whole-season CI [{lo:+.6f}, {hi:+.6f}]; Brier delta {r['brier_delta']:+.6f}; {r['seasons_improved']}/{r['seasons']} seasons improve.")
    lines+=['','## Fixed gates']+[f'- {k}: {v}' for k,v in rules.items()]
    lines+=['','## Limits']+['- '+v for v in SPEC['limitations']]
    lines+=['','`strength_ranges.csv` contains dated mean/SD/95% latent ranges, not claimed empirical coverage. `parameter_audit.csv` records prior-only variance estimation. `per_season_differences.csv`, `calibration.csv` and `predictions.csv.gz` contain detailed results. `title_simulation_demo.csv` demonstrates joint persistent draws; historical future fixtures are conditioned on, not reconstructed archived schedules.','',
        'Research inspiration: [Lopez, Matthews & Baumer](https://arxiv.org/abs/1701.05976), which uses betting-market state-space models. This experiment instead uses a small Gaussian point-margin filter plus an unchanged XGBoost game head in both arms; it is not a replication.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(decision=report['decision'],comparisons=comparisons,checkpoints=checkpoints,gate_checks=rules),indent=2),flush=True)

if __name__=='__main__':run()
