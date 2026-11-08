"""Reuse validated pre-playoff OOS predictions; calibrate strictly from earlier resolved seasons."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent
ROOT=HERE.parents[2]
OUT=HERE/'results';OUT.mkdir(exist_ok=True)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases():
    uncertain=pd.read_csv(EXPERIMENT/'results/predictions.csv.gz',dtype={'event':str})
    uncertain=uncertain.loc[uncertain.checkpoint.eq('pre_playoff')].copy()
    uncertain['model']=uncertain.arm
    production=pd.read_csv(ROOT/'experiments/player_minutes/production_results/predictions.csv.gz',dtype={'event':str,'input_quality':str},low_memory=False)
    production=production.loc[production.checkpoint.eq('pre_playoff')&production.model.eq('production_series_xgboost')].copy()
    if not production.input_quality.eq('complete_regular_season').all():raise AssertionError('Proxy production features')
    production['model']='production'
    columns=['season','kind','event','cutoff','phase','y','probability','model']
    frame=pd.concat([production[columns],uncertain[columns]],ignore_index=True)
    keys=['season','kind','event']
    baseline=frame.loc[frame.model.eq('production')]
    for model in ['fixed','uncertain']:
        pair=baseline.merge(frame.loc[frame.model.eq(model)],on=keys,suffixes=('_production','_candidate'),validate='one_to_one')
        if len(pair)!=1421 or not pair.cutoff_production.eq(pair.cutoff_candidate).all() or not pair.y_production.eq(pair.y_candidate).all():
            raise AssertionError('Different pre-playoff forecast cases')
    return frame


def audit_home_court():
    ranges=pd.read_csv(EXPERIMENT/'results/strength_ranges.csv')
    series=pd.read_csv(ROOT/'data/series.csv')
    aliases={'NJN':'BRK','NOH':'NOP','NOK':'NOP','SEA':'OKC','CHA':'CHO','BKN':'BRK','PHX':'PHO'}
    records=[]
    for year in range(2013,2027):
        snapshot=json.loads((ROOT/f'experiments/player_minutes/cache/snapshots_{year}.json').read_text())['pre_playoff']
        means=ranges.loc[ranges.season.eq(year)&ranges.checkpoint.eq('pre_playoff')].set_index('team')['mean']
        for row in series.loc[series.season.eq(year)].itertuples():
            a,b=sorted([aliases.get(row.winner,row.winner),aliases.get(row.loser,row.loser)])
            i,j=snapshot['index'][a],snapshot['index'][b]
            record_a=snapshot['wins'][i]/snapshot['gp'][i];record_b=snapshot['wins'][j]/snapshot['gp'][j]
            old=(record_a,snapshot['srs'][i],a)>(record_b,snapshot['srs'][j],b)
            new=(record_a,means[a],a)>(record_b,means[b],b)
            if old!=new:raise AssertionError('Home-court mismatch requires production replay')
            records.append(dict(season=year,round=row.round,team_a=a,team_b=b,common_home=a if old else b,neutral=year==2020))
    return pd.DataFrame(records)


def fit_slope(probability,y):
    z=logit(np.clip(np.asarray(probability),1e-6,1-1e-6));y=np.asarray(y)
    if np.sum(z*z)<1e-12:return 1.
    objective=lambda a:float(np.mean(np.logaddexp(0,a*z)-y*a*z))
    result=minimize_scalar(objective,bounds=(.25,4.),method='bounded',options={'xatol':1e-8})
    if not result.success:raise ValueError('Calibration failed')
    return float(result.x)


def apply_slope(probability,slope):
    if slope==1.:return np.asarray(probability).copy()
    return np.clip(expit(slope*logit(np.clip(probability,1e-6,1-1e-6))),1e-6,1-1e-6)


def calibrate(frame,season_ends):
    raw=frame.copy();raw['variant']=raw.model+'_raw'
    adjusted=[];audit=[]
    for (model,kind),group in frame.groupby(['model','kind']):
        for year,test in group.groupby('season'):
            train=group.loc[group.season.lt(year)]
            years=sorted(train.season.unique())
            cutoff=pd.Timestamp(test.cutoff.min())
            resolved=max((season_ends[int(y)] for y in years),default=None)
            if resolved is not None and resolved>=cutoff:raise AssertionError('Calibration labels not resolved')
            active=len(years)>=3
            slope=fit_slope(train.probability,train.y) if active else 1.
            result=test.copy();result.probability=apply_slope(test.probability,slope)
            result['variant']=model+'_calibrated';adjusted.append(result)
            audit.append(dict(model=model,kind=kind,test_season=int(year),train_seasons=[int(y) for y in years],
                train_last=None if not years else int(max(years)),train_cases=len(train),slope=slope,
                active=active,labels_resolved_through=None if resolved is None else str(resolved.date()),cutoff=str(cutoff.date())))
    return pd.concat([raw,pd.concat(adjusted)],ignore_index=True),audit


def bins(frame):
    index=np.minimum((frame.probability.to_numpy()*10).astype(int),9);rows=[]
    for i in range(10):
        part=frame.loc[index==i]
        rows.append(dict(bin=i,n=len(part),prediction=float(part.probability.mean()) if len(part) else None,
            outcome=float(part.y.mean()) if len(part) else None))
    return rows


def score(frame):
    ece=sum(r['n']*abs(r['prediction']-r['outcome']) for r in bins(frame) if r['n'])/len(frame)
    return dict(n=len(frame),log_loss=float(frame.loss.mean()),brier=float(frame.brier.mean()),ece=ece)


def comparison(frame,first,second):
    keys=['season','kind','event']
    paired=frame.loc[frame.variant.eq(first)].merge(frame.loc[frame.variant.eq(second)],on=keys,suffixes=('_baseline','_candidate'),validate='one_to_one')
    output={}
    for kind,part in paired.groupby('kind'):
        part=part.copy();part['delta']=part.loss_candidate-part.loss_baseline;part['bd']=part.brier_candidate-part.brier_baseline
        annual=part.groupby('season').agg(n=('delta','size'),loss=('delta','sum'),brier=('bd','sum'))
        rng=np.random.default_rng(0);sample=rng.integers(0,len(annual),(2000,len(annual)))
        size=annual.n.to_numpy()[sample].sum(axis=1)
        output[kind]=dict(log_loss_delta=float(part.delta.mean()),brier_delta=float(part.bd.mean()),
            log_loss_delta_95=np.quantile(annual.loss.to_numpy()[sample].sum(axis=1)/size,[.025,.975]).tolist(),
            brier_delta_95=np.quantile(annual.brier.to_numpy()[sample].sum(axis=1)/size,[.025,.975]).tolist(),
            seasons_improved=int((annual.loss<0).sum()),seasons=len(annual))
    return output


def run():
    sources=[EXPERIMENT/p for p in ['spec.json','run.py','ratings.py','probabilities.py','simulator.py','results/report.json','results/predictions.csv.gz']]
    old=json.loads((EXPERIMENT/'results/report.json').read_text())['protected_hashes']
    paths=sources+[ROOT/p for p in old]
    before={str(p.relative_to(ROOT)):digest(p) for p in paths}
    frame=load_cases();home=audit_home_court();home.to_csv(OUT/'home_court_audit.csv',index=False)
    ends={}
    for year in sorted(frame.season.unique()):
        team=pd.read_parquet(ROOT/f'experiments/player_minutes/cache/{year}.parquet',columns=['date'],filters=[('player','=','')])
        ends[int(year)]=pd.to_datetime(team.date).max()
    forecasts,audit=calibrate(frame,ends)
    forecasts['loss']=-forecasts.y*np.log(forecasts.probability)-(1-forecasts.y)*np.log(1-forecasts.probability)
    forecasts['brier']=(forecasts.probability-forecasts.y)**2
    forecasts.to_csv(OUT/'predictions.csv.gz',index=False)
    metrics=[]
    for columns,scope in [(['kind','variant'],'all_14_seasons'),(['season','kind','variant'],'season')]:
        for values,g in forecasts.groupby(columns):metrics.append(dict(scope=scope,**dict(zip(columns,values)),**score(g)))
    for (kind,variant),g in forecasts.loc[forecasts.season.ge(2016)].groupby(['kind','variant']):
        metrics.append(dict(scope='calibration_active_11_seasons',kind=kind,variant=variant,**score(g)))
    metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'metrics.csv',index=False)
    calibration=[]
    for (kind,variant),g in forecasts.groupby(['kind','variant']):calibration.extend(dict(kind=kind,variant=variant,**r) for r in bins(g))
    pd.DataFrame(calibration).to_csv(OUT/'calibration_bins.csv',index=False)
    comparisons={f'{b}_minus_{a}':comparison(forecasts,a,b) for a,b in [
        ('production_raw','uncertain_raw'),('production_calibrated','uncertain_raw'),('production_raw','uncertain_calibrated'),
        ('production_calibrated','uncertain_calibrated'),('fixed_calibrated','uncertain_calibrated'),
        ('uncertain_raw','uncertain_calibrated'),('production_raw','production_calibrated')]}
    after={str(p.relative_to(ROOT)):digest(p) for p in paths}
    if before!=after:raise AssertionError('Production or original experiments changed')
    report=dict(decision='HOLD_RETROSPECTIVE_CHECK_ONLY',production_unchanged=True,original_gates_unchanged=True,
        spec_sha256=digest(HERE/'spec.json'),protected_hashes=after,calibration_folds=audit,comparisons=comparisons,
        warmup='Identity calibration 2013-2015; three prior out-of-season pre-playoff seasons required. Fitted calibration 2016-2026.',
        limitation='Post-hoc game/series calibration is fitted separately; not integrated into coherent season/title simulations. Historical comparison is not a new promotion holdout.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    lines=['# Pre-playoff uncertainty vs production, with prior-only calibration','',
        '**Retrospective diagnostic only. Production, displayed odds and original experiment gates remain unchanged.**','',
        'Identical 14-season pre-playoff cases, outcomes, cutoffs and projected home courts, verified before reusing cached out-of-season forecasts. One positive logit slope per model/target is fitted on earlier complete-season pre-playoff OOS predictions only. No intercept. 2013–2015 use identity calibration; fitted slopes begin in 2016. Production receives the same calibration opportunity.','',
        '| Target | Variant | N | Log loss | Brier | ECE |','|---|---|---:|---:|---:|---:|']
    for r in metrics.loc[metrics.scope.eq('all_14_seasons')].itertuples():lines.append(f'| {r.kind} | {r.variant} | {r.n} | {r.log_loss:.6f} | {r.brier:.6f} | {r.ece:.6f} |')
    lines+=['','## Paired season-bootstrap comparisons: negative favors candidate','']
    for label,result in comparisons.items():
        for kind,r in result.items():
            lo,hi=r['log_loss_delta_95'];lines.append(f"- {label}, {kind}: delta {r['log_loss_delta']:+.6f}, 95% CI [{lo:+.6f}, {hi:+.6f}], Brier delta {r['brier_delta']:+.6f}, wins {r['seasons_improved']}/{r['seasons']} seasons.")
    lines+=['','Per-season and fitted-calibration-active results are in `metrics.csv`; coefficients and strictly earlier training years are in `report.json`; `predictions.csv.gz`, `calibration_bins.csv` and `home_court_audit.csv` contain the complete audit.','',
        'This output calibration is a diagnostic, separately fitted for games and series. It has not been wired into the persistent-strength season simulator and does not establish calibrated title probabilities. These reused historical outcomes do not satisfy a genuinely new prospective promotion cohort.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    print(metrics.loc[metrics.scope.eq('all_14_seasons')].to_string(index=False),flush=True)
    print(json.dumps(comparisons,indent=2),flush=True)

if __name__=='__main__':run()
