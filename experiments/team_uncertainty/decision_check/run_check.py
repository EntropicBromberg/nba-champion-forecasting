"""Paired equal-season comparison and descriptive title calibration; no promotion."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from common import ROOT,UNC,HERE,research,calibration,snapshot,NativeProduction,UncertainPaths,GameProbabilities
from paths import season_paths

OUT=HERE/'results';OUT.mkdir(exist_ok=True)
CACHE=HERE/'cache';CACHE.mkdir(exist_ok=True)
PATHS=10000

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prior_slope(history,point,kind,year,cutoff,ends):
    rows=history.loc[history.checkpoint.eq(point)&history.kind.eq(kind)&history.season.lt(year)]
    years=sorted(rows.season.unique())
    if years and max(ends[int(y)] for y in years)>=cutoff:raise AssertionError('Unresolved calibration labels')
    return (calibration.fit_slope(rows.probability,rows.y) if len(years)>=3 else 1.),[int(y) for y in years]


def annual_pair(annual):
    base=annual.loc[annual.model.eq('production_calibrated')]
    candidate=annual.loc[annual.model.eq('uncertainty')]
    pair=base.merge(candidate,on=['season','checkpoint','kind'],suffixes=('_production','_uncertainty'),validate='one_to_one')
    pair['log_loss_delta']=pair.log_loss_uncertainty-pair.log_loss_production
    pair['brier_delta']=pair.brier_uncertainty-pair.brier_production
    rows=[]
    for (kind,point),g in pair.groupby(['kind','checkpoint']):
        rng=np.random.default_rng(0);index=rng.integers(0,len(g),(2000,len(g)))
        loss=g.log_loss_delta.to_numpy();brier=g.brier_delta.to_numpy()
        rows.append(dict(kind=kind,checkpoint=point,seasons=len(g),seasons_improved=int((loss<0).sum()),
            log_loss_delta=float(loss.mean()),brier_delta=float(brier.mean()),
            log_loss_delta_95=np.quantile(loss[index].mean(axis=1),[.025,.975]).tolist(),
            brier_delta_95=np.quantile(brier[index].mean(axis=1),[.025,.975]).tolist()))
    return pair,rows


def run():
    protected=[ROOT/p for p in ['live.py','model.py','train.py','data_pipeline.py','artifacts/xgboost.json','artifacts/evaluation.json']]
    original={str(p.relative_to(ROOT)):digest(p) for p in protected}
    uncertainty=pd.read_csv(UNC/'results/predictions.csv.gz',dtype={'event':str})
    uncertainty=uncertainty.loc[uncertainty.arm.eq('uncertain')].copy();uncertainty['model']='uncertainty'
    meta=json.loads((ROOT/'artifacts/evaluation.json').read_text());folds={f['season']:f for f in meta['folds']}
    ends={y:research.load_games(y).date.max() for y in range(2013,2027)}
    original_production=pd.read_csv(ROOT/'experiments/player_minutes/production_results/predictions.csv.gz',dtype={'event':str,'input_quality':str},low_memory=False)
    original_production=original_production.loc[original_production.model.eq('production_series_xgboost')]
    output=[];production_history=[];titles=[];calibration_audit=[];home_audit=[]
    for year in range(2013,2027):
        game_model=XGBClassifier();game_model.load_model(UNC/f'cache/game_{year}.json')
        lookup=GameProbabilities(game_model)
        native=XGBClassifier();native.load_model(CACHE/f'production_{year}.json')
        points=uncertainty.loc[uncertainty.season.eq(year),'checkpoint'].unique()
        for point in points:
            state,games=snapshot(year,point);future=research.features(state,games.loc[games.date.ge(state['cutoff'])])
            teams=pd.read_parquet(CACHE/f'inputs_{year}_{point}.parquet').set_index('team').loc[state['codes']].reset_index()
            production=NativeProduction(native,folds[year]['features'],teams)
            pframe=original_production.loc[original_production.season.eq(year)&original_production.checkpoint.eq(point)].copy()
            for index,row in pframe.loc[pframe.kind.eq('series')].iterrows():
                _,rnd,a,b=row.event.split('-');i,j=state['index'][a],state['index'][b]
                ranking=state['wins']/np.maximum(state['gp'],1) if point=='pre_playoff' else (state['wins']+20/(1+np.exp(-state['mean']/6)))/(state['gp']+20)
                home_a=(ranking[i],state['mean'][i],a)>(ranking[j],state['mean'][j],b)
                pframe.loc[index,'probability']=production.series_probability(i,j,int(rnd),home_a,year==2020)
                home_audit.append(dict(season=year,checkpoint=point,event=row.event,home=a if home_a else b))
            # Game native probabilities are also recomputed, including neutral venues.
            for index,row in pframe.loc[pframe.kind.eq('game')].iterrows():
                g=future.loc[future.game_id.eq(row.event)].iloc[0]
                pframe.loc[index,'probability']=production.game_probability(state['index'][g.a],state['index'][g.b],g.home_court)
            pframe['model']='production_raw';production_history.append(pframe)
            history=pd.concat(production_history,ignore_index=True)
            slopes={}
            for kind in ['game','series']:
                slope,years=prior_slope(history,point,kind,year,state['cutoff'],ends)
                slopes[kind]=slope;calibration_audit.append(dict(season=year,checkpoint=point,kind=kind,slope=slope,training_seasons=years))
            prod=pframe.copy();prod['model']='production_calibrated'
            for kind in slopes:
                mask=prod.kind.eq(kind);prod.loc[mask,'probability']=calibration.apply_slope(prod.loc[mask,'probability'],slopes[kind])
            unc=uncertainty.loc[uncertainty.season.eq(year)&uncertainty.checkpoint.eq(point)].copy()
            pair=prod.merge(unc,on=['season','checkpoint','kind','event'],suffixes=('_p','_u'),validate='one_to_one')
            if len(pair)!=len(prod) or len(pair)!=len(unc) or not pair.y_p.eq(pair.y_u).all() or not pair.cutoff_p.eq(pair.cutoff_u).all():raise AssertionError('Different forecast cases')
            output.extend([prod,unc])
            production.configure(slopes['game'],slopes['series'])
            conferences=research.HISTORY.loc[research.HISTORY.season.eq(year)].set_index('team').loc[state['codes'],'conference'].tolist()
            regular=future.loc[future.phase.eq('regular')]
            candidate=UncertainPaths(state,lookup,PATHS,year=year)
            adapters=[('production_calibrated',production.game_probability,lambda a,b,r,h:production.series_probability(a,b,r,h,year==2020)),
                      ('uncertainty',candidate.game_probability,candidate.series_probability)]
            winner=research.ALIASES.get(research.LABELS.loc[research.LABELS.season.eq(year)&research.LABELS['round'].eq(4),'winner'].item())
            if winner is None:winner=research.LABELS.loc[research.LABELS.season.eq(year)&research.LABELS['round'].eq(4),'winner'].item()
            for name,game_probability,series_probability in adapters:
                result=season_paths(state,regular,conferences,game_probability,series_probability,PATHS,year=year)
                titles.extend(dict(season=year,checkpoint=point,model=name,team=c,y=int(c==winner),
                    probability=float(result['title_smoothed'][i]),raw_probability=float(result['title'][i]),sampling_se=float(result['title_se'][i])) for i,c in enumerate(state['codes']))
            print('Scored',year,point,flush=True)
    frame=pd.concat(output,ignore_index=True)
    frame.probability=frame.probability.clip(1e-6,1-1e-6)
    frame['loss']=-frame.y*np.log(frame.probability)-(1-frame.y)*np.log(1-frame.probability);frame['brier']=(frame.probability-frame.y)**2
    frame.to_csv(OUT/'event_predictions.csv.gz',index=False)
    title=pd.DataFrame(titles);title.to_csv(OUT/'title_predictions.csv',index=False)
    annual=[];calib=[]
    for (year,point,kind,name),g in frame.groupby(['season','checkpoint','kind','model']):annual.append(dict(season=year,checkpoint=point,kind=kind,model=name,**calibration.score(g)))
    for (year,point,name),g in title.groupby(['season','checkpoint','model']):
        if g.y.sum()!=1 or not np.isclose(g.probability.sum(),1):raise AssertionError('Invalid title distribution')
        annual.append(dict(season=year,checkpoint=point,kind='title',model=name,n=1,
            log_loss=float(-np.log(g.loc[g.y.eq(1),'probability'].item())),brier=float(((g.probability-g.y)**2).sum())))
    annual=pd.DataFrame(annual);annual.to_csv(OUT/'annual_scores.csv',index=False)
    paired,comparison=annual_pair(annual);paired.to_csv(OUT/'paired_season_differences.csv',index=False)
    for source in [frame.assign(target=frame.kind),title.assign(target='title')]:
        for (target,point,name),g in source.groupby(['target','checkpoint','model']):calib.extend(dict(target=target,checkpoint=point,model=name,**r) for r in calibration.bins(g))
    pd.DataFrame(calib).to_csv(OUT/'reliability_bins.csv',index=False)
    pd.DataFrame(home_audit).to_csv(OUT/'home_court_audit.csv',index=False)
    if {str(p.relative_to(ROOT)):digest(p) for p in protected}!=original:raise AssertionError('Production calculations changed')
    report=dict(decision='HOLD',promotion_authorized=False,comparisons=comparison,calibration_folds=calibration_audit,
        protected_production_hashes=original,spec_sha256=digest(HERE/'spec.json'),
        title_samples=PATHS,title_smoothing='additive 0.5 per team count',
        limits=json.loads((HERE/'spec.json').read_text())['limits'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    lines=['# Paired season-level check versus calibrated production','',
        '**HOLD. No promotion. Production remains primary; any experimental display is explicitly labeled.**','',
        'Each paired effect weights seasons equally, with 2000 paired whole-season bootstrap resamples. Positive deltas favor calibrated production. October/January production inputs are proxy-based and not decisive promotion evidence. Pre-playoff uses completed regular-season statistics.','',
        '| Target | Checkpoint | Seasons | Uncertainty − calibrated production log loss | 95% CI | Improved seasons |','|---|---|---:|---:|---|---:|']
    for r in comparison:lines.append(f"| {r['kind']} | {r['checkpoint']} | {r['seasons']} | {r['log_loss_delta']:+.5f} | [{r['log_loss_delta_95'][0]:+.5f}, {r['log_loss_delta_95'][1]:+.5f}] | {r['seasons_improved']} |")
    lines+=['','## Title calibration','',
        'Title log loss is multiclass champion log loss; Brier is summed over all teams within a season. Monte Carlo counts receive additive 0.5 smoothing to avoid infinite scores from finite-sample zeros. Reliability bins are descriptive: there are only 13/14 champions per checkpoint, not hundreds of independent team outcomes. Schedule revisions and pandemic formats are conditioned on as realized. This is not a genuinely new validation cohort.','',
        '`annual_scores.csv`, `paired_season_differences.csv`, `title_predictions.csv`, `reliability_bins.csv` and `report.json` contain full metrics, coefficients and audits.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(comparison,indent=2),flush=True)

if __name__=='__main__':run()
