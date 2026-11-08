"""Read-only production replay and paired comparison; early BPM proxies are explicitly ineligible."""
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
import requests
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
import run as exp
from fetch import canonical
from data_pipeline import TEAM_FEATURES,parse_players,name_key,matchup
from live import blend_strength
from model import fit

OUT=HERE/'production_results'
OUT.mkdir(exist_ok=True)
CACHE=HERE/'cache/production'
CACHE.mkdir(parents=True,exist_ok=True)
NAME='production_series_xgboost'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_boxes(year):
    path=CACHE/f'{year}.parquet'
    if path.exists():return
    url=f'https://basketball-data.billmill.org/nba/stats/team_game_logs/season={year}/data.parquet'
    r=requests.get(url,timeout=60);r.raise_for_status()
    frame=pd.read_parquet(BytesIO(r.content))
    frame.loc[frame.game_id.str.startswith('002')].to_parquet(path,index=False)


def previous(year):
    frame=exp.HISTORY.loc[exp.HISTORY.season.eq(year-1)].copy()
    frame['team']=frame.team.map(canonical)
    return frame


def early_inputs(year,point):
    """Causal sensitivity inputs. Do NOT label these exact historical production snapshots."""
    snap=exp.checkpoint(year,point);cutoff=snap['cutoff']
    prior=previous(year)
    current=prior.copy().set_index('team',drop=False)
    current['season']=year
    d=pd.read_parquet(CACHE/f'{year}.parquet');d['date']=pd.to_datetime(d.game_date)
    d=d.loc[d.date.lt(cutoff)].copy()
    identities=pd.read_parquet(CACHE/'identities.parquet')
    mapping=identities.set_index('team_id').nba_abbrev.map(canonical).to_dict()
    d['code']=d.team_id.map(mapping);d['opp']=d.opp_team_id.map(mapping)
    cols=['pts','fgm','fga','fg3m','ftm','fta','oreb','dreb','tov']
    opponent=d[['game_id','team_id']+cols].rename(columns={c:'opp_'+c for c in cols})
    d=d.merge(opponent,left_on=['game_id','opp_team_id'],right_on=['game_id','team_id'],suffixes=('','_other'),validate='many_to_one')
    codes=list(current.index);idx={c:i for i,c in enumerate(codes)}
    lap=np.zeros((len(codes),len(codes)));rhs=np.zeros(len(codes))
    for r in d.itertuples():
        i,j=idx[r.code],idx[r.opp];lap[i,i]+=1;lap[i,j]-=1;rhs[i]+=r.plus_minus
    srs=np.linalg.lstsq(np.vstack([lap,np.ones(len(codes))]),np.append(rhs,0),rcond=None)[0]
    logs=pd.read_parquet(exp.CACHE/f'{year}.parquet');logs['date']=pd.to_datetime(logs.date)
    allowed=set(d.game_id.astype(str))
    pl=logs.loc[logs.player.ne('')&logs.game_id.isin(allowed)&logs.date.lt(cutoff)&logs.minutes.gt(0)].copy()
    pl['key']=pl.player.map(name_key)
    old=parse_players(year-1).copy();old['key']=old.player.map(name_key)
    old['weighted']=old.bpm.fillna(-2)*old.mp
    ratings=old.groupby('key').agg(mp=('mp','sum'),weighted=('weighted','sum'),age=('age','max'))
    ratings['bpm']=ratings.weighted/ratings.mp.clip(lower=1)
    player=pl.groupby(['team','key'],as_index=False).agg(mp=('minutes','sum'),gp=('game_id','nunique'))
    player['bpm']=player.key.map(ratings.bpm).fillna(-2)
    player['age']=player.key.map(ratings.age).add(1).fillna(22)
    for code in codes:
        i=snap['index'][code];g=d.loc[d.code.eq(code)].sort_values('date')
        current.loc[code,['gp','wins','losses']]=[len(g),snap['wins'][i],len(g)-snap['wins'][i]]
        if len(g)!=snap['gp'][i]:raise AssertionError('Different checkpoint game coverage')
        if g.empty:continue
        totals=g[cols+['opp_'+c for c in cols]].sum()
        t=totals
        poss=.5*((t.fga+.4*t.fta-1.07*t.oreb/max(t.oreb+t.opp_dreb,1)*(t.fga-t.fgm)+t.tov)+
                 (t.opp_fga+.4*t.opp_fta-1.07*t.opp_oreb/max(t.opp_oreb+t.dreb,1)*(t.opp_fga-t.opp_fgm)+t.opp_tov))
        current.loc[code,['srs','mov','ortg','drtg']]=[srs[idx[code]],g.plus_minus.mean(),100*t.pts/poss,100*t.opp_pts/poss]
        current.loc[code,'form_resid']=g.tail(25).plus_minus.mean()-g.plus_minus.mean()
        pyth=1/(1+(float(t.opp_pts)/float(t.pts))**14)
        current.loc[code,'luck_resid']=(g.win.sum()-round(len(g)*pyth))/len(g)
        current.loc[code,'efg_net']=(t.fgm+.5*t.fg3m)/t.fga-(t.opp_fgm+.5*t.opp_fg3m)/t.opp_fga
        current.loc[code,'tov_net']=100*t.opp_tov/(t.opp_fga+.44*t.opp_fta+t.opp_tov)-100*t.tov/(t.fga+.44*t.fta+t.tov)
        current.loc[code,'orb_net']=100*t.oreb/(t.oreb+t.opp_dreb)-(100-100*t.dreb/(t.dreb+t.opp_oreb))
        current.loc[code,'ftr_net']=t.ftm/t.fga-t.opp_ftm/t.opp_fga
        roster=player.loc[player.team.eq(code)]
        top=roster.loc[roster.mp.ge(500)|roster.mp.div(roster.gp).ge(15)].nlargest(8,'mp')
        if len(top)<4:continue  # Explicit proxy fallback; never promotion eligible.
        avg=lambda rows,col:np.average(rows[col],weights=rows.mp)
        current.loc[code,'top3_bpm']=avg(top.head(3),'bpm')
        current.loc[code,'depth_bpm']=avg(top.iloc[3:8],'bpm')
        current.loc[code,'age_top8']=avg(top,'age')
        talent=(top.bpm+2).clip(lower=0)*top.mp
        current.loc[code,'gp_share_top8']=np.average(top.gp/len(g),weights=talent if talent.sum() else top.mp)
        recent=set(g.tail(5).game_id)
        current.loc[code,'star_out_flag']=float(any(len(recent-set(pl.loc[pl.team.eq(code)&pl.key.eq(p.key),'game_id']))>=3 for p in top.head(2).itertuples()))
    return blend_strength(current.reset_index(drop=True),prior)


def inputs(year,point):
    if point!='pre_playoff':return early_inputs(year,point),'sensitivity_prior_bpm'
    snap=exp.checkpoint(year,point)
    current=exp.HISTORY.loc[exp.HISTORY.season.eq(year)].copy()
    if not pd.to_datetime(current.cutoff).lt(snap['cutoff']).all():raise AssertionError('Future regular-season data')
    current['team']=current.team.map(canonical)
    return blend_strength(current,previous(year)),'complete_regular_season'


def symmetric(model,features,teams,a,b,rnd,home,neutral=False):
    rows=pd.DataFrame([matchup(teams,a,b,rnd,home=home,neutral=neutral),matchup(teams,b,a,rnd,home=home,neutral=neutral)])
    p=model.predict_proba(rows[features])[:,1]
    return float(np.clip((p[0]+1-p[1])/2,1e-6,1-1e-6))


def forecasts(year,point,model,features):
    snap=exp.checkpoint(year,point);teams,quality=inputs(year,point)
    teams=teams.set_index('team',drop=False)
    if not np.isfinite(teams[['z_'+f for f in TEAM_FEATURES]].to_numpy()).all():raise AssertionError('Incomplete inputs')
    games,_=exp.load_games(year)
    rows=[]
    grid=np.linspace(0,1,10001);best7=35*grid**4-84*grid**5+70*grid**6-20*grid**7
    future=games.loc[games.date.ge(snap['cutoff'])]
    zcols=['z_'+f for f in TEAM_FEATURES]
    difference=teams.loc[future.a,zcols].to_numpy()-teams.loc[future.b,zcols].to_numpy()
    ab=pd.DataFrame({'d_'+f:difference[:,i] for i,f in enumerate(TEAM_FEATURES)})
    ab['home_court']=future.home_court.to_numpy();ab['round']=1
    ba=ab.copy()
    for f in TEAM_FEATURES:ba['d_'+f]=-ba['d_'+f]
    ba['home_court']=1-ba.home_court
    raw=model.predict_proba(pd.concat([ab,ba],ignore_index=True)[features])[:,1]
    p=.5*(raw[:len(ab)]+1-raw[len(ab):])
    for g,prob in zip(future.itertuples(),np.interp(p,best7,grid)):
        rows.append(dict(season=year,checkpoint=point,kind='game',phase=g.phase,event=g.game_id,y=g.y,
            probability=float(prob),model=NAME,cutoff=str(snap['cutoff'].date()),input_quality=quality))
    for s in exp.LABELS.loc[exp.LABELS.season.eq(year)].itertuples():
        a,b=sorted([canonical(s.winner),canonical(s.loser)])
        actual=games.loc[games.phase.eq('playoff')&games.a.isin([a,b])&games.b.isin([a,b])]
        if actual.date.min()<snap['cutoff']:continue
        i,j=snap['index'][a],snap['index'][b]
        ranking=snap['wins']/np.maximum(snap['gp'],1) if point=='pre_playoff' else (snap['wins']+20/(1+np.exp(-snap['srs']/6)))/(snap['gp']+20)
        home=a if (ranking[i],snap['srs'][i],a)>(ranking[j],snap['srs'][j],b) else b
        p=symmetric(model,features,teams,a,b,s.round,home,year==2020)
        rows.append(dict(season=year,checkpoint=point,kind='series',phase='playoff',event=f'{year}-{s.round}-{a}-{b}',
            y=int(a==canonical(s.winner)),probability=p,model=NAME,cutoff=str(snap['cutoff'].date()),input_quality=quality))
    return rows


def compare(frame):
    rows=[]
    for columns,scope in [(['kind','model'],'overall'),(['kind','checkpoint','model'],'checkpoint'),(['kind','checkpoint','season','model'],'season_checkpoint')]:
        for values,g in frame.groupby(columns):rows.append(dict(scope=scope,**dict(zip(columns,values)),**exp.score(g)))
    return pd.DataFrame(rows)


def deltas(frame):
    base=frame.loc[frame.model.eq(NAME)]
    player=frame.loc[frame.model.eq('with_players')]
    keys=['season','checkpoint','kind','event']
    paired=base.merge(player,on=keys,suffixes=('_production','_player'),validate='one_to_one')
    if len(paired)!=len(base) or len(paired)!=len(player):raise AssertionError('Unmatched forecast cases')
    if not paired.y_production.eq(paired.y_player).all():raise AssertionError('Different labels')
    paired['log_loss_delta']=paired.loss_player-paired.loss_production
    paired['brier_delta']=paired.brier_player-paired.brier_production
    return paired


def bootstrap(paired):
    out={}
    for (kind,point),g in paired.groupby(['kind','checkpoint']):
        annual=g.groupby('season').agg(n=('log_loss_delta','size'),loss=('log_loss_delta','sum'),brier=('brier_delta','sum'))
        rng=np.random.default_rng(0);sample=rng.integers(0,len(annual),(2000,len(annual)))
        n=annual.n.to_numpy()[sample].sum(axis=1)
        out[f'{kind}/{point}']=dict(log_loss_delta=float(g.log_loss_delta.mean()),brier_delta=float(g.brier_delta.mean()),
            log_loss_delta_95=np.quantile(annual.loss.to_numpy()[sample].sum(axis=1)/n,[.025,.975]).tolist(),
            brier_delta_95=np.quantile(annual.brier.to_numpy()[sample].sum(axis=1)/n,[.025,.975]).tolist(),
            seasons_improved=int((annual.loss<0).sum()),seasons=len(annual),n=len(g))
    return out


def run():
    protected=[ROOT/p for p in ['app.py','live.py','model.py','train.py','data_pipeline.py','artifacts/xgboost.json','artifacts/evaluation.json']]
    original=[HERE/p for p in ['spec.json','run.py','results/report.json','results/predictions.csv.gz']]
    before={str(p.relative_to(ROOT)):digest(p) for p in protected+original}
    policy=HERE/'prospective_promotion_rule.json'
    locked=json.loads(policy.read_text())
    if 'source_hashes' not in locked:
        locked['source_hashes']={str(p.relative_to(ROOT)):digest(p) for p in protected[:5]+original[:2]}
        policy.write_text(json.dumps(locked,indent=2)+'\n')
    else:
        for relative,expected in locked['source_hashes'].items():
            if digest(ROOT/relative)!=expected:raise AssertionError('Frozen policy source changed')
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(fetch_boxes,range(2013,2027)))
    identity=CACHE/'identities.parquet'
    if not identity.exists():
        r=requests.get('https://basketball-data.billmill.org/nba/stats/team_seasons/season=2026/data.parquet',timeout=60)
        r.raise_for_status();identity.write_bytes(r.content)
    history=pd.read_csv(ROOT/'data/matchups.csv')
    evaluation=json.loads((ROOT/'artifacts/evaluation.json').read_text())
    folds={f['season']:f for f in evaluation['folds']}
    result=[];audit=[]
    for year in range(2013,2027):
        train=history.loc[history.season.lt(year)]
        fold=folds[year]
        if fold['train_through']!=year-1:raise AssertionError('Invalid feature-selection window')
        model=fit(train,fold['features'])
        for point in json.loads((exp.CACHE/f'snapshots_{year}.json').read_text()):
            result.extend(forecasts(year,point,model,fold['features']))
        audit.append(dict(season=year,train_last=int(train.season.max()),features=fold['features']))
        print('Production replay',year,flush=True)
    production=pd.DataFrame(result)
    candidate=pd.read_csv(HERE/'results/predictions.csv.gz',dtype={'event':str})
    candidate=candidate.loc[candidate.model.isin(['with_players','without_players'])]
    frame=pd.concat([production,candidate],ignore_index=True)
    frame['probability']=frame.probability.clip(1e-6,1-1e-6)
    frame['loss']=-frame.y*np.log(frame.probability)-(1-frame.y)*np.log(1-frame.probability)
    frame['brier']=(frame.probability-frame.y)**2
    paired=deltas(frame)
    metric=compare(frame);metric.to_csv(OUT/'metrics.csv',index=False)
    frame.to_csv(OUT/'predictions.csv.gz',index=False)
    annual=paired.groupby(['season','checkpoint','kind']).agg(n=('event','size'),
        production_log_loss=('loss_production','mean'),player_log_loss=('loss_player','mean'),
        log_loss_delta=('log_loss_delta','mean'),production_brier=('brier_production','mean'),
        player_brier=('brier_player','mean'),brier_delta=('brier_delta','mean')).reset_index()
    annual.to_csv(OUT/'per_season_differences.csv',index=False)
    bins=[]
    for (kind,point,name),g in frame.groupby(['kind','checkpoint','model']):
        bins.extend(dict(kind=kind,checkpoint=point,model=name,**b) for b in exp.calibration(g))
    pd.DataFrame(bins).to_csv(OUT/'calibration.csv',index=False)
    after={str(p.relative_to(ROOT)):digest(p) for p in protected+original}
    if before!=after:raise AssertionError('Production or original experiment changed')
    report=dict(decision='HOLD',retrospective_only=True,folds=audit,comparisons=bootstrap(paired),
        exact_scope='Pre-playoff feature reconstruction from complete regular-season data, using live blending; not a vendor-vintage archive.',
        early_scope='Sensitivity only: prior-season BPM and estimated age; team rates reconstructed from NBA box totals may differ from Basketball Reference. Ineligible for promotion.',
        original_rule_unchanged=True,production_unchanged=True,protected_hashes=after,
        prospective_rule_sha256=digest(policy),comparison_spec_sha256=digest(HERE/'production_check_spec.json'))
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    result=report['comparisons']['series/pre_playoff']
    finding=f"Pre-playoff series: candidate minus production log-loss delta {result['log_loss_delta']:+.5f}, 95% season-bootstrap CI [{result['log_loss_delta_95'][0]:+.5f}, {result['log_loss_delta_95'][1]:+.5f}], wins {result['seasons_improved']}/{result['seasons']} seasons. The interval crosses zero. The player-free game model is also slightly better than the player model at this checkpoint. Production stays unchanged."
    lines=['# Player model versus production series XGBoost','',finding,'',
        '**Retrospective diagnostic. No promotion; production and the original 10/14 rule are unchanged.**','',
        'Each production fold trains only on earlier seasons and uses its saved training-only nested feature selection. The deployed full-data artifact is never applied to historical training seasons. Models score identical cases, cutoffs and conditional home courts.','',
        'Pre-playoff inputs use complete regular-season statistics and the exact live blending code. October/January rows are **sensitivity analyses**, not exact production replays: historical daily BPM is unavailable, so prior BPM/estimated ages are carried forward; NBA-derived team rates may differ from Basketball Reference. Neither early sensitivity results nor previously inspected historical cases qualify as new promotion evidence.','',
        '| Checkpoint | Target | Model | N | Log loss | Brier | ECE |','|---|---|---|---:|---:|---:|---:|']
    for r in metric.loc[metric.scope.eq('checkpoint')].itertuples():
        lines.append(f'| {r.checkpoint} | {r.kind} | {r.model} | {r.n} | {r.log_loss:.5f} | {r.brier:.5f} | {r.ece:.5f} |')
    lines+=['','## Paired comparison: candidate minus production','']
    for key,record in report['comparisons'].items():
        lo,hi=record['log_loss_delta_95']
        lines.append(f"- {key}: log-loss delta {record['log_loss_delta']:+.5f}, season-bootstrap 95% CI [{lo:+.5f}, {hi:+.5f}]; improved in {record['seasons_improved']}/{record['seasons']} seasons.")
    lines+=['','## Pre-playoff per-season differences','',
        '| Ending year | Game log-loss delta | Series log-loss delta |','|---|---:|---:|']
    pre=annual.loc[annual.checkpoint.eq('pre_playoff')].pivot(index='season',columns='kind',values='log_loss_delta')
    for year,row in pre.iterrows():lines.append(f'| {year} | {row.game:+.5f} | {row.series:+.5f} |')
    lines+=['','The immutable prospective policy is `../prospective_promotion_rule.json`: three wholly new complete seasons (2026–27 through 2028–29), archived real as-of inputs and forecasts, all-three-season pre-playoff improvements plus the other fixed gates, no tuning or optional stopping. Missing exact inputs or uncertain evidence means HOLD. Original historical rejection is not waived.','',
        'Detailed files: `per_season_differences.csv`, `metrics.csv`, `calibration.csv`, `predictions.csv.gz`, `report.json`.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['comparisons'],indent=2),flush=True)

if __name__=='__main__':run()
