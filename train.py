"""Nested season-based ablations and walk-forward evaluation; target is never trained on."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from data_pipeline import DATA, TARGET, FEATURES, CORE, FACTORS, OPTIONAL
from model import fit, paired_predictions, title_odds, PARAMS

ARTIFACTS = Path(__file__).resolve().parent / 'artifacts'

def metrics(y,p):
    return dict(log_loss=float(log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1])),
                brier=float(brier_score_loss(y,p)),accuracy=float(accuracy_score(y,p>=0.5)))

def choose_features(rows):
    """Ablate blocks, then at most one core feature, using only inner validation seasons."""
    years = sorted(rows.season.unique())
    inner, validation = rows.loc[rows.season.lt(years[-3])], rows.loc[rows.season.ge(years[-3])]
    y = validation.iloc[::2].y
    scores = []
    def score(name,features):
        fitted = fit(inner,features)
        loss = metrics(y,paired_predictions(fitted,validation,features))['log_loss']
        scores.append(dict(variant=name,features=features,inner_log_loss=loss))
        return loss
    candidates = [('all',FEATURES),('without_C',CORE+FACTORS),('without_B',CORE+OPTIONAL),('core',CORE)]
    for name, features in candidates:
        score(name,features)
    best_loss = min(s['inner_log_loss'] for s in scores)
    # Prefer fewer features if the inner log-loss improvement is below 0.003.
    best = min((s for s in scores if s['inner_log_loss'] <= best_loss+0.003),key=lambda s:len(s['features']))
    base = best['features']
    for feature in CORE:
        score('without_'+feature,[f for f in base if f != feature])
    deletions = [s for s in scores[4:] if s['inner_log_loss'] < best['inner_log_loss']-0.003]
    if deletions:
        best = min(deletions,key=lambda s:s['inner_log_loss'])
    return best['features'],scores

def bootstrap(predictions):
    """Resample whole seasons; mirrored rows are never counted twice."""
    rng = np.random.default_rng(0)
    results = {}
    for name, frame in predictions.groupby('model'):
        seasons = [g for _,g in frame.groupby('season')]
        values = []
        for _ in range(1000):
            sample = pd.concat([seasons[i] for i in rng.integers(0,len(seasons),len(seasons))])
            values.append([np.mean(sample.log_loss),np.mean(sample.brier)])
        values = np.asarray(values)
        results[name] = {**metrics(frame.y,frame.probability),
            'log_loss_95':np.quantile(values[:,0],[.025,.975]).tolist(),
            'brier_95':np.quantile(values[:,1],[.025,.975]).tolist()}
    return results

def run():
    ARTIFACTS.mkdir(exist_ok=True)
    rows = pd.read_csv(DATA/'matchups.csv')
    teams, series = pd.read_csv(DATA/'teams.csv'), pd.read_csv(DATA/'series.csv')
    historical = rows.loc[rows.season.lt(TARGET)]
    years = sorted(historical.season.unique())
    predictions, folds, ablations, titles = [], [], [], []
    for year in years[10:]:
        training, test = historical.loc[historical.season.lt(year)], historical.loc[historical.season.eq(year)]
        features, scores = choose_features(training)
        fitted = fit(training,features)
        y = test.iloc[::2].y.to_numpy()
        p = paired_predictions(fitted,test,features)
        logistic = LogisticRegression(C=1.0,random_state=0).fit(training[['d_srs']],training.y)
        baseline = logistic.predict_proba(test[['d_srs']])[:,1]
        baseline = .5*(baseline[::2]+1-baseline[1::2])
        hard = np.where(test.iloc[::2].d_srs.to_numpy()>0,1.,np.where(test.iloc[::2].d_srs.to_numpy()<0,0.,.5))
        for name,probability in [('XGBoost',p),('SRS logistic (evaluation only)',baseline),('Higher SRS wins',hard)]:
            for sid,actual,prob in zip(test.iloc[::2].series_id,y,probability):
                predictions.append(dict(season=year,series_id=sid,model=name,y=int(actual),probability=float(prob),
                    log_loss=float(-actual*np.log(np.clip(prob,1e-6,1-1e-6))-(1-actual)*np.log(np.clip(1-prob,1e-6,1-1e-6))),brier=float((prob-actual)**2)))
        odds = title_odds(fitted,features,teams.loc[teams.season.eq(year)],series.loc[series.season.eq(year)])
        champion = series.loc[series.season.eq(year) & series['round'].eq(4),'winner'].item()
        odds['season'], odds['actual_champion'] = year, odds.team.eq(champion)
        titles.append(odds)
        folds.append(dict(season=int(year),train_through=int(year-1),features=features,**metrics(y,p)))
        ablations.extend(dict(outer_season=int(year),**s) for s in scores)
        print('Evaluated',year,metrics(y,p),flush=True)
    features, final_scores = choose_features(historical)
    fitted = fit(historical,features)
    fitted.save_model(ARTIFACTS/'xgboost.json')
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(ARTIFACTS/'walk_forward.csv',index=False)
    pd.concat(titles).to_csv(ARTIFACTS/'historical_title_odds.csv',index=False)
    report = dict(target_season=TARGET,trained_through=int(max(years)),training_seasons=len(years),training_series=int(historical.series_id.nunique()),features=features,params=PARAMS,
        protocol='10 initial training seasons; outer walk-forward through 2026; inner final 3 training seasons for ablations; fixed 600 trees; 1000 whole-season bootstrap resamples.',
        selection='Block B/C ablations, then at most one core-feature deletion; improvement threshold 0.003 inner log loss. No outer or current-season outcome selects features.',
        baseline_note='Deterministic SRS log loss clips probabilities to [1e-6, 1-1e-6]; logistic baseline is evaluation-only.',
        metrics=bootstrap(predictions),folds=folds,inner_ablations=ablations,final_inner_ablations=final_scores)
    (ARTIFACTS/'evaluation.json').write_text(json.dumps(report,indent=2))
    print('Selected',features,flush=True)
    print(json.dumps(report['metrics'],indent=2),flush=True)

if __name__ == '__main__':
    run()
