"""Publish aligned aggregate scenario outputs without source records or injury text."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.publish_forecast import public_forecast
MODELS={
    'uncertainty':('experimental_snapshot.json','Experimental — uncertainty-aware'),
    'injury':('injury_snapshot.json','Experimental — injury-adjusted'),
    'stress':('injury_uncertainty_snapshot.json','Experimental — uncertainty + injury stress'),
}


def fingerprint(primary):
    clean=public_forecast(primary)
    identity=dict(season=clean['season'],games_through=clean['games_through'],
                  odds=sorted(clean['odds'],key=lambda r:r['team']))
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def scenario(identifier,snapshot,primary,uncertainty=None):
    if not snapshot.get('label','').startswith(MODELS[identifier][1]):
        raise ValueError('Wrong scenario identity')
    if snapshot['season']!=primary['season'] or snapshot['games_through']!=primary['games_through']:
        raise ValueError('Scenario and primary forecast cutoffs differ')
    cleaned=public_forecast(dict(season=primary['season'],trained_through=primary['trained_through'],
        training_seasons=primary['training_seasons'],checked_at=snapshot['computed_at'],
        games_through=snapshot['games_through'],simulations=snapshot['simulations'],odds=snapshot['odds']))
    names={r['team']:r['name'] for r in primary['odds']}
    if {r['team']:r['name'] for r in cleaned['odds']}!=names:
        raise ValueError('Scenario team identities differ')
    baseline={r['team']:r['title_probability'] for r in primary['odds']}
    if identifier=='stress':
        if uncertainty is None:raise ValueError('Stress comparison requires aligned uncertainty output')
        baseline={r['team']:r['title_probability'] for r in uncertainty['odds']}
    original={r['team']:r for r in snapshot['odds']}
    baseline_key={'injury':'production_title_probability','stress':'uncertainty_title_probability'}.get(identifier)
    if baseline_key and any(not math.isclose(float(original[t][baseline_key]),p,abs_tol=1e-8) for t,p in baseline.items()):
        raise ValueError('Recorded paired baseline does not match published baseline')
    rows=[dict(r,baseline_title_probability=baseline[r['team']]) for r in cleaned['odds']]
    result=dict(id=identifier,status='available',computed_at=cleaned['checked_at'],
                games_through=cleaned['games_through'],simulations=cleaned['simulations'],
                baseline='uncertainty' if identifier=='stress' else 'primary',odds=rows)
    if identifier=='stress':
        settings=snapshot['settings']
        result['settings']={key:settings[key] for key in ['severity','recovery_days']}
        result['settings']['team_specific_adjustments']=bool(settings.get('team_multipliers') or settings.get('return_overrides'))
    return result


def build(primary,data_dir):
    primary=public_forecast(primary)
    models={};uncertainty=None
    for identifier,(filename,_) in MODELS.items():
        try:
            snapshot=json.loads((data_dir/filename).read_text())
            result=scenario(identifier,snapshot,primary,uncertainty)
            if identifier=='uncertainty':uncertainty=result
            models[identifier]=result
        except (OSError,ValueError,KeyError,TypeError):
            models[identifier]=dict(id=identifier,status='unavailable',
                reason='No matching scenario output is available for this forecast.')
    return dict(schema_version=1,primary_fingerprint=fingerprint(primary),models=models)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--primary',type=Path,required=True)
    args=parser.parse_args()
    bundle=build(json.loads(args.primary.read_text()),args.data_dir)
    output=ROOT/'reports/experimental_forecasts.json'
    output.write_text(json.dumps(bundle,indent=2,allow_nan=False)+'\n')
    print('Published scenario availability:',{k:v['status'] for k,v in bundle['models'].items()})
