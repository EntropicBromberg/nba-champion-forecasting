"""Isolate legacy experiment module names; export inputs/models without touching old artifacts."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'experiments/player_minutes'
sys.path[:0]=[str(OLD),str(ROOT)]
import compare_production as production
from model import fit

OUT=Path(__file__).resolve().parent/'cache';OUT.mkdir(exist_ok=True)
meta=json.loads((ROOT/'artifacts/evaluation.json').read_text())
rows=pd.read_csv(ROOT/'data/matchups.csv')
ranges=pd.read_csv(ROOT/'experiments/team_uncertainty/results/strength_ranges.csv')
for fold in meta['folds']:
    year=fold['season']
    if not 2013<=year<=2026:continue
    path=OUT/f'production_{year}.json'
    if not path.exists():fit(rows.loc[rows.season.lt(year)],fold['features']).save_model(path)
    for point in json.loads((OLD/f'cache/snapshots_{year}.json').read_text()):
        target=OUT/f'inputs_{year}_{point}.parquet'
        if not target.exists():
            teams,quality=production.inputs(year,point)
            teams.to_parquet(target,index=False)
print('Production fold models and input caches ready',flush=True)
