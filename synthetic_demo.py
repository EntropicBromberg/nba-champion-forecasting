"""Invented inputs for an offline demonstration of the primary simulation engine."""
import numpy as np
import pandas as pd
from data_pipeline import FEATURES, TEAM_FEATURES
from live import simulate


class SyntheticMatchupModel:
    """Illustrative logistic rule, not a fitted NBA model."""
    def predict_proba(self, rows):
        probability=1/(1+np.exp(-np.asarray(rows['d_srs'],dtype=float)))
        return np.column_stack([1-probability,probability])


def synthetic_forecast(simulations=5000, strength_penalty=0.):
    if not 100<=simulations<=50000:
        raise ValueError('Choose 100 to 50,000 simulations')
    if not 0<=strength_penalty<=3:
        raise ValueError('Synthetic strength penalty must be between 0 and 3')
    teams=pd.DataFrame([dict(team=f'T{i+1:02}',name=f'Synthetic Team {i+1:02}',
        conference='E' if i<15 else 'W',wins=41,losses=41,gp=82,
        standing_order=i%15,season=2027,
        **{'z_'+feature:0. for feature in TEAM_FEATURES}) for i in range(30)])
    teams['z_srs']=np.tile(np.linspace(1.5,-1.5,15),2)
    teams.loc[teams.team.eq('T01'),'z_srs']-=strength_penalty
    odds,_=simulate(teams,SyntheticMatchupModel(),FEATURES,[],set(),simulations=simulations)
    return odds
