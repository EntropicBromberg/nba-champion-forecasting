"""Adapters keep native production and persistent-strength probabilities separate."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
from scipy.special import expit,logit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
UNC=HERE.parent

def load_module(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

sys.path.insert(0,str(UNC))
research=load_module('_uncertainty_research',UNC/'run.py')
calibration=load_module('_earlier_calibration',UNC/'calibration_check/check.py')
from probabilities import GameProbabilities,best7
from ratings import draw_strengths


def snapshot(year,point):
    ranges=pd.read_csv(UNC/'results/strength_ranges.csv')
    part=ranges.loc[ranges.season.eq(year)&ranges.checkpoint.eq(point)]
    cutoff=pd.Timestamp(part.cutoff.iloc[0]);games=research.load_games(year)
    codes=sorted(research.HISTORY.loc[research.HISTORY.season.eq(year),'team'])
    index={c:i for i,c in enumerate(codes)};wins=np.zeros(len(codes));gp=np.zeros(len(codes));last={c:pd.NaT for c in codes}
    for g in games.loc[games.date.lt(cutoff)].itertuples():
        last[g.a]=last[g.b]=g.date
        if g.phase=='regular':
            i,j=index[g.a],index[g.b];gp[[i,j]]+=1;wins[i]+=g.y;wins[j]+=1-g.y
    posterior=np.load(UNC/f'cache/{year}_{point}.npz')
    return dict(codes=codes,index=index,mean=posterior['mean'],covariance=posterior['covariance'],wins=wins,gp=gp,last=last,
        cutoff=cutoff,max_input_date=pd.Timestamp(part.max_input_date.max())),games


class NativeProduction:
    def __init__(self,model,features,teams):
        self.codes=teams.team.tolist();n=len(teams)
        a=np.repeat(np.arange(n),n);b=np.tile(np.arange(n),n)
        names=[f[2:] for f in features if f.startswith('d_')]
        z=teams[['z_'+f for f in names]].to_numpy();difference=z[a]-z[b]
        parts=[]
        for rnd in range(1,5):
            for home in [0.,.5,1.]:
                ab=pd.DataFrame({'d_'+f:difference[:,i] for i,f in enumerate(names)})
                ab['home_court']=home;ab['round']=rnd
                ba=ab.copy()
                for f in names:ba['d_'+f]=-ba['d_'+f]
                ba.home_court=1-ba.home_court
                parts.extend([ab[features],ba[features]])
        raw=model.predict_proba(pd.concat(parts,ignore_index=True))[:,1].reshape(4,3,2,n*n)
        self.series_raw=np.clip(.5*(raw[:,:,0]+1-raw[:,:,1]),1e-6,1-1e-6).reshape(4,3,n,n)
        grid=np.linspace(0,1,10001);curve=35*grid**4-84*grid**5+70*grid**6-20*grid**7
        self.game_raw=np.interp(self.series_raw[0],curve,grid)
        self.configure(1.,1.)
    def configure(self,game_slope,series_slope):
        self.game=calibration.apply_slope(self.game_raw,game_slope)
        self.series=calibration.apply_slope(self.series_raw,series_slope)
    def game_probability(self,a,b,home,rest=0):return self.game[np.asarray(np.rint(2*np.asarray(home)),dtype=int),a,b]
    def series_probability(self,a,b,rnd,home_a,neutral=False):
        h=1 if neutral else 2*np.asarray(home_a,dtype=int)
        return self.series[rnd-1,h,a,b]


class UncertainPaths:
    def __init__(self,snapshot,model,paths=10000,seed=1,year=2027):
        self.strength=draw_strengths(snapshot['mean'],snapshot['covariance'],paths,seed)
        self.probabilities=model;self.rows=np.arange(paths);self.year=year
    def difference(self,a,b):return self.strength[self.rows,a]-self.strength[self.rows,b]
    def game_probability(self,a,b,home,rest=0):return self.probabilities.conditional(self.difference(a,b),home,rest)
    def series_probability(self,a,b,rnd,home_a):
        delta=self.difference(a,b)
        h=.5 if self.year==2020 else 1.;away=.5 if self.year==2020 else 0.
        p_home=self.probabilities.conditional(delta,h);p_away=self.probabilities.conditional(delta,away)
        pattern=[1,1,0,0,0,1,1] if self.year<=2013 and rnd==4 else [1,1,0,0,1,0,1]
        return best7([np.where(home_a,p_home,p_away) if flag else np.where(home_a,p_away,p_home) for flag in pattern])
