"""Small opponent-adjusted Gaussian filter; no player data or production dependencies."""
import numpy as np
import pandas as pd

class Ratings:
    def __init__(self,codes,prior_mean,prior_variance,observation_variance,process_variance=.04):
        self.codes=list(codes);self.index={code:i for i,code in enumerate(codes)}
        n=len(codes);self.center=np.eye(n)-np.ones((n,n))/n
        self.mean=self.center@np.asarray(prior_mean,dtype=float)
        self.covariance=prior_variance*self.center
        self.observation_variance=float(observation_variance)
        self.process_variance=float(process_variance)
        self.max_date=None
        self.wins=np.zeros(n);self.gp=np.zeros(n)
        self.last={c:pd.NaT for c in codes}

    def update(self,a,b,margin,home_court,date,regular=True):
        i,j=self.index[a],self.index[b]
        noise=np.zeros(len(self.codes));noise[[i,j]]=self.process_variance
        predicted=self.covariance+self.center@np.diag(noise)@self.center
        contrast=predicted[:,i]-predicted[:,j]
        variance=predicted[i,i]+predicted[j,j]-2*predicted[i,j]+self.observation_variance
        residual=margin-3*(2*home_court-1)-(self.mean[i]-self.mean[j])
        self.mean=self.center@(self.mean+contrast*residual/variance)
        self.covariance=predicted-np.outer(contrast,contrast)/variance
        self.covariance=self.center@((self.covariance+self.covariance.T)/2)@self.center
        date=pd.Timestamp(date)
        self.max_date=date if self.max_date is None else max(self.max_date,date)
        self.last[a]=self.last[b]=date
        if regular:
            self.gp[[i,j]]+=1;self.wins[i]+=margin>0;self.wins[j]+=margin<0

    def snapshot(self,cutoff):
        cutoff=pd.Timestamp(cutoff)
        if self.max_date is not None and self.max_date>=cutoff:raise AssertionError('Same-day/future input')
        return dict(codes=self.codes,index=self.index.copy(),mean=self.mean.copy(),covariance=self.covariance.copy(),
            cutoff=cutoff,max_input_date=self.max_date,wins=self.wins.copy(),gp=self.gp.copy(),last=self.last.copy(),
            observation_variance=self.observation_variance)


def draw_strengths(mean,covariance,paths=10000,seed=1):
    """One row per complete season path; antithetic draws retain the posterior mean."""
    if paths%2:raise ValueError('Use an even number of paths')
    values,vectors=np.linalg.eigh(covariance)
    if values.min() < -1e-8:raise ValueError('Invalid covariance')
    rng=np.random.default_rng(seed);z=rng.normal(size=(paths//2,len(mean)))
    shocks=np.vstack([z,-z])@(vectors*np.sqrt(values.clip(min=0))).T
    return np.asarray(mean)+shocks
