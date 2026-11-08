"""Exact Gaussian marginalization for a shallow, piecewise-constant XGBoost game model."""
import numpy as np
import pandas as pd
from scipy.special import ndtr

FEATURES=['d_strength','home_court','d_rest']


def best7(probabilities):
    """Seven possible game probabilities; scalar or equal-shaped arrays."""
    shape=np.asarray(probabilities[0]).shape
    dp=np.zeros((5,5)+shape);dp[0,0]=1
    for won in range(4):
        for lost in range(4):
            p=probabilities[won+lost]
            dp[won+1,lost]+=dp[won,lost]*p
            dp[won,lost+1]+=dp[won,lost]*(1-p)
    return dp[4,:4].sum(axis=0)


class GameProbabilities:
    def __init__(self,model):
        self.model=model
        tree=model.get_booster().trees_to_dataframe()
        cut=tree.loc[tree.Feature.eq('d_strength'),'Split'].to_numpy(dtype=np.float32).astype(float)
        self.cuts=np.unique(np.concatenate([cut,-cut]))
        if len(self.cuts):
            reps=np.concatenate([[self.cuts[0]-1],(self.cuts[:-1]+self.cuts[1:])/2,[self.cuts[-1]+1]])
        else:reps=np.array([0.])
        self.interval={};self.at_cut={}
        for home in [0.,.5,1.]:
            for rest in range(-7,8):
                self.interval[(home,rest)]=self.direct(reps,home,rest)
                self.at_cut[(home,rest)]=self.direct(self.cuts,home,rest) if len(self.cuts) else np.array([])

    def direct(self,difference,home,rest):
        diff=np.atleast_1d(difference)
        ab=pd.DataFrame(dict(d_strength=diff,home_court=np.broadcast_to(home,diff.shape),d_rest=np.broadcast_to(rest,diff.shape)))
        ba=ab.copy();ba.d_strength=-ba.d_strength;ba.home_court=1-ba.home_court;ba.d_rest=-ba.d_rest
        raw=self.model.predict_proba(pd.concat([ab,ba],ignore_index=True)[FEATURES])[:,1]
        return np.clip(.5*(raw[:len(ab)]+1-raw[len(ab):]),1e-6,1-1e-6)

    def conditional(self,difference,home,rest=0):
        difference=np.asarray(difference,dtype=np.float32).astype(float)
        home=np.broadcast_to(home,difference.shape);rest=np.broadcast_to(rest,difference.shape)
        answer=np.empty(difference.shape)
        for h in np.unique(home):
            for r in np.unique(rest):
                mask=(home==h)&(rest==r)
                if not mask.any():continue
                values=difference[mask]
                bins=np.searchsorted(self.cuts,values,side='right')
                p=self.interval[(float(h),int(r))][bins].copy()
                if len(self.cuts):
                    index=np.searchsorted(self.cuts,values)
                    hit=(index<len(self.cuts))&(self.cuts[np.minimum(index,len(self.cuts)-1)]==values)
                    p[hit]=self.at_cut[(float(h),int(r))][index[hit]]
                answer[mask]=p
        return answer

    def integrate(self,mean,sd,values,fixed):
        """Integrate the forest steps; no Monte Carlo error in reported probabilities."""
        mean=np.atleast_1d(mean);sd=np.broadcast_to(sd,mean.shape)
        result=np.asarray(fixed,dtype=float).copy().reshape(mean.shape)
        active=sd>1e-10
        if active.any():
            probability=ndtr((mean[active,None]-self.cuts[None,:])/sd[active,None])
            result[active]=values[0]+probability@np.diff(values)
        return np.clip(result,1e-6,1-1e-6)

    def games(self,mean,variance,home,rest,uncertain):
        mean=np.asarray(mean);home=np.asarray(home);rest=np.asarray(rest)
        fixed=self.conditional(mean,home,rest)
        if not uncertain:return fixed
        output=fixed.copy()
        for h in np.unique(home):
            for r in np.unique(rest):
                mask=(home==h)&(rest==r)
                if mask.any():output[mask]=self.integrate(mean[mask],np.sqrt(np.maximum(variance[mask],0)),
                    self.interval[(float(h),int(r))],fixed[mask])
        return output

    def series(self,mean,variance,home_a=True,neutral=False,old_finals=False,uncertain=False):
        pattern=[True,True,False,False,False,True,True] if old_finals else [True,True,False,False,True,False,True]
        home=[.5 if neutral else float(h==home_a) for h in pattern]
        fixed=float(best7([self.conditional(np.array([mean]),h)[0] for h in home]))
        if not uncertain:return fixed
        values=best7([self.interval[(h,0)] for h in home])
        return float(self.integrate(np.array([mean]),np.sqrt(max(variance,0)),values,np.array([fixed]))[0])
