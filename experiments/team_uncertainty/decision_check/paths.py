"""One generic path engine for paired historical diagnostics and the labeled experimental view."""
import numpy as np


def record_score(wins,games,tiebreak):
    return wins/np.maximum(games,1)+tiebreak*1e-7

def games_behind(wins_eight,wins_nine,gp_eight,gp_nine):
    return wins_eight-wins_nine+.5*(gp_nine-gp_eight)

def season_paths(snapshot,fixtures,conferences,game_probability,series_probability,paths=10000,seed=0,year=2027,tbd=None,quality=None,date_hook=None):
    rng=np.random.default_rng(seed);n=len(snapshot['codes']);index=snapshot['index']
    wins=np.tile(snapshot['wins'].astype(int),(paths,1));row_index=np.arange(paths)
    final_gp=snapshot['gp'].astype(float).copy()
    for g in fixtures.itertuples():final_gp[[index[g.a],index[g.b]]]+=1
    if tbd is not None:final_gp+=np.asarray(tbd)
    for g in fixtures.itertuples():
        if date_hook is not None:date_hook(g.date)
        a,b=index[g.a],index[g.b]
        p=game_probability(a,b,g.home_court,g.d_rest)
        won=rng.random(paths)<p;wins[:,a]+=won;wins[:,b]+=~won
    if tbd is not None:
        if date_hook is not None:date_hook(f'{year-1}-12-15')
        for a,count in enumerate(tbd):
            if count:
                p=sum(.5*(game_probability(a,b,1.,0)+game_probability(a,b,0.,0)) for b in range(n) if b!=a)/(n-1)
                wins[:,a]+=rng.binomial(int(count),p,size=paths)
    if date_hook is not None:date_hook(f'{year}-04-14')
    qualified=np.zeros(n);finalists=[]
    def play(a,b,home):return np.where(rng.random(paths)<game_probability(a,b,home,0),a,b)
    for conference in ['E','W']:
        members=np.flatnonzero(np.asarray(conferences)==conference)
        ranked=members[np.argsort(-record_score(wins[:,members],final_gp[members],rng.random((paths,len(members)))),axis=1)]
        seed7,seed8=ranked[:,6],ranked[:,7]
        if year>=2021:
            seed7=play(ranked[:,6],ranked[:,7],1.)
            loser=np.where(seed7==ranked[:,6],ranked[:,7],ranked[:,6])
            lower=play(ranked[:,8],ranked[:,9],1.)
            seed8=play(loser,lower,1.)
        elif year==2020:
            eight,nine=ranked[:,7],ranked[:,8]
            first=play(eight,nine,.5);second=play(eight,nine,.5)
            close=games_behind(wins[row_index,eight],wins[row_index,nine],final_gp[eight],final_gp[nine])<=4
            seed8=np.where(close&(first==nine)&(second==nine),nine,eight)
        bracket=np.column_stack([ranked[:,0],seed8,ranked[:,3],ranked[:,4],ranked[:,1],seed7,ranked[:,2],ranked[:,5]])
        qualified+=np.bincount(bracket.ravel(),minlength=n)
        for round_number in range(1,4):
            next_round=[]
            for column in range(0,bracket.shape[1],2):
                a,b=bracket[:,column],bracket[:,column+1]
                home_a=np.argmax(ranked==a[:,None],axis=1)<np.argmax(ranked==b[:,None],axis=1)
                p=series_probability(a,b,round_number,home_a)
                next_round.append(np.where(rng.random(paths)<p,a,b))
            bracket=np.column_stack(next_round)
        finalists.append(bracket[:,0])
    a,b=finalists;wa=wins[row_index,a]/np.maximum(final_gp[a],1);wb=wins[row_index,b]/np.maximum(final_gp[b],1)
    home_a=(wa>wb)|((wa==wb)&(rng.random(paths)<.5))
    p=series_probability(a,b,4,home_a)
    champion=np.where(rng.random(paths)<p,a,b)
    counts=np.bincount(champion,minlength=n)
    return dict(title=counts/paths,title_smoothed=(counts+.5)/(paths+.5*n),playoff=qualified/paths,
        projected_wins=wins.mean(axis=0),title_se=np.sqrt((counts/paths)*(1-counts/paths)/paths))
