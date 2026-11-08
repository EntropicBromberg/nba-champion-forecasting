"""Contained modern-season demo: draw team quality once, reuse through the title game."""
import numpy as np
from ratings import draw_strengths
from probabilities import best7


def simulate(snapshot,fixtures,conferences,probabilities,uncertain,paths=10000,seed=0):
    codes=snapshot['codes'];index=snapshot['index'];n=len(codes)
    if n!=30 or sorted(conferences.count(c) for c in ['E','W'])!=[15,15]:raise ValueError('Modern 30-team format only')
    strength=draw_strengths(snapshot['mean'],snapshot['covariance'],paths,seed+1) if uncertain else np.tile(snapshot['mean'],(paths,1))
    rng=np.random.default_rng(seed)  # Common outcome draws in both arms; quality uses a separate generator.
    wins=np.tile(snapshot['wins'].astype(int),(paths,1))
    for row in fixtures.itertuples():
        i,j=index[row.a],index[row.b]
        p=probabilities.conditional(strength[:,i]-strength[:,j],row.home_court,row.d_rest)
        won=rng.random(paths)<p;wins[:,i]+=won;wins[:,j]+=~won
    playoff=np.zeros(n);conference_winners=[]
    def game(a,b,home):
        p=probabilities.conditional(strength[np.arange(paths),a]-strength[np.arange(paths),b],home)
        return np.where(rng.random(paths)<p,a,b)
    def series(a,b,home_a):
        difference=strength[np.arange(paths),a]-strength[np.arange(paths),b]
        p_home=probabilities.conditional(difference,1.)
        p_away=probabilities.conditional(difference,0.)
        p=best7([np.where(home_a,p_home,p_away) if h else np.where(home_a,p_away,p_home)
                 for h in [True,True,False,False,True,False,True]])
        return np.where(rng.random(paths)<p,a,b)
    for conference in ['E','W']:
        members=np.flatnonzero(np.array(conferences)==conference)
        ranked=members[np.argsort(-(wins[:,members]+rng.random((paths,15))*.99),axis=1)]
        seven,eight,nine,ten=[ranked[:,i] for i in [6,7,8,9]]
        seed7=game(seven,eight,1.)
        lower=game(nine,ten,1.)
        loser=np.where(seed7==seven,eight,seven)
        seed8=game(loser,lower,1.)
        bracket=np.column_stack([ranked[:,0],seed8,ranked[:,3],ranked[:,4],ranked[:,1],seed7,ranked[:,2],ranked[:,5]])
        playoff+=np.bincount(bracket.ravel(),minlength=n)
        for _ in range(3):
            next_round=[]
            for column in range(0,bracket.shape[1],2):
                a,b=bracket[:,column],bracket[:,column+1]
                home_a=np.argmax(ranked==a[:,None],axis=1)<np.argmax(ranked==b[:,None],axis=1)
                next_round.append(series(a,b,home_a))
            bracket=np.column_stack(next_round)
        conference_winners.append(bracket[:,0])
    a,b=conference_winners
    count_a=wins[np.arange(paths),a];count_b=wins[np.arange(paths),b]
    home_a=(count_a>count_b)|((count_a==count_b)&(rng.random(paths)<.5))
    champion=series(a,b,home_a)
    title=np.bincount(champion,minlength=n)/paths
    if not np.isclose(title.sum(),1):raise AssertionError('Title probabilities do not sum to one')
    return dict(title=title,playoff=playoff/paths,projected_wins=wins.mean(axis=0),
                title_se=np.sqrt(title*(1-title)/paths))
