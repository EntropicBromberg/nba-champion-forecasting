"""Refresh current regular-season inputs and forecast qualification, play-in and title odds."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import time
import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier
from data_pipeline import DATA, TARGET, TEAM_FEATURES, season_features, standardize, team_key, matchup
from scripts.fetch_reference import fetch
from scripts.fetch_logs import season as fetch_logs
from train import ARTIFACTS

SNAPSHOT = DATA / 'live_snapshot.json'
PRIOR_GAMES = 20
SIMULATIONS = 50000


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def blend_strength(current, previous):
    """20 prior-season pseudo-games damp noisy October inputs; observed records stay intact."""
    result = current.copy()
    prior = previous.set_index('team')
    weight = result.gp / (result.gp + PRIOR_GAMES)
    # SRS is ill-conditioned while the opponent graph is sparse in the opening weeks.
    result.loc[result.gp.lt(10),'srs'] = result.loc[result.gp.lt(10),'mov']
    # Availability and age describe the current roster; do not smooth them into last year's roster.
    columns = ['srs','ortg','drtg','mov','top3_bpm','depth_bpm','form_resid','luck_resid',
               'efg_net','tov_net','orb_net','ftr_net']
    for col in columns:
        old = result.team.map(prior[col])
        result[col] = weight * result[col] + (1-weight) * old.fillna(result[col])
    return standardize(result)


def get_schedule(teams):
    """ESPN publishes 80 games before NBA Cup matchups are assigned; keep those gaps explicit."""
    cache = DATA/'raw'/f'schedule_{TARGET}.json'
    if cache.exists() and time.time() - cache.stat().st_mtime < 86400:
        return json.loads(cache.read_text())
    url = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
    response = requests.get(url+'/teams', params={'limit':100}, timeout=20)
    response.raise_for_status()
    identities = response.json()['sports'][0]['leagues'][0]['teams']
    lookup = {team_key(r['team']['displayName']):r['team']['id'] for r in identities}
    codes = {team_key(t.name):t.team for t in teams.itertuples()}
    if not set(codes).issubset(lookup):
        raise ValueError('Schedule identities do not cover all 30 teams')
    def download(name):
        r = requests.get(url+f'/teams/{lookup[name]}/schedule', params={'season':TARGET,'seasontype':2}, timeout=20)
        r.raise_for_status()
        return r.json().get('events',[])
    games = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for events in pool.map(download,codes):
            for event in events:
                if int(event.get('season',{}).get('year',0)) != TARGET:
                    continue
                if int(event.get('seasonType',{}).get('type',2)) != 2:
                    continue
                competition = event['competitions'][0]
                notes = ' '.join(n.get('headline','') for n in competition.get('notes',[])).lower()
                if 'championship' in notes and ('cup' in notes or 'tournament' in notes):
                    continue  # NBA Cup final does not count in regular-season records.
                competitors = competition['competitors']
                by_venue = {c['homeAway']:codes.get(team_key(c['team']['displayName'])) for c in competitors}
                if not by_venue.get('home') or not by_venue.get('away'):
                    continue
                date = pd.Timestamp(event['date']).tz_convert('America/New_York').date().isoformat()
                games[event['id']] = dict(date=date,home=by_venue['home'],away=by_venue['away'])
    result = sorted(games.values(),key=lambda x:(x['date'],x['home'],x['away']))
    coverage = {code:sum(g['home']==code or g['away']==code for g in result) for code in codes.values()}
    if any(count < 75 or count > 82 for count in coverage.values()):
        raise ValueError(f'Incomplete or duplicate regular-season schedule: {coverage}')
    atomic_json(cache,result)
    return result


def probability_tables(model, features, teams):
    """One batched XGBoost inference supplies all possible series and single-game matchups."""
    indexed = teams.set_index('team',drop=False)
    codes = teams.team.tolist()
    rows, keys = [], []
    for rnd in range(1,5):
        for i in range(len(codes)):
            for j in range(i+1,len(codes)):
                for home in [i,j]:
                    for a,b in [(i,j),(j,i)]:
                        rows.append(matchup(indexed,codes[a],codes[b],rnd,home=codes[home]))
                        keys.append((rnd-1,a,b,home))
    raw = model.predict_proba(pd.DataFrame(rows)[features])[:,1]
    probabilities = np.full((4,len(codes),len(codes),len(codes)),.5,dtype=np.float64)
    for k in range(0,len(keys),2):
        rnd,a,b,home = keys[k]
        p = float((raw[k]+1-raw[k+1])/2)
        probabilities[rnd,a,b,home] = p
        probabilities[rnd,b,a,home] = 1-p
    # Invert P(win best-of-seven) under an iid game approximation for regular/play-in games.
    grid = np.linspace(0,1,10001)
    best7 = 35*grid**4 - 84*grid**5 + 70*grid**6 - 20*grid**7
    single = np.interp(probabilities[0],best7,grid)
    return probabilities,single


def simulate(teams, model, features, schedule, played, simulations=SIMULATIONS, seed=0, dated_tables=None):
    teams = teams.reset_index(drop=True)
    if len(teams)!=30 or teams.team.nunique()!=30 or teams.conference.value_counts().to_dict()!={'E':15,'W':15}:
        raise ValueError('Live forecast needs 15 teams per conference')
    if not teams.gp.between(0,82).all():
        raise ValueError('Invalid regular-season game counts')
    rng = np.random.default_rng(seed)
    codes = {code:i for i,code in enumerate(teams.team)}
    series, single = probability_tables(model,features,teams)
    wins = np.tile(teams.wins.to_numpy(dtype=np.int16),(simulations,1))
    remaining = (82 - teams.gp.to_numpy()).astype(int)
    for game in schedule:
        home,away = codes[game['home']],codes[game['away']]
        if (game['home'],game['date']) in played and (game['away'],game['date']) in played:
            continue
        if remaining[home]<=0 or remaining[away]<=0:
            raise ValueError('Schedule and observed game counts disagree')
        game_single = single if dated_tables is None else dated_tables(game['date'])[1]
        won = rng.random(simulations)<game_single[home,away,home]
        wins[:,home] += won
        wins[:,away] += ~won
        remaining[home]-=1
        remaining[away]-=1
    if remaining.min()<0 or remaining.max()>4:
        raise ValueError(f'Too many unassigned remaining games: {remaining.tolist()}')
    # Cup/TBD opponents are not yet published. Fill only those slots against a league-average opponent.
    if dated_tables is not None:
        # Cup/TBD slots fall in December; use the dated injury state there too.
        _, single = dated_tables(f'{TARGET-1}-12-15')
    for i,count in enumerate(remaining):
        if count:
            p = np.mean([.5*(single[i,j,i]+single[i,j,j]) for j in range(30) if i!=j])
            wins[:,i] += rng.binomial(int(count),p,size=simulations)
    projected = wins.mean(axis=0)
    qualifiers = np.zeros(30,dtype=np.int64)
    def draw(a,b,rnd=None,home=None):
        home = a if home is None else home
        active_series,active_single=series,single
        if dated_tables is not None:
            # Approximate round dates; statuses expire at the feed's estimated return.
            dates={None:f'{TARGET}-04-14',1:f'{TARGET}-04-20',2:f'{TARGET}-05-06',3:f'{TARGET}-05-22',4:f'{TARGET}-06-06'}
            active_series,active_single=dated_tables(dates[rnd])
        p = active_single[a,b,home] if rnd is None else active_series[rnd-1,a,b,home]
        return np.where(rng.random(simulations)<p,a,b)
    conference_winners=[]
    for conference in ['E','W']:
        members = np.flatnonzero(teams.conference.eq(conference).to_numpy())
        # Tied projected records receive random tiebreaks, not the full NBA tiebreak hierarchy.
        order = np.argsort(-(wins[:,members]+rng.random((simulations,15))*.99),axis=1)
        seeds = members[order]
        seven,eight,nine,ten=(seeds[:,i] for i in [6,7,8,9])
        seed7=draw(seven,eight)
        loser78=np.where(seed7==seven,eight,seven)
        winner910=draw(nine,ten)
        seed8=draw(loser78,winner910)
        bracket = np.column_stack([seeds[:,0],seed8,seeds[:,3],seeds[:,4],seeds[:,1],seed7,seeds[:,2],seeds[:,5]])
        qualifiers += np.bincount(bracket.ravel(),minlength=30)
        for rnd in range(1,4):
            next_round=[]
            for column in range(0,bracket.shape[1],2):
                a,b=bracket[:,column],bracket[:,column+1]
                # Seed 7/8 slots have been restored; standings rank establishes conference home court.
                rank_a=np.argmax(seeds==a[:,None],axis=1)
                rank_b=np.argmax(seeds==b[:,None],axis=1)
                home=np.where(rank_a<rank_b,a,b)
                next_round.append(draw(a,b,rnd,home))
            bracket=np.column_stack(next_round)
        conference_winners.append(bracket[:,0])
    a,b=conference_winners
    home=np.where(wins[np.arange(simulations),a]>wins[np.arange(simulations),b],a,b)
    tied=wins[np.arange(simulations),a]==wins[np.arange(simulations),b]
    home=np.where(tied & (rng.random(simulations)<.5),a,home)
    champion=draw(a,b,4,home)
    output=teams[['team','name','conference','wins','losses','gp']].copy()
    output['projected_wins']=projected
    output['playoff_probability']=qualifiers/simulations
    output['title_probability']=np.bincount(champion,minlength=30)/simulations
    output['simulation_se']=np.sqrt(output.title_probability*(1-output.title_probability)/simulations)
    if not np.isclose(output.title_probability.sum(),1) or not np.isclose(output.playoff_probability.sum(),16):
        raise ValueError('Invalid simulated probability totals')
    return output.sort_values('title_probability',ascending=False).reset_index(drop=True),int(remaining.sum())


def refresh():
    report=json.loads((ARTIFACTS/'evaluation.json').read_text())
    if report['trained_through']!=TARGET-1 or report['training_seasons']!=25:
        raise ValueError('Run python train.py to train on all 25 completed seasons first')
    for suffix in ['', '_advanced']:
        fetch(f'/leagues/NBA_{TARGET}{suffix}.html',force=True)
    fetch_logs(TARGET,force=True)
    current=season_features(TARGET)
    if current.season.nunique()!=1 or current.season.iloc[0]!=TARGET:
        raise ValueError('Source returned the wrong season')
    previous=pd.read_csv(DATA/'teams.csv')
    previous=previous.loc[previous.season.eq(TARGET-1)]
    strength=blend_strength(current,previous)
    if not np.isfinite(strength[['z_'+f for f in TEAM_FEATURES]].to_numpy()).all():
        raise ValueError('Current-season features are incomplete')
    logs=pd.read_parquet(DATA/'raw/logs'/f'{TARGET}.parquet')
    codes={team_key(t.name):t.team for t in current.itertuples()}
    games=logs.loc[logs.player.eq('')]
    played={(codes[team_key(g.team)],str(g.date)[:10]) for g in games.itertuples()}
    model=XGBClassifier()
    model.load_model(ARTIFACTS/'xgboost.json')
    odds,unassigned=simulate(strength,model,report['features'],get_schedule(current),played)
    result=dict(season=TARGET,checked_at=datetime.now(timezone.utc).isoformat(),
        games_through=current.cutoff.max(),trained_through=report['trained_through'],
        training_seasons=report['training_seasons'],training_series=report['training_series'],
        simulations=SIMULATIONS,seed=0,prior_games=PRIOR_GAMES,unassigned_team_games=unassigned,
        odds=json.loads(odds.to_json(orient='records')),teams=json.loads(current.to_json(orient='records')),
        sources=['https://www.basketball-reference.com/leagues/NBA_2027.html',
                 'https://basketball-data.billmill.org/nba/metadata.json',
                 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams'])
    atomic_json(SNAPSHOT,result)
    return result


def refresh_or_cached():
    try:
        return refresh(),None
    except (requests.RequestException,ValueError,KeyError,TypeError,FileNotFoundError,OSError) as error:
        if not SNAPSHOT.exists():
            raise RuntimeError(f'Live refresh failed and no saved snapshot exists: {error}') from error
        snapshot=json.loads(SNAPSHOT.read_text())
        if snapshot.get('season')!=TARGET or snapshot.get('trained_through')!=TARGET-1:
            raise RuntimeError('Saved snapshot does not match the current season/model') from error
        return snapshot,f'Refresh failed ({type(error).__name__}); showing the last successful snapshot.'

if __name__=='__main__':
    snapshot=refresh()
    print(pd.DataFrame(snapshot['odds'])[['name','wins','losses','title_probability']].head(10).to_string(index=False))
    print('Games through',snapshot['games_through'],'checked',snapshot['checked_at'])
