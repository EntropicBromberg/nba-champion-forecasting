"""Regular-season features and paired playoff-series labels. Season = ending year."""
from pathlib import Path
import json
import re
import unicodedata
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
YEARS = range(2002, 2027)
TARGET = 2027
CORE = ['d_srs','d_top3_bpm','d_depth_bpm_resid','d_def_share','d_form_resid',
        'd_luck_resid','d_gp_share_top8','d_star_out_flag','home_court','round']
FACTORS = ['d_efg_net','d_tov_net','d_orb_net','d_ftr_net']
OPTIONAL = ['d_age_top8']
DIRECTIONS = dict(zip(CORE + FACTORS + OPTIONAL, [1,1,1,0,0,0,1,-1,1,0,1,1,1,1,0]))
FEATURES = CORE + FACTORS + OPTIONAL
TEAM_FEATURES = [f[2:] for f in FEATURES if f.startswith('d_')]

def name_key(name):
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    name = re.sub(r'[^a-z ]', '', name)
    name = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', name)
    name = ''.join(name.split())
    return {'darrelarmstrong':'darrellarmstrong','nenehilario':'nene','ronartest':'mettaworldpeace','mettaartest':'mettaworldpeace',
            'pattimills':'pattymills','patrickmills':'pattymills', 'pjwashington':'pjwashington'}.get(name, name)

def team_key(name):
    name = ' '.join(name.split()).replace('NO/Oklahoma City Hornets', 'New Orleans Hornets')
    return name.replace('LA Clippers','Los Angeles Clippers').replace('New Orleans/Oklahoma City Hornets', 'New Orleans Hornets')

def soup(year, suffix=''):
    path = DATA / 'raw/leagues' / f'NBA_{year}{suffix}.html'
    text = path.read_text()
    try:
        text = text.encode('latin1').decode('utf-8')
    except UnicodeError:
        pass
    return BeautifulSoup(text.replace('<!--','').replace('-->',''), 'html.parser')

def cells(row):
    return {c.get('data-stat'): c.get_text(strip=True) for c in row.select('th[data-stat],td[data-stat]')}

def number(row, key):
    return float(row[key].replace(',', '')) if row[key] else float('nan')

def table(source, *ids):
    for key in ids:
        result = source.find('table', id=key)
        if result is not None:
            return result
    raise ValueError(f'Missing source table: {ids}')

def parse_teams(year):
    source = soup(year)
    conferences = {}
    ranks = {}
    for conf in ['E','W']:
        standings = table(source, f'confs_standings_{conf}', f'divs_standings_{conf}')
        i = 0
        for row in standings.select('tbody > tr'):
            a = row.select_one('a[href^="/teams/"]')
            if a:
                code = a['href'].split('/')[2]
                conferences[code] = conf
                ranks[code] = i
                i += 1
    rows = []
    advanced = table(source, 'advanced-team', 'misc_stats')
    for row in advanced.select('tbody > tr'):
        a = row.select_one('a[href^="/teams/"]')
        if not a:
            continue
        c = cells(row)
        code = a['href'].split('/')[2]
        get = lambda key: number(c, key)
        wins, losses = get('wins'), get('losses')
        rows.append(dict(season=year, team=code, name=a.get_text(strip=True), conference=conferences[code],
            standing_order=ranks[code], gp=wins+losses, wins=wins, losses=losses,
            srs=get('srs'), ortg=get('off_rtg'), drtg=get('def_rtg'), mov=get('mov'),
            luck_resid=wins/(wins+losses)-get('wins_pyth')/(wins+losses),
            efg_net=get('efg_pct')-get('opp_efg_pct'),
            tov_net=get('opp_tov_pct')-get('tov_pct'),
            orb_net=get('orb_pct')-(100-get('drb_pct')),
            ftr_net=get('ft_rate')-get('opp_ft_rate')))
    teams = pd.DataFrame(rows)
    if len(teams) not in [29,30]:
        raise ValueError(f'{year}: expected all NBA teams, got {len(teams)}')
    return teams

def parse_players(year):
    rows = []
    for row in table(soup(year, '_advanced'), 'advanced').select('tbody > tr'):
        c = cells(row)
        a = row.select_one('a[href^="/players/"]')
        t = row.select_one('a[href^="/teams/"]')
        if a is None or t is None:  # Excludes TOT/2TM/3TM aggregates.
            continue
        rows.append(dict(player=a.get_text(strip=True), player_id=a['href'].split('/')[-1][:-5],
            team=t['href'].split('/')[2], mp=number(c,'mp'), gp=number(c,'games'),
            bpm=number(c,'bpm'), age=number(c,'age')))
    return pd.DataFrame(rows).drop_duplicates(['team','player_id'])

def parse_series(year):
    """Only labels, entrants, and bracket structure use postseason source tables."""
    source = soup(year)
    series = []
    for row in table(source, 'all_playoffs').select('tbody > tr'):
        label = row.select_one('span.opener')
        if label is None:
            continue
        links = row.select('a[href^="/teams/"]')
        if len(links) != 2:
            continue
        text = label.get_text()
        rnd = 1 if 'First Round' in text else 2 if 'Semifinals' in text else 3 if 'Conference Finals' in text else 4
        winner, loser = [a['href'].split('/')[2] for a in links]
        games = source.find(id=label['data-id'])
        box = games.select_one('a[href^="/boxscores/"]') if games else None
        if box is None:
            raise ValueError(f'{year}: missing home-court source')
        home = box['href'].split('/')[-1][-8:-5]
        series.append(dict(season=year, round=rnd, winner=winner, loser=loser,
                           home=home, neutral=year == 2020))
    if len(series) != 15:
        raise ValueError(f'{year}: expected 15 series, got {len(series)}')
    return pd.DataFrame(series)

def zscore(series):
    std = series.std(ddof=0)
    return (series - series.mean()) / std if std > 0 else series * 0

def season_features(year):
    teams, players = parse_teams(year), parse_players(year)
    logs = pd.read_parquet(DATA / 'raw/logs' / f'{year}.parquet')
    logs['team'] = logs.team.map(team_key)
    logs['date'] = pd.to_datetime(logs.date)
    team_logs = logs.loc[logs.player.eq('')].drop_duplicates(['game_id','team'])
    player_logs = logs.loc[logs.player.ne('') & logs.minutes.gt(0)].copy()
    player_logs['key'] = player_logs.player.map(name_key)
    result = []
    for team in teams.itertuples():
        games = team_logs.loc[team_logs.team.eq(team_key(team.name))].sort_values(['date','game_id'])
        if len(games) != team.gp:
            raise ValueError(f'{year} {team.name}: {len(games)} game logs vs {team.gp} games')
        # Either eligibility threshold is sufficient; rankings always use team-stint minutes.
        roster = players.loc[players.team.eq(team.team)].copy()
        eligible = roster.loc[roster.mp.ge(500) | roster.mp.div(roster.gp).ge(15)]
        top = eligible.nlargest(8, 'mp')
        if len(top) < 4:
            raise ValueError(f'{year} {team.name}: insufficient eligible players')
        weighted = lambda frame, col: np.average(frame[col], weights=frame.mp)
        talent = (top.bpm + 2).clip(lower=0) * top.mp
        share = np.average(top.gp / team.gp, weights=talent) if talent.sum() else weighted(top,'gp')/team.gp
        recent = set(games.tail(5).game_id)
        star_flags = []
        for star in top.head(2).itertuples():
            played = player_logs.loc[player_logs.team.eq(team_key(team.name)) & player_logs.key.eq(name_key(star.player))]
            if played.empty:
                raise ValueError(f'{year}: unmatched top-two player {star.player} ({team.name})')
            star_flags.append(len(recent - set(played.game_id)) >= 3)
        result.append(dict(team=team.team, top3_bpm=weighted(top.head(3),'bpm'),
            depth_bpm=weighted(top.iloc[3:8],'bpm'), gp_share_top8=share,
            star_out_flag=float(any(star_flags)), age_top8=weighted(top,'age'),
            form_resid=games.tail(25).mov.mean()-games.mov.mean(),
            cutoff=games.date.max().date().isoformat()))
    teams = teams.merge(pd.DataFrame(result), on='team', validate='one_to_one')
    return standardize(teams)

def standardize(teams):
    teams = teams.copy()
    x = np.column_stack([np.ones(len(teams)), teams.top3_bpm])
    teams['depth_bpm_resid'] = teams.depth_bpm - x @ np.linalg.lstsq(x, teams.depth_bpm, rcond=None)[0]
    teams['def_share'] = zscore(-teams.drtg) - zscore(teams.ortg)
    for feature in TEAM_FEATURES:
        teams['z_' + feature] = zscore(teams[feature])
    return teams

def matchup(teams, a, b, rnd, home=None, neutral=False):
    a, b = teams.loc[a], teams.loc[b]
    row = {'d_'+f: a['z_'+f] - b['z_'+f] for f in TEAM_FEATURES}
    if home is None:
        # Record decides home court; source standings order breaks tied conference records.
        home_a = (a.wins/a.gp, -a.standing_order, a.team) > (b.wins/b.gp, -b.standing_order, b.team)
    else:
        home_a = home == a.team
    row.update(home_court=0.5 if neutral else float(home_a), round=rnd)
    return row

def build():
    DATA.mkdir(exist_ok=True)
    teams_all, series_all, paired = [], [], []
    for year in YEARS:
        teams = season_features(year)
        series = parse_series(year)
        indexed = teams.set_index('team', drop=False)
        for i, s in enumerate(series.itertuples()):
            a, b = sorted([s.winner, s.loser])
            label = int(a == s.winner)
            for a, b, y in [(a,b,label),(b,a,1-label)]:
                paired.append(dict(season=year, series_id=f'{year}-{i}', team_a=a, team_b=b, y=y,
                    **matchup(indexed,a,b,s.round,s.home,s.neutral)))
        teams_all.append(teams)
        series_all.append(series)
        print('Built',year, flush=True)
    pd.concat(teams_all).to_csv(DATA/'teams.csv',index=False)
    pd.concat(series_all).to_csv(DATA/'series.csv',index=False)
    pd.DataFrame(paired).to_csv(DATA/'matchups.csv',index=False)
    (DATA/'sources.json').write_text(json.dumps({
        'seasons':'2001–02 through 2025–26 (ending years 2002–2026)',
        'regular_season_tables':'https://www.basketball-reference.com/leagues/NBA_{year}.html',
        'regular_season_players':'https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html',
        'logs_2002_2009':'https://github.com/sportsdataverse/hoopR-data',
        'logs_2010_2026':'https://github.com/llimllib/nba_data',
        'excluded_optional':{'d_playoff_exp':'Complete pre-2002 career playoff minutes are not available in the bulk logs. Omitted rather than substituting truncated experience.'},
        'availability':'Top two by eligible regular-season team-stint minutes; missed >=3 of final 5 team regular-season games. Rest and injury are not distinguished.',
        'target':'2026–27 live forecast. All 25 completed seasons through 2025–26 are training data; current-season outcomes are never training labels.'
    },indent=2))

if __name__ == '__main__':
    build()
