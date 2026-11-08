"""Cache compact public bulk game logs for the 25 seasons ending in 2026."""
from pathlib import Path
from io import BytesIO
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/raw/logs'
CACHE.mkdir(parents=True, exist_ok=True)

def read(url):
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return pd.read_parquet(BytesIO(response.content))

def season(year, force=False):
    target = CACHE / f'{year}.parquet'
    if target.exists() and not force:
        return
    if year < 2010:
        base = 'https://raw.githubusercontent.com/sportsdataverse/hoopR-data/main/nba'
        teams = read(f'{base}/team_box/parquet/team_box_{year}.parquet')
        players = read(f'{base}/player_box/parquet/player_box_{year}.parquet')
        teams = teams.loc[teams.season_type.eq(2)].copy()
        players = players.loc[players.season_type.eq(2)].copy()
        teams['mov'] = teams.team_score - teams.opponent_team_score
        teams = teams.rename(columns={'team_display_name':'team', 'game_date':'date'})
        players = players.rename(columns={'team_display_name':'team', 'game_date':'date',
                                         'athlete_display_name':'player', 'minutes':'minutes'})
    else:
        base = f'https://basketball-data.billmill.org/nba/stats'
        teams = read(f'{base}/team_game_logs/season={year}/data.parquet')
        players = read(f'{base}/player_game_logs/season={year}/data.parquet')
        names = read(f'{base}/player_seasons/season={year}/data.parquet')
        team_names = read(f'{base}/team_seasons/season={year}/data.parquet')
        teams = teams.loc[teams.game_id.str.startswith('002')].copy()
        players = players.loc[players.game_id.str.startswith('002')].copy()
        teams = teams.merge(team_names[['team_id','full_name']], on='team_id', validate='many_to_one')
        players = players.merge(names[['player_id','name']], on='player_id', validate='many_to_one')
        players = players.merge(team_names[['team_id','full_name']], on='team_id', validate='many_to_one')
        teams = teams.rename(columns={'full_name':'team','game_date':'date','plus_minus':'mov'})
        players = players.rename(columns={'full_name':'team','game_date':'date','name':'player','min':'minutes'})
    teams['player'] = ''
    teams['minutes'] = float('nan')
    players['mov'] = float('nan')
    columns = ['game_id','date','team','player','minutes','mov']
    logs = pd.concat([teams[columns], players[columns]], ignore_index=True)
    logs['game_id'] = logs.game_id.astype(str)
    temporary = target.with_suffix('.tmp')
    logs.to_parquet(temporary, index=False)
    temporary.replace(target)
    print(year, len(teams), len(players), flush=True)

if __name__ == '__main__':
    for year in range(2002, 2027):
        season(year)
