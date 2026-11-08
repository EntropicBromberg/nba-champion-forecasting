"""Isolated compact cache; does not modify application data or artifacts."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import sys
import requests
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from data_pipeline import team_key
CACHE=Path(__file__).resolve().parent/'cache'
CACHE.mkdir(exist_ok=True)
HISTORY=pd.read_csv(ROOT/'data/teams.csv')
ALIASES={'NJN':'BRK','NOH':'NOP','NOK':'NOP','SEA':'OKC','CHA':'CHO','BKN':'BRK','PHX':'PHO'}

def canonical(code):return ALIASES.get(code,code)

def read(url):
    r=requests.get(url,timeout=90);r.raise_for_status()
    return pd.read_parquet(BytesIO(r.content))

def download(year):
    path=CACHE/f'{year}.parquet'
    if path.exists():return
    if year<2010:
        base='https://raw.githubusercontent.com/sportsdataverse/hoopR-data/main/nba'
        t=read(f'{base}/team_box/parquet/team_box_{year}.parquet')
        p=read(f'{base}/player_box/parquet/player_box_{year}.parquet')
        names={team_key(r.name):canonical(r.team) for r in HISTORY.loc[HISTORY.season.eq(year)].itertuples()}
        t=t.loc[t.season_type.isin([2,3]) & t.team_display_name.map(team_key).isin(names)].copy()
        p=p.loc[p.season_type.isin([2,3]) & p.team_display_name.map(team_key).isin(names)].copy()
        t['team']=t.team_display_name.map(team_key).map(names)
        t['opponent']=t.opponent_team_display_name.map(team_key).map(names)
        t['home']=t.team_home_away.eq('home')
        t['mov']=pd.to_numeric(t.team_score)-pd.to_numeric(t.opponent_team_score)
        t['phase']=t.season_type.map({2:'regular',3:'playoff'})
        t=t.rename(columns={'game_date':'date'})
        p['team']=p.team_display_name.map(team_key).map(names)
        p=p.rename(columns={'game_date':'date','athlete_display_name':'player','minutes':'minutes'})
    else:
        base='https://basketball-data.billmill.org/nba/stats'
        t=read(f'{base}/team_game_logs/season={year}/data.parquet')
        p=read(f'{base}/player_game_logs/season={year}/data.parquet')
        n=read(f'{base}/player_seasons/season={year}/data.parquet')
        identities=read(f'{base}/team_seasons/season={year}/data.parquet')
        names=identities.set_index('team_id').nba_abbrev.map(canonical).to_dict()
        t=t.loc[t.game_id.str[:3].isin(['002','004','005'])].copy()
        p=p.loc[p.game_id.isin(t.game_id)].merge(n[['player_id','name']],on='player_id',validate='many_to_one')
        t['team']=t.team_id.map(names);t['opponent']=t.opp_team_id.map(names)
        t['phase']=t.game_id.str[:3].map({'002':'regular','004':'playoff','005':'playin'})
        t=t.rename(columns={'game_date':'date','plus_minus':'mov'})
        p['team']=p.team_id.map(names)
        p=p.rename(columns={'game_date':'date','name':'player','min':'minutes'})
    t=t[['game_id','date','team','opponent','home','mov','phase']].drop_duplicates(['game_id','team'])
    p=p[['game_id','date','team','player','minutes']].copy()
    p['minutes']=pd.to_numeric(p.minutes,errors='coerce')
    p=p.loc[p.minutes.gt(0)]
    t['player']='';t['minutes']=float('nan')
    p['opponent']='';p['home']=None;p['mov']=float('nan');p['phase']=''
    columns=['game_id','date','team','opponent','home','mov','phase','player','minutes']
    output=pd.concat([t[columns],p[columns]],ignore_index=True)
    output['game_id']=output.game_id.astype(str)
    output.to_parquet(path,index=False)
    print(year,len(t),'team logs',len(p),'player appearances',flush=True)

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download,range(2003,2027)))
