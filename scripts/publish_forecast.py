"""Render only primary model outputs into the public README and chart."""
import argparse
from datetime import date, datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
START='<!-- FORECAST:START -->'
END='<!-- FORECAST:END -->'


def public_forecast(snapshot):
    season=snapshot['season']
    if snapshot.get('label') or snapshot.get('report') or snapshot.get('settings'):
        raise ValueError('Publish the primary forecast only, not an experimental scenario')
    if season!=2027 or snapshot['trained_through']!=season-1 or snapshot['training_seasons']!=25:
        raise ValueError('Unexpected season or model training boundary')
    checked=datetime.fromisoformat(snapshot['checked_at'].replace('Z','+00:00'))
    if checked.tzinfo is None or (checked-datetime.now(timezone.utc)).total_seconds()>300:
        raise ValueError('Forecast timestamp must be timezone-aware and not in the future')
    games=date.fromisoformat(snapshot['games_through'])
    if games>checked.date():raise ValueError('Game cutoff cannot follow the computation date')
    if not isinstance(snapshot['simulations'],int) or snapshot['simulations']<=0:
        raise ValueError('A positive simulation count is required')
    rows=snapshot['odds']
    if len(rows)!=30 or len({row['team'] for row in rows})!=30:
        raise ValueError('Expected 30 unique NBA teams')
    clean=[]
    for row in rows:
        probability=float(row['title_probability'])
        if not math.isfinite(probability) or not 0<=probability<=1:
            raise ValueError('Invalid championship probability')
        if not re.fullmatch('[A-Z]{2,3}',row['team']):raise ValueError('Invalid team identifier')
        name=row['name']
        if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z0-9 .'-]{3,60}",name):
            raise ValueError('Invalid team name')
        clean.append(dict(team=row['team'],name=name,title_probability=probability))
    if not math.isclose(sum(row['title_probability'] for row in clean),1,abs_tol=1e-7):
        raise ValueError('Championship probabilities must sum to one')
    return dict(season=season,trained_through=snapshot['trained_through'],training_seasons=25,
                games_through=games.isoformat(),checked_at=checked.isoformat(),
                simulations=snapshot['simulations'],forecast='primary',
                odds=sorted(clean,key=lambda row:(-row['title_probability'],row['team'])))


def chart(data):
    rows=data['odds'][:12];width=960;height=120+len(rows)*34
    scale=max(.1,math.ceil(rows[0]['title_probability']*10)/10)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
           '<title id="title">NBA championship probabilities — primary forecast</title>',
           f'<desc id="description">Top twelve teams. Games through {data["games_through"]}. All thirty probabilities appear in the README table.</desc>',
           f'<rect width="{width}" height="{height}" rx="16" fill="#101c2c"/>',
           '<g font-family="Arial, sans-serif">',
           '<text x="28" y="38" fill="#f5f8fc" font-size="23" font-weight="700">NBA Championship Forecast</text>',
           f'<text x="28" y="66" fill="#b5c7da" font-size="14">Games through {data["games_through"]} · {data["simulations"]:,} season paths · Primary model</text>']
    for i,row in enumerate(rows):
        y=90+i*34;bar=round(520*row['title_probability']/scale,2)
        parts.extend([f'<text x="28" y="{y+17}" fill="#e4edf7" font-size="14">{escape(row["name"])}</text>',
                      f'<rect x="286" y="{y}" width="520" height="23" rx="4" fill="#203349"/>',
                      f'<rect x="286" y="{y}" width="{bar}" height="23" rx="4" fill="#4ed1bd"/>',
                      f'<text x="828" y="{y+17}" fill="#f5f8fc" font-size="15" font-weight="700">{100*row["title_probability"]:.2f}%</text>'])
    parts.extend([f'<text x="28" y="{height-18}" fill="#b5c7da" font-size="12">Model estimates, not sportsbook odds. Bar scale: 0–{scale*100:.0f}%. Forecast timestamp is shown in the README.</text>','</g></svg>'])
    return '\n'.join(parts)+'\n'


def publish(snapshot,root=ROOT):
    data=public_forecast(snapshot)
    readme=root/'README.md';text=readme.read_text()
    if text.count(START)!=1 or text.count(END)!=1 or text.index(START)>text.index(END):
        raise ValueError('README needs exactly one ordered forecast marker pair')
    checked=datetime.fromisoformat(data['checked_at']).astimezone(timezone.utc)
    block=[START,f'**Latest successful primary forecast** · Games through **{data["games_through"]}** · Refreshed **{checked:%Y-%m-%d %H:%M UTC}**',
           '', '![Primary model championship probabilities](reports/title_probabilities.svg)', '',
           '<details>','<summary>All 30 teams</summary>','','| Team | Championship probability |','|---|---:|']
    block.extend(f'| {row["name"]} | {100*row["title_probability"]:.2f}% |' for row in data['odds'])
    block.extend(['','</details>','','Daily refresh is scheduled. The timestamp above advances only after a successful rebuild; this is not a real-time feed. A displayed 0.00% may reflect simulation sampling.',END])
    updated=text[:text.index(START)]+'\n'.join(block)+text[text.index(END)+len(END):]
    out=root/'reports';out.mkdir(exist_ok=True)
    (out/'forecast.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (out/'title_probabilities.svg').write_text(chart(data))
    readme.write_text(updated)
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    args=parser.parse_args()
    result=publish(json.loads(args.snapshot.read_text()))
    print(f'Published primary forecast: {len(result["odds"])} teams, games through {result["games_through"]}.')
