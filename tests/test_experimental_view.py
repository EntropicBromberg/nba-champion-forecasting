import json
from pathlib import Path
from io import BytesIO
from unittest.mock import patch
import unittest
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
import experimental_live

ROOT=Path(__file__).resolve().parents[1]

class ExperimentalViewTests(unittest.TestCase):
    @unittest.skipUnless(all((ROOT/p).exists() for p in ['data/live_snapshot.json', 'data/experimental_snapshot.json', 'artifacts/xgboost.json', 'data/teams.csv', 'data/matchups.csv']), 'Optional local-data integration check; see docs/SETUP.md')
    def test_primary_default_and_explicit_experimental_selection(self):
        st.cache_data.clear()
        primary=json.loads((ROOT/'data/live_snapshot.json').read_text())
        alternate=json.loads((ROOT/'data/experimental_snapshot.json').read_text())
        with patch('live.refresh_or_cached',return_value=(primary,None)),patch('experimental_live.forecast',return_value=alternate):
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=20)
            self.assertEqual(len(app.exception),0)
            radio=next(r for r in app.radio if r.label=='Forecast view')
            self.assertEqual(radio.value,'Production (primary)')
            np.testing.assert_allclose(next(t.value for t in app.dataframe if 'Title probability (%)' in t.value.columns)['Title probability (%)']/100,[r['title_probability'] for r in primary['odds']])
            radio.set_value('Experimental — uncertainty-aware').run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('not promoted' in w.value for w in app.warning))
            np.testing.assert_allclose(next(t.value for t in app.dataframe if 'Title probability (%)' in t.value.columns)['Title probability (%)']/100,[r['title_probability'] for r in alternate['odds']])

    @unittest.skipUnless(all((ROOT/p).exists() for p in ['data/live_snapshot.json', 'artifacts/xgboost.json', 'data/teams.csv', 'data/matchups.csv']), 'Optional local-data integration check; see docs/SETUP.md')
    def test_experimental_failure_never_relabels_primary(self):
        st.cache_data.clear();primary=json.loads((ROOT/'data/live_snapshot.json').read_text())
        with patch('live.refresh_or_cached',return_value=(primary,None)),patch('experimental_live.forecast',side_effect=ValueError('date mismatch')):
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=20)
            next(r for r in app.radio if r.label=='Forecast view').set_value('Experimental — uncertainty-aware').run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('unavailable' in w.value for w in app.warning));self.assertEqual(len(app.dataframe),0)
            next(r for r in app.radio if r.label=='Forecast view').set_value('Production (primary)').run()
            self.assertEqual(len(app.dataframe[0].value),30)

    def test_live_games_are_cut_off_at_primary_snapshot(self):
        def encoded(frame):
            stream=BytesIO();frame.to_parquet(stream,index=False);return stream.getvalue()
        rows=[]
        for game,date in [('0022600001','2026-10-24'),('0022600002','2026-10-25')]:
            for team,opponent,home,win,mov in [('1','2',True,True,5),('2','1',False,False,-5)]:
                rows.append(dict(game_id=game,game_date=pd.Timestamp(date).date(),team_id=team,opp_team_id=opponent,home=home,win=win,plus_minus=mov))
        games=encoded(pd.DataFrame(rows));identities=encoded(pd.DataFrame(dict(team_id=['1','2'],nba_abbrev=['BOS','LAL'])))
        class Response:
            def __init__(self,content):self.content=content
            def raise_for_status(self):pass
        primary=dict(season=2027,teams=[dict(team='BOS',gp=1,wins=1,cutoff='2026-10-24'),dict(team='LAL',gp=1,wins=0,cutoff='2026-10-24')])
        with patch('experimental_live.requests.get',side_effect=lambda url,**kwargs:Response(games if 'team_game_logs' in url else identities)):
            result=experimental_live.current_games(primary)
        self.assertEqual(len(result),1);self.assertEqual(result.date.max(),pd.Timestamp('2026-10-24'))

if __name__=='__main__':unittest.main()
