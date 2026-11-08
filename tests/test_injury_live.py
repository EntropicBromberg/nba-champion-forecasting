import json
from pathlib import Path
from unittest.mock import patch
import unittest
import pandas as pd
import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest
import injury_live

ROOT=Path(__file__).resolve().parents[1]

class InjuryTests(unittest.TestCase):
    def test_williams_is_matched_and_unknown_duration_is_not_season_long(self):
        primary=dict(teams=[dict(name='Oklahoma City Thunder',team='OKC',gp=3)])
        prior=pd.DataFrame([dict(player='Jalen Williams',team='OKC',mp=936.,gp=33.,bpm=2.1)])
        current=prior.copy();current['mp']=95.;current['gp']=3.
        row=dict(team_name='Oklahoma City Thunder',player='Jalen Williams',status='Out',estimated_return='2027-03-02',season_out=False)
        report=dict(fetched_at='2026-10-26T20:00:00Z',reports=[row])
        with patch('injury_live.parse_players',side_effect=[current,prior]):out=injury_live.build_adjustments(primary,report)[0]
        self.assertTrue(out['applied']);self.assertGreater(out['strength_penalty'],0);self.assertEqual(out['active_until'],'2027-03-02')
        report['reports'][0]['estimated_return']=None
        with patch('injury_live.parse_players',side_effect=[current,prior]):out=injury_live.build_adjustments(primary,report)[0]
        self.assertEqual(out['active_until'],'2026-10-27');self.assertTrue(out['unknown_duration'])

    def test_recovery_removes_penalty_and_same_state_is_cached(self):
        base=pd.DataFrame(dict(team=['OKC'],srs=[10.]))
        rows=[dict(team='OKC',applied=True,active_until='2027-03-02',strength_penalty=1.2)]
        with patch('injury_live.standardize',side_effect=lambda x:x),patch('injury_live.probability_tables',side_effect=lambda m,f,t:(None,t.srs.iloc[0])) as inference:
            tables=injury_live.InjuryTables(base,None,[],rows)
            self.assertAlmostEqual(tables('2026-10-26')[1],8.8)
            self.assertAlmostEqual(tables('2027-01-16')[1],8.8)
            self.assertEqual(tables('2027-03-02')[1],10.)
            self.assertEqual(inference.call_count,2)
        self.assertEqual(base.srs.iloc[0],10.)

    @unittest.skipUnless(all((ROOT/p).exists() for p in ['data/live_snapshot.json', 'data/injury_snapshot.json', 'artifacts/xgboost.json', 'data/teams.csv', 'data/matchups.csv']), 'Optional local-data integration check; see docs/SETUP.md')
    def test_injury_view_displays_real_report_and_production_comparison(self):
        primary=json.loads((ROOT/'data/live_snapshot.json').read_text())
        alternate=json.loads((ROOT/'data/injury_snapshot.json').read_text())
        st.cache_data.clear()
        with patch('live.refresh_or_cached',return_value=(primary,None)),patch('injury_live.forecast',return_value=alternate):
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=20)
            next(r for r in app.radio if r.label=='Forecast view').set_value('Experimental — injury-adjusted').run(timeout=20)
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('unvalidated' in w.value for w in app.warning))
            final=app.dataframe[-1].value
            self.assertIn('Injury change (pp)',final)
            np.testing.assert_allclose(final['Title probability (%)']/100,[r['title_probability'] for r in alternate['odds']])
            actual=next(r for r in alternate['adjustments'] if r['player']=='Jalen Williams')
            self.assertTrue(actual['applied']);self.assertEqual(actual['estimated_return'],'2027-03-02')

if __name__=='__main__':unittest.main()
