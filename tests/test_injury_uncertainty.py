import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
import injury_uncertainty as injury

ROOT=Path(__file__).resolve().parents[1]

class InjuryUncertaintyTests(unittest.TestCase):
    def test_recovery_one_month_before_playoffs_still_costs_strength(self):
        row=dict(base_penalty=5.,return_date='2027-03-20',duration_days=120,recovery_fraction=.65)
        self.assertEqual(injury.penalty(row,'2027-03-19',90),5.)
        residual=injury.penalty(row,'2027-04-20',90)
        self.assertGreater(residual,2.)
        self.assertLess(injury.penalty(row,'2027-06-20',90),residual)
        self.assertGreater(injury.penalty(row,'2027-04-20',180),residual)

    def test_depth_and_concentration_each_raise_penalty(self):
        self.assertGreater(injury.roster_fragility(-2,.6),injury.roster_fragility(2,.6))
        self.assertGreater(injury.roster_fragility(0,.9),injury.roster_fragility(0,.4))

    def test_episode_removal_retains_recovery_and_first_observed_onset(self):
        report=dict(fetched_at='2026-10-26T20:00:00Z',reports=[dict(team_name='Thunder',player='Star',status='Out',reported_at='2026-10-25T10:00:00Z',estimated_return='2027-03-02')])
        with tempfile.TemporaryDirectory() as temp,patch.object(injury,'DATA',Path(temp)):
            injury.track_recovery(report)
            update=copy.deepcopy(report);update['fetched_at']='2026-11-26T20:00:00Z';update['reports'][0]['reported_at']='2026-11-26T10:00:00Z'
            result=injury.track_recovery(update)
            self.assertEqual(result['reports'][0]['_first_seen'],'2026-10-25')
            result=injury.track_recovery(dict(fetched_at='2027-03-03T20:00:00Z',reports=[]))
            row=result['reports'][0];self.assertTrue(row['_recovery']);self.assertEqual(row['_recovery_return'],'2027-03-03')

    def test_zero_severity_exactly_restores_paired_uncertainty_forecast(self):
        # Exercise the real paired path engines with fixed external inputs. Never
        # read a live feed, local raw tables, or an existing forecast cache.
        codes=[str(i) for i in range(30)]
        teams=pd.DataFrame(dict(team=codes,name=codes,conference=['E']*15+['W']*15,
                                wins=40,losses=41,gp=81))
        state=dict(codes=codes,index={c:i for i,c in enumerate(codes)},
                   mean=np.linspace(4.,-4.,30),covariance=np.eye(30),
                   wins=np.full(30,40),gp=np.full(30,81),cutoff='2027-03-01')
        fixtures=pd.DataFrame([dict(a=str(i),b=str(i+15),home_court=1.,
                                    d_rest=0,date='2027-04-10') for i in range(15)])
        primary=dict(season=2027,games_through='2027-03-01',teams=teams.to_dict('records'))
        players=pd.DataFrame([dict(team='0',player=name,mp=mpg*78,gp=78,bpm=bpm)
                              for name,mpg,bpm in [('First Star',38,6),('Second Star',35,5),
                                                   ('Jalen Williams',32,4),('Reserve',20,0)]])
        report=dict(fetched_at='2027-03-01T20:00:00Z',reports=[dict(
            team_name='0',player='Jalen Williams',status='Out',
            reported_at='2027-02-01T10:00:00Z',estimated_return='2027-05-01',season_out=False)])

        class FixedProbabilities:
            def conditional(self,difference,home,rest=0):
                return 1/(1+np.exp(-(np.asarray(difference)+3*(2*np.asarray(home)-1))/6))

        with tempfile.TemporaryDirectory() as temp, \
             patch.object(injury,'DATA',Path(temp)), \
             patch.object(injury,'OUTPUT',Path(temp)/'forecast.json'), \
             patch.object(injury,'PATHS',512), \
             patch.object(injury,'current_report',return_value=report), \
             patch.object(injury,'parse_players',side_effect=lambda year:players.copy()), \
             patch.object(injury,'live_inputs',return_value=(state,teams,fixtures,np.zeros(30,dtype=int))), \
             patch.object(injury,'full_game_model',return_value=FixedProbabilities()), \
             patch('requests.sessions.Session.request',side_effect=AssertionError('Unexpected network request')):
            result=injury.forecast(primary,severity=0.)
            rows=pd.DataFrame(result['odds']).set_index('team').sort_index()
            np.testing.assert_array_equal(rows.title_probability,rows.uncertainty_title_probability)
            self.assertAlmostEqual(rows.title_probability.sum(),1.)
            self.assertTrue(rows.playoff_strength_penalty.eq(0).all())
            self.assertTrue(rows.injury_change_pp.eq(0).all())
            william=next(r for r in result['adjustments'] if r['player']=='Jalen Williams')
            self.assertTrue(william['top3']);self.assertEqual(william['role_rank'],3)
            self.assertTrue(william['applied']);self.assertGreater(william['base_penalty'],0)
            # A nonzero control prevents a vacuous pass from ignoring the injury.
            stressed=injury.forecast(primary,severity=2.)
            stressed_rows=pd.DataFrame(stressed['odds']).set_index('team').sort_index()
            self.assertGreater(stressed_rows.loc['0','playoff_strength_penalty'],0)
            np.testing.assert_array_equal(stressed_rows.uncertainty_title_probability,rows.title_probability)
            self.assertFalse(stressed_rows.title_probability.equals(rows.title_probability))
            self.assertTrue((Path(temp)/'forecast.json').exists())

    @unittest.skipUnless(all((ROOT/p).exists() for p in ['data/live_snapshot.json', 'data/injury_uncertainty_snapshot.json', 'artifacts/xgboost.json', 'data/teams.csv', 'data/matchups.csv']), 'Optional local-data integration check; see docs/SETUP.md')
    def test_app_controls_and_matching_uncertainty_comparison(self):
        primary=json.loads((ROOT/'data/live_snapshot.json').read_text())
        result=json.loads((ROOT/'data/injury_uncertainty_snapshot.json').read_text())
        st.cache_data.clear()
        with patch('live.refresh_or_cached',return_value=(primary,None)),patch('injury_uncertainty.forecast',return_value=result),patch('injury_uncertainty.current_report',return_value=result['report']):
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=20)
            next(r for r in app.radio if r.label=='Forecast view').set_value('Experimental — uncertainty + injury stress').run(timeout=20)
            self.assertEqual(len(app.exception),0)
            self.assertEqual(next(s for s in app.slider if s.label=='Injury severity multiplier').value,2.)
            self.assertIn('Uncertainty baseline (%)',app.dataframe[-1].value)
            np.testing.assert_allclose(app.dataframe[-1].value['Title probability (%)']/100,[r['title_probability'] for r in result['odds']])

if __name__=='__main__':unittest.main()
