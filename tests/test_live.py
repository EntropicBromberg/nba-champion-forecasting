import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import requests
import live
from data_pipeline import FEATURES, TEAM_FEATURES
from test_model import ConstantModel

class LiveTests(unittest.TestCase):
    def teams(self):
        return pd.DataFrame([dict(team=str(i),name=str(i),conference='E' if i<15 else 'W',
            wins=41,losses=41,gp=82,standing_order=i%15,season=2027,
            **{'z_'+f:0. for f in TEAM_FEATURES}) for i in range(30)])

    def test_equal_teams_and_probability_totals(self):
        odds,missing=live.simulate(self.teams(),ConstantModel(),FEATURES,[],set(),simulations=12000)
        self.assertEqual(missing,0)
        self.assertAlmostEqual(odds.title_probability.sum(),1)
        self.assertAlmostEqual(odds.playoff_probability.sum(),16)
        np.testing.assert_allclose(odds.title_probability,1/30,atol=.008)
        self.assertTrue(odds.projected_wins.eq(41).all())
        self.assertTrue(odds.title_probability.gt(0).all())

    def test_completed_games_not_replayed(self):
        teams=self.teams()
        schedule=[dict(home='0',away='15',date='2026-10-24')]
        played={('0','2026-10-24'),('15','2026-10-24')}
        odds,_=live.simulate(teams,ConstantModel(),FEATURES,schedule,played,simulations=1000)
        self.assertTrue(odds.projected_wins.eq(41).all())

    def test_large_schedule_gaps_fail(self):
        teams=self.teams().assign(gp=70,wins=35,losses=35)
        with self.assertRaisesRegex(ValueError,'unassigned'):
            live.simulate(teams,ConstantModel(),FEATURES,[],set(),simulations=10)

    def test_network_failure_keeps_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'snapshot.json'
            payload=dict(season=2027,trained_through=2026,checked_at='2026-10-24T12:00:00+00:00')
            path.write_text(json.dumps(payload))
            with patch.object(live,'SNAPSHOT',path),patch.object(live,'refresh',side_effect=requests.ConnectionError('offline')):
                data,warning=live.refresh_or_cached()
                self.assertEqual(data,payload)
                self.assertIn('last successful',warning)
                self.assertEqual(json.loads(path.read_text()),payload)

    def test_missing_cache_never_returns_old_season(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(live,'SNAPSHOT',Path(directory)/'missing'),patch.object(live,'refresh',side_effect=requests.ConnectionError('offline')):
                with self.assertRaisesRegex(RuntimeError,'no saved snapshot'):
                    live.refresh_or_cached()

    def test_early_strength_uses_prior_without_overwriting_records(self):
        strength_columns=['srs','ortg','drtg','mov','top3_bpm','depth_bpm','form_resid',
                          'luck_resid','efg_net','tov_net','orb_net','ftr_net']
        prior=self.teams().assign(season=2026)
        for feature in TEAM_FEATURES:
            prior[feature]=np.linspace(-2,2,len(prior))
        for i,column in enumerate(strength_columns):
            prior[column]=np.linspace(-2,2,len(prior))+i
        current=prior.copy().assign(season=2027,gp=2,wins=1,losses=1,srs=100,mov=2)
        blended=live.blend_strength(current,prior)
        expected=(2*2+20*prior.srs)/22
        np.testing.assert_allclose(blended.srs,expected)
        self.assertTrue(blended.wins.eq(1).all())
        self.assertTrue(blended.gp.eq(2).all())
        np.testing.assert_allclose(blended.z_srs.mean(),0,atol=1e-10)

if __name__=='__main__':
    unittest.main()
