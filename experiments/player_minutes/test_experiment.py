import json
import math
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run as experiment

class CausalityTests(unittest.TestCase):
    def test_snapshots_strictly_past(self):
        rows=pd.read_parquet(HERE/'cache/pregame.parquet')
        self.assertTrue(pd.to_datetime(rows.max_input_date).lt(pd.to_datetime(rows.date)).all())
        self.assertFalse(rows[['d_team_strength','d_player_strength','home_court','d_rest']].isna().any().any())
        self.assertEqual(len(rows),rows.game_id.nunique())

    def test_projected_minutes_and_rating_dates(self):
        audit=pd.read_csv(HERE/'results/projected_minutes_audit.csv.gz')
        self.assertTrue(audit.rating_season.lt(audit.season).all())
        self.assertTrue(pd.to_datetime(audit.max_input_date).lt(pd.to_datetime(audit.cutoff)).all())
        totals=audit.groupby(['season','checkpoint','team']).projected_minutes.sum()
        np.testing.assert_allclose(totals,240,atol=1e-8)

    def test_future_result_cannot_change_frozen_inputs(self):
        snap=experiment.checkpoint(2026,'january')
        games,_=experiment.load_games(2026)
        games['season']=2026
        target=games.loc[games.date.ge(snap['cutoff'])].head(1).copy()
        first=experiment.fixtures(snap,target)
        target['mov']=-1000;target['y']=1-target.y
        second=experiment.fixtures(snap,target)
        cols=experiment.ADDED+['d_srs_z']
        pd.testing.assert_frame_equal(first[cols],second[cols])

    def test_state_rejects_same_day_results(self):
        state=experiment.State(2026)
        state.max_date=pd.Timestamp('2026-01-15')
        with self.assertRaises(AssertionError):state.snapshot('2026-01-15')

    def test_series_dp(self):
        self.assertAlmostEqual(experiment.series_probability([.5]*7),.5)
        expected=sum(math.comb(7,k)*.6**k*.4**(7-k) for k in range(4,8))
        self.assertAlmostEqual(experiment.series_probability([.6]*7),expected)
        p=[.7,.7,.5,.5,.7,.5,.7]
        self.assertAlmostEqual(experiment.series_probability(p)+experiment.series_probability([1-x for x in p]),1)

    def test_training_boundaries_and_spec_hash(self):
        import hashlib
        report=json.loads((HERE/'results/report.json').read_text())
        self.assertEqual(len(report['folds']),14)
        self.assertTrue(all(f['train_last']<f['test_season'] for f in report['folds']))
        self.assertEqual(report['spec_sha256'],hashlib.sha256((HERE/'spec.json').read_bytes()).hexdigest())
        for path,digest in report['production_hashes'].items():
            self.assertEqual(hashlib.sha256((experiment.ROOT/path).read_bytes()).hexdigest(),digest)

if __name__=='__main__':unittest.main()
