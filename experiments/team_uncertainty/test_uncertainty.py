import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from ratings import Ratings,draw_strengths
from probabilities import GameProbabilities,best7
import simulator

class RatingTests(unittest.TestCase):
    def test_covariance_update_and_opponent_adjustment(self):
        rating=Ratings(['A','B','C'],[3.,-1.,-2.],16,144,process_variance=0)
        before=rating.snapshot('2026-10-01')
        rating.update('A','B',7,1.,'2026-10-01')  # Exactly expected margin: 4 quality + 3 home.
        np.testing.assert_allclose(rating.mean,before['mean'],atol=1e-12)
        self.assertLess(rating.covariance[0,0]+rating.covariance[1,1]-2*rating.covariance[0,1],32)
        self.assertGreaterEqual(np.linalg.eigvalsh(rating.covariance).min(),-1e-10)
        np.testing.assert_allclose(rating.covariance.sum(axis=0),0,atol=1e-10)
        self.assertAlmostEqual(rating.mean.sum(),0)

    def test_future_update_cannot_change_saved_snapshot(self):
        rating=Ratings(['A','B'],[0.,0.],16,144)
        before=rating.snapshot('2026-10-01')
        rating.update('A','B',100,1.,'2026-10-01')
        np.testing.assert_array_equal(before['mean'],[0,0])
        self.assertGreater(rating.mean[0],0)
        with self.assertRaises(AssertionError):rating.snapshot('2026-10-01')

    def test_joint_draws_preserve_mean_and_covariance(self):
        rating=Ratings(['A','B','C'],[3.,-1.,-2.],16,144)
        sampled=draw_strengths(rating.mean,rating.covariance,20000)
        np.testing.assert_allclose(sampled.mean(axis=0),rating.mean,atol=1e-12)
        np.testing.assert_allclose(sampled.sum(axis=1),0,atol=1e-6)
        # A known identity-covariance design verifies the transform exactly, without a flaky Monte Carlo tolerance.
        with patch('ratings.np.random.default_rng') as rng:
            rng.return_value.normal.return_value=np.sqrt(3)*np.eye(3)
            exact=draw_strengths(rating.mean,rating.covariance,6)-rating.mean
        np.testing.assert_allclose(exact.T@exact/6,rating.covariance,atol=1e-12)

class ProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model=XGBClassifier();model.load_model(HERE/'cache/game_2026.json')
        cls.probabilities=GameProbabilities(model)

    def test_interval_lookup_matches_actual_booster(self):
        p=self.probabilities
        values=np.concatenate([np.random.default_rng(0).normal(0,12,1000),p.cuts,[0.]])
        for home,rest in [(1.,0),(.5,-3),(0.,7)]:
            np.testing.assert_allclose(p.conditional(values,home,rest),p.direct(values,home,rest),atol=1e-7)

    def test_zero_uncertainty_reproduces_control(self):
        p=self.probabilities;mean=np.array([-7.,0.,4.]);home=np.array([1.,.5,0.]);rest=np.array([0,-1,2])
        np.testing.assert_array_equal(p.games(mean,np.zeros(3),home,rest,False),p.games(mean,np.zeros(3),home,rest,True))
        self.assertAlmostEqual(p.series(3,0,uncertain=False),p.series(3,0,uncertain=True))

    def test_integrated_series_uses_one_persistent_difference(self):
        p=self.probabilities
        sampled=np.random.default_rng(0).normal(3,4,200000)
        home=p.conditional(sampled,1.);away=p.conditional(sampled,0.)
        conditional=best7([home,home,away,away,home,away,home])
        integrated=p.series(3,16,uncertain=True)
        self.assertAlmostEqual(integrated,float(conditional.mean()),delta=.003)
        # Integrating a seven-game series is NOT combining independent marginal game probabilities.
        independent=float(best7([home.mean(),home.mean(),away.mean(),away.mean(),home.mean(),away.mean(),home.mean()]))
        self.assertGreater(abs(integrated-independent),.001)

    def test_simulator_draws_quality_once_and_reuses_it(self):
        n=30;paths=1000;codes=[str(i) for i in range(n)]
        snapshot=dict(codes=codes,index={c:i for i,c in enumerate(codes)},mean=np.zeros(n),
            covariance=np.eye(n),wins=np.full(n,40),gp=np.full(n,80))
        fixtures=pd.DataFrame([dict(a='0',b='15',home_court=1.,d_rest=0)]*2)
        seen=[]
        class Recorder:
            def conditional(self,difference,home,rest=0):
                seen.append(np.asarray(difference).copy());return np.full(np.asarray(difference).shape,.5)
        values=np.tile(np.arange(n),(paths,1))*np.linspace(.5,1.5,paths)[:,None]
        with patch.object(simulator,'draw_strengths',return_value=values) as draw:
            result=simulator.simulate(snapshot,fixtures,['E']*15+['W']*15,Recorder(),True,paths=paths)
        draw.assert_called_once()
        np.testing.assert_array_equal(seen[0],seen[1])
        np.testing.assert_array_equal(seen[0],values[:,0]-values[:,15])
        self.assertAlmostEqual(result['title'].sum(),1)
        self.assertAlmostEqual(result['playoff'].sum(),16)

class AuditTests(unittest.TestCase):
    def test_training_and_snapshot_dates(self):
        rows=pd.read_parquet(HERE/'cache/pregame.parquet')
        self.assertTrue(pd.to_datetime(rows.max_input_date).lt(rows.date).all())
        audit=pd.read_csv(HERE/'results/parameter_audit.csv')
        self.assertTrue(audit.learned_through.lt(audit.season).all())
        ranges=pd.read_csv(HERE/'results/strength_ranges.csv')
        self.assertTrue(pd.to_datetime(ranges.max_input_date).lt(pd.to_datetime(ranges.cutoff)).all())
        self.assertTrue(ranges.sd.gt(0).all())

    def test_identical_cases_and_protected_files(self):
        frame=pd.read_csv(HERE/'results/predictions.csv.gz',dtype={'event':str})
        keys=['season','checkpoint','kind','event']
        paired=frame.loc[frame.arm.eq('fixed')].merge(frame.loc[frame.arm.eq('uncertain')],on=keys,suffixes=('_fixed','_uncertain'),validate='one_to_one')
        self.assertEqual(len(paired)*2,len(frame))
        self.assertTrue(paired.y_fixed.eq(paired.y_uncertain).all())
        self.assertTrue(paired.cutoff_fixed.eq(paired.cutoff_uncertain).all())
        report=json.loads((HERE/'results/report.json').read_text())
        root=HERE.parents[1]
        for relative,expected in report['protected_hashes'].items():
            self.assertEqual(hashlib.sha256((root/relative).read_bytes()).hexdigest(),expected)
        self.assertEqual(hashlib.sha256((HERE/'spec.json').read_bytes()).hexdigest(),report['spec_sha256'])
        self.assertTrue(all(f['same_model_both_arms'] and f['train_through']<f['season'] for f in report['folds']))

if __name__=='__main__':unittest.main()
