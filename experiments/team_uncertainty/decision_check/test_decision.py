from pathlib import Path
import hashlib
import json
import sys
import unittest
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_check as decision
from paths import record_score,games_behind

class DecisionTests(unittest.TestCase):
    def test_unequal_game_count_standings(self):
        scores=record_score(np.array([[41,41],[20,21]]),np.array([[81,82],[60,82]]),np.array([[0.,1.],[0.,1.]]))
        self.assertTrue((scores[:,0]>scores[:,1]).all())
        self.assertAlmostEqual(float(games_behind(35,34,74,73)),.5)

    def test_equal_season_weighting(self):
        rows=[]
        for year,delta,n in [(2020,-.2,100),(2021,.2,1)]:
            for model,value in [('production_calibrated',1.),('uncertainty',1.+delta)]:
                rows.append(dict(season=year,checkpoint='pre_playoff',kind='game',model=model,n=n,log_loss=value,brier=value/3))
        _,comparison=decision.annual_pair(pd.DataFrame(rows))
        self.assertAlmostEqual(comparison[0]['log_loss_delta'],0)

    def test_same_cases_and_prior_only_calibration(self):
        frame=pd.read_csv(decision.OUT/'event_predictions.csv.gz',dtype={'event':str})
        keys=['season','checkpoint','kind','event']
        pair=frame.loc[frame.model.eq('production_calibrated')].merge(frame.loc[frame.model.eq('uncertainty')],on=keys,suffixes=('_p','_u'),validate='one_to_one')
        self.assertEqual(len(pair)*2,len(frame));self.assertTrue(pair.y_p.eq(pair.y_u).all())
        self.assertTrue(pair.cutoff_p.eq(pair.cutoff_u).all())
        report=json.loads((decision.OUT/'report.json').read_text())
        self.assertTrue(all(all(y<f['season'] for y in f['training_seasons']) for f in report['calibration_folds']))
        for relative,digest in report['protected_production_hashes'].items():
            self.assertEqual(hashlib.sha256((decision.ROOT/relative).read_bytes()).hexdigest(),digest)
        self.assertEqual(hashlib.sha256((HERE/'spec.json').read_bytes()).hexdigest(),report['spec_sha256'])

    def test_title_distributions_and_effective_sample_size(self):
        title=pd.read_csv(decision.OUT/'title_predictions.csv')
        for _,part in title.groupby(['season','checkpoint','model']):
            self.assertEqual(len(part),30);self.assertEqual(part.y.sum(),1)
            self.assertAlmostEqual(part.probability.sum(),1);self.assertAlmostEqual(part.raw_probability.sum(),1)
        report=json.loads((decision.OUT/'report.json').read_text())
        self.assertFalse(report['promotion_authorized'])
        self.assertTrue(all(x['seasons']<=14 for x in report['comparisons']))

if __name__=='__main__':unittest.main()
