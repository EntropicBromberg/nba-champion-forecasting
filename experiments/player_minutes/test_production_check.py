import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import compare_production as check
from model import fit,paired_predictions
from live import blend_strength

class ProductionComparisonTests(unittest.TestCase):
    def test_identical_cases_and_cutoffs(self):
        frame=pd.read_csv(check.OUT/'predictions.csv.gz',dtype={'event':str})
        base=frame.loc[frame.model.eq(check.NAME)]
        player=frame.loc[frame.model.eq('with_players')]
        keys=['season','checkpoint','kind','event']
        pair=base.merge(player,on=keys,suffixes=('_prod','_player'),validate='one_to_one')
        self.assertEqual(len(pair),len(base));self.assertEqual(len(pair),len(player))
        self.assertTrue(pair.cutoff_prod.eq(pair.cutoff_player).all())
        self.assertTrue(pair.y_prod.eq(pair.y_player).all())
        self.assertEqual(base.kind.value_counts().to_dict(),{'game':27300,'series':615})
        self.assertTrue(base.loc[base.checkpoint.eq('pre_playoff')].input_quality.eq('complete_regular_season').all())
        self.assertTrue(base.loc[~base.checkpoint.eq('pre_playoff')].input_quality.eq('sensitivity_prior_bpm').all())

    def test_pre_playoff_uses_actual_production_blending(self):
        year=2026
        current=check.exp.HISTORY.loc[check.exp.HISTORY.season.eq(year)].copy()
        current['team']=current.team.map(check.canonical)
        expected=blend_strength(current,check.previous(year))
        actual,quality=check.inputs(year,'pre_playoff')
        pd.testing.assert_frame_equal(actual,expected)
        self.assertEqual(quality,'complete_regular_season')
        snap=check.exp.checkpoint(year,'pre_playoff')
        self.assertTrue(pd.to_datetime(current.cutoff).lt(snap['cutoff']).all())

    def test_replayed_model_matches_recorded_production_fold(self):
        year=2026
        evaluation=json.loads((check.ROOT/'artifacts/evaluation.json').read_text())
        fold=next(f for f in evaluation['folds'] if f['season']==year)
        rows=pd.read_csv(check.ROOT/'data/matchups.csv')
        model=fit(rows.loc[rows.season.lt(year)],fold['features'])
        test=rows.loc[rows.season.eq(year)]
        p=paired_predictions(model,test,fold['features'])
        original=pd.read_csv(check.ROOT/'artifacts/walk_forward.csv')
        original=original.loc[original.season.eq(year)&original.model.eq('XGBoost')].set_index('series_id')
        expected=original.loc[test.iloc[::2].series_id,'probability'].to_numpy()
        np.testing.assert_allclose(p,expected,atol=1e-7)

    def test_old_rule_and_production_unchanged(self):
        report=json.loads((check.OUT/'report.json').read_text())
        for relative,digest in report['protected_hashes'].items():
            self.assertEqual(hashlib.sha256((check.ROOT/relative).read_bytes()).hexdigest(),digest)
        old=json.loads((HERE/'results/report.json').read_text())
        self.assertEqual(old['comparisons']['series']['seasons_improved'],9)
        self.assertFalse(old['promotion_checks']['series_season_consistency'])
        self.assertEqual(old['decision'],'DO_NOT_PROMOTE')
        self.assertTrue(all(f['train_last']<f['season'] for f in report['folds']))

    def test_future_rule_is_separate_and_locked(self):
        path=HERE/'prospective_promotion_rule.json'
        rule=json.loads(path.read_text())
        report=json.loads((check.OUT/'report.json').read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),report['prospective_rule_sha256'])
        self.assertLess(pd.Timestamp(rule['locked_at_utc']),pd.Timestamp(rule['cohort']['first_checkpoint_utc']))
        self.assertEqual(rule['cohort']['ending_seasons'],[2027,2028,2029])
        self.assertEqual(rule['cohort']['minimum_unique_pre_playoff_series'],45)
        for relative,digest in rule['source_hashes'].items():
            self.assertEqual(hashlib.sha256((check.ROOT/relative).read_bytes()).hexdigest(),digest)

if __name__=='__main__':unittest.main()
