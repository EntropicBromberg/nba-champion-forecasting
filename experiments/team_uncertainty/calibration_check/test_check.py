import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('calibration_check',HERE/'check.py')
check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)

class CalibrationTests(unittest.TestCase):
    def test_adjustment_symmetry_and_identity(self):
        p=np.array([.001,.1,.3,.5,.8,.999])
        np.testing.assert_allclose(check.apply_slope(p,.7)+check.apply_slope(1-p,.7),1,atol=1e-12)
        np.testing.assert_array_equal(check.apply_slope(p,1.),p)
        self.assertTrue(np.diff(check.apply_slope(p,.7)).min()>0)

    def test_earlier_labels_only(self):
        frame=check.load_cases()
        ends={y:pd.Timestamp(year=y,month=10,day=1) for y in range(2013,2027)}
        first,audit=check.calibrate(frame,ends)
        altered=frame.copy();altered.loc[altered.season.eq(2026),'y']=1-altered.loc[altered.season.eq(2026),'y']
        second,other=check.calibrate(altered,ends)
        a=first.loc[first.season.eq(2026)&first.variant.str.endswith('_calibrated')].probability
        b=second.loc[second.season.eq(2026)&second.variant.str.endswith('_calibrated')].probability
        np.testing.assert_array_equal(a,b)
        self.assertTrue(all(f['train_last'] is None or f['train_last']<f['test_season'] for f in audit))

    def test_unresolved_labels_rejected(self):
        frame=check.load_cases()
        ends={y:pd.Timestamp('2030-01-01') for y in range(2013,2027)}
        with self.assertRaisesRegex(AssertionError,'resolved'):check.calibrate(frame,ends)

    def test_saved_audit_and_warmup(self):
        report=json.loads((check.OUT/'report.json').read_text())
        for fold in report['calibration_folds']:
            if fold['test_season']<=2015:
                self.assertFalse(fold['active']);self.assertEqual(fold['slope'],1.)
            else:
                self.assertTrue(fold['active']);self.assertGreaterEqual(fold['slope'],.25);self.assertLessEqual(fold['slope'],4.)
            self.assertNotIn(fold['test_season'],fold['train_seasons'])
            if fold['labels_resolved_through']:
                self.assertLess(pd.Timestamp(fold['labels_resolved_through']),pd.Timestamp(fold['cutoff']))
        for relative,expected in report['protected_hashes'].items():
            self.assertEqual(hashlib.sha256((check.ROOT/relative).read_bytes()).hexdigest(),expected)
        self.assertEqual(hashlib.sha256((HERE/'spec.json').read_bytes()).hexdigest(),report['spec_sha256'])
        frame=pd.read_csv(check.OUT/'predictions.csv.gz',dtype={'event':str})
        warm=frame.loc[frame.season.le(2015)]
        for model in ['production','fixed','uncertain']:
            keys=['season','kind','event']
            pair=warm.loc[warm.variant.eq(model+'_raw')].merge(warm.loc[warm.variant.eq(model+'_calibrated')],on=keys,suffixes=('_raw','_calibrated'),validate='one_to_one')
            np.testing.assert_array_equal(pair.probability_raw.to_numpy(),pair.probability_calibrated.to_numpy())

if __name__=='__main__':unittest.main()
