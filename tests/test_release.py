from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import requests
import hosted_dashboard
from streamlit.testing.v1 import AppTest
from synthetic_demo import synthetic_forecast

ROOT=Path(__file__).resolve().parents[1]

class ReleaseTests(unittest.TestCase):
    def test_synthetic_demo_is_offline_reproducible_and_sensitive(self):
        with patch('requests.sessions.Session.request',side_effect=AssertionError('Unexpected network')):
            baseline=synthetic_forecast(1000).set_index('team').sort_index()
            repeat=synthetic_forecast(1000).set_index('team').sort_index()
            stressed=synthetic_forecast(1000,3.).set_index('team').sort_index()
        np.testing.assert_array_equal(baseline.title_probability,repeat.title_probability)
        self.assertAlmostEqual(baseline.title_probability.sum(),1.)
        self.assertAlmostEqual(stressed.title_probability.sum(),1.)
        self.assertLess(stressed.loc['T01','title_probability'],baseline.loc['T01','title_probability'])

    def test_demo_app_runs_without_source_data(self):
        with patch('requests.sessions.Session.request',side_effect=AssertionError('Unexpected network')):
            app=AppTest.from_file(str(ROOT/'demo.py')).run(timeout=20)
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('Synthetic demonstration' in w.value for w in app.warning))
            self.assertEqual(len(app.dataframe[0].value),30)
            app.slider[0].set_value(2.).run(timeout=20)
            self.assertEqual(len(app.exception),0)

    @unittest.skipIf((ROOT/'artifacts/xgboost.json').exists(), 'Real-data environment has been prepared')
    def test_hosted_real_forecast_runs_without_training_inputs_or_network(self):
        hosted_dashboard.cached_forecast.clear()
        with patch('hosted_dashboard.requests.get',side_effect=requests.ConnectionError('offline')):
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=20)
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.title[0].value,'NBA championship probabilities')
        self.assertEqual(len(app.dataframe[0].value),30)
        self.assertTrue(any('Games through' in item.value for item in app.caption))

if __name__=='__main__':unittest.main()
