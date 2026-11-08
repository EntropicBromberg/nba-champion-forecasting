import json
from pathlib import Path
import unittest
from unittest.mock import patch
import requests
from streamlit.testing.v1 import AppTest
import hosted_dashboard as dashboard

ROOT=Path(__file__).resolve().parents[1]

class HostedDashboardTests(unittest.TestCase):
    def test_failed_download_preserves_real_forecast_timestamp(self):
        local=json.loads((ROOT/'reports/forecast.json').read_text())
        with patch('hosted_dashboard.requests.get',side_effect=requests.ConnectionError('offline')):
            result,warning=dashboard.load_forecast()
        self.assertEqual(result['checked_at'],local['checked_at'])
        self.assertEqual(result['odds'],local['odds'])
        self.assertIn('last committed primary',warning)

    def test_invalid_remote_forecast_is_not_displayed(self):
        local=json.loads((ROOT/'reports/forecast.json').read_text())
        invalid=dict(local,odds=local['odds'][:-1])
        with patch('hosted_dashboard.requests.get') as get:
            get.return_value.json.return_value=invalid
            result,warning=dashboard.load_forecast()
        self.assertEqual(len(result['odds']),30);self.assertIsNotNone(warning)

    def test_real_dashboard_and_team_filter(self):
        local=json.loads((ROOT/'reports/forecast.json').read_text())
        dashboard.cached_forecast.clear()
        with patch('hosted_dashboard.load_forecast',return_value=(local,None)):
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=20)
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.title[0].value,'NBA championship probabilities')
            expected=[row['title_probability']*100 for row in local['odds']]
            self.assertEqual(app.dataframe[0].value['Championship probability (%)'].tolist(),expected)
            app.multiselect[0].set_value([]).run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('Select at least one team' in item.value for item in app.info))

if __name__=='__main__':unittest.main()
