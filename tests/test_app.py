"""Exercise both Streamlit pages and feature selection against shipped artifacts."""
import unittest
import json
from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

@unittest.skipUnless(all((ROOT/p).exists() for p in ['artifacts/evaluation.json','artifacts/xgboost.json','data/teams.csv','data/matchups.csv','data/live_snapshot.json']), 'Optional local-data integration check; see docs/SETUP.md')
class AppTests(unittest.TestCase):
    def test_probabilities_and_correlations(self):
        snapshot=json.loads((ROOT/'data/live_snapshot.json').read_text())
        with patch('live.refresh_or_cached',return_value=(snapshot,None)):
            app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=20)
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.title[0].value,'2026–27 championship probabilities')
        self.assertEqual(len(app.dataframe[0].value),30)
        app.sidebar.radio[0].set_value('Feature correlations').run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.title[0].value,'Feature correlations')
        app.selectbox[0].set_value('d_srs').run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.selectbox[0].value,'d_srs')

if __name__=='__main__':
    unittest.main()
