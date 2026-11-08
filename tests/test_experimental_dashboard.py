import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
import hosted_dashboard
from experimental_dashboard import validate_bundle,load_bundle
from scripts.publish_experiments import build,MODELS

ROOT=Path(__file__).resolve().parents[1]

class ExperimentalDashboardTests(unittest.TestCase):
    def setUp(self):
        self.primary=json.loads((ROOT/'reports/forecast.json').read_text())
        # Fixtures remain available even if a future scheduled real experiment fails.
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            for identifier,(filename,label) in MODELS.items():
                rows=[dict(r,production_title_probability=r['title_probability'],
                           uncertainty_title_probability=r['title_probability']) for r in self.primary['odds']]
                snapshot=dict(season=2027,label=label,games_through=self.primary['games_through'],
                              computed_at=self.primary['checked_at'],simulations=10000,odds=rows,
                              settings=dict(severity=2.,recovery_days=90))
                (path/filename).write_text(json.dumps(snapshot))
            self.bundle=build(self.primary,path)

    def test_models_are_aligned_and_have_correct_baselines(self):
        self.assertIs(validate_bundle(self.bundle,self.primary),self.bundle)
        self.assertEqual(set(self.bundle['models']),{'uncertainty','injury','stress'})
        for model in self.bundle['models'].values():
            self.assertEqual(model['status'],'available')
            self.assertEqual(len(model['odds']),30)
            self.assertAlmostEqual(sum(r['title_probability'] for r in model['odds']),1)
        self.assertEqual(self.bundle['models']['stress']['baseline'],'uncertainty')

    def test_changed_primary_or_baseline_is_rejected(self):
        primary=copy.deepcopy(self.primary)
        primary['odds'][0]['title_probability']-=.001
        primary['odds'][1]['title_probability']+=.001
        with self.assertRaises(ValueError):validate_bundle(self.bundle,primary)
        bundle=copy.deepcopy(self.bundle)
        bundle['models']['stress']['odds'][0]['baseline_title_probability']+=.01
        with self.assertRaises(ValueError):validate_bundle(bundle,self.primary)

    def test_invalid_remote_bundle_keeps_aligned_local_results(self):
        load_bundle.clear()
        with patch('experimental_dashboard.requests.get') as get:
            get.return_value.json.return_value=dict(self.bundle,primary_fingerprint='wrong')
            bundle,warning=load_bundle(self.primary)
        self.assertIsNotNone(bundle);self.assertIn('saved experimental',warning)
        load_bundle.clear()

    def test_export_strips_source_records_and_marks_missing_variations(self):
        snapshot=dict(season=2027,label=MODELS['uncertainty'][1],games_through=self.primary['games_through'],
                      computed_at=self.primary['checked_at'],simulations=10000,
                      odds=[dict(r,source_text='private feed text') for r in self.primary['odds']],
                      report={'private':'injury text'},adjustments=[{'player':'private details'}])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            (path/'experimental_snapshot.json').write_text(json.dumps(snapshot))
            bundle=build(self.primary,path)
        self.assertEqual(bundle['models']['uncertainty']['status'],'available')
        self.assertEqual(bundle['models']['injury']['status'],'unavailable')
        self.assertNotIn('private',json.dumps(bundle))
        self.assertNotIn('source_text',json.dumps(bundle))

    def test_app_displays_three_experimental_tabs_below_primary(self):
        hosted_dashboard.cached_forecast.clear();load_bundle.clear()
        with patch('hosted_dashboard.load_forecast',return_value=(self.primary,None)), \
             patch('experimental_dashboard.load_bundle',return_value=(self.bundle,None)):
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=20)
        self.assertEqual(len(app.exception),0)
        self.assertEqual([tab.label for tab in app.tabs],['Team-strength uncertainty','Injury adjustment','Injury + recovery stress'])
        self.assertEqual(len(app.dataframe),4)
        self.assertIn('Uncertainty baseline (%)',app.dataframe[3].value)
        self.assertTrue(any(h.value=='Experimental variations' for h in app.header))

if __name__=='__main__':unittest.main()
