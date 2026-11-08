import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.publish_forecast import public_forecast,publish,START,END

class PublishForecastTests(unittest.TestCase):
    def snapshot(self):
        return dict(season=2027,trained_through=2026,training_seasons=25,
                    games_through='2026-11-07',checked_at='2026-11-08T12:00:00+00:00',simulations=50000,
                    teams=[{'private':'source records'}],
                    odds=[dict(team=chr(65+i//26)+chr(65+i%26),name='Example Team '+str(i+1),
                               title_probability=1/30,wins=10,source_text='not for publication') for i in range(30)])

    def test_only_derived_outputs_are_exported(self):
        data=public_forecast(self.snapshot())
        self.assertNotIn('teams',data)
        self.assertEqual(set(data['odds'][0]),{'team','name','title_probability'})
        self.assertNotIn('source_text',json.dumps(data))

    def test_invalid_and_experimental_forecasts_are_rejected(self):
        for change in [dict(label='Injury scenario'),dict(trained_through=2025),dict(simulations=0),
                       dict(checked_at='2999-01-01T00:00:00+00:00')]:
            data=self.snapshot();data.update(change)
            with self.assertRaises(ValueError):public_forecast(data)
        for value in [float('nan'),-1.,2.]:
            data=self.snapshot();data['odds'][0]['title_probability']=value
            with self.assertRaises(ValueError):public_forecast(data)
        data=self.snapshot();data['odds'][0]['title_probability']=.9
        with self.assertRaises(ValueError):public_forecast(data)

    def test_publication_preserves_surrounding_readme_and_old_output_on_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);readme=root/'README.md'
            readme.write_text('Before\n'+START+'\nold\n'+END+'\nAfter\n')
            publish(self.snapshot(),root)
            text=readme.read_text()
            self.assertTrue(text.startswith('Before\n'));self.assertTrue(text.endswith('\nAfter\n'))
            self.assertIn('2026-11-07',text);self.assertIn('2026-11-08 12:00 UTC',text)
            self.assertEqual(len(json.loads((root/'reports/forecast.json').read_text())['odds']),30)
            before=(root/'reports/title_probabilities.svg').read_bytes()
            bad=copy.deepcopy(self.snapshot());bad['odds'].pop()
            with self.assertRaises(ValueError):publish(bad,root)
            self.assertEqual(readme.read_text(),text)
            self.assertEqual((root/'reports/title_probabilities.svg').read_bytes(),before)

if __name__=='__main__':unittest.main()
