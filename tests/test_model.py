import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline import FEATURES, TEAM_FEATURES, matchup, zscore, name_key
from model import bracket_tree, title_odds, paired_predictions

class ConstantModel:
    def predict_proba(self,x):
        return np.tile([.3,.7],(len(x),1))

class ModelTests(unittest.TestCase):
    def teams(self):
        return pd.DataFrame([dict(team=str(i),name=str(i),conference='E' if i<8 else 'W',
            wins=82-i,losses=i,gp=82,standing_order=i,season=2026,
            **{'z_'+f:float(i) for f in TEAM_FEATURES}) for i in range(16)])

    def series(self):
        alive = [str(i) for i in range(16)]
        rows=[]
        for rnd in range(1,5):
            nxt=[]
            for a,b in zip(alive[::2],alive[1::2]):
                rows.append(dict(round=rnd,winner=a,loser=b))
                nxt.append(a)
            alive=nxt
        return pd.DataFrame(rows)

    def test_mirroring(self):
        teams=self.teams().set_index('team',drop=False)
        ab,ba=matchup(teams,'0','1',1),matchup(teams,'1','0',1)
        for f in TEAM_FEATURES:
            self.assertEqual(ab['d_'+f],-ba['d_'+f])
        self.assertEqual(ab['home_court']+ba['home_court'],1)
        self.assertEqual(ab['round'],ba['round'])

    def test_neutral(self):
        teams=self.teams().set_index('team',drop=False)
        self.assertEqual(matchup(teams,'0','1',4,neutral=True)['home_court'],.5)

    def test_zscore(self):
        x=zscore(pd.Series([1,2,3]))
        self.assertAlmostEqual(x.mean(),0)
        self.assertAlmostEqual(x.std(ddof=0),1)
        self.assertTrue(zscore(pd.Series([1,1])).eq(0).all())

    def test_symmetric_probability(self):
        rows=pd.DataFrame([dict(series_id='s',**{f:0 for f in FEATURES})]*2)
        self.assertAlmostEqual(paired_predictions(ConstantModel(),rows,FEATURES)[0],.5)

    def test_uniform_bracket(self):
        odds=title_odds(ConstantModel(),FEATURES,self.teams(),self.series())
        np.testing.assert_allclose(odds.title_probability,1/16)
        self.assertAlmostEqual(odds.title_probability.sum(),1)

    def test_topology_does_not_condition_on_winners(self):
        a=self.series()
        b=a.copy()
        # Exchange the identities throughout the results. All entrant odds remain uniform.
        b[['winner','loser']]=b[['winner','loser']].replace({'0':'1','1':'0'})
        x=title_odds(ConstantModel(),FEATURES,self.teams(),a).sort_values('team')
        y=title_odds(ConstantModel(),FEATURES,self.teams(),b).sort_values('team')
        np.testing.assert_allclose(x.title_probability,y.title_probability)

    def test_name_normalization(self):
        self.assertEqual(name_key('Nikola Jokić'),name_key('Nikola Jokic'))
        self.assertEqual(name_key('Darrel Armstrong'),name_key('Darrell Armstrong'))

class DatasetTests(unittest.TestCase):
    def test_built_dataset(self):
        from data_pipeline import DATA,TARGET
        if not (DATA/'matchups.csv').exists():
            self.skipTest('Data not built yet')
        rows=pd.read_csv(DATA/'matchups.csv')
        teams=pd.read_csv(DATA/'teams.csv')
        self.assertFalse(rows[FEATURES].isna().any().any())
        self.assertEqual(rows.groupby('season').size().unique().tolist(),[30])
        self.assertEqual(rows.groupby('series_id').y.sum().unique().tolist(),[1])
        self.assertEqual(rows.season.max(),TARGET-1)
        current=teams.loc[teams.season.eq(TARGET-1)]
        self.assertEqual(len(current),30)
        self.assertTrue(current.gp.eq(82).all())
        for f in TEAM_FEATURES:
            np.testing.assert_allclose(rows.iloc[::2]['d_'+f].to_numpy(),-rows.iloc[1::2]['d_'+f].to_numpy(),atol=1e-10)

@unittest.skipUnless(all((Path(__file__).resolve().parents[1]/p).exists() for p in ['artifacts/evaluation.json','artifacts/walk_forward.csv','artifacts/xgboost.json','data/live_snapshot.json']), 'Optional local-data artifact check; see docs/SETUP.md')
class ArtifactTests(unittest.TestCase):
    def test_target_forecast_and_training_boundary(self):
        import json
        from pathlib import Path
        from xgboost import XGBClassifier
        from data_pipeline import DATA, TARGET
        artifacts=Path(__file__).resolve().parents[1]/'artifacts'
        report=json.loads((artifacts/'evaluation.json').read_text())
        self.assertEqual(report['trained_through'],2026)
        self.assertEqual(report['training_seasons'],25)
        self.assertEqual(report['training_series'],375)
        self.assertTrue(all(f['train_through'] < f['season'] < TARGET for f in report['folds']))
        heldout=pd.read_csv(artifacts/'walk_forward.csv')
        self.assertEqual(heldout.loc[heldout.model.eq('XGBoost')].shape[0],225)
        self.assertLess(heldout.season.max(),TARGET)
        model=XGBClassifier()
        model.load_model(artifacts/'xgboost.json')
        self.assertEqual(model.n_features_in_,len(report['features']))
        snapshot=json.loads((DATA/'live_snapshot.json').read_text())
        self.assertEqual(snapshot['season'],TARGET)
        self.assertEqual(snapshot['trained_through'],2026)
        odds=pd.DataFrame(snapshot['odds'])
        self.assertEqual(len(odds),30)
        self.assertAlmostEqual(odds.title_probability.sum(),1)
        self.assertAlmostEqual(odds.playoff_probability.sum(),16)

if __name__=='__main__':
    unittest.main()
