import unittest
import pandas as pd
import numpy as np

# Assuming the uploaded file is named 'ps_s06e01_feature_engineering.py'
# and resides in the same directory.
from ps_s06e02_feature_engineering import FeatureFactory

class TestFeatureFactory(unittest.TestCase):
    
    def setUp(self):
        """
        Creates the DataFrame representing the first 10 rows of the provided dataset.
        Resets data before every test method.
        """
        data = {
            'id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'Age': [58, 52, 56, 44, 58, 38, 59, 60, 48, 44],
            'Sex': [1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            'Chest pain type': [4, 1, 2, 3, 4, 4, 4, 3, 4, 4],
            'BP': [152, 125, 160, 134, 140, 138, 130, 120, 140, 150],
            'Cholesterol': [239, 325, 188, 229, 234, 283, 246, 245, 212, 197],
            'FBS over 120': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'EKG results': [0, 2, 2, 2, 2, 0, 2, 0, 2, 0],
            'Max HR': [158, 171, 151, 150, 125, 147, 152, 151, 125, 150],
            'Exercise angina': [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            'ST depression': [3.6, 0.0, 0.0, 1.0, 3.8, 1.6, 0.8, 1.2, 0.0, 0.0],
            'Slope of ST': [2, 1, 1, 2, 2, 2, 2, 1, 1, 2],
            'Number of vessels fluro': [2, 0, 0, 0, 3, 2, 2, 0, 0, 0],
            'Thallium': [7, 3, 3, 3, 3, 7, 3, 3, 3, 3],
            'Heart Disease': [
                'Presence', 'Absence', 'Absence', 'Absence', 'Presence', 
                'Presence', 'Presence', 'Absence', 'Absence', 'Absence'
            ]        }
        self.df = pd.DataFrame(data)          # small fixture for exact row checks
        self.df_big = pd.concat([self.df] * 6, ignore_index=True)  # >= 50 rows for quantile bins
        self.seed = 10301

    def test_initialization(self):
        """Test that the class initializes with the correct categorical list."""
        ff = FeatureFactory(strategies=['drop_id'], target='Heart Disease')
        expected_cats = [
            'Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
            'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
        ]
        assert ff.cat_cols == expected_cats
    
    def test_init_validation(self):
        """Test that invalid strategies raise a ValueError."""
        with self.assertRaises(ValueError):
            FeatureFactory(strategies=['invalid_strategy'])

        # Should not raise
        FeatureFactory(strategies=[
            'drop_id', 'cast_categoricals', 'numeric_transforms', 'bins',
            'cat_crosses', 'interactions', 'one_hot_encoding', 'standard_scaling'
        ])

    def test_drop_id(self):
        """Test 'drop_id' strategy removes id and target columns."""
        ff = FeatureFactory(strategies=['drop_id'], target='Heart Disease')
        ff.fit(self.df)
        df_trans = ff.transform(self.df)
        
        self.assertNotIn('id', df_trans.columns)
        self.assertNotIn('Heart Disease', df_trans.columns)
        self.assertIn('Age', df_trans.columns) # Ensure other cols remain

    def test_interactions(self):
        """Test 'interactions' creates combined features."""
        ff = FeatureFactory(strategies=['interactions'])
        ff.fit(self.df)
        df_trans = ff.transform(self.df)
        
        # Check 1: Theoretical Max HR (One Hot * Hours)
        # Row 1 is 'Age' (52). 'Theoretical_Max_HR' should be 168
        self.assertIn('Theoretical_Max_HR', df_trans.columns)
        self.assertAlmostEqual(df_trans.loc[1, 'Theoretical_Max_HR'], 168, places=4)
        # Row 1 'HR_Deficiency' should be 0
        self.assertIn('HR_Deficiency', df_trans.columns)
        self.assertEqual(df_trans.loc[1, 'HR_Deficiency'], -3)
        
        # Check 2: Cholesterol_Age_Ratio ('Cholesterol' / ('Age' + 1))
        # Row 0: 239 / 59 = 9.8
        self.assertIn('Cholesterol_Age_Ratio', df_trans.columns)
        self.assertAlmostEqual(df_trans.loc[0, 'Cholesterol_Age_Ratio'], 4.0508, places=4)
        
        # Check 3: BP_Age_Factor
        # Row 0: 152 * 58
        self.assertIn('BP_Age_Factor', df_trans.columns)
        self.assertAlmostEqual(df_trans.loc[0, 'BP_Age_Factor'], 8816, places=5)

        # Check 4: Exercise Angina & Chest Pain Interaction
        # Row 0: 1 * 4 = 4
        self.assertIn('Angina_Pain_Interaction', df_trans.columns)
        self.assertAlmostEqual(df_trans.loc[0, 'Angina_Pain_Interaction'], 4, places=5)

    def test_cast_categoricals(self):
        """Test 'cast_categoricals' marks base categorical columns as dtype category (without needing 'interactions')."""
        ff = FeatureFactory(strategies=['cast_categoricals'], target='Heart Disease')
        ff.fit(self.df)
        df_trans = ff.transform(self.df)

        # Spot-check a couple
        self.assertTrue(str(df_trans['Sex'].dtype).startswith('category'))
        self.assertTrue(str(df_trans['Thallium'].dtype).startswith('category'))

    def test_numeric_transforms(self):
        """Test 'numeric_transforms' creates expected numeric engineered features and remains finite."""
        ff = FeatureFactory(strategies=['interactions', 'numeric_transforms'], target='Heart Disease')
        ff.fit(self.df)
        df_trans = ff.transform(self.df)

        self.assertIn('log1p_Cholesterol', df_trans.columns)
        self.assertIn('ST_depression_is_zero', df_trans.columns)
        self.assertIn('log1p_ST_depression', df_trans.columns)
        self.assertIn('MaxHR_ratio', df_trans.columns)
        self.assertIn('ST_by_MaxHR', df_trans.columns)

        # Row 1 has ST depression = 0.0 in fixture
        self.assertEqual(int(df_trans.loc[1, 'ST_depression_is_zero']), 1)

        # No inf / NaN in created numeric columns for this fixture
        for c in ['log1p_Cholesterol', 'log1p_ST_depression', 'MaxHR_ratio', 'ST_by_MaxHR']:
            self.assertTrue(np.isfinite(df_trans[c].astype(float)).all())

    def test_bins(self):
        """Test 'bins' creates bin columns (Age always; others require >=50 rows)."""
        ff = FeatureFactory(
            strategies=['interactions', 'numeric_transforms', 'bins'],
            target='Heart Disease'
        )
        ff.fit(self.df_big)
        df_trans = ff.transform(self.df_big)

        # Age bins should always exist (fixed edges in fit)
        self.assertIn('Age_bin', df_trans.columns)
        self.assertTrue(str(df_trans['Age_bin'].dtype).startswith('category'))

        # With >=50 rows, quantile bins should also be learned/materialized
        self.assertIn('Cholesterol_bin', df_trans.columns)
        self.assertIn('BP_bin', df_trans.columns)
        self.assertIn('ST_depression_bin', df_trans.columns)
        self.assertIn('MaxHR_ratio_bin', df_trans.columns)

    def test_cat_crosses(self):
        """Test 'cat_crosses' creates crossed categorical columns."""
        ff = FeatureFactory(
            strategies=['cat_crosses'],
            target='Heart Disease'
        )
        ff.fit(self.df)
        df_trans = ff.transform(self.df)

        self.assertIn('Vessels_x_Thallium', df_trans.columns)
        self.assertIn('ChestPain_x_SlopeST', df_trans.columns)
        self.assertTrue(str(df_trans['Vessels_x_Thallium'].dtype).startswith('category'))

    def test_one_hot_encoding(self):
        """Test 'one_hot_encoding' converts nominal cats to binaries."""
        ff = FeatureFactory(strategies=['one_hot_encoding'])
        ff.fit(self.df)
        df_trans = ff.transform(self.df)

        # Check original columns are gone
        self.assertNotIn('Sex', df_trans.columns)
        self.assertNotIn('Chest pain type', df_trans.columns)

        # Check new columns exist (sklearn output names)
        # e.g., gender_female, gender_male, gender_other
        self.assertIn('Sex_0', df_trans.columns)
        self.assertIn('Sex_1', df_trans.columns)
        self.assertIn('EKG results_2', df_trans.columns)

        # Check value mapping
        # Row 0 has Sex == 1 in fixture, so Sex_1 should be 1
        self.assertEqual(df_trans.loc[0, 'Sex_0'], 0.0)
        self.assertEqual(df_trans.loc[0, 'Sex_1'], 1.0)

        # Check handle_unknown='ignore'
        # Pass a df with a brand new category. It should result in all 0s for that feature group.
        new_data = pd.DataFrame(self.df.iloc[0:1].to_dict())

        # Make Sex explicitly categorical/string for this test so assigning a new label is valid
        new_data['Sex'] = new_data['Sex'].astype(str)
        new_data.loc[0, 'Sex'] = 'alien_species' # Unknown category
        
        df_new_trans = ff.transform(new_data)
        self.assertEqual(df_new_trans.loc[0, 'Sex_0'], 0.0)
        self.assertEqual(df_new_trans.loc[0, 'Sex_1'], 0.0)
        
    def test_one_hot_encoding_includes_engineered_cats(self):
        """OHE should include engineered categorical columns from bins/crosses when those strategies are active."""
        ff = FeatureFactory(
            strategies=['interactions', 'numeric_transforms', 'bins', 'cat_crosses', 'one_hot_encoding'],
            target='Heart Disease'
        )
        ff.fit(self.df_big)
        df_trans = ff.transform(self.df_big)

        # Engineered cats should have been one-hot encoded (original cols removed)
        self.assertNotIn('Vessels_x_Thallium', df_trans.columns)
        self.assertNotIn('Age_bin', df_trans.columns)

        # But at least one OHE column derived from them should exist
        engineered_ohe_cols = [c for c in df_trans.columns if c.startswith('Vessels_x_Thallium_')]
        self.assertTrue(len(engineered_ohe_cols) > 0)

        agebin_ohe_cols = [c for c in df_trans.columns if c.startswith('Age_bin_')]
        self.assertTrue(len(agebin_ohe_cols) > 0)

    def test_standard_scaling(self):
        """Test 'standard_scaling' centers numeric data around 0."""
        ff = FeatureFactory(strategies=['standard_scaling'], target='Heart Disease')
        ff.fit(self.df)
        df_trans = ff.transform(self.df)

        # Check scaling logic on a feature
        # Note: Sklearn uses biased estimator (divide by N), pandas uses unbiased (N-1)
        # We just check that transformation happened and mean is approx 0.
        mean_age = df_trans['Age'].mean()
        self.assertAlmostEqual(mean_age, 0.0, places=5)
        
        # Ensure values actually changed
        self.assertNotEqual(df_trans.loc[0, 'Age'], self.df.loc[0, 'Age'])

        # Check target 'Heart Disease' was NOT scaled
        # Row 0 Heart Disease is 78.3
        self.assertEqual(df_trans.loc[0, 'Heart Disease'], 'Presence')

    def test_linear_pipeline_combo(self):
        """Test the combination of OHE and Scaling (Ridge pipeline)."""
        # This confirms that OHE columns created during the process are subsequently scaled.
        strategies = ['interactions', 'numeric_transforms', 'bins', 'cat_crosses', 'one_hot_encoding', 'standard_scaling']
        ff = FeatureFactory(strategies=strategies, target='Heart Disease')
        
        df_trans = ff.fit_transform(self.df_big)
        
        # Check OHE column presence
        self.assertIn('Sex_0', df_trans.columns)
        
        # Check Scaling on OHE column
        # An OHE column usually has values 0 or 1. 
        # After scaling, they should be floats (centered around 0).
        val_0 = df_trans.loc[0, 'Sex_0']
        
        # Should NOT be exactly 0.0 or 1.0 if scaling worked
        self.assertNotEqual(val_0, 0.0)
        self.assertNotEqual(val_0, 1.0)
        
        # Check Scaling on standard numeric
        self.assertAlmostEqual(df_trans['Age'].mean(), 0.0, places=5)

        # Engineered OHE columns should exist in the linear pipeline too
        self.assertTrue(any(c.startswith('Age_bin_') for c in df_trans.columns))
        self.assertTrue(any(c.startswith('Vessels_x_Thallium_') for c in df_trans.columns))
        
if __name__ == '__main__':
    unittest.main()