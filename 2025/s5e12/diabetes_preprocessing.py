# %% [code]
import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class FeatureFactory(BaseEstimator, TransformerMixin):
    """
    Centralized factory for creating new numeric and engineered features.
    
    It focuses on creating new numeric features (ratios, polynomials) and
    transforming existing numeric features (log transforms).
    """
    valid_strategies = [
        'drop_id',
        'ordinal_encoding',
        'medical_metrics',
        'interactions',
        'ratios',
        'log',
        'polynomials',
        'one_hot_encoding'
    ]
        
    def __init__(self, strategies=None, target='diagnosed_diabetes', verbose=False):
        if strategies is None:
            strategies = []
        
        self.strategies = []
        self.target = target
        self.verbose = verbose
                
        invalid_strategies = set()
        
        # Validate and store only the requested strategies
        for strategy in strategies:
            if strategy in self.valid_strategies:
                self.strategies.append(strategy)
            else:
                invalid_strategies.add(strategy)
        
        if len(invalid_strategies) > 0:
            invalid_list_str = ','.join(invalid_strategies)
            raise ValueError(f'Invalid FeatureFactory strategies requested: {invalid_list_str}')
    
    def fit(self, df: pd.DataFrame, y=None):
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print(f"Applying FeatureFactory with strategies: {', '.join(self.strategies)}")
        
        df_new = df.copy()
        
        if 'drop_id' in self.strategies:
            df_new = self._drop_ids(df_new)
        
        if 'ordinal_encoding' in self.strategies:
            df_new = self._add_ordinal_encoding(df_new)
        
        if 'medical_metrics' in self.strategies:
            df_new = self._add_medical_metrics(df_new)
        
        if 'interactions' in self.strategies:
            df_new = self._add_interactions(df_new)
        
        if 'ratios' in self.strategies:
            df_new = self._add_ratios(df_new)
        
        if 'log' in self.strategies:
            df_new = self._add_log_transforms(df_new)
        
        if 'polynomials' in self.strategies:
            df_new = self._add_polynomials(df_new)
        
        if 'one_hot_encoding' in self.strategies:
            df_new = self._add_one_hot_encodings(df_new)
        
        return df_new
    
    def strategies(self) -> []:
        return self.strategies
    
    def _drop_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        # Dropping the 'id' column
        if self.verbose:
            print("  -> Dropping the 'id' column...")
        
        df = df.drop('id', axis=1)
        
        return df
        
    def _add_ordinal_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        # Preserves the rank order of Education and Income
        if self.verbose:
            print('  -> Adding ordinal encodings...')
        
        # Education Mapping
        # Assuming typical hierarchy: No Schooling < Elementary < Highschool < Graduate
        edu_map = {
            'No Schooling': 0,
            'Elementary': 1,
            'Highschool': 2,
            'Graduate': 3
        }
        
        # Income Mapping
        income_map = {
            'Low': 0,
            'Lower-Middle': 1,
            'Middle': 2,
            'Upper-Middle': 3,
            'High': 4
        }
        
        if 'education_level' in df.columns:
            df['education_level_ord'] = df['education_level'].map(edu_map).fillna(-1)
            
        if 'income_level' in df.columns:
            df['income_level_ord'] = df['income_level'].map(income_map).fillna(-1)
            
        return df
    
    def _add_medical_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # Adds biologically relevant cardiovascular metrics
        if self.verbose:
            print('  -> Adding medical metrics (MAP, Pulse Pressure)...')
        
        # Pulse Pressure = Systolic - Diastolic
        # Indicates arterial stiffness
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        
        # Mean Arterial Pressure (MAP)
        # MAP = Diastolic + (1/3 * Pulse Pressure)
        # Represents the average pressure in a patient's arteries during one cardiac cycle
        df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
        
        # Non-HDL Cholesterol
        # Often a better risk indicator than LDL alone
        df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
        
        return df
    
    def _add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        # Adds interaction terms between features
        if self.verbose:
            print('  -> Adding interaction features...')
        
        # Age * BMI Interaction
        # The metabolic impact of BMI often worsens with age
        df['age_bmi_interaction'] = df['age'] * df['bmi']
        
        # Sedentary Ratio
        # Ratio of screen time (hrs) to physical activity (mins).
        # Note: Units differ, but the relative ratio remains a valid index.
        # Adding 1 to denominator to avoid division by zero.
        df['sedentary_ratio'] = df['screen_time_hours_per_day'] / (df['physical_activity_minutes_per_week'] + 1)
        
        return df
        
    def _add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ratios based on physiological risk factors
        if self.verbose:
            print('  -> Adding ratio features...')
        
        # Total Cholesterol / HDL (a key risk ratio)
        df['cholesterol_risk_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
        
        # Systolic BP / Diastolic BP
        df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1e-5)
        
        # BMI related ratio (waist to hip ratio and BMI are highly correlated)
        df['whr_bmi_product'] = df['waist_to_hip_ratio'] * df['bmi']
        
        return df
        
    def _add_log_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        # Log transformation is useful for right-skewed features
        if self.verbose:
            print('  -> Adding log transforms...')
        
        # Triglycerides is often highly skewed
        df['triglycerides_log'] = np.log1p(df['triglycerides'])
        
        # Screen time might also benefit
        df['screen_time_log'] = np.log1p(df['screen_time_hours_per_day'])
        
        return df
        
    def _add_polynomials(self, df: pd.DataFrame) -> pd.DataFrame:
        # Adds simple squares of key features
        if self.verbose:
            print('  -> Adding polynomial features (squares of key features)...')
        
        df['age_sq'] = df['age'] ** 2
        df['bmi_sq'] = df['bmi'] ** 2
        
        return df
        
    def _add_one_hot_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        # Adds one hot encodings of categorical features
        if self.verbose:
            print('  -> Adding one hot encodings of categorical features')
        
        # Identify Feature Types
        # We exclude the target from the features list
        features = [c for c in df.columns if c != self.target]
        
        # Automatically select categorical columns for encoding
        # (gender, ethnicity, education_level, income_level, smoking_status, employment_status)
        cat_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
        
        df = pd.get_dummies(df, columns=cat_features, prefix_sep='_', dtype=int)
        
        return df;
