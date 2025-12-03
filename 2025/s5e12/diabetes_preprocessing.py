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
    
    This factory should be run BEFORE the ColumnTransformer in your model pipeline.
    It focuses on creating new numeric features (ratios, polynomials) and 
    transforming existing numeric features (log transforms).
    """
    valid_strategies = ['drop_id', 'ratios', 'log', 'polynomials', 'one_hot_encoding']
    
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
        
        for strategy in self.strategies:
            if strategy == 'drop_id':
                df_new = self._drop_ids(df_new)
            if strategy == 'ratios':
                df_new = self._add_ratios(df_new)
            elif strategy == 'log':
                df_new = self._add_log_transforms(df_new)
            elif strategy == 'polynomials':
                df_new = self._add_polynomials(df_new)
            elif strategy == 'one_hot_encoding':
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
