# %% [code]
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pandas.api.types import is_numeric_dtype
from pandas.api.types import CategoricalDtype

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

@dataclass
class _AggSpec:
    group_cols: Tuple[str, ...]
    value_cols: Tuple[str, ...]
    prefix: str

class FeatureFactory(BaseEstimator, TransformerMixin):
    valid_strategies = [
        'drop_id',
        'encoding',
        'ratios',
        'interactions',
        'binning',
        'drop_non_numeric'
    ]
    
    def __init__(
        self, 
        *,
        strategies=None,
        seed=10301,
        target='Churn',
        verbose=False,
    ):
        if strategies is None:
            strategies = []
        
        self.strategies = []
        self.seed = seed
        self.target = target
        self.verbose = verbose
        self._engineered_cat_cols_ = []
        
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

        # Service-related categories
        self.service_cols = [
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection', 
            'TechSupport',
            'StreamingTV',
            'StreamingMovies'
        
        ]
        
        # Learned state (set in fit)
        self._is_fit: bool = False

    
    # ----------------------------
    # Public API
    # ----------------------------
    def fit(self, df: pd.DataFrame = None) -> 'FeatureFactory':
        if self.verbose:
            print('  -> Fitting DataFrame...')
            
        self._is_fit = True
        return self

    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError('FeatureFactory must be fit() before transform().')

        if self.verbose:
            print(f'Applying FeatureFactory with strategies: {", ".join(self.strategies)}')
        
        df_new = df.copy()
            
        if 'drop_id' in self.strategies:
            df_new = self._drop_ids(df_new)
        
        if 'encoding' in self.strategies:
            df_new = self._add_encoding(df_new)
        
        if 'ratios' in self.strategies:
            df_new = self._add_ratios(df_new)

        if 'interactions' in self.strategies:
            df_new = self._add_interactions(df_new)
            
        if 'binning' in self.strategies:
            df_new = self._add_binning(df_new)
            
        if 'drop_non_numeric' in self.strategies:
            df_new = self._add_drop_non_numeric(df_new)
            
        return df_new

    
    def fit_transform(self, df:pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
        
    def get_strategies(self) -> []:
        return self.strategies

    # -------------------------
    # Internals
    # -------------------------
    def _drop_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        # Dropping the 'id' column
        if self.verbose:
            print('  -> Dropping the "id" column and cleaning TotalCharges...')

        if 'id' in df.columns:
            df = df.drop('id', axis=1)

        # Cleaning: TotalCharges has commas and empty spaces (from EDA)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = df['TotalCharges'].astype(str).str.replace(',', '')
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

        # Get rid of the target if it is in the dataset
        if self.target in df.columns:
            df = df.drop(self.target, axis=1)
            
        return df

    
    def _add_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles label mapping for the target and ordinal encoding for 
        high-signal categorical features.
        """
        if self.verbose:
            print('  -> Adding encoding features...')

        eps = 1e-6

        # Target Mapping (Internal for training/validation)
        if 'Churn' in df.columns and df['Churn'].dtype == 'object':
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

        # Service Engagement Features
        # EDA showed high signal in service-related categories
        existing_services = [c for c in self.service_cols if c in df.columns]
        if existing_services:
            # Count active services
            df['Total_Addon_Services'] = (df[existing_services] == 'Yes').sum(axis=1)

        # High-Signal Categorical Ordinal Encoding
        if 'Contract' in df.columns:
            col_name = 'Contract_Code'
            df[col_name] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
            if col_name not in self._engineered_cat_cols_:
                self._engineered_cat_cols_.append(col_name)
                
        return df
        

    def _add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Purely derived financial ratios to capture customer value over time.
        NaN-safe for use before scaling/clustering.
        """
        if self.verbose:
            print('  -> Adding ratio features...')

        # Financial Ratios
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            # Monthly cost relative to total time
            df['Avg_Cost_Per_Month_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
            
            # Identifying potential discounts or price hikes
            if 'TotalCharges' in df.columns:
                df['Expected_Total'] = df['MonthlyCharges'] * df['tenure']
                df['Charges_Gap'] = df['TotalCharges'] - df['Expected_Total']

        # Cost per addon service
        if 'MonthlyCharges' in df.columns and 'Total_Addon_Services' in df.columns:
            df['Cost_Per_Service'] = df['MonthlyCharges'] / (df['Total_Addon_Services'] + 1)

        # Average cost per service (TotalCharges / Number of Services)
        df['num_services'] = (df[self.service_cols] == 'Yes').sum(axis=1)
        df['cost_per_service'] = df['MonthlyCharges'] / (df['num_services'] + 1)
        
        # Tenure-Weighted Charges
        df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

        return df
        

    def _add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates features based on service density and interactions between 
        high-signal categorical features and numeric predictors.
        """
        if self.verbose:
            print('  -> Adding interactions features...')

        # Total Services (including Phone and Internet)
        all_services = self.service_cols + ['PhoneService', 'MultipleLines', 'InternetService']
        df['total_service_count'] = (df[all_services].isin(['Yes', 'Fiber optic', 'DSL'])).sum(axis=1)

        # Interaction between high-signal Contract and Tenure
        # This helps the model see if tenure signal changes for month-to-month vs long-term
        df['tenure_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int) * df['tenure']

        return df
        

    def _add_binning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bins continuous tenure into lifecycle stages to smooth out noise
        and improve model generalization.
        """
        if self.verbose:
            print('  -> Adding binning features...')

        # Based on industry standards: New (<1yr), Junior (1-2yrs), Established (2-5yrs), Loyal (>5yrs)
        col_name = 'tenure_group'
        df['tenure_group'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 60, 100], 
                                    labels=['New', 'Junior', 'Established', 'Loyal']).astype('category')
        
        if col_name not in self._engineered_cat_cols_:
                self._engineered_cat_cols_.append(col_name)
            
        return df
        

    def _add_drop_non_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes IDs and non-numeric columns to prepare the dataframe 
        for XGBoost/LGBM.
        """
        if self.verbose:
            print('  -> Adding numeric transforms...')
            
        # Drop non-numeric for XGBoost
        cols_to_drop = ['id', 'customerID', 'gender', 'Partner', 'Dependents', 
                        'PhoneService', 'MultipleLines', 'InternetService', 
                        'PaymentMethod', 'PaperlessBilling', 'Contract']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        
        return df


    def get_cat_features(self, df: pd.DataFrame) -> List[str]:
        """
        Prefer explicit cat features:
          - object/category columns
          - your binned classes + cluster label
        Avoid auto-marking low-cardinality numeric columns as categorical unless intended.
        """
        cat_cols = []

        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
                cat_cols.append(col)

        # Explicit engineered categorical columns we track
        for c in self._engineered_cat_cols_:
            if c in df.columns:
                cat_cols.append(c)

        # Ensure uniqueness, stable order
        seen = set()
        out = []
        for c in cat_cols:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out