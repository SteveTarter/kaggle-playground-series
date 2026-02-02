# %% [code]
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

@dataclass
class _AggSpec:
    group_cols: Tuple[str, ...]
    value_cols: Tuple[str, ...]
    prefix: str

class FeatureFactory(BaseEstimator, TransformerMixin):
    """
    Feature engineering with optional:
      - binning of key numeric columns
      - interaction features (including stable study_method dummy interactions)
      - clustering features via StandardScaler
      - one-hot encoding (for linear models)
      - standard scaling (for linear models/NNs)
    Critical invariants:
      - Any columns created from categoricals must be stable between fit/transform.
      - Any 'learned' transformations (scaler) must be fit only on training data.
    """
    valid_strategies = [
        'drop_id',
        'one_hot_encoding',
        'interactions',
        'standard_scaling'
    ]
    
    def __init__(
        self, 
        *,
        strategies=None,
        seed=10301,
        target='Heart Disease',
        verbose=False,
        n_clusters=5,
    ):
        if strategies is None:
            strategies = []
        
        self.strategies = []
        self.seed = seed
        self.target = target
        self.verbose = verbose
        self.n_clusters = n_clusters

        # Learned state (fit-time)
        self._scaler: Optional[StandardScaler] = None
        self._ohe: Optional[OneHotEncoder] = None
        self._feature_scaler: Optional[StandardScaler] = None
        
        self._ohe_cols_ = None
        self._scaling_cols_ = None
        
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

        # Columns that are actually categorical codes
        self.cat_cols = [
            'Sex', 
            'Chest pain type', 
            'FBS over 120', 
            'EKG results', 
            'Exercise angina', 
            'Slope of ST', 
            'Number of vessels fluro', 
            'Thallium'
        ]
        
        # Learned state (set in fit)
        self._is_fit: bool = False

    
    # ----------------------------
    # Public API
    # ----------------------------
    def fit(self, df: pd.DataFrame = None) -> 'FeatureFactory':
        if self.verbose:
            print('  -> Fitting DataFrame...')
            
        df_new = df.copy()

        if 'one_hot_encoding' in self.strategies:
            self._add_one_hot_encoding_fit(df_new)
            
        # We run this last or near last to scale generated features too
        if 'standard_scaling' in self.strategies:
            # We need to temporarily apply transformations that create columns
            # to know what columns exist for scaling.
            # However, for simplicity here, we assume scaling handles existing numeric cols
            # at the time of transform.
            # Better approach: We calculate mean/std during the fit phase.
            # To do that accurately, we must replicate the generation steps on df_new
            # before fitting the scaler.
            if 'drop_id' in self.strategies:
                df_new = self._drop_ids(df_new)

            # Apply OHE transformation to df_new so we scale the OHE columns too if needed    
            if 'one_hot_encoding' in self.strategies:
                df_new = self._add_one_hot_encoding_transform(df_new)
            
            self._add_standard_scaling_fit(df_new)
            
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
        
        if 'interactions' in self.strategies:
            df_new = self._add_interactions(df_new)
        
        if 'one_hot_encoding' in self.strategies:
            df_new = self._add_one_hot_encoding_transform(df_new)

        if 'standard_scaling' in self.strategies:
            df_new = self._add_standard_scaling_transform(df_new)
            
        return df_new

    
    def fit_transform(self, df:pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
        
    def get_strategies(self) -> []:
        return self.strategies

    # -------------------------
    # Internals
    # -------------------------
    @staticmethod
    def _safe_cut_codes(s: pd.Series, bins, labels=None, include_lowest=True) -> pd.Series:
        """
        Returns integer codes, with NaN -> -1, safe for missing values.
        """
        cat = pd.cut(s, bins=bins, labels=labels, include_lowest=include_lowest)
        # cat.codes gives -1 for NaN
        return cat.cat.codes.astype('int16')
        
        
    @staticmethod
    def _normalize_name(col: str) -> str:
        """
        Normalize column names so they are stable and model-safe.
        Example:
            'effort_self-study' -> 'effort_self_study'
            'effort_group study' -> 'effort_group_study'
        """
        return (
            col.strip()
               .replace(' ', '_')
               .replace('-', '_')
        )

    
    def _drop_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        # Dropping the 'id' column
        if self.verbose:
            print('  -> Dropping the "id" column...')

        if 'id' in df.columns:
            df = df.drop('id', axis=1)

        if self.target in df.columns:
            df = df.drop(self.target, axis=1)
            
        return df

    

    def _add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Core interactions (purely derived, no learned state),
        but must be NaN-safe because clustering/scaling may use these values.
        """
        if self.verbose:
            print('  -> Adding interactions features...')

        eps = 1e-6

        # ---------------------------------------------------------
        # Domain Knowledge Features (Heart Disease Specific)
        # ---------------------------------------------------------
        
        # Theoretical Max Heart Rate (approx 220 - Age)
        df['Theoretical_Max_HR'] = 220 - df['Age']
        
        # Heart Rate Reserve / Deficiency
        # How far is the patient's Max HR from their theoretical max?
        df['HR_Deficiency'] = df['Theoretical_Max_HR'] - df['Max HR']
        
        # Cholesterol Ratio
        # Cholesterol generally increases with age, normalizing might help
        df['Cholesterol_Age_Ratio'] = df['Cholesterol'] / (df['Age'] + 1)
        
        # Blood Pressure Factor
        # High BP combined with Age can be a compounding risk
        df['BP_Age_Factor'] = df['BP'] * df['Age']
        
        # Exercise Angina & Chest Pain Interaction
        # (1 = Yes Angina) * Chest Pain Type creates a risk severity scale
        if 'Exercise angina' in df.columns and 'Chest pain type' in df.columns:
            df['Angina_Pain_Interaction'] = df['Exercise angina'] * df['Chest pain type']

        # ---------------------------------------------------------
        # Categorical Type Casting
        # ---------------------------------------------------------
        # Gradient Boosting models (XGB/LGBM/Cat) work best when 
        # distinct integer categories are explicitly cast as 'category'
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df
        
        
    def _add_one_hot_encoding_fit(self, df: pd.DataFrame) -> None:
        """
        Fit-time one hot encoding:
          - A OneHotEncoder is created
          - categorical features are identified and fit
          - OneHotEncoder is stored
        """
        if self.verbose:
            print('  -> One-Hot Encoding: fitting...')
            
        cat_cols = self.cat_cols
        self._ohe_cols_ = cat_cols
        
        if not cat_cols:
            return
            
        self._ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.int8)
        self._ohe.fit(df[cat_cols])

        
    def _add_one_hot_encoding_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform-time one hot encoding:
          - Stored OneHotEncoder is used to encode categorical features
          - Original categorical features are replaced with enccoded features. 
        """
        if self.verbose:
            print('  -> One-Hot Encoding: transform...')
            
        if self._ohe is None or not self._ohe_cols_:
            return df
        
        # Ensure cols exist
        present = [c for c in self._ohe_cols_ if c in df.columns]
        if len(present) != len(self._ohe_cols_):
            return df

        encoded_data = self._ohe.transform(df[self._ohe_cols_])
        feature_names = self._ohe.get_feature_names_out(self._ohe_cols_)
        
        # Create DataFrame with proper index
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
        
        # Drop original cats and concat
        df = df.drop(columns=self._ohe_cols_)
        df = pd.concat([df, encoded_df], axis=1)
        return df

    def _add_standard_scaling_fit(self, df: pd.DataFrame) -> None:
        """
        Fit-time one standard scaling:
          - numeric features are identified and fit
          - A StandardScaler is created and stored
          - StandardScaler is fit on numeric features
        """
        if self.verbose:
            print('  -> Standard Scaling: fitting...')
            
        # Numeric columns only
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target in num_cols:
            num_cols.remove(self.target)
            
        self._scaling_cols_ = num_cols
        
        if not num_cols:
            return
        
        self._feature_scaler = StandardScaler()
        
        # Handle NaNs before fitting scaler.
        # Simple fill for scaling stability.
        df_fill = df[num_cols].fillna(df[num_cols].mean())
        self._feature_scaler.fit(df_fill)

        
    def _add_standard_scaling_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform-time standard scaling:
          - Stored StandardScaler is used to transform numeric features
          - Replace any NaNs with 0. 
        """
        if self.verbose: print('  -> Standard Scaling: transform...')
        if self._feature_scaler is None:
            return df
        
        valid_cols = [c for c in self._scaling_cols_ if c in df.columns]
        if not valid_cols:
            return df
        
        # Note: Scikit-learn scalers don't handle NaNs by default in older versions, 
        # but modern ones propagate them. We fill NaNs to be safe for linear models.
        # Using the same means learned in fit would be ideal, but simple fillna(0) after scaling
        # (since mean is 0) works often.
        # Here we assume df is filled or let scaler handle it.
        df[valid_cols] = self._feature_scaler.transform(df[valid_cols])
        # Linear models hate NaNs, so we fill any remaining with 0 (mean)
        df[valid_cols] = df[valid_cols].fillna(0)
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

        # Explicit discrete engineered columns
        for c in ['class_attendance_class', 'sleep_hours_class', 'study_hours_class', 'cluster_label']:
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