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
        'cast_categoricals',
        'numeric_transforms',
        'bins',
        'cat_crosses',
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
        self._bin_edges_: Dict[str, np.ndarray] = {}
        self._engineered_cat_cols_: List[str] = []
        
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

        # Replicate generation steps so learned transforms (bins/OHE/scaler)
        # see the same columns that will exist at transform-time.
        df_new = self._apply_generation_steps_for_fit(df_new)

        if 'one_hot_encoding' in self.strategies:
            self._add_one_hot_encoding_fit(df_new)
            
        if 'standard_scaling' in self.strategies:
            # scale after OHE, so linear/NN can scale OHE columns too
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
        
        # Make categorical typing independent of 'interactions'
        if 'cast_categoricals' in self.strategies:
            df_new = self._cast_categoricals(df_new)

        if 'interactions' in self.strategies:
            df_new = self._add_interactions(df_new)
        
        if 'numeric_transforms' in self.strategies:
            df_new = self._add_numeric_transforms(df_new)

        if 'bins' in self.strategies:
            df_new = self._add_bins_transform(df_new)

        if 'cat_crosses' in self.strategies:
            df_new = self._add_cat_crosses(df_new)

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


    def _as_numeric_series(self, s: pd.Series) -> pd.Series:
        if isinstance(s.dtype, CategoricalDtype):
            # if categories are numeric strings/ints, try numeric; else codes
            try:
                return pd.to_numeric(s.astype(str), errors='raise')
            except Exception:
                return s.cat.codes.astype('float64')
                
        return pd.to_numeric(s, errors='coerce')

    
    def _apply_generation_steps_for_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same feature-generation steps used in transform(),
        but without any learned transforms that depend on already-fitted state.
        This is used so OHE/scaler are fit on the final feature set.
        """
        df_new = df.copy()
        if 'drop_id' in self.strategies:
            df_new = self._drop_ids(df_new)
        if 'cast_categoricals' in self.strategies:
            df_new = self._cast_categoricals(df_new)
        if 'interactions' in self.strategies:
            df_new = self._add_interactions(df_new)
        if 'numeric_transforms' in self.strategies:
            df_new = self._add_numeric_transforms(df_new)
        if 'bins' in self.strategies:
            df_new = self._add_bins_fit(df_new)      # learns bin edges
            df_new = self._add_bins_transform(df_new) # materialize bin columns
        if 'cat_crosses' in self.strategies:
            df_new = self._add_cat_crosses(df_new)
        return df_new


    def _drop_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        # Dropping the 'id' column
        if self.verbose:
            print('  -> Dropping the "id" column...')

        if 'id' in df.columns:
            df = df.drop('id', axis=1)

        if self.target in df.columns:
            df = df.drop(self.target, axis=1)
            
        return df

    
    def _cast_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Casting categoricals...')
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
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
            df['Angina_Pain_Interaction'] = self._as_numeric_series(df['Exercise angina']) * self._as_numeric_series(df['Chest pain type'])
            
        return df
        

    def _add_numeric_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NaN-safe numeric transforms that help across XGB/LGBM/Cat/NN.
        """
        if self.verbose:
            print('  -> Adding numeric transforms...')
            
        # log transforms (guard for negatives; these should be >=0 in this dataset)
        eps = 1e-6
        if 'Cholesterol' in df.columns:
            df['log1p_Cholesterol'] = np.log1p(np.clip(df['Cholesterol'].astype(float), 0, None))
            
        if 'ST depression' in df.columns:
            st = df['ST depression'].astype(float)
            df['ST_depression_is_zero'] = (st.fillna(0) == 0).astype('int8')
            df['log1p_ST_depression'] = np.log1p(np.clip(st, 0, None))
            
        # ratios
        if 'Max HR' in df.columns and 'Theoretical_Max_HR' in df.columns:
            df['MaxHR_ratio'] = df['Max HR'] / (df['Theoretical_Max_HR'].astype(float) + eps)
            
        if 'ST depression' in df.columns and 'Max HR' in df.columns:
            df['ST_by_MaxHR'] = df['ST depression'].astype(float) / (df['Max HR'].astype(float) + eps)
            
        return df


    def _add_bins_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Learn bin edges on training data only (fold-safe).
        Store edges in self._bin_edges_.
        """
        if self.verbose:
            print('  -> Learning bin edges...')
            
        self._bin_edges_.clear()
        
        # Fixed bins for Age (domain-ish)
        if 'Age' in df.columns:
            self._bin_edges_['Age'] = np.array([0, 30, 40, 50, 60, 70, 80, 200], dtype=float)
            
        # Quantile bins for skewed/continuous-ish columns
        for col, q in [('Cholesterol', 10), ('BP', 10), ('ST depression', 10), ('MaxHR_ratio', 10)]:
            if col in df.columns and is_numeric_dtype(df[col]):
                s = df[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
                if len(s) < 50:
                    continue
                    
                qs = np.linspace(0, 1, q + 1)
                edges = np.unique(np.quantile(s, qs))
                if len(edges) >= 3:
                    # expand ends a hair so pd.cut includes extremes
                    edges[0] = edges[0] - 1e-9
                    edges[-1] = edges[-1] + 1e-9
                    self._bin_edges_[col] = edges
                    
        return df


    def _add_bins_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Materialize *_bin columns using learned edges. Output are categorical codes.
        """
        if self.verbose:
            print('  -> Adding bins...')
            
        engineered = []
        for col, edges in self._bin_edges_.items():
            if col not in df.columns:
                continue
                
            out_col = f'{self._normalize_name(col)}_bin'
            df[out_col] = self._safe_cut_codes(df[col].astype(float), bins=edges)
            df[out_col] = df[out_col].astype('category')
            engineered.append(out_col)
            
        # track engineered cat cols for CatBoost/LGBM convenience
        self._engineered_cat_cols_ = sorted(set(self._engineered_cat_cols_ + engineered))
        
        return df


    def _add_cat_crosses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Low-cardinality categorical crosses as single token/category.
        Good for CatBoost and NN embeddings; OK for LGBM/XGB native-cat.
        """
        if self.verbose:
            print('  -> Adding categorical crosses...')
            
        engineered = []
        crosses = [
            ('Number of vessels fluro', 'Thallium', 'Vessels_x_Thallium'),
            ('Chest pain type', 'Slope of ST', 'ChestPain_x_SlopeST'),
        ]
        
        for a, b, name in crosses:
            if a in df.columns and b in df.columns:
                s = df[a].astype(str) + '|' + df[b].astype(str)
                df[name] = s.astype('category')
                engineered.append(name)
                
        self._engineered_cat_cols_ = sorted(set(self._engineered_cat_cols_ + engineered))
        
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
            
        # OHE should include engineered categorical columns too
        cat_cols = [c for c in (self.cat_cols + self._engineered_cat_cols_) if c in df.columns]
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