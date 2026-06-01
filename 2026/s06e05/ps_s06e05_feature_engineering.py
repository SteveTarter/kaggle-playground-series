from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureFactory(BaseEstimator, TransformerMixin):
    valid_strategies = ['encoding', 'race_dynamics', 'tire_physics', 'rolling_stats']
    
    def __init__(self, strategies=None, seed=10301, target='PitNextLap', verbose=False):
        self.strategies = strategies if strategies else ['encoding', 'race_dynamics']
        self.seed = seed
        self.target = target
        self.verbose = verbose
        self.cat_cols = ['Driver', 'Track', 'Compound', 'Race', 'Team']
        self._is_fit = False
        self._engineered_cat_cols_ = []

    def fit(self, df: pd.DataFrame = None) -> 'FeatureFactory':
        self._is_fit = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit: raise RuntimeError('Factory must be fit.')
        df_new = df.copy()
        
        # 1. Race Dynamics: How far into the race/stint are we?
        if 'race_dynamics' in self.strategies:
            # Total # of laps in race
            df_new['TotalLaps'] = df_new['LapNumber'] / df_new['RaceProgress']
            # How much of the race is left
            df_new['LapsRemaining'] = df_new['TotalLaps'] - df_new['LapNumber']
            # Position volatility
            df_new['PositionVolatility'] = df_new['Position_Change'].abs()
            # Progress remaining
            df_new['RemainingRace'] = 1.0 - df_new['RaceProgress']
            # Percentage of stint relative to average (logic placeholder)
            df_new['StintProgress'] = df_new['TyreLife'] / (df_new['LapNumber'] + 1)
            # Volatility of position
            df_new['PositionChange'] = df_new['Position'].diff().fillna(0)
                
        # 2. Tire Physics: Degradation rates
        if 'tire_physics' in self.strategies:
            # Degradation per lap
            df_new['DegradationRate'] = df_new['Cumulative_Degradation'] / (df_new['TyreLife'] + 1)
            # Log transform for cumulative degradation (handles outliers)
            df_new['Log_Degradation'] = np.sign(df_new['Cumulative_Degradation']) * np.log1p(df_new['Cumulative_Degradation'].abs())
            # Log transform for cumulative wear
            df_new['LogWear'] = np.log1p(df_new['TyreLife'])
            
        # 3. Encoding
        if 'encoding' in self.strategies:
            for col in self.cat_cols:
                if col in df_new.columns:
                    df_new[col] = df_new[col].astype('category').cat.codes

        # Cleanup
        if 'id' in df_new.columns: df_new.drop('id', axis=1, inplace=True)
        if self.target in df_new.columns: df_new.drop(self.target, axis=1, inplace=True)
            
        return df_new

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

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