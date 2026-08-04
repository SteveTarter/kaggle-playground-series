from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


class FeatureFactory(BaseEstimator, TransformerMixin):
    """
    Feature engineering pipeline for PS-S06E08: Predicting Smartphone Addiction.

    Target classes: binary classification (0 = not addicted, 1 = addicted)
    Raw features: age, daily_screen_time_hours, social_media_hours, gaming_hours,
                  work_study_hours, sleep_hours, notifications_per_day,
                  app_opens_per_day, weekend_screen_time, gender, stress_level,
                  academic_work_impact
    """

    valid_strategies = [
        'encoding',
        'screen_time_ratios',
        'sleep_stress',
        'risk_score',
        'missing_flags',
        'numeric_expansion',
    ]

    def __init__(
        self,
        *,
        strategies=None,
        impute_strategy: Optional[str] = None,
        seed=10301,
        target='addicted_label',
        verbose=False
    ):
        if strategies is None:
            strategies = []

        self.strategies = []
        self.impute_strategy = impute_strategy
        self.seed = seed
        self.target = target
        self.verbose = verbose
        self._engineered_cat_cols_: List[str] = []

        invalid_strategies = set()
        for strategy in strategies:
            if strategy in self.valid_strategies:
                self.strategies.append(strategy)
            else:
                invalid_strategies.add(strategy)

        if invalid_strategies:
            raise ValueError(
                f'Invalid FeatureFactory strategies requested: {",".join(invalid_strategies)}'
            )

        self.cat_cols = [
            'gender',
            'stress_level',
            'academic_work_impact'
        ]

        self.num_cols = [
            'age',
            'daily_screen_time_hours',
            'social_media_hours',
            'gaming_hours',
            'work_study_hours',
            'sleep_hours',
            'notifications_per_day',
            'app_opens_per_day',
            'weekend_screen_time'
        ]

        self._is_fit: bool = False
        self.medians_: Dict[str, float] = {}
        self.modes_: Dict[str, str] = {}

    # ----------------------------
    # Public API
    # ----------------------------
    def fit(self, df: pd.DataFrame) -> 'FeatureFactory':
        if self.verbose:
            print('  -> Fitting DataFrame...')
            
        df_fit = df.copy()
        # Always drop id and target
        df_fit.drop('id', axis=1, inplace=True, errors='ignore')
        df_fit.drop(self.target, axis=1, inplace=True, errors='ignore')
        
        self.num_features_ = df_fit.select_dtypes(exclude=['object', 'bool', 'category']).columns.tolist()
        self.cat_features_ = df_fit.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

        # Learn imputation parameters if requested
        if self.impute_strategy == 'median_mode':
            for col in self.num_features_:
                self.medians_[col] = float(df_fit[col].median(skipna=True))
            for col in self.cat_features_:
                # Compute mode, handle empty case
                mode_series = df_fit[col].mode(dropna=True)
                self.modes_[col] = str(mode_series.iloc[0]) if not mode_series.empty else ''

        self._is_fit = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError('FeatureFactory must be fit() before transform().')

        if self.verbose:
            print(f'Applying FeatureFactory with strategies: {", ".join(self.strategies)}')

        df_new = df.copy()

        # Always drop id and target
        df_new.drop('id', axis=1, inplace=True, errors='ignore')
        if self.target in df_new.columns:
            df_new.drop(self.target, axis=1, inplace=True, errors='ignore')

        # Impute missing values if requested
        if self.impute_strategy == 'median_mode':
            if self.verbose:
                print('  -> Imputing missing values using learned medians and modes...')
            for col, val in self.medians_.items():
                if col in df_new.columns:
                    df_new[col] = df_new[col].fillna(val)
            for col, val in self.modes_.items():
                if col in df_new.columns:
                    df_new[col] = df_new[col].fillna(val)

        # Apply feature engineering strategies
        if 'missing_flags' in self.strategies:
            df_new = self._add_missing_flags(df_new)

        if 'encoding' in self.strategies:
            df_new = self._add_encoding(df_new)

        if 'screen_time_ratios' in self.strategies:
            df_new = self._add_screen_time_ratios(df_new)

        if 'sleep_stress' in self.strategies:
            df_new = self._add_sleep_stress(df_new)

        if 'risk_score' in self.strategies:
            df_new = self._add_risk_score(df_new)

        if 'numeric_expansion' in self.strategies:
            df_new = self._add_numeric_expansion(df_new)

        return df_new

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_strategies(self) -> list:
        return self.strategies

    # -------------------------
    # Internals
    # -------------------------
    def _add_missing_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary indicators for features that are missing.
        """
        if self.verbose:
            print('  -> Adding missing flags...')
        for col in self.num_cols + self.cat_cols:
            if col in df.columns:
                df[f'is_missing_{col}'] = df[col].isnull().astype(np.int8)
        return df

    def _add_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert object columns to pandas Categorical type with logical orders.
        """
        if self.verbose:
            print('  -> Encoding categorical variables...')

        # Ordinal stress_level: Low < Medium < High
        if 'stress_level' in df.columns:
            df['stress_level'] = pd.Categorical(
                df['stress_level'], categories=['Low', 'Medium', 'High'], ordered=True
            )

        # Ordinal academic_work_impact: No < Yes
        if 'academic_work_impact' in df.columns:
            df['academic_work_impact'] = pd.Categorical(
                df['academic_work_impact'], categories=['No', 'Yes'], ordered=True
            )

        # Nominal features cast to category
        for col in ['gender']:
            if col in df.columns:
                df[col] = df[col].astype('category')

        return df

    def _add_screen_time_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates screen time fractions and ratios.
        """
        if self.verbose:
            print('  -> Adding screen time ratio features...')

        screen_time = df['daily_screen_time_hours'] if 'daily_screen_time_hours' in df.columns else None

        if screen_time is not None:
            if 'social_media_hours' in df.columns:
                df['social_media_fraction'] = df['social_media_hours'] / (screen_time + 1e-5)
            if 'gaming_hours' in df.columns:
                df['gaming_fraction'] = df['gaming_hours'] / (screen_time + 1e-5)
            if 'social_media_hours' in df.columns and 'gaming_hours' in df.columns:
                df['total_social_gaming_hours'] = df['social_media_hours'] + df['gaming_hours']
                df['non_social_gaming_hours'] = (screen_time - df['total_social_gaming_hours']).clip(lower=0)

            if 'weekend_screen_time' in df.columns:
                df['weekend_to_weekday_ratio'] = df['weekend_screen_time'] / (screen_time + 1e-5)

            if 'app_opens_per_day' in df.columns:
                df['app_opens_per_hour'] = df['app_opens_per_day'] / (screen_time + 1e-5)
            
            if 'notifications_per_day' in df.columns:
                df['notifications_per_hour'] = df['notifications_per_day'] / (screen_time + 1e-5)

            if 'age' in df.columns:
                df['screen_time_per_age'] = screen_time / (df['age'] + 1e-5)
                if 'app_opens_per_day' in df.columns:
                    df['app_opens_per_age'] = df['app_opens_per_day'] / (df['age'] + 1e-5)
                if 'notifications_per_day' in df.columns:
                    df['notifications_per_age'] = df['notifications_per_day'] / (df['age'] + 1e-5)

        if 'notifications_per_day' in df.columns and 'app_opens_per_day' in df.columns:
            df['notifications_per_app_open'] = df['notifications_per_day'] / (df['app_opens_per_day'] + 1e-5)
            df['app_opens_to_notifications_ratio'] = df['app_opens_per_day'] / (df['notifications_per_day'] + 1e-5)

        return df

    def _add_sleep_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sleep and stress interactions.
        """
        if self.verbose:
            print('  -> Adding sleep_stress interaction features...')

        # Custom numerical mapping for calculation
        stress_num = df['stress_level'].astype(str).map({'Low': 1, 'Medium': 2, 'High': 3}).fillna(2)

        if 'sleep_hours' in df.columns:
            # Sleep shortage below recommendation (8.0 hours)
            df['sleep_shortage'] = (8.0 - df['sleep_hours']).clip(lower=0)
            df['stress_to_sleep_ratio'] = stress_num / (df['sleep_hours'] + 1e-5)
            
            if 'daily_screen_time_hours' in df.columns:
                df['screen_time_to_sleep_ratio'] = df['daily_screen_time_hours'] / (df['sleep_hours'] + 1e-5)
            
            if 'social_media_hours' in df.columns:
                df['social_media_to_sleep_ratio'] = df['social_media_hours'] / (df['sleep_hours'] + 1e-5)
                
            if 'gaming_hours' in df.columns:
                df['gaming_to_sleep_ratio'] = df['gaming_hours'] / (df['sleep_hours'] + 1e-5)

        return df

    def _add_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Additive heuristic risk score from unhealthy indicators.
        """
        if self.verbose:
            print('  -> Adding lifestyle risk score features...')

        risk = pd.Series(0.0, index=df.index)

        if 'daily_screen_time_hours' in df.columns:
            risk += (df['daily_screen_time_hours'] > 6.0).astype(float).fillna(0.0)
        if 'stress_level' in df.columns:
            risk += (df['stress_level'].astype(str) == 'High').astype(float).fillna(0.0)
        if 'academic_work_impact' in df.columns:
            risk += (df['academic_work_impact'].astype(str) == 'Yes').astype(float).fillna(0.0)
        if 'sleep_hours' in df.columns:
            risk += (df['sleep_hours'] < 6.0).astype(float).fillna(0.0)
        if 'app_opens_per_day' in df.columns:
            risk += (df['app_opens_per_day'] > 100.0).astype(float).fillna(0.0)

        df['addiction_risk_score'] = risk
        return df

    def _add_numeric_expansion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Logs, squares, and square roots of numerical columns.
        """
        if self.verbose:
            print('  -> Adding numeric expansion features...')

        for c in self.num_features_:
            if c in df.columns:
                df[f"Log_{c}"] = np.log1p(np.clip(df[c], -1 + 1e-6, None))
                df[f"{c}_sq"] = df[c] ** 2            
                df[f"{c}_sqrt"] = np.sqrt(np.clip(df[c], 0, None))
        
        return df

    # -------------------------
    # Utility
    # -------------------------
    def get_cat_features(self, df: pd.DataFrame) -> List[str]:
        """
        Returns the list of categorical columns present in df, including
        any engineered categorical columns tracked by this instance.
        """
        cat_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
                cat_cols.append(col)

        for c in self._engineered_cat_cols_:
            if c in df.columns and c not in cat_cols:
                cat_cols.append(c)

        # Stable deduplicated order
        seen: set = set()
        out: List[str] = []
        for c in cat_cols:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out
