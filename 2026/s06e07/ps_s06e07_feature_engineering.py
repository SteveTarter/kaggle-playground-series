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
    Feature engineering pipeline for PS-S06E07: Predicting Student Health Risk.

    Target classes: at-risk (0), unhealthy (1), fit (2)
    Raw features: sleep_duration, heart_rate, bmi, calorie_expenditure, step_count,
                  exercise_duration, water_intake, diet_type, stress_level,
                  sleep_quality, physical_activity_level, smoking_alcohol, gender
    """

    valid_strategies = [
        'encoding',
        'sleep_stress',
        'activity_energy',
        'bmi_features',
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
        target='health_condition',
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
            'diet_type',
            'stress_level',
            'sleep_quality',
            'physical_activity_level',
            'smoking_alcohol',
            'gender'
        ]

        self.num_cols = [
            'sleep_duration',
            'heart_rate',
            'bmi',
            'calorie_expenditure',
            'step_count',
            'exercise_duration',
            'water_intake'
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

        if 'sleep_stress' in self.strategies:
            df_new = self._add_sleep_stress(df_new)

        if 'activity_energy' in self.strategies:
            df_new = self._add_activity_energy(df_new)

        if 'bmi_features' in self.strategies:
            df_new = self._add_bmi_features(df_new)

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

        # Ordinal stress_level: low < medium < high
        if 'stress_level' in df.columns:
            df['stress_level'] = pd.Categorical(
                df['stress_level'], categories=['low', 'medium', 'high'], ordered=True
            )

        # Ordinal sleep_quality: poor < average < good
        if 'sleep_quality' in df.columns:
            df['sleep_quality'] = pd.Categorical(
                df['sleep_quality'], categories=['poor', 'average', 'good'], ordered=True
            )

        # Ordinal physical_activity_level: sedentary < moderate < active
        if 'physical_activity_level' in df.columns:
            df['physical_activity_level'] = pd.Categorical(
                df['physical_activity_level'], categories=['sedentary', 'moderate', 'active'], ordered=True
            )

        # Ordinal smoking_alcohol: no < occasional < yes
        if 'smoking_alcohol' in df.columns:
            df['smoking_alcohol'] = pd.Categorical(
                df['smoking_alcohol'], categories=['no', 'occasional', 'yes'], ordered=True
            )

        # Nominal features cast to category
        for col in ['diet_type', 'gender']:
            if col in df.columns:
                df[col] = df[col].astype('category')

        return df

    def _add_sleep_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sleep and stress interactions.
        """
        if self.verbose:
            print('  -> Adding sleep_stress interaction features...')

        # Custom numerical mapping for calculation (uses pd.Series.map which works with categories)
        stress_num = df['stress_level'].astype(str).map({'low': 1, 'medium': 2, 'high': 3})
        sleep_qual_num = df['sleep_quality'].astype(str).map({'poor': 1, 'average': 2, 'good': 3})

        if 'sleep_duration' in df.columns:
            df['sleep_quality_index'] = df['sleep_duration'] * sleep_qual_num
            df['sleep_to_stress_ratio'] = df['sleep_duration'] / (stress_num + 1e-5)
            # Sleep shortage below recommendation (8.0 hours)
            df['sleep_shortage'] = (8.0 - df['sleep_duration']).clip(lower=0)

        return df

    def _add_activity_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ratios and interaction metrics between steps, exercise, and calories.
        """
        if self.verbose:
            print('  -> Adding activity_energy features...')

        activity_num = df['physical_activity_level'].astype(str).map({'sedentary': 1, 'moderate': 2, 'active': 3})

        has_steps = 'step_count' in df.columns
        has_cal = 'calorie_expenditure' in df.columns
        has_ex = 'exercise_duration' in df.columns

        if has_steps and has_cal:
            df['calories_per_step'] = df['calorie_expenditure'] / (df['step_count'] + 1.0)
        
        if has_cal and has_ex:
            df['calories_per_exercise_min'] = df['calorie_expenditure'] / (df['exercise_duration'] + 1.0)

        if has_steps and has_ex:
            df['steps_per_exercise_min'] = df['step_count'] / (df['exercise_duration'] + 1.0)

        if has_ex:
            df['exercise_intensity_proxy'] = df['exercise_duration'] * activity_num

        return df

    def _add_bmi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BMI groupings and interaction features.
        """
        if self.verbose:
            print('  -> Adding bmi features...')

        if 'bmi' in df.columns:
            # BMI category binning
            # Underweight (< 18.5), Normal (18.5 - 24.9), Overweight (25 - 29.9), Obese (>= 30)
            df['bmi_category'] = pd.cut(
                df['bmi'],
                bins=[-np.inf, 18.5, 24.9, 29.9, np.inf],
                labels=['underweight', 'normal', 'overweight', 'obese']
            ).astype('category')
            
            if 'calorie_expenditure' in df.columns:
                df['bmi_x_calories'] = df['bmi'] * df['calorie_expenditure']
            if 'step_count' in df.columns:
                df['bmi_per_step'] = df['bmi'] / (df['step_count'] + 1.0)

            self._engineered_cat_cols_.append('bmi_category')

        return df

    def _add_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Additive heuristic risk score from unhealthy indicators.
        """
        if self.verbose:
            print('  -> Adding lifestyle risk score features...')

        stress_num = df['stress_level'].astype(str).map({'low': 1, 'medium': 2, 'high': 3})
        sleep_qual_num = df['sleep_quality'].astype(str).map({'poor': 1, 'average': 2, 'good': 3})
        activity_num = df['physical_activity_level'].astype(str).map({'sedentary': 1, 'moderate': 2, 'active': 3})

        # Calculate a sum of risk points:
        # High stress (+1), poor sleep (+1), sedentary (+1), smoking/alcohol yes (+1), BMI >= 25 (+1), Sleep < 6.0 (+1)
        risk = pd.Series(0.0, index=df.index)

        risk += (stress_num == 3).astype(float).fillna(0.0)
        risk += (sleep_qual_num == 1).astype(float).fillna(0.0)
        risk += (activity_num == 1).astype(float).fillna(0.0)
        
        if 'smoking_alcohol' in df.columns:
            risk += (df['smoking_alcohol'].astype(str) == 'yes').astype(float).fillna(0.0)
        if 'bmi' in df.columns:
            risk += (df['bmi'] >= 25.0).astype(float).fillna(0.0)
        if 'sleep_duration' in df.columns:
            risk += (df['sleep_duration'] < 6.0).astype(float).fillna(0.0)

        df['lifestyle_risk_score'] = risk
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
