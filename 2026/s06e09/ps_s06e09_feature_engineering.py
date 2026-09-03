from __future__ import annotations
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


class FeatureFactory(BaseEstimator, TransformerMixin):
    """
    Feature engineering pipeline for PS-S06E09: Electric Vehicle Purchases.

    Target classes: binary classification ('No', 'Yes' -> 0, 1)
    Raw features: Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned,
                  Charging_Stations_Near_Home, Charging_Stations_Near_Work,
                  Environmental_Concern_Level, Gender, City_Type, Current_Car_Type,
                  Home_Charging_Possible, Subsidy_Available, Range_Anxiety_Level
    """

    valid_strategies = [
        'encoding',
        'quantile_binning',
        'ev_ratios',
        'charging_accessibility',
        'financial_incentives',
        'range_anxiety_interactions',
        'buyer_readiness_score',
        'income_age_benchmarks',
        'missing_flags',
        'numeric_expansion',
    ]

    def __init__(
        self,
        *,
        strategies=None,
        impute_strategy: Optional[str] = None,
        seed=10301,
        target='Will_Buy_EV',
        verbose=False
    ):
        """
        Initializes the FeatureFactory pipeline with selected strategies.

        Parameters
        ----------
        strategies : list of str, optional
            List of feature engineering strategy names to execute during transform.
        impute_strategy : str, optional
            Imputation strategy to apply ('median_mode' or None).
        seed : int, default=10301
            Random state seed for reproducibility.
        target : str, default='Will_Buy_EV'
            Name of the target column in the dataset.
        verbose : bool, default=False
            If True, prints progress messages during execution.
        """
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
            'Gender',
            'City_Type',
            'Current_Car_Type',
            'Home_Charging_Possible',
            'Subsidy_Available',
            'Range_Anxiety_Level'
        ]

        self.num_cols = [
            'Age',
            'Annual_Income_USD',
            'Daily_Commute_km',
            'Number_of_Cars_Owned',
            'Charging_Stations_Near_Home',
            'Charging_Stations_Near_Work',
            'Environmental_Concern_Level'
        ]

        self._is_fit: bool = False
        self.medians_: Dict[str, float] = {}
        self.modes_: Dict[str, str] = {}

    # ----------------------------
    # Public API
    # ----------------------------
    def fit(self, df: pd.DataFrame) -> 'FeatureFactory':
        """
        Fits the FeatureFactory by discovering feature dtypes and computing
        imputation parameters (medians/modes) if configured.
        """
        if self.verbose:
            print('  -> Fitting DataFrame...')
            
        df_fit = df.copy()
        # Always drop id and target
        df_fit.drop('id', axis=1, inplace=True, errors='ignore')
        df_fit.drop('Buyer_ID', axis=1, inplace=True, errors='ignore')
        df_fit.drop(self.target, axis=1, inplace=True, errors='ignore')
        
        self.num_features_ = df_fit.select_dtypes(exclude=['object', 'bool', 'category']).columns.tolist()
        self.cat_features_ = df_fit.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

        # Learn imputation parameters if requested
        if self.impute_strategy == 'median_mode':
            for col in self.num_features_:
                self.medians_[col] = float(df_fit[col].median(skipna=True))
            for col in self.cat_features_:
                mode_series = df_fit[col].mode(dropna=True)
                self.modes_[col] = str(mode_series.iloc[0]) if not mode_series.empty else ''

        # Fit quantile binner if requested
        self.bin_cols_ = [c for c in ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level'] if c in df_fit.columns]
        if 'quantile_binning' in self.strategies and self.bin_cols_:
            self.binner_ = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile', subsample=None)
            df_binner_in = df_fit[self.bin_cols_].copy()
            for c in self.bin_cols_:
                med = self.medians_.get(c, float(df_binner_in[c].median()))
                df_binner_in[c] = df_binner_in[c].fillna(med)
            self.binner_.fit(df_binner_in)

        self._is_fit = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by sequentially applying configured
        feature engineering strategies.
        """
        if not self._is_fit:
            raise RuntimeError('FeatureFactory must be fit() before transform().')

        if self.verbose:
            print(f'Applying FeatureFactory with strategies: {", ".join(self.strategies)}')

        df_new = df.copy()

        # Always drop id, Buyer_ID and target
        df_new.drop('id', axis=1, inplace=True, errors='ignore')
        df_new.drop('Buyer_ID', axis=1, inplace=True, errors='ignore')
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

        if 'ev_ratios' in self.strategies:
            df_new = self._add_ev_ratios(df_new)

        if 'charging_accessibility' in self.strategies:
            df_new = self._add_charging_accessibility(df_new)

        if 'financial_incentives' in self.strategies:
            df_new = self._add_financial_incentives(df_new)

        if 'range_anxiety_interactions' in self.strategies:
            df_new = self._add_range_anxiety_interactions(df_new)

        if 'buyer_readiness_score' in self.strategies:
            df_new = self._add_buyer_readiness_score(df_new)

        if 'quantile_binning' in self.strategies:
            df_new = self._add_quantile_binning(df_new)

        if 'income_age_benchmarks' in self.strategies:
            df_new = self._add_income_age_benchmarks(df_new)

        if 'numeric_expansion' in self.strategies:
            df_new = self._add_numeric_expansion(df_new)

        return df_new

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits to data, then transforms it.
        """
        return self.fit(df).transform(df)

    def get_strategies(self) -> list:
        """
        Returns active strategies.
        """
        return self.strategies

    # -------------------------
    # Internals
    # -------------------------
    def _add_missing_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding missing flags...')
        for col in self.num_cols + self.cat_cols:
            if col in df.columns:
                df[f'is_missing_{col}'] = df[col].isnull().astype(np.int8)
        return df

    def _add_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Encoding categorical variables...')

        # Ordinal Range_Anxiety_Level: Low < Medium < High
        if 'Range_Anxiety_Level' in df.columns:
            df['Range_Anxiety_Level'] = pd.Categorical(
                df['Range_Anxiety_Level'], categories=['Low', 'Medium', 'High'], ordered=True
            )

        # Ordinal Home_Charging_Possible: No < Yes
        if 'Home_Charging_Possible' in df.columns:
            df['Home_Charging_Possible'] = pd.Categorical(
                df['Home_Charging_Possible'], categories=['No', 'Yes'], ordered=True
            )

        # Ordinal Subsidy_Available: No < Yes
        if 'Subsidy_Available' in df.columns:
            df['Subsidy_Available'] = pd.Categorical(
                df['Subsidy_Available'], categories=['No', 'Yes'], ordered=True
            )

        # Nominal features cast to category
        for col in ['Gender', 'City_Type', 'Current_Car_Type']:
            if col in df.columns:
                df[col] = df[col].astype('category')

        return df

    def _add_ev_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding EV ratio features...')

        if 'Charging_Stations_Near_Home' in df.columns and 'Charging_Stations_Near_Work' in df.columns:
            df['total_charging_stations'] = df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
            df['home_work_station_ratio'] = df['Charging_Stations_Near_Home'] / (df['Charging_Stations_Near_Work'] + 1e-5)

        if 'Annual_Income_USD' in df.columns:
            if 'Number_of_Cars_Owned' in df.columns:
                df['income_per_car'] = df['Annual_Income_USD'] / (df['Number_of_Cars_Owned'] + 1.0)
            if 'Daily_Commute_km' in df.columns:
                df['income_per_commute_km'] = df['Annual_Income_USD'] / (df['Daily_Commute_km'] + 1e-5)
            if 'Age' in df.columns:
                df['income_per_age'] = df['Annual_Income_USD'] / (df['Age'] + 1e-5)

        if 'Daily_Commute_km' in df.columns:
            if 'Age' in df.columns:
                df['commute_per_age'] = df['Daily_Commute_km'] / (df['Age'] + 1e-5)
            if 'total_charging_stations' in df.columns:
                df['commute_per_charging_station'] = df['Daily_Commute_km'] / (df['total_charging_stations'] + 1e-5)

        return df

    def _add_charging_accessibility(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding charging accessibility features...')

        home_charging_num = df['Home_Charging_Possible'].astype(str).map({'Yes': 1, 'No': 0}).fillna(0)
        
        stations_home = df['Charging_Stations_Near_Home'] if 'Charging_Stations_Near_Home' in df.columns else 0
        stations_work = df['Charging_Stations_Near_Work'] if 'Charging_Stations_Near_Work' in df.columns else 0
        
        df['charging_accessibility_score'] = (home_charging_num * 3.0) + (stations_home * 1.5) + (stations_work * 1.0)
        df['has_any_charging_option'] = ((home_charging_num > 0) | (stations_home > 0) | (stations_work > 0)).astype(int)

        return df

    def _add_financial_incentives(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding financial incentive features...')

        subsidy_num = df['Subsidy_Available'].astype(str).map({'Yes': 1, 'No': 0}).fillna(0)

        if 'Annual_Income_USD' in df.columns:
            df['affordability_with_subsidy'] = df['Annual_Income_USD'] * subsidy_num
            median_income = float(self.medians_.get('Annual_Income_USD', df['Annual_Income_USD'].median()))
            df['high_income_buyer'] = (df['Annual_Income_USD'] > median_income).astype(int)

        return df

    def _add_range_anxiety_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding range anxiety interaction features...')

        anxiety_num = df['Range_Anxiety_Level'].astype(str).map({'Low': 1, 'Medium': 2, 'High': 3}).fillna(2)
        home_charging_num = df['Home_Charging_Possible'].astype(str).map({'Yes': 1, 'No': 0}).fillna(0)

        if 'Daily_Commute_km' in df.columns:
            df['anxiety_x_commute'] = anxiety_num * df['Daily_Commute_km']

        if 'total_charging_stations' in df.columns:
            df['anxiety_per_station'] = anxiety_num / (df['total_charging_stations'] + 1e-5)

        df['anxiety_without_home_charging'] = anxiety_num * (1 - home_charging_num)

        return df

    def _add_buyer_readiness_score(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding buyer readiness score features...')

        readiness = pd.Series(0.0, index=df.index)

        if 'Environmental_Concern_Level' in df.columns:
            readiness += (df['Environmental_Concern_Level'] >= 4.0).astype(float).fillna(0.0)
        if 'Subsidy_Available' in df.columns:
            readiness += (df['Subsidy_Available'].astype(str) == 'Yes').astype(float).fillna(0.0)
        if 'Range_Anxiety_Level' in df.columns:
            readiness += (df['Range_Anxiety_Level'].astype(str) == 'Low').astype(float).fillna(0.0)
        if 'Home_Charging_Possible' in df.columns:
            readiness += (df['Home_Charging_Possible'].astype(str) == 'Yes').astype(float).fillna(0.0)
        if 'Annual_Income_USD' in df.columns:
            readiness += (df['Annual_Income_USD'] > 60000.0).astype(float).fillna(0.0)

        df['ev_buyer_readiness_score'] = readiness
        return df

    def _add_income_age_benchmarks(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding income and age benchmark features...')

        if 'Age' in df.columns:
            age_bins = pd.cut(df['Age'].fillna(df['Age'].median()), bins=[0, 25, 35, 50, 65, 100], labels=[1, 2, 3, 4, 5])
            
            if 'Annual_Income_USD' in df.columns:
                income_mean = df.groupby(age_bins, observed=False)['Annual_Income_USD'].transform('mean')
                df['income_vs_age_peer_diff'] = df['Annual_Income_USD'] - income_mean

            if 'Daily_Commute_km' in df.columns:
                commute_mean = df.groupby(age_bins, observed=False)['Daily_Commute_km'].transform('mean')
                df['commute_vs_age_peer_diff'] = df['Daily_Commute_km'] - commute_mean

        return df

    def _add_numeric_expansion(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Expanding numeric features...')

        numeric_cols = [c for c in self.num_cols if c in df.columns]
        for col in numeric_cols:
            vals = df[col].astype(float)
            df[f'{col}_log'] = np.log1p(np.maximum(0, vals))
            df[f'{col}_sqrt'] = np.sqrt(np.maximum(0, vals))
            df[f'{col}_sq'] = vals ** 2

        return df

    def get_cat_features(self, df: pd.DataFrame) -> List[str]:
        cats = df.select_dtypes(include=['category', 'object', 'bool']).columns.tolist()
        return [c for c in cats if c != self.target and c != 'id' and c != 'Buyer_ID']

    def get_num_features(self, df: pd.DataFrame) -> List[str]:
        nums = df.select_dtypes(exclude=['category', 'object', 'bool']).columns.tolist()
        return [c for c in nums if c != self.target and c != 'id' and c != 'Buyer_ID']

    def _add_quantile_binning(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding quantile-binned features...')

        if hasattr(self, 'binner_') and hasattr(self, 'bin_cols_') and self.bin_cols_:
            df_in = df[self.bin_cols_].copy()
            for c in self.bin_cols_:
                med = self.medians_.get(c, float(df_in[c].median()))
                df_in[c] = df_in[c].fillna(med)
            binned = self.binner_.transform(df_in)
            for i, col in enumerate(self.bin_cols_):
                df[f'{col}_binned'] = binned[:, i]

        return df
