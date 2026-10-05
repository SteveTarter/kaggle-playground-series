from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Sequence, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer

class FeatureFactory(BaseEstimator, TransformerMixin):
    """
    Clean, leak-free feature engineering pipeline for PS-S06E10: Predicting Airline Satisfaction.

    Target classes: binary classification ('satisfaction': False/True -> 0, 1)
    Raw features:
        - Continuous / Discrete: Age, Flight Distance, Departure Delay in Minutes, Arrival Delay in Minutes
        - Categorical: Gender, Customer Type, Type of Travel, Class
        - Rating Scales (0-5): Inflight wifi service, Departure/Arrival time convenient,
          Ease of Online booking, Gate location, Food and drink, Online boarding,
          Seat comfort, Inflight entertainment, On-board service, Leg room service,
          Baggage handling, Checkin service, Cleanliness
    """

    valid_strategies = [
        'encoding',
        'service_aggregations',
        'delay_features',
        'satisfaction_scores',
        'flight_ratios',
        'group_aggregations',
        'quantile_binning',
        'numeric_expansion',
        'missing_flags',
        'rating_counts'
    ]

    def __init__(
        self,
        *,
        strategies=None,
        impute_strategy: Optional[str] = None,
        seed=10301,
        target='satisfaction',
        verbose=False
    ):
        if strategies is None:
            strategies = [
                'encoding',
                'service_aggregations',
                'delay_features',
                'satisfaction_scores',
                'flight_ratios',
                'group_aggregations',
                'missing_flags',
                'rating_counts'
            ]

        self.strategies = []
        self.impute_strategy = impute_strategy
        self.seed = seed
        self.target = target
        self.verbose = verbose

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
            'Customer Type',
            'Type of Travel',
            'Class'
        ]

        self.rating_cols = [
            'Inflight wifi service',
            'Departure/Arrival time convenient',
            'Ease of Online booking',
            'Gate location',
            'Food and drink',
            'Online boarding',
            'Seat comfort',
            'Inflight entertainment',
            'On-board service',
            'Leg room service',
            'Baggage handling',
            'Checkin service',
            'Cleanliness'
        ]

        self.num_cols = [
            'Age',
            'Flight Distance',
            'Departure Delay in Minutes',
            'Arrival Delay in Minutes'
        ]

        self._is_fit: bool = False
        self.medians_: Dict[str, float] = {}
        self.modes_: Dict[str, str] = {}

    def fit(self, df: pd.DataFrame) -> 'FeatureFactory':
        if self.verbose:
            print('  -> Fitting FeatureFactory...')

        df_fit = df.copy()
        df_fit.drop('id', axis=1, inplace=True, errors='ignore')
        df_fit.drop(self.target, axis=1, inplace=True, errors='ignore')

        self.num_features_ = df_fit.select_dtypes(exclude=['object', 'bool', 'category']).columns.tolist()
        self.cat_features_ = df_fit.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

        if self.impute_strategy == 'median_mode':
            for col in self.num_features_:
                self.medians_[col] = float(df_fit[col].median(skipna=True))
            for col in self.cat_features_:
                mode_series = df_fit[col].mode(dropna=True)
                self.modes_[col] = str(mode_series.iloc[0]) if not mode_series.empty else ''

        if 'group_aggregations' in self.strategies:
            self.group_stats_ = {}
            group_keys_list = [
                ['Class'],
                ['Type of Travel'],
                ['Customer Type'],
                ['Class', 'Type of Travel']
            ]
            agg_cols = [c for c in ['Online boarding', 'Inflight wifi service', 'Seat comfort', 'Flight Distance'] if c in df_fit.columns]
            for keys in group_keys_list:
                if all(k in df_fit.columns for k in keys):
                    group_key_name = '_'.join(keys)
                    for col in agg_cols:
                        grouped = df_fit.groupby(keys, observed=False)[col]
                        stats = grouped.agg(['mean', 'std']).reset_index()
                        global_mean = float(df_fit[col].mean(skipna=True))
                        global_std = float(df_fit[col].std(skipna=True))
                        self.group_stats_[(group_key_name, col)] = {
                            'keys': keys,
                            'stats': stats,
                            'global_mean': global_mean,
                            'global_std': global_std if global_std > 0 else 1.0
                        }

        self.bin_cols_ = [c for c in ['Age', 'Flight Distance', 'Arrival Delay in Minutes'] if c in df_fit.columns]
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
        if not self._is_fit:
            raise RuntimeError('FeatureFactory must be fit() before transform().')

        if self.verbose:
            print(f'Applying FeatureFactory with strategies: {", ".join(self.strategies)}')

        df_new = df.copy()
        df_new.drop('id', axis=1, inplace=True, errors='ignore')
        if self.target in df_new.columns:
            df_new.drop(self.target, axis=1, inplace=True, errors='ignore')

        if self.impute_strategy == 'median_mode':
            if self.verbose:
                print('  -> Imputing missing values...')
            for col, val in self.medians_.items():
                if col in df_new.columns:
                    df_new[col] = df_new[col].fillna(val)
            for col, val in self.modes_.items():
                if col in df_new.columns:
                    df_new[col] = df_new[col].fillna(val)

        new_cols = {}

        if 'missing_flags' in self.strategies:
            self._add_missing_flags(df_new, new_cols)

        if 'rating_counts' in self.strategies:
            self._add_rating_counts(df_new, new_cols)

        if 'encoding' in self.strategies:
            self._add_encoding(df_new)

        if 'service_aggregations' in self.strategies:
            self._add_service_aggregations(df_new, new_cols)

        if 'delay_features' in self.strategies:
            self._add_delay_features(df_new, new_cols)

        if 'satisfaction_scores' in self.strategies:
            self._add_satisfaction_scores(df_new, new_cols)

        if 'flight_ratios' in self.strategies:
            self._add_flight_ratios(df_new, new_cols)

        if 'group_aggregations' in self.strategies:
            self._add_group_aggregations(df_new, new_cols)

        if 'quantile_binning' in self.strategies:
            self._add_quantile_binning(df_new, new_cols)

        if 'numeric_expansion' in self.strategies:
            self._add_numeric_expansion(df_new, new_cols)

        if new_cols:
            df_new = pd.concat([df_new, pd.DataFrame(new_cols, index=df_new.index)], axis=1)

        return df_new

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_strategies(self) -> list:
        return self.strategies

    # -------------------------
    # Internal Strategy Implementation
    # -------------------------
    def _add_missing_flags(self, df: pd.DataFrame, new_cols: dict):
        for col in self.num_cols + self.cat_cols:
            if col in df.columns and df[col].isnull().any():
                new_cols[f'is_missing_{col}'] = df[col].isnull().astype(np.int8)

    def _add_rating_counts(self, df: pd.DataFrame, new_cols: dict):
        available_ratings = [c for c in self.rating_cols if c in df.columns]

        if available_ratings:
            n_na = (df[available_ratings] == 0).sum(axis=1)
            n_terrible = (df[available_ratings] == 1).sum(axis=1)
            n_delight = (df[available_ratings] == 5).sum(axis=1)

            new_cols['n_services_not_applicable'] = n_na
            new_cols['n_services_terrible'] = n_terrible
            new_cols['n_services_delight'] = n_delight

            new_cols['has_any_terrible_service'] = (n_terrible > 0).astype(np.int8)
            new_cols['dealbreaker_ratio'] = n_terrible / (n_delight + 1.0)

    def _add_encoding(self, df: pd.DataFrame):
        if 'Class' in df.columns:
            df['Class'] = pd.Categorical(
                df['Class'], categories=['Eco', 'Eco Plus', 'Business'], ordered=True
            )
        if 'Customer Type' in df.columns:
            df['Customer Type'] = pd.Categorical(
                df['Customer Type'], categories=['disloyal Customer', 'Loyal Customer'], ordered=True
            )
        if 'Type of Travel' in df.columns:
            df['Type of Travel'] = pd.Categorical(
                df['Type of Travel'], categories=['Personal Travel', 'Business travel'], ordered=True
            )
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].astype('category')

    def _add_service_aggregations(self, df: pd.DataFrame, new_cols: dict):
        available_ratings = [c for c in self.rating_cols if c in df.columns]
        if available_ratings:
            new_cols['rating_mean'] = df[available_ratings].mean(axis=1)
            new_cols['rating_std'] = df[available_ratings].std(axis=1)
            new_cols['rating_min'] = df[available_ratings].min(axis=1)
            new_cols['rating_max'] = df[available_ratings].max(axis=1)
            new_cols['rating_sum'] = df[available_ratings].sum(axis=1)

            inflight_cols = [c for c in ['Inflight wifi service', 'Inflight entertainment', 'Food and drink', 'Seat comfort', 'Cleanliness'] if c in df.columns]
            if inflight_cols:
                new_cols['inflight_rating_mean'] = df[inflight_cols].mean(axis=1)

            preflight_cols = [c for c in ['Online boarding', 'Ease of Online booking', 'Checkin service', 'Gate location'] if c in df.columns]
            if preflight_cols:
                new_cols['preflight_rating_mean'] = df[preflight_cols].mean(axis=1)

            comfort_cols = [c for c in ['Seat comfort', 'Leg room service', 'Cleanliness', 'Inflight entertainment'] if c in df.columns]
            if comfort_cols:
                new_cols['comfort_rating_mean'] = df[comfort_cols].mean(axis=1)

            if inflight_cols and preflight_cols:
                new_cols['ground_vs_cabin_diff'] = new_cols['preflight_rating_mean'] - new_cols['inflight_rating_mean']

    def _add_delay_features(self, df: pd.DataFrame, new_cols: dict):
        dep_delay = df['Departure Delay in Minutes'] if 'Departure Delay in Minutes' in df.columns else 0
        arr_delay = df['Arrival Delay in Minutes'].fillna(dep_delay) if 'Arrival Delay in Minutes' in df.columns else 0

        tot = dep_delay + arr_delay
        new_cols['total_delay_minutes'] = tot
        new_cols['delay_diff_minutes'] = arr_delay - dep_delay
        new_cols['has_departure_delay'] = (dep_delay > 0).astype(np.int8)
        new_cols['has_arrival_delay'] = (arr_delay > 0).astype(np.int8)
        new_cols['has_any_delay'] = ((dep_delay > 0) | (arr_delay > 0)).astype(np.int8)

    def _add_satisfaction_scores(self, df: pd.DataFrame, new_cols: dict):
        score = pd.Series(0.0, index=df.index)

        if 'Online boarding' in df.columns:
            score += (df['Online boarding'] >= 4).astype(float)
        if 'Inflight wifi service' in df.columns:
            score += (df['Inflight wifi service'] >= 4).astype(float)
        if 'Inflight entertainment' in df.columns:
            score += (df['Inflight entertainment'] >= 4).astype(float)
        if 'Seat comfort' in df.columns:
            score += (df['Seat comfort'] >= 4).astype(float)

        if 'Inflight wifi service' in df.columns and 'Online boarding' in df.columns:
            new_cols['wifi_x_boarding_interaction'] = df['Inflight wifi service'] * df['Online boarding']

        new_cols['composite_satisfaction_score'] = score

    def _add_flight_ratios(self, df: pd.DataFrame, new_cols: dict):
        if 'Flight Distance' in df.columns:
            if 'Age' in df.columns:
                new_cols['distance_per_age'] = df['Flight Distance'] / (df['Age'] + 1e-5)
            if 'total_delay_minutes' in new_cols:
                new_cols['delay_per_distance'] = new_cols['total_delay_minutes'] / (df['Flight Distance'] + 1.0)

    def _add_group_aggregations(self, df: pd.DataFrame, new_cols: dict):
        if not hasattr(self, 'group_stats_'):
            return

        for (group_key_name, col), info in self.group_stats_.items():
            keys = info['keys']
            stats_df = info['stats']
            g_mean = info['global_mean']
            g_std = info['global_std']

            if len(keys) == 1:
                k = keys[0]
                mean_map = stats_df.set_index(k)['mean'].to_dict()
                std_map = stats_df.set_index(k)['std'].to_dict()
                mean_vals = pd.Series(df[k].astype(object).map(mean_map)).astype(float).fillna(g_mean).values
                std_vals = pd.Series(df[k].astype(object).map(std_map)).astype(float).fillna(g_std).replace(0, g_std).values
            else:
                stats_indexed = stats_df.copy()
                for k in keys:
                    stats_indexed[k] = stats_indexed[k].astype(object)
                stats_indexed = stats_indexed.set_index(keys)
                df_keys = df[keys].copy()
                for k in keys:
                    df_keys[k] = df_keys[k].astype(object)
                joined = df_keys.join(stats_indexed, on=keys, how='left')
                mean_vals = joined['mean'].astype(float).fillna(g_mean).values
                std_vals = joined['std'].astype(float).fillna(g_std).replace(0, g_std).values

            if col in df.columns:
                val = df[col].astype(float).fillna(g_mean).values
                new_cols[f'{col}_mean_by_{group_key_name}'] = mean_vals
                new_cols[f'{col}_diff_from_{group_key_name}_mean'] = val - mean_vals
                new_cols[f'{col}_ratio_to_{group_key_name}_mean'] = val / (mean_vals + 1e-5)
                new_cols[f'{col}_zscore_{group_key_name}'] = (val - mean_vals) / (std_vals + 1e-5)

    def _add_quantile_binning(self, df: pd.DataFrame, new_cols: dict):
        if hasattr(self, 'binner_') and hasattr(self, 'bin_cols_') and self.bin_cols_:
            df_in = df[self.bin_cols_].copy()
            for c in self.bin_cols_:
                med = self.medians_.get(c, float(df_in[c].median()))
                df_in[c] = df_in[c].fillna(med)
            binned = self.binner_.transform(df_in)
            for i, col in enumerate(self.bin_cols_):
                new_cols[f'{col}_binned'] = binned[:, i]

    def _add_numeric_expansion(self, df: pd.DataFrame, new_cols: dict):
        numeric_cols = [c for c in self.num_cols if c in df.columns]
        for col in numeric_cols:
            vals = df[col].astype(float)
            new_cols[f'{col}_log'] = np.log1p(np.maximum(0, vals.fillna(0)))
            new_cols[f'{col}_sqrt'] = np.sqrt(np.maximum(0, vals.fillna(0)))
            new_cols[f'{col}_sq'] = vals.fillna(0) ** 2

    def get_cat_features(self, df: pd.DataFrame) -> List[str]:
        cats = df.select_dtypes(include=['category', 'object', 'bool']).columns.tolist()
        return [c for c in cats if c != self.target and c != 'id']

    def get_num_features(self, df: pd.DataFrame) -> List[str]:
        nums = df.select_dtypes(exclude=['category', 'object', 'bool']).columns.tolist()
        return [c for c in nums if c != self.target and c != 'id']


# Alias for compatibility
FeatureEngineering = FeatureFactory
