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
        'age_benchmarks',
        'behavioral_indices',
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
        target : str, default='addicted_label'
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
        """
        Fits the FeatureFactory by discovering feature dtypes and computing
        imputation parameters (medians/modes) if configured.

        Parameters
        ----------
        df : pd.DataFrame
            The input training DataFrame.

        Returns
        -------
        self : FeatureFactory
            The fitted transformer instance.
        """
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
        """
        Transforms the input DataFrame by sequentially applying configured
        feature engineering strategies.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            DataFrame augmented with engineered features.
        """
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

        if 'age_benchmarks' in self.strategies:
            df_new = self._add_age_benchmarks(df_new)

        if 'behavioral_indices' in self.strategies:
            df_new = self._add_behavioral_indices(df_new)

        if 'numeric_expansion' in self.strategies:
            df_new = self._add_numeric_expansion(df_new)

        return df_new

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits to data, then transforms it.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        return self.fit(df).transform(df)

    def get_strategies(self) -> list:
        """
        Returns the list of active strategies configured for this transformer instance.

        Returns
        -------
        list of str
            Active strategy names.
        """
        return self.strategies

    # -------------------------
    # Internals
    # -------------------------
    def _add_missing_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds binary indicator variables (0/1) for columns containing missing values.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with binary `is_missing_<col>` flags appended.
        """
        if self.verbose:
            print('  -> Adding missing flags...')
        for col in self.num_cols + self.cat_cols:
            if col in df.columns:
                df[f'is_missing_{col}'] = df[col].isnull().astype(np.int8)
        return df

    def _add_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical variables into ordered pandas Categorical types or nominal dtypes.

        Categorical mappings:
        - `stress_level`: Ordinal mapping ('Low' < 'Medium' < 'High')
        - `academic_work_impact`: Ordinal mapping ('No' < 'Yes')
        - `gender`: Nominal category cast

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with encoded categorical columns.
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
        Calculates screen time breakdown fractions and intensity ratios.

        Engineered features:
        - `social_media_fraction`: Proportion of daily screen time spent on social media.
        - `gaming_fraction`: Proportion of daily screen time spent on gaming.
        - `total_social_gaming_hours`: Combined social media + gaming hours.
        - `non_social_gaming_hours`: Remaining screen time outside social/gaming.
        - `weekend_to_weekday_ratio`: Weekend screen time vs daily screen time.
        - `app_opens_per_hour`: App opens per screen hour.
        - `notifications_per_hour`: Notifications received per screen hour.
        - `screen_time_per_age`, `app_opens_per_age`, `notifications_per_age`: Age-scaled metrics.
        - `notifications_per_app_open`, `app_opens_to_notifications_ratio`: Inter-notification dynamics.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with screen time ratio features added.
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
        Calculates sleep deficit, stress interactions, and sleep-to-screen ratios.

        Engineered features:
        - `sleep_shortage`: Deficit below 8.0 hours of recommended sleep.
        - `stress_to_sleep_ratio`: Ratio of numerical stress level (1=Low, 2=Medium, 3=High) to sleep hours.
        - `screen_time_to_sleep_ratio`: Daily screen time relative to sleep duration.
        - `social_media_to_sleep_ratio`: Social media time relative to sleep duration.
        - `gaming_to_sleep_ratio`: Gaming time relative to sleep duration.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with sleep/stress features added.
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
        Constructs an additive heuristic lifestyle risk score based on unhealthy indicators:
        - Screen time > 6.0 hours (+1.0)
        - High stress level (+1.0)
        - Academic/work impact reported (+1.0)
        - Sleep hours < 6.0 (+1.0)
        - App opens > 100 per day (+1.0)

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with `addiction_risk_score` added.
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

    def _add_age_benchmarks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives peer-norm benchmark features relative to age cohorts.

        Behavioral metrics (screen time, sleep, app opens) vary significantly across
        age cohorts. Calculating deviations from age cohort averages isolates anomalous
        behavior relative to peers.

        Engineered features:
        - `<col>_diff_from_age_mean`: Individual value minus age cohort mean.
        - `<col>_ratio_to_age_mean`: Individual value divided by age cohort mean.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with age benchmark features added.
        """
        if self.verbose:
            print('  -> Adding age benchmark features...')

        if 'age' in df.columns:
            # Create age groups (5-year / life stage cohorts)
            age_bins = pd.cut(
                df['age'],
                bins=[0, 18, 25, 35, 50, 100],
                labels=['<18', '18-24', '25-34', '35-49', '50+'],
                include_lowest=True
            )
            
            for col in ['daily_screen_time_hours', 'sleep_hours', 'social_media_hours', 'app_opens_per_day']:
                if col in df.columns:
                    group_means = df.groupby(age_bins, observed=False)[col].transform('mean')
                    df[f'{col}_diff_from_age_mean'] = df[col] - group_means
                    df[f'{col}_ratio_to_age_mean'] = df[col] / (group_means + 1e-5)

        return df

    def _add_behavioral_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates high-level behavioral indices capturing compulsivity, productivity loss,
        and lifestyle balance.

        Engineered features:
        - `compulsive_checking_frequency`: App opens divided by daily screen time hours.
        - `unproductive_leisure_ratio`: Leisure screen time (social media + gaming) divided by work/study hours.
        - `non_screen_waking_hours`: Hours spent awake without screen exposure.
        - `screen_to_waking_ratio`: Screen time as a fraction of total waking hours.
        - `screen_sleep_total_hours`: Combined hours spent sleeping or using screens in a 24-hour day.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with behavioral index features added.
        """
        if self.verbose:
            print('  -> Adding behavioral indices...')

        # Compulsive checking frequency: opens per active screen hour
        if 'app_opens_per_day' in df.columns and 'daily_screen_time_hours' in df.columns:
            df['compulsive_checking_frequency'] = df['app_opens_per_day'] / (df['daily_screen_time_hours'] + 1e-5)

        # Unproductive leisure ratio: (social_media + gaming) vs work/study
        if all(c in df.columns for c in ['social_media_hours', 'gaming_hours', 'work_study_hours']):
            leisure_hours = df['social_media_hours'] + df['gaming_hours']
            df['unproductive_leisure_ratio'] = leisure_hours / (df['work_study_hours'] + 1e-5)

        # Non-screen waking hours: time spent awake without screen exposure
        if 'sleep_hours' in df.columns and 'daily_screen_time_hours' in df.columns:
            waking_hours = (24.0 - df['sleep_hours']).clip(lower=0)
            df['non_screen_waking_hours'] = (waking_hours - df['daily_screen_time_hours']).clip(lower=0)
            df['screen_to_waking_ratio'] = df['daily_screen_time_hours'] / (waking_hours + 1e-5)

        # Screen + sleep balance across total 24 hours
        if 'sleep_hours' in df.columns and 'daily_screen_time_hours' in df.columns:
            df['screen_sleep_total_hours'] = df['daily_screen_time_hours'] + df['sleep_hours']

        return df

    def _add_numeric_expansion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates mathematical transformations (Log, Square, Square Root) for numerical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with numeric expansion features added.
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

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        List[str]
            Deduplicated list of categorical column names.
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
