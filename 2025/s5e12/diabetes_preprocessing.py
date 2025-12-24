# %% [code]
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

@dataclass
class _AggSpec:
    group_cols: Tuple[str, ...]
    value_cols: Tuple[str, ...]
    prefix: str

class FeatureFactory(BaseEstimator, TransformerMixin):
    """
    Centralized factory for creating new numeric and engineered features.
    
    It focuses on creating new numeric features (ratios, polynomials) and
    transforming existing numeric features (log transforms).
    """
    valid_strategies = [
        'drop_id',
        'ordinal_encoding',
        'medical_metrics',
        'clinical_indices',
        'binning',
        'clustering',
        'interactions',
        'ratios',
        'log',
        'polynomials',
        'one_hot_encoding',
        'group_aggregations',
        'cohort_deviations'
    ]

    
    def __init__(
        self, 
        *,
        strategies=None,
        seed=10301,
        target='diagnosed_diabetes',
        verbose=False,
        groupby_cols: Sequence[str] = ('gender',),
        group_agg_cols: Sequence[str] = ('bmi', 'blood_glucose', 'hbA1c_level'),
        cohort_target_cols: Sequence[str] = ('bmi', 'blood_glucose', 'hbA1c_level'),
        age_bins: int = 10,
        age_bin_strategy: str = 'quantile',  # 'quantile' or 'fixed'
        fixed_age_edges: Optional[Sequence[float]] = None,
        
    ):
        if strategies is None:
            strategies = []
        
        self.strategies = []
        self.seed = seed
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
            
        self.groupby_cols = tuple(groupby_cols)
        self.group_agg_cols = tuple(group_agg_cols)
        self.cohort_target_cols = tuple(cohort_target_cols)

        self.age_bins = int(age_bins)
        self.age_bin_strategy = age_bin_strategy
        self.fixed_age_edges = list(fixed_age_edges) if fixed_age_edges is not None else None
        
        # Select features for clustering (scale them first)
        # We use a subset of key physiological markers
        self.cluster_cols = ['bmi', 'age', 'systolic_bp', 'triglycerides', 'hdl_cholesterol']

        # Learned state (set in fit)
        self._is_fit: bool = False
        self._train_global_means: Dict[str, float] = {}
        self._group_means_map: Optional[pd.DataFrame] = None     # indexed by group keys
        self._cohort_means_map: Optional[pd.DataFrame] = None    # indexed by (gender, age_decile)
        self._age_edges: Optional[np.ndarray] = None
        self._scaler: Optional[RobustScaler] = None
        self._kmeans: Optional[RobustScaler] = None

    def fit(self, df: pd.DataFrame = None) -> 'FeatureFactory':
        df = df.copy()

        # Train-global backoffs (used when unseen groups/cohorts appear)
        for c in set(self.group_agg_cols).union(self.cohort_target_cols):
            if c in df.columns:
                self._train_global_means[c] = float(pd.to_numeric(df[c], errors='coerce').mean())

        # Fit age bins (stable across folds)
        if 'age' in df.columns and ('cohort_deviations' in self.strategies):
            if self.verbose:
                print("  -> Fitting Age Cohort Deviations...")

            age = pd.to_numeric(df['age'], errors='coerce')

            if self.age_bin_strategy == 'fixed':
                if not self.fixed_age_edges:
                    raise ValueError('fixed_age_edges must be provided when age_bin_strategy="fixed"')
                edges = np.asarray(self.fixed_age_edges, dtype=float)
            else:
                # Quantile edges learned from TRAIN only and stored.
                # Use duplicates='drop' logic by jittering slightly if needed.
                qs = np.linspace(0.0, 1.0, self.age_bins + 1)
                edges = np.nanquantile(age.to_numpy(), qs)

                # Ensure strictly increasing edges; if not, fall back to pd.qcut codes with duplicates='drop'
                # by making edges unique and monotonic.
                edges = np.unique(edges)
                if len(edges) < 2:
                    # degenerate; fall back to a simple min/max
                    mn, mx = np.nanmin(age.to_numpy()), np.nanmax(age.to_numpy())
                    edges = np.array([mn, mx], dtype=float)

            # Expand outer edges a bit so cut includes min/max
            edges[0] = edges[0] - 1e-9    
            edges[-1] = edges[-1] + 1e-9
            self._age_edges = edges

        # Fit group aggregations (TRAIN only)
        if 'group_aggregations' in self.strategies and self.groupby_cols:
            if self.verbose:
                print("  -> Fitting Group Aggregations...")

            # Compute means for each group key for requested columns
            cols = [c for c in self.group_agg_cols if c in df.columns]
            if cols:
                g = (
                    df.loc[:, list(self.groupby_cols) + cols]
                    .assign(**{c: pd.to_numeric(df[c], errors='coerce') for c in cols})
                    .groupby(list(self.groupby_cols), dropna=False)[cols]
                    .mean()
                )
                # DataFrame indexed by group keys, columns are mean values
                self._group_means_map = g

        # Fit cohort means (TRAIN only): (gender, age_decile) -> mean(target_cols)
        if 'cohort_deviations' in self.strategies and ('gender' in df.columns) and ('age' in df.columns):
            if self.verbose:
                print("  -> Fitting Gender Cohort Deviations...")

            df_tmp = df.copy()
            df_tmp['age_decile'] = self._age_to_bin_codes(df_tmp['age'])

            cols = [c for c in self.cohort_target_cols if c in df_tmp.columns]
            if cols:
                cmeans = (
                    df_tmp.loc[:, ['gender', 'age_decile'] + cols]
                    .assign(**{c: pd.to_numeric(df_tmp[c], errors='coerce') for c in cols})
                    .groupby(['gender', 'age_decile'], dropna=False)[cols]
                    .mean()
                )
                self._cohort_means_map = cmeans

        # Fit clustering using KMeans (TRAIN only)
        # Handle missing columns gracefully
        cluster_cols = [c for c in self.cluster_cols if c in df.columns]
        if 'clustering' in self.strategies and cluster_cols:
            if self.verbose:
                print("  -> Fitting K-Means Clustering...")

            self._scaler = RobustScaler()
            self._kmeans = KMeans(n_clusters=7, random_state=self.seed, n_init=10)
                
            X_cluster = self._scaler.fit_transform(df[cluster_cols])
            self._kmeans.fit(X_cluster)

        self._is_fit = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError('FeatureFactory must be fit() before transform().')

        if self.verbose:
            print(f"Applying FeatureFactory with strategies: {', '.join(self.strategies)}")
        
        df_new = df.copy()
        
        if 'drop_id' in self.strategies:
            df_new = self._drop_ids(df_new)
        
        if 'ordinal_encoding' in self.strategies:
            df_new = self._add_ordinal_encoding(df_new)
        
        if 'medical_metrics' in self.strategies:
            df_new = self._add_medical_metrics(df_new)
        
        if 'clinical_indices' in self.strategies:
            df_new = self._add_clinical_indices(df_new)

        if 'binning' in self.strategies:
            df_new = self._add_binning(df_new)
            
        if 'interactions' in self.strategies:
            df_new = self._add_interactions(df_new)
        
        if 'ratios' in self.strategies:
            df_new = self._add_ratios(df_new)
        
        if 'log' in self.strategies:
            df_new = self._add_log_transforms(df_new)
        
        if 'polynomials' in self.strategies:
            df_new = self._add_polynomials(df_new)
        
        if 'clustering' in self.strategies:
            df_new = self._add_clustering(df_new)

        if 'group_aggregations' in self.strategies:
            df_new = self._add_group_aggregations(df_new)

        if 'cohort_deviations' in self.strategies:
            df_new = self._add_cohort_deviations(df_new)

        # This strategy destroys columns, so do it last
        if 'one_hot_encoding' in self.strategies:
            df_new = self._add_one_hot_encodings(df_new)
            
        return df_new

    
    def get_strategies(self) -> []:
        return self.strategies

    # -------------------------
    # Internals
    # -------------------------

    def _age_to_bin_codes(self, age_series: pd.Series) -> pd.Series:
        """
        Convert age -> stable bin code using fitted edges.
        """
        if self._age_edges is None:
            raise RuntimeError('Age bin edges not fit. Call fit() first or disable cohort_deviations.')

        age = pd.to_numeric(age_series, errors='coerce')
        # pd.cut returns categorical; labels=False yields integer codes [0..n-1]
        codes = pd.cut(age, bins=self._age_edges, labels=False, include_lowest=True)
        return codes.astype('Int64')  # nullable integer
        
    def _drop_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        # Dropping the 'id' column
        if self.verbose:
            print("  -> Dropping the 'id' column...")

        if 'id' in df.columns:
            df = df.drop('id', axis=1)

        if self.target in df.columns:
            df = df.drop(self.target, axis=1)
            
        return df

    
    def _add_ordinal_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        # Preserves the rank order of Education and Income
        if self.verbose:
            print('  -> Adding ordinal encodings...')
        
        # Education Mapping
        # Assuming typical hierarchy: No Schooling < Elementary < Highschool < Graduate
        edu_map = {
            'No Schooling': 0,
            'Elementary': 1,
            'Highschool': 2,
            'Graduate': 3
        }
        
        # Income Mapping
        income_map = {
            'Low': 0,
            'Lower-Middle': 1,
            'Middle': 2,
            'Upper-Middle': 3,
            'High': 4
        }
        
        if 'education_level' in df.columns:
            df['education_level_ord'] = df['education_level'].map(edu_map).fillna(-1)
            
        if 'income_level' in df.columns:
            df['income_level_ord'] = df['income_level'].map(income_map).fillna(-1)
            
        return df

    
    def _add_medical_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # Adds biologically relevant cardiovascular metrics
        if self.verbose:
            print('  -> Adding medical metrics (MAP, Pulse Pressure)...')
        
        # Pulse Pressure = Systolic - Diastolic
        # Indicates arterial stiffness
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        
        # Mean Arterial Pressure (MAP)
        # MAP = Diastolic + (1/3 * Pulse Pressure)
        # Represents the average pressure in a patient's arteries during one cardiac cycle
        df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
        
        # Non-HDL Cholesterol
        # Often a better risk indicator than LDL alone
        df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
        
        return df

    
    def _add_clinical_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding advanced clinical indices (VAI, LAP proxies)...')
            
        # Visceral Adiposity Index (VAI) Proxy
        # Standard VAI uses Waist Circumference (WC). We use Waist-to-Hip Ratio (WHR) as proxy.
        # Formula Concept: (Adiposity) * (Lipid Toxicity)
        # VAI ~ (BMI * WHR) * (Triglycerides / HDL)
        # Adding epsilon to avoid div/0
        df['vai_proxy'] = (df['bmi'] * df['waist_to_hip_ratio']) * \
                          (df['triglycerides'] / (df['hdl_cholesterol'] + 1e-5))
        
        # Lipid Accumulation Product (LAP) Proxy
        # LAP indicates lipid overaccumulation.
        # Standard: (WC - 65) * TG. We substitute WC with BMI * WHR * constant scaling
        df['lap_proxy'] = (df['bmi'] * df['waist_to_hip_ratio']) * df['triglycerides']
        
        return df

        
    def _add_group_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds {col}_mean_by_{group} and {col}_dev_from_{group} using TRAIN-fitted group means.
        Does NOT recompute means on df.
        """
        if self.verbose:
            print('  -> Adding group aggregations...')
        
        if self._group_means_map is None:
            return df

        cols = [c for c in self.group_agg_cols if c in df.columns]
        if not cols:
            return df

        # Merge group means onto df
        # _group_means_map is indexed by group keys; reset_index() to merge.
        means_df = self._group_means_map.reset_index()
        means_df = means_df.rename(columns={c: f'{c}_mean_by_group' for c in cols})

        # If transform() might be called multiple times on same df,
        # drop existing mean columns to avoid _x/_y suffixes.
        mean_cols = [f'{c}_mean_by_group' for c in cols]
        df = df.drop(columns=[c for c in mean_cols if c in df.columns], errors='ignore')

        df = df.merge(means_df, on=list(self.groupby_cols), how='left')

        for c in cols:
            mcol = f'{c}_mean_by_group'
            backoff = self._train_global_means.get(c, np.nan)
            df[mcol] = df[mcol].fillna(backoff)
            df[f'{c}_dev_from_group'] = pd.to_numeric(df[c], errors='coerce') - df[mcol]

        return df

    def _add_cohort_deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds stable age_decile + {col}_cohort_mean + {col}_dev_from_cohort
        where cohort is (gender, age_decile) and means are TRAIN-fitted.
        Does NOT recompute cohort means on df.
        """
        if self._cohort_means_map is None:
            return df

        if 'gender' not in df.columns or 'age' not in df.columns:
            return df
            
        if self.verbose:
            print('  -> Adding cohort deviations...')
            print(f'Cols before:{df.columns}')

        cols = [c for c in self.cohort_target_cols if c in df.columns]
        if not cols:
            return df

        df = df.copy()
        df['age_decile'] = self._age_to_bin_codes(df['age'])

        cohort_df = self._cohort_means_map.reset_index()
        cohort_df = cohort_df.rename(columns={c: f'{c}_cohort_mean' for c in cols})

        df = df.merge(cohort_df, on=['gender', 'age_decile'], how='left')

        for c in cols:
            mcol = f'{c}_cohort_mean'
            backoff = self._train_global_means.get(c, np.nan)
            df[mcol] = df[mcol].fillna(backoff)
            df[f'{c}_dev_from_cohort'] = pd.to_numeric(df[c], errors='coerce') - df[mcol]

        if self.verbose:
            print(f'Cols after:{df.columns}')
        
        return df
        
    def _add_binning(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding medical binning (BMI Class, BP Class)...')
            
        # BMI Classes (WHO Standards)
        # 0: Underweight (<18.5), 1: Normal, 2: Overweight, 3: Obese (>=30)
        df['bmi_class'] = pd.cut(
            df['bmi'], 
            bins=[-1, 18.5, 24.9, 29.9, 100], 
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Blood Pressure Classes (AHA Standards - simplified)
        # 0: Normal (<120), 1: Elevated, 2: Hypertension Stage 1, 3: Stage 2 (>=140)
        df['bp_class'] = pd.cut(
            df['systolic_bp'],
            bins=[-1, 120, 129, 139, 300],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        return df

        
    def _add_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding unsupervised clusters...')
        
        # We must use the SAME features used in fit()
        cluster_cols = [c for c in self.cluster_cols if c in df.columns]
        
        if hasattr(self, '_kmeans') and cluster_cols:
            df_cluster = self._scaler.transform(df[cluster_cols])
            df['cluster_label'] = self._kmeans.predict(df_cluster)
        
        return df

        
    def _add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        # Adds interaction terms between features
        if self.verbose:
            print('  -> Adding interaction features...')
        
        # Age * BMI Interaction
        # The metabolic impact of BMI often worsens with age
        df['age_bmi_interaction'] = df['age'] * df['bmi']
        
        # Sedentary Ratio
        # Ratio of screen time (hrs) to physical activity (mins).
        # Note: Units differ, but the relative ratio remains a valid index.
        # Adding 1 to denominator to avoid division by zero.
        df['sedentary_ratio'] = df['screen_time_hours_per_day'] / (df['physical_activity_minutes_per_week'] + 1)
        
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
            print(f'Cols before:{df.columns}')
        
        # Identify Feature Types
        # We exclude the target from the features list
        features = [c for c in df.columns if c != self.target]
        
        # Automatically select categorical columns for encoding
        # (gender, ethnicity, education_level, income_level, smoking_status, employment_status)
        cat_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
        if self.verbose:
            print(f'Categorical features: {cat_features}')
        
        
        df = pd.get_dummies(df, columns=cat_features, prefix_sep='_', dtype=int)
        
        if self.verbose:
            print(f'Cols after:{df.columns}')
        
        return df;
