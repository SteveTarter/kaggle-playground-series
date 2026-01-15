# %% [code]
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
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
    Feature engineering with optional:
      - binning of key numeric columns
      - interaction features (including stable study_method dummy interactions)
      - clustering features via StandardScaler + KMeans

    Critical invariants:
      - Any columns created from categoricals must be stable between fit/transform.
      - Any "learned" transformations (scaler/kmeans) must be fit only on training data.
    """
    valid_strategies = [
        'drop_id',
        'ordinal_encoding',
        'binning',
        'interactions',
        'clustering',
        'polynomials',
        'log',
        'manual_formula'
    ]
    
    def __init__(
        self, 
        *,
        strategies=None,
        seed=10301,
        target='exam_score',
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
        self._kmeans: Optional[KMeans] = None

        # Learned stable dummy columns for study_method
        self._study_method_dummy_cols: Optional[List[str]] = None

        self._cluster_cols_fit_ = None
        self._cluster_fill_values_ = None

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

        # Column lists
        self.base_cols = [
            'study_hours',
            'sleep_hours',
            'class_attendance',
            'exam_score',
        ]
        self.cluster_cols = [
            'study_hours',
            'sleep_hours',
            'class_attendance',
        ]
        
        # If interactions are active, add 'restoration_index' to clustering columns.
        if 'interactions' in self.strategies:
            self.cluster_cols.append('restoration_index')
            
        # Learned state (set in fit)
        self._is_fit: bool = False

    
    # ----------------------------
    # Public API
    # ----------------------------
    def fit(self, df: pd.DataFrame = None) -> 'FeatureFactory':
        if self.verbose:
            print("  -> Fitting DataFrame...")
            
        df_new = df.copy()

        if 'binning' in self.strategies:
            df_new = self._add_binning(df_new)
            
        if 'interactions' in self.strategies:
            # This is pure feature construction EXCEPT the stable dummy column set;
            # we learn the dummy column set here.
            df_new = self._add_interactions_fit(df_new)
            
        if 'clustering' in self.strategies:
            self._add_clustering_fit(df_new)

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
            
        if 'binning' in self.strategies:
            df_new = self._add_binning(df_new)

        if 'log' in self.strategies:
            df_new = self._add_log(df_new)

        if 'polynomials' in self.strategies:
            df_new = self._add_polynomials(df_new)
            
        if 'interactions' in self.strategies:
            df_new = self._add_interactions_transform(df_new)
        
        if 'clustering' in self.strategies:
            df_new = self._add_clustering_transform(df_new)

        if 'manual_formula' in self.strategies:
            df_new = self._add_manual_formula(df_new)
            
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
        return cat.cat.codes.astype("int16")
        
        
    @staticmethod
    def _normalize_dummy_name(col: str) -> str:
        """
        Normalize dummy column names so they are stable and model-safe.
        Example:
            'effort_self-study' -> 'effort_self_study'
            'effort_group study' -> 'effort_group_study'
        """
        return (
            col.strip()
               .replace(" ", "_")
               .replace("-", "_")
        )

    
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
        
        # Sleep Quality
        # Assuming typical hierarchy: Poor < Average < Good
        sleep_quality_map = {
            'poor': 0,
            'average': 1,
            'good': 2
        }
        
        # Facility Rating
        facility_rating_map = {
            'low': 0,
            'medium': 1,
            'high': 2
        }

        # Exam Difficulty
        exam_difficulty_map = {
            'easy': 0,
            'moderate': 1,
            'hard': 2
        }
        
        if 'sleep_quality' in df.columns:
            df['sleep_quality_ord'] = df['sleep_quality'].map(sleep_quality_map).fillna(-1)
            
        if 'facility_rating' in df.columns:
            df['facility_rating_ord'] = df['facility_rating'].map(facility_rating_map).fillna(-1)
            
        if 'exam_difficulty' in df.columns:
            df['exam_difficulty_ord'] = df['exam_difficulty'].map(exam_difficulty_map).fillna(-1)
            
        return df
        
    def _add_binning(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding binning (Class Attendance, Sleep Hours, Study Hours)...')
            
        # Class Attendance
        # In many universities, dropping below 75% attendance triggers an automatic failure or exclusion from exams.
        # 0: Critical (<60%), 1: Warning (60-80), 2: Safe (>= 80%)
        if 'class_attendance' in df.columns:
            df['class_attendance_class'] = self._safe_cut_codes(
                df['class_attendance'], 
                bins=[-1, 60.0, 80.0, 100.1], 
                labels=[0, 1, 2]
            ).astype(int)
        
        # Sleep Hours Classes
        # Sleep usually has a U-shaped relationship with performance.
        # 0: Deprived (< 6 hours), 1: Optimal (6-9 hours), 2: Excess (> 9 hours)
        if 'sleep_hours' in df.columns:
            df['sleep_hours_class'] = self._safe_cut_codes(
                df['sleep_hours'],
                bins=[-1, 6.0, 9.0, 24.0],
                labels=[0, 1, 2]
            ).astype(int)
        
        # Study Hours Classes
        # Diminishing Returns.  Why bin it? The benefit of studying is not infinite.
        # 0: Low (< 2 hours), 1: Target (2-6 hours), 2: Burnout (> 6 hours)
        if 'study_hours' in df.columns:
            df['study_hours_class'] = self._safe_cut_codes(
                df['study_hours'],
                bins=[-1, 2.0, 6.0, 100.0],
                labels=[0, 1, 2]
            ).astype(int)
        
        return df


    def _add_log(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding log of selected features...')

        if 'age' in df.columns:
            df['log_age'] = np.log1p(df['age'])

        if 'study_hours' in df.columns:
            df['log_study_hours'] = np.log1p(df['study_hours'])

        if 'class_attendance' in df.columns:
            df['log_class_attendance'] = np.log1p(df['class_attendance'])

        if 'sleep_hours' in df.columns:
            df['log_sleep_hours'] = np.log1p(df['sleep_hours'])

        return df

        
    def _add_polynomials(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding polynomials of selected features...')

        if 'age' in df.columns:
            df['age_sq'] = df['age'] * df['age']

        if 'study_hours' in df.columns:
            df['study_hours_sq'] = df['study_hours'] * df['study_hours']

        if 'class_attendance' in df.columns:
            df['class_attendance_sq'] = df['class_attendance'] * df['class_attendance']

        if 'sleep_hours' in df.columns:
            df['sleep_hours_sq'] = df['sleep_hours'] * df['sleep_hours']

        return df


    def _add_manual_formula(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This feature was discussed here: https://www.kaggle.com/competitions/playground-series-s6e1/discussion/666695
        This appears to be a powerful feature.
        """
        if self.verbose:
            print('  -> Adding manual formula of selected features...')

        # Return without doing anything if all of the necessary features aren't present
        needed_cols = ['sleep_quality', 'facility_rating', 'study_method', 'study_hours', 'class_attendance']
        for col in needed_cols:
            if col not in df.columns:
                return df
        
        LUT = {
            'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
            'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
            'study_method': {
                'coaching': 10,
                'mixed': 5,
                'group study': 2,
                'online videos': 1,
                'self-study': 0
            }
        }

        df['manual_formula'] = 6.0 * df['study_hours'] + 0.35 * df['class_attendance'] \
                + 1.5 * df['sleep_hours'] \
                + df['sleep_quality'].map(LUT['sleep_quality']) \
                + df['study_method'].map(LUT['study_method']) \
                + df['facility_rating'].map(LUT['facility_rating'])

        return df

        
    def _add_interactions_core(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Core interactions (purely derived, no learned state),
        but must be NaN-safe because clustering/scaling may use these values.
        """
        eps = 1e-6

        # Ratio features
        if 'study_hours' in df.columns:
            df['high_study'] = (df['study_hours'] >= 7).astype(int)
            
            if 'sleep_hours' in df.columns:
                df['study_to_sleep_ratio'] = df['study_hours'] / (df['sleep_hours'] + eps)
                
            if 'class_attendance' in df.columns:
                df['attendance_to_study_ratio'] = df['class_attendance'] / (df['study_hours'] + eps)
                
                # Engagement Multiplier (Num x Num)
                # Logic: High attendance AND High study hours suggests a top performer.
                # We use multiplication to highlight the compound effect.
                df['total_engagement'] = df['study_hours'] * df['class_attendance']
                
                # "Autodidact Ratio". High study hours but LOW attendance.
                # We add a small epsilon (1e-5) to avoid division by zero.
                df['autodidact_ratio'] = df['study_hours'] / (df['class_attendance'] + 1e-5)

            if 'facility_rating' in df.columns:
                # Resource Efficiency (Num x Ordinal)
                # Logic: Facility rating acts as a friction coefficient.
                # We map facility to a scale. If facility is 'Low' (1), study hours might count for less.
                facility_map = {
                    'low': 0.8, 'medium': 1.0, 'high': 1.2
                }
                df['facility_weight'] = df.get('facility_rating').map(facility_map).astype("float32")
                df['facility_weight'] = df['facility_weight'].fillna(1.0)
                df['weighted_study_hours'] = df['study_hours'].astype("float32") * df['facility_weight']

        if 'sleep_quality' in df.columns and 'sleep_hours' in df.columns:
            # Restoration Index (Num x Ordinal)
            # Logic: Sleep duration needs to be qualified by sleep quality.
            # We map the categorical quality to a numeric scale (Ordinal Encoding)
            # and multiply to get a "Total Rest" volume.
            sleep_map = {
                'poor': 1, 'average': 2, 'good': 3
            }
            df['sleep_quality_num'] = df.get('sleep_quality').map(sleep_map).astype("float32")
            df['sleep_quality_num'] = df['sleep_quality_num'].fillna(0.0)
            df['restoration_index'] = df['sleep_hours'].astype("float32") * df['sleep_quality_num']

        if 'study_hours' in df.columns and 'class_attendance' in df.columns:
            df['study_att'] = df['study_hours'] * df['class_attendance']
            
        # Cleanup: Drop intermediate columns
        drop_cols = ['sleep_quality_num', 'facility_weight'] 
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        return df

    
    def _add_interactions_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit-time version:
          - compute numeric interactions
          - learn the stable set of dummy columns for study_method
        """
        df = self._add_interactions_core(df)

        if 'study_method' in df.columns and 'study_hours' in df.columns:
            raw_dummies = pd.get_dummies(
                df.get('study_method'),
                prefix='effort',
                dummy_na=False
            )
        
            # Normalize dummy column names ONCE and store them
            rename_map = {
                col: self._normalize_dummy_name(col)
                for col in raw_dummies.columns
            }
        
            dummies = raw_dummies.rename(columns=rename_map)
        
            # Persist the normalized column list (stable feature space)
            self._study_method_dummy_cols = list(dummies.columns)
        
            # Create interaction features
            for col in self._study_method_dummy_cols:
                df[f"{col}_hours"] = (
                    dummies[col].astype("float32") *
                    df['study_hours'].astype("float32")
                )
    
        return df

    
    def _add_interactions_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform-time version:
          - compute numeric interactions
          - create the SAME dummy columns learned in fit(), in the SAME order
        """
        if self._is_fit is False:
            raise RuntimeError("FeatureFactory.transform called before fit (study_method dummy columns not learned).")

        df = self._add_interactions_core(df)

        if self._study_method_dummy_cols is None:
            return df

        if 'study_method' in df.columns and 'study_hours' in df.columns:
            raw_dummies = pd.get_dummies(
                df.get('study_method'),
                prefix='effort',
                dummy_na=False
            )
        
            # Apply the SAME normalization
            raw_dummies = raw_dummies.rename(
                columns={col: self._normalize_dummy_name(col) for col in raw_dummies.columns}
            )
        
            # Enforce stable column set and order
            dummies = raw_dummies.reindex(
                columns=self._study_method_dummy_cols,
                fill_value=0
            )
        
            for col in self._study_method_dummy_cols:
                df[f"{col}_hours"] = (
                    dummies[col].astype("float32") *
                    df['study_hours'].astype("float32")
                )
        
        return df

        
    def _add_clustering_fit(self, df: pd.DataFrame) -> None:
        if self.verbose:
            print(f'cluster_cols before: {self.cluster_cols}')

        # Determine which clustering columns are actually present
        cluster_cols_present = [c for c in self.cluster_cols if c in df.columns]
    
        if not cluster_cols_present:
            raise ValueError(
                "Clustering requested but none of the clustering columns are present. "
                f"Expected one of: {self.cluster_cols}"
            )
    
        self._cluster_cols_fit_ = cluster_cols_present
    
        X_cluster = df[cluster_cols_present].copy()


        # KMeans / StandardScaler cannot handle NaN
        self._cluster_fill_values_ = X_cluster.median(numeric_only=True)
        X_cluster = X_cluster.fillna(self._cluster_fill_values_)
    
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_cluster)
    
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            n_init=10,
        )
        self._kmeans.fit(X_scaled)

    
    def _add_clustering_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None or self._kmeans is None or self._cluster_cols_fit_ is None:
            raise RuntimeError("Clustering requested but FeatureFactory was not fit properly.")

        missing = [c for c in self._cluster_cols_fit_ if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing clustering columns at transform-time: {missing}"
            )

        X_cluster = df[self.cluster_cols].copy()
        X_cluster = X_cluster.fillna(self._cluster_fill_values_)

        X_scaled = self._scaler.transform(X_cluster)
        df['cluster_label'] = self._kmeans.predict(X_scaled).astype("int16")
        
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
            if df[col].dtype == 'object' or str(df[col].dtype).startswith("category"):
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