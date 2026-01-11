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
    Centralized factory for creating new numeric and engineered features.
    
    It focuses on creating new numeric features (ratios, polynomials) and
    transforming existing numeric features (log transforms).
    """
    valid_strategies = [
        'drop_id',
        'ordinal_encoding',
        'binning',
        'interactions',
        'clustering',
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
            
        # Select features for clustering
        self.cluster_cols = ['study_hours', 'class_attendance']

        # If interactions are active, add 'restoration_index'
        if 'interactions' in self.strategies:
            self.cluster_cols.append('restoration_index')
            
        # Learned state (set in fit)
        self._is_fit: bool = False
        self._scaler: Optional[StandardScaler] = None
        self._kmeans: Optional[StandardScaler] = None

    def fit(self, df: pd.DataFrame = None) -> 'FeatureFactory':
        df = df.copy()
        
        # If we need interaction columns for clustering, we must create them first
        if 'interactions' in self.strategies:
            df = self._add_interactions(df)
            
        cluster_cols = [c for c in self.cluster_cols if c in df.columns]

        if 'clustering' in self.strategies and cluster_cols:
            if self.verbose:
                print("  -> Fitting K-Means Clustering...")

            # SAFEGUARD: Ensure we don't request more clusters than samples
            n_samples = len(df)
            actual_n_clusters = min(self.n_clusters, n_samples - 1) if n_samples > 1 else 1
            
            # Scale the data
            # K-Means is distance-based; different scales distort the geometry.
            self._scaler = StandardScaler()
            # Scale fitting data
            X_cluster = self._scaler.fit_transform(df[cluster_cols])
            
            self._kmeans = KMeans(n_clusters=actual_n_clusters, random_state=self.seed, n_init=10)
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
            
        if 'interactions' in self.strategies:
            df_new = self._add_interactions(df_new)
        
        if 'binning' in self.strategies:
            df_new = self._add_binning(df_new)
        
        if 'clustering' in self.strategies:
            df_new = self._add_clustering(df_new)
            
        return df_new

    def get_strategies(self) -> []:
        return self.strategies

    # -------------------------
    # Internals
    # -------------------------
        
        
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
        df['class_attendance_class'] = pd.cut(
            df['class_attendance'], 
            bins=[-1, 60.0, 80.0, 100.1], 
            labels=[0, 1, 2]
        ).astype(int)
        
        # Sleep Hours Classes
        # Sleep usually has a U-shaped relationship with performance.
        # 0: Deprived (< 6 hours), 1: Optimal (6-9 hours), 2: Excess (> 9 hours)
        df['sleep_hours_class'] = pd.cut(
            df['sleep_hours'],
            bins=[-1, 6.0, 9.0, 24.0],
            labels=[0, 1, 2]
        ).astype(int)
        
        # Study Hours Classes
        # Diminishing Returns.  Why bin it? The benefit of studying is not infinite.
        # 0: Low (< 2 hours), 1: Target (2-6 hours), 2: Burnout (> 6 hours)
        df['study_hours_class'] = pd.cut(
            df['study_hours'],
            bins=[-1, 2.0, 6.0, 100.0],
            labels=[0, 1, 2]
        ).astype(int)
        
        return df
    
    def _add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding interactions (Effective Effort, Restoration Index, Engagement Multiplier, Resource Efficiency)...')

        # Interaction 1: Effective Effort (Num x Cat)
        # Logic: 10 hours of 'Cramming' != 10 hours of 'Spaced Repetition'.
        # We create specific features for each method weighted by hours.
        # This helps the model isolate the "slope" of study_hours for specific methods.
        methods = pd.get_dummies(df['study_method'], prefix='effort')
        for col in methods.columns:
            # Clean the name: 'effort_self-study' -> 'effort_self_study'
            clean_base = col.replace(' ', '_').replace('-', '_')
            if self.verbose:
                print(f'    Adding {clean_base}_hours')
                
            # e.g., creates 'effort_Spaced' = study_hours * 1 (if Spaced) else 0
            df[f'{clean_base}_hours'] = methods[col] * df['study_hours']
            
        # Interaction 2: Restoration Index (Num x Ordinal)
        # Logic: Sleep duration needs to be qualified by sleep quality.
        # We map the categorical quality to a numeric scale (Ordinal Encoding)
        # and multiply to get a "Total Rest" volume.
        quality_map = {'poor': 1, 'average': 2, 'good': 3}
        
        # Create temp numeric column for calculation
        df['sleep_quality_num'] = df['sleep_quality'].map(quality_map)
        df['restoration_index'] = df['sleep_hours'] * df['sleep_quality_num']
    
        # Interaction 3: Engagement Multiplier (Num x Num)
        # Logic: High attendance AND High study hours suggests a top performer.
        # We use multiplication to highlight the compound effect.
        df['total_engagement'] = df['study_hours'] * df['class_attendance']
    
        # Logic: "Autodidact Ratio". High study hours but LOW attendance.
        # We add a small epsilon (1e-5) to avoid division by zero.
        df['autodidact_ratio'] = df['study_hours'] / (df['class_attendance'] + 1e-5)
    
        # Interaction 4: Resource Efficiency (Num x Ordinal)
        # Logic: Facility rating acts as a friction coefficient.
        # We map facility to a scale. If facility is 'Low' (1), study hours might count for less.
        facility_map = {'low': 0.8, 'medium': 1.0, 'high': 1.2} # Custom weights based on domain intuition
        
        df['facility_weight'] = df['facility_rating'].map(facility_map)
        df['weighted_study_hours'] = df['study_hours'] * df['facility_weight']
    
        # Cleanup: Drop intermediate columns
        drop_cols = ['sleep_quality_num', 'facility_weight'] 
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        return df

    def _add_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.verbose:
            print('  -> Adding unsupervised clusters...')
        
        # We must use the SAME features used in fit()
        cluster_cols = [c for c in self.cluster_cols if c in df.columns]
        
        if hasattr(self, '_kmeans') and self._kmeans is not None and cluster_cols:
            # Must handle NaN if interactions created NaNs (though we handle map fillna usually)
            df[cluster_cols] = df[cluster_cols].fillna(0)
            
            df_cluster = self._scaler.transform(df[cluster_cols])
            df['cluster_label'] = self._kmeans.predict(df_cluster)
        
        return df