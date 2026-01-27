# %% [code]
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
      - one-hot encoding (for linear models)
      - standard scaling (for linear models/NNs)
    Critical invariants:
      - Any columns created from categoricals must be stable between fit/transform.
      - Any "learned" transformations (scaler/kmeans) must be fit only on training data.
    """
    valid_strategies = [
        'drop_id',
        'ordinal_encoding',
        'one_hot_encoding',
        'binning',
        'interactions',
        'clustering',
        'polynomials',
        'log',
        'manual_formula',
        'cyclical',
        'frequency',
        'standard_scaling'
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
        self._ohe: Optional[OneHotEncoder] = None
        self._feature_scaler: Optional[StandardScaler] = None
        
        # Learned stable dummy columns for study_method
        self._study_method_dummy_cols: Optional[List[str]] = None

        self._cluster_cols_fit_ = None
        self._cluster_fill_values_ = None
        self._ohe_cols_ = None
        self._scaling_cols_ = None

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

        if 'frequency' in self.strategies:
            self._add_frequency_fit(df_new)
            
        if 'one_hot_encoding' in self.strategies:
            self._add_one_hot_encoding_fit(df_new)            
        
        # We run this last or near last to scale generated features too
        if 'standard_scaling' in self.strategies:
            # We need to temporarily apply transformations that create columns
            # to know what columns exist for scaling.
            # However, for simplicity here, we assume scaling handles existing numeric cols
            # at the time of transform.
            # Better approach: We calculate mean/std during the fit phase.
            # To do that accurately, we must replicate the generation steps on df_new
            # before fitting the scaler.
            if 'drop_id' in self.strategies:
                df_new = self._drop_ids(df_new)

            if 'log' in self.strategies:
                df_new = self._add_log(df_new)
                
            if 'polynomials' in self.strategies:
                df_new = self._add_polynomials(df_new)
                
            if 'manual_formula' in self.strategies:
                df_new = self._add_manual_formula(df_new)
                
            if 'cyclical' in self.strategies:
                df_new = self._add_cyclical(df_new)

            # Apply OHE transformation to df_new so we scale the OHE columns too if needed    
            if 'one_hot_encoding' in self.strategies:
                df_new = self._add_one_hot_encoding_transform(df_new)
            
            self._add_standard_scaling_fit(df_new)
            
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

        if 'cyclical' in self.strategies:
            df_new = self._add_cyclical(df_new)

        if 'frequency' in self.strategies:
            df_new = self._add_frequency_transform(df_new)

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

        if 'course' in df.columns and 'study_hours' in df.columns:
            if self.verbose:
                print('  -> Adding effort_gap feature...')
                
            # Highlights "slacking" relative to peers in the same difficulty bracket.
            df['course_max_study'] = df.groupby('course')['study_hours'].transform('max')
            df['effort_gap'] = df['course_max_study'] - df['study_hours']
            df.drop('course_max_study', inplace=True, errors='ignore')
            
        # Return without doing anything if all of the necessary features aren't present
        needed_cols = ['sleep_quality', 'facility_rating', 'study_method', 'study_hours', 'class_attendance']
        for col in needed_cols:
            if col not in df.columns:
                return df
        
        if self.verbose:
            print('  -> Adding manual_formula feature...')
        
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
        if self.verbose:
            print('  -> Iteraction features: fitting...')

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
        if self.verbose:
            print('  -> Iteraction features: transform...')

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
            print('  -> Clustering features: fitting...')

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
        if self.verbose:
            print('  -> Clustering features: transform...')
            
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

    
    def _add_frequency_fit(self, df: pd.DataFrame) -> None:
        if self.verbose:
            print('  -> Frequency features: fitting...')

        # Dictionary to store counts for each column
        self._freq_maps_ = {}

        # Select categorical columns
        cat_cols = self.get_cat_features(df)

        for col in cat_cols:
            # Calculate counts on TRAINING data only
            # normalize=True gives percentage, False gives raw count. 
            # Raw count is usually better for tree models to judge sample size.
            counts = df[col].astype(str).value_counts().to_dict()
            self._freq_maps_[col] = counts
                

    def _add_frequency_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps the frequencies learned during fit() to the current dataframe.
        """
        if self.verbose:
            print('  -> Frequency features: fitting...')

        if not hasattr(self, '_freq_maps_'):
            raise RuntimeError("Frequency requested but FeatureFactory was not fit properly.")
            
        for col, freq_map in self._freq_maps_.items():
            if col in df.columns:
                # Map the saved counts. fillna(0) handles categories seen in Test but not Train.
                df[f"{col}_freq"] = df[col].astype(str).map(freq_map).fillna(0).astype('int32')
                
        return df

        
    def _add_cyclical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Sine transformations to specific time-based or cyclical columns.
        """
        if self.verbose:
            print('  -> Adding cyclicals of selected features...')
            
        # Sine features for Study Hours (assuming 12-hour cycle logic from snippet)
        if 'study_hours' in df.columns:
            df['study_hours_sin'] = np.sin(2 * np.pi * df['study_hours'] / 12).astype('float32')
            
        # Sine features for Class Attendance
        if 'class_attendance' in df.columns:
            df['class_attendance_sin'] = np.sin(2 * np.pi * df['class_attendance'] / 12).astype('float32')
            
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
            
        cat_cols = self.get_cat_features(df)
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
        cat_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype).startswith("category"):
                cat_cols.append(col)
        # Discrete engineered
        for c in ['class_attendance_class', 'sleep_hours_class', 'study_hours_class', 'cluster_label']:
            if c in df.columns: cat_cols.append(c)
            
        seen = set()
        out = []
        for c in cat_cols:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

    
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