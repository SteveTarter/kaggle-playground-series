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
    Feature engineering pipeline for PS-S06E06: Predicting Stellar Class.

    Target classes: GALAXY (2), QSO (0), STAR (1)
    Raw features: alpha, delta, u, g, r, i, z, redshift, spectral_type, galaxy_population
    """

    valid_strategies = [
        'encoding',
        'colors',             # Photometric color indices (magnitude differences)
        'ratios',             # Flux/magnitude ratios and spectral shape features
        'interactions',       # Cross-feature interaction terms
        'redshift',           # Redshift-derived cosmological features
        'position',           # Sky-position features
        'flux',               # Derived flux features
        'numeric_expansion',  # Log, sqrt, ratios, and differences
    ]

    def __init__(
        self,
        *,
        strategies=None,
        seed=10301,
        target='class',
        verbose=False
    ):
        if strategies is None:
            strategies = []

        self.strategies = []
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

        # Categorical columns present in the raw data
        self.cat_cols = ['spectral_type', 'galaxy_population']

        # Learned state (set in fit)
        self._is_fit: bool = False

    # ----------------------------
    # Public API
    # ----------------------------
    def fit(self, df: pd.DataFrame = None) -> 'FeatureFactory':
        if self.verbose:
            print('  -> Fitting DataFrame...')
            
        self.num_features_ = df.select_dtypes(exclude=['object', 'bool', 'category']).columns.tolist()
        self.cat_features_ = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

        self._is_fit = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError('FeatureFactory must be fit() before transform().')

        if self.verbose:
            print(f'Applying FeatureFactory with strategies: {", ".join(self.strategies)}')

        df_new = df.copy()

        # Strategy order matters: colors first (they feed ratios/interactions)
        if 'colors' in self.strategies:
            df_new = self._add_colors(df_new)

        if 'ratios' in self.strategies:
            df_new = self._add_ratios(df_new)

        if 'redshift' in self.strategies:
            df_new = self._add_redshift_features(df_new)

        if 'position' in self.strategies:
            df_new = self._add_position_features(df_new)

        if 'interactions' in self.strategies:
            df_new = self._add_interactions(df_new)

        if 'encoding' in self.strategies:
            df_new = self._add_encoding(df_new)

        if 'flux' in self.strategies:
            df_new = self._add_flux(df_new)

        if 'numeric_expansion' in self.strategies:
            df_new = self._add_numeric_expansion(df_new)
            
        # Always drop id and target
        if 'id' in df_new.columns:
            df_new = df_new.drop('id', axis=1)

        if self.target in df_new.columns:
            df_new = df_new.drop(self.target, axis=1)

        # If any feature has just one value, it's just noise.
        drop = [c for c in df_new.columns if df_new[c].nunique(dropna=False) == 1]
        if drop:
            df_new.drop(drop, axis=1, inplace=True)

        return df_new

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_strategies(self) -> list:
        return self.strategies


    # -------------------------
    # Internals
    # -------------------------
    def _add_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ordinal-encode the two categorical features so XGBoost native-cat
        mode can consume them, or so they survive a category→int fallback.
        The columns are cast to pandas Categorical dtype; XGBoost with
        enable_categorical=True will handle them natively.
        """
        if self.verbose:
            print('  -> Adding encoding features...')

        # spectral_type: O/B < A < F < G < K < M  (rough temperature order)
        spectral_order = ['O/B', 'A', 'F', 'G', 'K', 'M']
        if 'spectral_type' in df.columns:
            df['spectral_type'] = pd.Categorical(
                df['spectral_type'], categories=spectral_order, ordered=True
            )

        # galaxy_population: no strong intrinsic order → unordered categorical
        if 'galaxy_population' in df.columns:
            df['galaxy_population'] = df['galaxy_population'].astype('category')

        return df

    
    def _add_colors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standard SDSS photometric color indices (magnitude differences).
        These are the primary discriminators between GALAXY, QSO, and STAR
        in the Sloan Digital Sky Survey color-color space.

        Bands ordered by wavelength: u < g < r < i < z
        """
        if self.verbose:
            print('  -> Adding photometric color features...')

        bands = ['u', 'g', 'r', 'i', 'z']
        # Consecutive colors
        df['u_minus_g'] = df['u'] - df['g']
        df['g_minus_r'] = df['g'] - df['r']
        df['r_minus_i'] = df['r'] - df['i']
        df['i_minus_z'] = df['i'] - df['z']

        # Wider-baseline colors (strong QSO vs STAR discriminators)
        df['u_minus_r'] = df['u'] - df['r']
        df['u_minus_z'] = df['u'] - df['z']
        df['g_minus_z'] = df['g'] - df['z']
        df['g_minus_i'] = df['g'] - df['i']

        # Total color span across all bands
        df['color_range'] = df['u'] - df['z']

        return df

    
    def _add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Magnitude-based ratio and spectral-shape features.
        Because SDSS magnitudes are on an AB system (log-flux), differences
        are equivalent to flux ratios in log-space; we also derive explicit
        curvature proxies.
        """
        if self.verbose:
            print('  -> Adding ratio/spectral-shape features...')

        # Spectral slope proxies (blue vs red)
        if 'u_minus_g' in df.columns and 'r_minus_i' in df.columns:
            df['blue_red_slope'] = df['u_minus_g'] - df['r_minus_i']

        # color curvature (concavity of the SED in g-r-i)
        if all(c in df.columns for c in ['g_minus_r', 'r_minus_i']):
            df['color_curvature'] = df['g_minus_r'] - df['r_minus_i']

        # Mean magnitude (overall brightness)
        bands = ['u', 'g', 'r', 'i', 'z']
        df['mean_mag'] = df[bands].mean(axis=1)

        # Standard deviation across bands (SED flatness)
        df['std_mag'] = df[bands].std(axis=1)

        # Ratio of outer bands to central band (SED shape)
        df['outer_to_r'] = (df['u'] + df['z']) / (2.0 * df['r'] + 1e-5)

        return df

    
    def _add_redshift_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features derived from redshift.  Redshift is the single strongest
        discriminator in this dataset:
          - STARs:    z ≈ 0
          - GALAXYs:  z ~ 0.1–1
          - QSOs:     z ~ 0.1–5+

        We also interact redshift with photometric colors because the
        4000-Å break shifts through different SDSS bands at different redshifts.
        """
        if self.verbose:
            print('  -> Adding redshift features...')

        z = df['redshift']

        df['redshift_log1p'] = np.log1p(np.clip(z, 0, None))
        df['redshift_sq']    = z ** 2
        df['is_near_zero_z'] = (np.abs(z) < 0.004).astype(np.int8)   # strong STAR flag
        df['is_high_z']      = (z > 1.0).astype(np.int8)             # QSO-dominated region

        # Redshift × color interactions
        if 'u_minus_g' in df.columns:
            df['z_x_u_g'] = z * df['u_minus_g']
        if 'g_minus_r' in df.columns:
            df['z_x_g_r'] = z * df['g_minus_r']
        if 'color_range' in df.columns:
            df['z_x_color_range'] = z * df['color_range']

        df['is_mid_z'] = ((z >= 0.1) & (z <= 0.5)).astype(np.int8)

        return df

    
    def _add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sky-coordinate features from right ascension (alpha) and
        declination (delta).  The Galactic plane is a natural boundary
        between STAR-rich and extragalactic-rich sky regions.
        """
        if self.verbose:
            print('  -> Adding sky-position features...')

        alpha_rad = np.deg2rad(df['alpha'])
        delta_rad = np.deg2rad(df['delta'])

        # Cartesian projection onto the unit sphere
        df['pos_x'] = np.cos(delta_rad) * np.cos(alpha_rad)
        df['pos_y'] = np.cos(delta_rad) * np.sin(alpha_rad)
        df['pos_z'] = np.sin(delta_rad)

        # Galactic latitude proxy: SDSS avoids |b| < ~30°, but small
        # residual latitude effects remain
        df['abs_delta'] = np.abs(df['delta'])

        return df

        
    def _add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Higher-order interactions between the most discriminating features.
        """
        if self.verbose:
            print('  -> Adding interaction features...')

        z = df['redshift']

        # Redshift × brightness
        df['z_x_mean_mag'] = z * df['mean_mag'] if 'mean_mag' in df.columns else z * df['r']

        # color-color products (analogous to color-color diagram axes)
        if 'u_minus_g' in df.columns and 'g_minus_r' in df.columns:
            df['ug_x_gr'] = df['u_minus_g'] * df['g_minus_r']

        if 'g_minus_r' in df.columns and 'r_minus_i' in df.columns:
            df['gr_x_ri'] = df['g_minus_r'] * df['r_minus_i']

        # SED flatness × redshift
        if 'std_mag' in df.columns:
            df['std_mag_x_z'] = df['std_mag'] * z

        if 'u_minus_g' in df.columns and 'i_minus_z' in df.columns and 'std_mag' in df.columns:
            df['color_skew'] = (df['u_minus_g'] - df['i_minus_z']) / (df['std_mag'] + 1e-5)
        
        return df

    
    def _add_flux(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flux-derived features.
        """
        if self.verbose:
            print('  -> Adding flux features...')
            
        # Band pair ratios in linear flux space
        for band in ['u', 'g', 'r', 'i', 'z']:
            df[f'flux_{band}'] = np.power(10, -0.4 * df[band])
        df['flux_u_over_g'] = df['flux_u'] / (df['flux_g'] + 1e-9)
        df['total_flux']    = df[['flux_u','flux_g','flux_r','flux_i','flux_z']].sum(axis=1)

        return df

    
    def _add_numeric_expansion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add square, square root, log, ratios, and differences 
        of all of the original numeric features
        """
        if self.verbose:
            print('  -> Adding blegga features...')

        for c in self.num_features_:
            df[f"Log_{c}"] = np.log1p(df[c])
            df[f"{c}_sq"] = df[c]**2            
            df[f"{c}_sqrt"] = df[c]**0.5
        
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