# %% [code]
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pandas.api.types import is_numeric_dtype
from pandas.api.types import CategoricalDtype

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

@dataclass
class _AggSpec:
    group_cols: Tuple[str, ...]
    value_cols: Tuple[str, ...]
    prefix: str

class FeatureFactory(BaseEstimator, TransformerMixin):
    valid_strategies = [
        'encoding',
        'ratios',
        'interactions',
        'binning',
        'magic_formula'
    ]
    
    def __init__(
        self, 
        *,
        strategies=None,
        seed=10301,
        target='Irrigation_Need',
        verbose=False,
    ):
        if strategies is None:
            strategies = []
        
        self.strategies = []
        self.seed = seed
        self.target = target
        self.verbose = verbose
        self._engineered_cat_cols_ = []
        
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

        # Categorical columns
        self.cat_cols = [
            'Soil_Type',
            'Crop_Type',
            'Crop_Growth_Stage', 
            'Season',
            'Irrigation_Type',
            'Water_Source',
            'Region'
        ]
        
        # Learned state (set in fit)
        self._is_fit: bool = False

    
    # ----------------------------
    # Public API
    # ----------------------------
    def fit(self, df: pd.DataFrame = None) -> 'FeatureFactory':
        if self.verbose:
            print('  -> Fitting DataFrame...')
            
        self._is_fit = True
        return self

    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError('FeatureFactory must be fit() before transform().')

        if self.verbose:
            print(f'Applying FeatureFactory with strategies: {", ".join(self.strategies)}')
        
        df_new = df.copy()
            
        if 'ratios' in self.strategies:
            df_new = self._add_ratios(df_new)

        if 'interactions' in self.strategies:
            df_new = self._add_interactions(df_new)
            
        if 'binning' in self.strategies:
            df_new = self._add_binning(df_new)

        if 'magic_formula' in self.strategies:
            df_new = self._add_magic_formula(df_new)
            
        if 'encoding' in self.strategies:
            df_new = self._add_encoding(df_new)
        
        # Regardless of strategies, there's a few things that are always needed
        
        # get rid of the id column
        if 'id' in df_new.columns:
            df_new = df_new.drop('id', axis=1)

        # If the target is in the dataset, drop it.
        if self.target in df_new.columns:
            df_new = df_new.drop(self.target, axis=1)

        return df_new

    
    def fit_transform(self, df:pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
        
    def get_strategies(self) -> []:
        return self.strategies

    # -------------------------
    # Internals
    # -------------------------
    def _add_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles label mapping for the target and ordinal encoding for 
        high-signal categorical features.
        """
        if self.verbose:
            print('  -> Adding encoding features...')

        # Stub; to be filled out later
                
        return df
        

    def _add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Purely derived financial ratios to capture customer value over time.
        NaN-safe for use before scaling/clustering.
        """
        if self.verbose:
            print('  -> Adding ratio features...')

        # Physics-based weather features
        if 'Temperature_C' in df.columns and 'Humidity' in df.columns:
            # Saturation Vapor Pressure (approximate)
            df['SVP'] = 0.611 * np.exp((17.27 * df['Temperature_C']) / (df['Temperature_C'] + 237.3))
            # Vapor Pressure Deficit (How much water the air can pull from soil)
            df['VPD'] = df['SVP'] * (1 - (df['Humidity'] / 100))
            
        if 'Rainfall_mm' in df.columns and 'Temperature_C' in df.columns:
            # Simple Water Balance Index
            df['Water_Balance_Index'] = df['Rainfall_mm'] - (df['Temperature_C'] * 0.2)

        # Interactions from https://www.kaggle.com/code/ektarr/cv-0-975-lb-0-97444-irrigation-need
        df["moisture_sq"] = df["Soil_Moisture"] ** 2
        df["wind_sq"] = df["Wind_Speed_kmh"] ** 2
        df["temp_sq"] = df["Temperature_C"] ** 2
    
        df["rainfall_log"] = np.log1p(df["Rainfall_mm"])
        df["prev_irrig_log"] = np.log1p(df["Previous_Irrigation_mm"])
        df["field_area_log"] = np.log1p(df["Field_Area_hectare"])
        df["moisture_rank"] = df["Soil_Moisture"].rank()

        return df
        

    def _add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates features based on service density and interactions between 
        high-signal categorical features and numeric predictors.
        """
        if self.verbose:
            print('  -> Adding interactions features...')

        # Manually map Soil_Type to numeric values
        soil_mapping = {
            'Sandy': 1, 
            'Loamy': 2, 
            'Silt':  3, 
            'Clay':  4
        }

        if 'Soil_Type' in df.columns:
            soil_numeric = df['Soil_Type'].map(soil_mapping).fillna(0)
        
            if 'Soil_Moisture' in df.columns:
                # How moisture level behaves per soil type
                df['Soil_Moisture_Interaction'] = soil_numeric * df['Soil_Moisture']

        # Interactions from https://www.kaggle.com/code/ektarr/cv-0-975-lb-0-97444-irrigation-need
        df["water_balance"] = df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
        df["temp_humidity_ratio"] = df["Temperature_C"] / (df["Humidity"] + 1e-5)
    
        df["evaporation"] = df["Temperature_C"] * df["Sunlight_Hours"] * (1 - df["Humidity"] / 100)
        df["water_deficit"] = df["evaporation"] - df["water_balance"]
    
        df["soil_health"] = df["Organic_Carbon"] * df["Soil_Moisture"]
        df["moisture_per_area"] = df["Soil_Moisture"] / df["Field_Area_hectare"]
        df["irrigation_efficiency"] = df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 1e-5)
    
        df["ET_proxy"]       = (df["Temperature_C"] * df["Wind_Speed_kmh"] * df["Sunlight_Hours"]) / (df["Humidity"] + 1)
        df["heat_stress"]    = df["Temperature_C"] * df["Sunlight_Hours"]
        df["drying_force"]   = df["Wind_Speed_kmh"] * df["Temperature_C"] / (df["Humidity"] + 1)
        df["holding_cap"]    = df["Soil_Moisture"] * df["Organic_Carbon"]
        df["soil_quality"]   = df["Organic_Carbon"] / (df["Electrical_Conductivity"] + 1e-5)
    
        df["moisture_rainfall"] = df["Soil_Moisture"] * df["Rainfall_mm"]
        df["temp_x_humidity"] = df["Temperature_C"] * df["Humidity"]
        df["sun_temp"] = df["Sunlight_Hours"] * df["Temperature_C"]
        df["moist_x_wind"]   = df["Soil_Moisture"] * df["Wind_Speed_kmh"]
        df["moist_x_temp"]   = df["Soil_Moisture"] * df["Temperature_C"]
        df["wind_x_temp"]    = df["Wind_Speed_kmh"] * df["Temperature_C"]
        

        return df
        

    def _add_magic_formula(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds "magic" values.
        From https://www.kaggle.com/code/wguesdon/ps6e4-14-model-gbdt-ensemble
        """
        if self.verbose:
            print('  -> Adding magic_formula features...')

        df["magic_soil_dry"] = (df["Soil_Moisture"] < 25).astype(np.int8)
        df["magic_rain_low"] = (df["Rainfall_mm"] < 300).astype(np.int8)
        df["magic_temp_hot"] = (df["Temperature_C"] > 30).astype(np.int8)
        df["magic_wind_high"] = (df["Wind_Speed_kmh"] > 10).astype(np.int8)
        df["magic_harvest"] = (df["Crop_Growth_Stage"] == "Harvest").astype(np.int8)
        df["magic_sowing"] = (df["Crop_Growth_Stage"] == "Sowing").astype(np.int8)
        df["magic_mulch_yes"] = (df["Mulching_Used"] == "Yes").astype(np.int8)
    
        df["magic_high_score"] = (
            df["magic_soil_dry"] * 2 + df["magic_rain_low"] * 2
            + df["magic_temp_hot"] + df["magic_wind_high"]
        )
        df["magic_low_score"] = (
            df["magic_harvest"] * 2 + df["magic_sowing"] * 2 + df["magic_mulch_yes"]
        )
        df["magic_score"] = df["magic_high_score"] - df["magic_low_score"]
        df["magic_dist_to_boundary"] = np.minimum(
            np.abs(df["magic_score"] - 0), np.abs(df["magic_score"] - 3)
        )
        df["margin_soil_25"] = df["Soil_Moisture"] - 25
        df["margin_rain_300"] = df["Rainfall_mm"] - 300
        df["margin_temp_30"] = df["Temperature_C"] - 30
        df["margin_wind_10"] = df["Wind_Speed_kmh"] - 10

        return df

        
    def _add_binning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bins continuous tenure into lifecycle stages to smooth out noise
        and improve model generalization.
        """
        if self.verbose:
            print('  -> Adding binning features...')

        # Stub; to be filled out later

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
            if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
                cat_cols.append(col)

        # Explicit engineered categorical columns we track
        for c in self._engineered_cat_cols_:
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