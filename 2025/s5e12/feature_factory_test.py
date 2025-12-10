import pandas as pd
import numpy as np
import sys
from diabetes_preprocessing import FeatureFactory

# Ensure the module can be found if running in a different directory structure
sys.path.append('.')

seed = 10301
# In a notebook, this variable is usually global. 
# We inject it into the builtins so the class can find it if it's not passed in __init__
import builtins
builtins.seed = seed

# Setup the Class (Paste your full class here if testing in a new notebook)
# Ensure your class definition includes the new _add_one_hot_encodings method 
# and the updated transform loop we discussed.

# Create Dummy Data
# We create a small dataframe with just enough columns to trigger the logic
test_data = {
    'id':                                 [1001, 1002, 1003, 1004, 1005, 1006, 1007],
    'cholesterol_total':                  [200, 180, 220, 190, 210, 205, 195],
    'hdl_cholesterol':                    [50, 45, 60, 55, 40, 48, 52],
    'systolic_bp':                        [120, 130, 110, 125, 140, 122, 118],
    'diastolic_bp':                       [80, 85, 70, 82, 90, 78, 76],
    'triglycerides':                      [150, 200, 100, 160, 210, 155, 145],
    'age':                                [30, 50, 40, 35, 60, 45, 55],
    'waist_to_hip_ratio':                 [0.930, 0.830, 0.950, 0.900, 0.880, 0.890, 0.910],
    'screen_time_hours_per_day':          [6.2, 3.4, 1.2, 4.0, 5.5, 4.5, 3.0],
    'physical_activity_minutes_per_week': [120, 60, 200, 150, 45, 90, 110],
    'bmi':                                [33.4, 23.8, 28.9, 26.0, 31.5, 27.5, 29.0],
    'gender':                             ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'ethnicity':                          ['White', 'White', 'Hispanic', 'Asian', 'Black', 'Other', 'White'],
    'education_level':                    ['Highschool', 'Graduate', 'Elementary', 'Highschool', 'No Schooling', 'Graduate', 'Highschool'],
    'income_level':                       ['Middle', 'High', 'Low', 'Lower-Middle', 'Low', 'Middle', 'Upper-Middle'],
    'diagnosed_diabetes':                 [1, 0, 0, 0, 1, 0, 1] 
}
df_test = pd.DataFrame(test_data)
df_test = pd.DataFrame(test_data)

print(f"Test Data Shape: {df_test.shape}")
print("Original Columns:", df_test.columns.tolist())

# ---------------------------------------------------------
# Test Case A: Numeric Features (Ratios, Log, Polynomials)
# ---------------------------------------------------------
print("\n--- Test A: Ratios, Log, Polynomials ---")
ff_numeric = FeatureFactory(strategies=['ratios', 'log', 'polynomials'])
df_numeric_result = ff_numeric.transform(df_test)

new_cols_a = [c for c in df_numeric_result.columns if c not in df_test.columns]

# Check Polynomials
if 'age_sq' in new_cols_a:
    print("SUCCESS: Polynomials (age_sq) created.")
else:
    print("FAILURE: Polynomials missing.")

# Check Log
if 'triglycerides_log' in new_cols_a:
    print("SUCCESS: Log transforms (triglycerides_log) created.")
else:
    print("FAILURE: Log transforms missing.")
    
# ---------------------------------------------------------
# Test B: One-Hot Encoding
# ---------------------------------------------------------
print("\n--- Test B: One Hot Encoding ---")
ff_ohe = FeatureFactory(strategies=['one_hot_encoding'])
df_ohe_result = ff_ohe.transform(df_test)

# Check if 'gender' was converted
if 'gender_Male' in df_ohe_result.columns:
    print("SUCCESS: 'gender' column was successfully One-Hot Encoded.")
else:
    print(f"FAILURE: OHE columns missing. Present columns: {df_ohe.columns.tolist()}")

# ---------------------------------------------------------
# Test C: Error Handling
# ---------------------------------------------------------
print("\n--- Test C: Invalid Strategy Check ---")
try:
    ff_invalid = FeatureFactory(strategies=['magic_wand'])
except ValueError as e:
    print(f"SUCCESS: Caught expected error -> {e}")

# ---------------------------------------------------------
# Test D: Clinical Indices
# ---------------------------------------------------------
print("\n--- Test D: Clinical Indices (VAI, LAP) ---")
ff_indices = FeatureFactory(strategies=['clinical_indices'])
df_indices = ff_indices.fit_transform(df_test)

if 'vai_proxy' in df_indices.columns and 'lap_proxy' in df_indices.columns:
    print(f"SUCCESS: Clinical indices created.")
    print(f"Sample VAI: {df_indices['vai_proxy'].values}")
else:
    print("FAILURE: Clinical indices missing.")

# ---------------------------------------------------------
# Test E: Binning
# ---------------------------------------------------------
print("\n--- Test E: Medical Binning (BMI, BP Class) ---")
ff_binning = FeatureFactory(strategies=['binning'])
df_binning = ff_binning.fit_transform(df_test)

# Check that they are integers/categories
if 'bmi_class' in df_binning.columns and 'bp_class' in df_binning.columns:
    print(f"SUCCESS: Bins created.")
    print(f"Sample BMI Class: {df_binning['bmi_class'].values}")
else:
    print("FAILURE: Binning columns missing.")


# ---------------------------------------------------------
# Test F: Clustering
# ---------------------------------------------------------
print("\n--- Test F: Unsupervised Clustering ---")
# Clustering requires the 'fit' method to initialize KMeans
ff_cluster = FeatureFactory(strategies=['clustering'], verbose=True)

# We simulate a train/test split scenario
# Fit on df_test, Transform df_test
ff_cluster.fit(df_test)
df_cluster = ff_cluster.transform(df_test)

if 'cluster_label' in df_cluster.columns:
    print("SUCCESS: Cluster labels created.")
    print(f"Labels: {df_cluster['cluster_label'].values}")
else:
    print("FAILURE: Cluster label missing.")


# ---------------------------------------------------------
# Test G: Ordinal Encoding
# ---------------------------------------------------------
print("\n--- Test G: Ordinal Encoding ---")
ff_ordinal = FeatureFactory(strategies=['ordinal_encoding'])
df_ordinal = ff_ordinal.fit_transform(df_test)

if 'education_level_ord' in df_ordinal.columns:
    print("SUCCESS: Ordinal features created.")
    # 'Highschool' should be mapped to 2
    print(f"Mapped Education: {df_ordinal['education_level_ord'].values}")
else:
    print("FAILURE: Ordinal columns missing.")


# ---------------------------------------------------------
# Test I: Group Aggregations
# ---------------------------------------------------------
print("\n--- Test I: Group Aggregations ---")
ff_group = FeatureFactory(strategies=['group_aggregations'])

# We need to handle potential pandas fragmentation warnings or settings 
# that might arise from multiple transforms in a loop, but here it's fine.
df_group = ff_group.transform(df_test)

# Check for a specific aggregated column
# The code creates {target}_diff_gender_age
expected_col = 'bmi_diff_gender_age'

if expected_col in df_group.columns:
    print(f"SUCCESS: Group aggregation column '{expected_col}' created.")
    # Check that values are actually different (i.e., calculation happened)
    print(f"Sample Values: {df_group[expected_col].values[:3]}")
else:
    print(f"FAILURE: Group aggregations missing. Columns found: {df_group.columns.tolist()}")

# ---------------------------------------------------------
# Test J: Cohort Deviations
# ---------------------------------------------------------
print("\n--- Test J: Cohort Deviations ---")
# Note: In diabetes_preprocessing.py this might be labelled 'cohort deviations' or 'cohort_deviations'
# We try to use the key used in the transform method logic.
ff_cohort = FeatureFactory(strategies=['cohort_deviations'], verbose=False)

try:
    df_cohort = ff_cohort.transform(df_test)
    
    # The code calculates deviation from age decile/gender group
    # Columns like {target}_cohort_deviation
    dev_col = 'bmi_cohort_deviation'

    if dev_col in df_cohort.columns:
        print(f"SUCCESS: Cohort deviation column '{dev_col}' created.")
        print(f"Sample Deviation: {df_cohort[dev_col].values[:3]}")
    else:
        print(f"FAILURE: Cohort deviation column missing.")
        
except Exception as e:
    print(f"FAILURE: Cohort deviation transform crashed: {e}")

# ---------------------------------------------------------
# Test H: Integrated Pipeline (ALL Features)
# ---------------------------------------------------------
print("\n--- Test H: Full Pipeline Integration ---")
all_strategies = [
    'drop_id', 'ordinal_encoding', 'medical_metrics', 'clinical_indices',
    'binning', 'clustering', 'interactions', 'ratios', 'one_hot_encoding',
    'log', 'polynomials', 'cohort_deviations', 'group_aggregations'
]

ff_full = FeatureFactory(strategies=all_strategies, verbose=False)

try:
    # Fit and Transform
    df_full = ff_full.fit(df_test).transform(df_test)

    print(f"Final Shape: {df_full.shape}")
    
    # 1. Verify ID dropped
    if 'id' not in df_full.columns:
        print("CHECK: ID column dropped.")
    
    # 2. Verify OHE didn't consume Ordinal/Binning columns
    if 'bmi_class' in df_full.columns:
        print("CHECK: Binned columns preserved.")
        
    # 3. Verify New Features present in full pipeline
    if 'bmi_diff_gender_age' in df_full.columns:
        print("CHECK: Group Aggregations present in full pipeline.")

    print("SUCCESS: Full pipeline integration complete.")

except Exception as e:
    print(f"FAILURE: Integrated pipeline crashed -> {e}")