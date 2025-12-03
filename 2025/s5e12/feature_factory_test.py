import pandas as pd
import numpy as np
from diabetes_preprocessing import FeatureFactory

# Setup the Class (Paste your full class here if testing in a new notebook)
# Ensure your class definition includes the new _add_one_hot_encodings method 
# and the updated transform loop we discussed.

# Create Dummy Data
# We create a small dataframe with just enough columns to trigger the logic
test_data = {
    'cholesterol_total':         [200, 180, 220],
    'hdl_cholesterol':           [50, 45, 60],
    'systolic_bp':               [120, 130, 110],
    'diastolic_bp':              [80, 85, 70],
    'triglycerides':             [150, 200, 100],
    'age':                       [30, 50, 40],
    'waist_to_hip_ratio':        [0.930, 0.830, 0.950],
    'screen_time_hours_per_day': [6.2, 3.4, 1.2],
    'bmi':                       [33.4, 23.8, 28.9],
    'gender':                    ['Male', 'Female', 'Male'],
    'ethnicity':                 ['White', 'White', 'Hispanic']
}
df_test = pd.DataFrame(test_data)

print("Original Columns:", df_test.columns.tolist())

# Test Case A: Numeric Features
print("\n--- Test A: Ratios, Log, Polynomials ---")
# Initialize factory with numeric strategies
ff_numeric = FeatureFactory(strategies=['ratios', 'log', 'polynomials'])

# Run transform
df_numeric_result = ff_numeric.transform(df_test)

# Verify results
new_cols = [c for c in df_numeric_result.columns if c not in df_test.columns]
print(f"Strategies Applied: {ff_numeric.strategies}")
print(f"New Columns Created: {new_cols}")

# Check a specific calculation (e.g., Age Squared)
print("Age Squared Verification:", df_numeric_result['age_sq'].values)


# Test Case B: One-Hot Encoding
print("\n--- Test B: One Hot Encoding ---")
ff_ohe = FeatureFactory(strategies=['one_hot_encoding'])
df_ohe_result = ff_ohe.transform(df_test)

print(f"Strategies Applied: {ff_ohe.strategies}")
print("Columns after OHE:", df_ohe_result.columns.tolist())

# Check if 'gender' was converted
if 'gender_Male' in df_ohe_result.columns:
    print("SUCCESS: 'gender' column was successfully One-Hot Encoded.")
else:
    print("FAILURE: OHE columns missing.")


# Test Case C: Error Handling
print("\n--- Test C: Invalid Strategy Check ---")
try:
    ff_invalid = FeatureFactory(strategies=['magic_wand'])
except ValueError as e:
    print(f"SUCCESS: Caught expected error -> {e}")