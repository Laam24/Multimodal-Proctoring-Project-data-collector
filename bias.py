import pandas as pd
import numpy as np

df = pd.read_csv("FINAL_TRAINING_DATASET.csv")
print("--- Data Integrity Check ---")
nan_counts = df.isnull().sum().sum()
inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

print(f"Total NaN values: {nan_counts}")
print(f"Total Infinity values: {inf_counts}")

if nan_counts > 0 or inf_counts > 0:
    print("\n⚠️ FOUND ISSUES! Cleaning data...")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df.to_csv("FINAL_TRAINING_DATASET_CLEAN.csv", index=False)
    print("✅ Cleaned data saved as 'FINAL_TRAINING_DATASET_CLEAN.csv'")
else:
    print("✅ Data looks clean. The issue is likely the training parameters.")