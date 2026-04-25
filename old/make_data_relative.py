import pandas as pd

# 1. Load the cleaned data
print("Loading FINAL_TRAINING_DATASET_CLEAN.csv...")
try:
    df = pd.read_csv("FINAL_TRAINING_DATASET_CLEAN.csv")
except FileNotFoundError:
    print("❌ Error: Could not find 'FINAL_TRAINING_DATASET_CLEAN.csv'. Please ensure it is in the same folder.")
    exit()

# 2. Identify the absolute spatial features that need to be baselined
# (We DO NOT baseline Velocity or Variance, as they are naturally shift-invariant)
spatial_features = ['Pitch', 'Yaw', 'Roll', 'Gaze_X', 'Gaze_Y']

print(f"Calculating Zero-State Baselines for {len(df['Session_ID'].unique())} unique sessions...")
relative_df_list = []

# 3. Process each session independently
for session_id, group in df.groupby('Session_ID'):
    # Sort by frame chronologically to ensure the first 60 are actually the beginning
    group = group.sort_values(by='Frame').copy()
    
    # Calculate the baseline using the first 60 frames
    baseline_frames = group.head(60)
    baselines = baseline_frames[spatial_features].mean()
    
    # Subtract the baseline from the entire session for these specific columns
    for feature in spatial_features:
        group[feature] = group[feature] - baselines[feature]
        
    relative_df_list.append(group)

# 4. Recombine and save
final_relative_df = pd.concat(relative_df_list, ignore_index=True)
final_relative_df.to_csv("RELATIVE_TRAINING_DATASET.csv", index=False)

print("\n✅ Success! Covariate Shift neutralized.")
print(f"✅ Saved as 'RELATIVE_TRAINING_DATASET.csv'")
print(f"Total rows processed: {len(final_relative_df)}")