import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("1. Loading the Multimodal Dataset...")
try:
    df = pd.read_csv("FINAL_MULTIMODAL_DATASET.csv")
except FileNotFoundError:
    print("❌ Error: Could not find 'FINAL_MULTIMODAL_DATASET.csv'.")
    exit()

# We only calculate baselines for the absolute spatial coordinates.
# Velocity and Variance are already relative by the laws of physics.
spatial_features = ['Pitch', 'Yaw', 'Roll', 'Gaze_X', 'Gaze_Y']

print("2. Calculating Intelligent Baselines per Session...")
relative_dfs = []

for session_id, group in df.groupby('Session_ID'):
    # Filter for frames where the student is actively interacting AND labeled NORMAL
    # This is the safest bet for a true "screen-focused" zero state.
    active_frames = group[(group['Is_Interacting'] == 1) & (group['Target_Label'] == 'NORMAL')]
    
    if len(active_frames) >= 20:
        # If we have enough valid typing frames, use them for the baseline
        baselines = active_frames[spatial_features].mean()
        print(f"   [{session_id}] Baseline calculated perfectly from {len(active_frames)} active typing frames.")
    else:
        # Fallback: If they didn't type much at all, use the first 60 frames
        print(f"   [{session_id}] ⚠️ Insufficient typing frames. Falling back to first 60 frames.")
        baselines = group.head(60)[spatial_features].mean()
        
    # Create a copy so we don't trigger Pandas warnings
    group_rel = group.copy()
    
    # Subtract the baselines to neutralize Covariate Shift
    for feature in spatial_features:
        group_rel[feature] = group_rel[feature] - baselines[feature]
        
    relative_dfs.append(group_rel)

print("\n3. Finalizing Relative Dataset...")
final_rel_df = pd.concat(relative_dfs, ignore_index=True)

output_name = "RELATIVE_MULTIMODAL_DATASET.csv"
final_rel_df.to_csv(output_name, index=False)

print(f"✅ Success! Intelligent Covariate Shift Neutralized.")
print(f"✅ Saved as: {output_name}")