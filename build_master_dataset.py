import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. Define the root path
BASE_DIR = "Honeypot_Sessions"
OUTPUT_FILE = "FINAL_MULTIMODAL_DATASET.csv"

if not os.path.exists(BASE_DIR):
    print(f"❌ Error: Cannot find directory '{BASE_DIR}'. Make sure you run this from the root project folder.")
    exit()

all_sessions_data = []

print(f"🔍 Scanning '{BASE_DIR}' and all subfolders for valid sessions...\n")

# os.walk() searches everything inside Honeypot_Sessions automatically
for root_path, dirs, files in os.walk(BASE_DIR):
    # Check if this specific folder contains the holy trinity of files
    if "visual_features.csv" in files and "events_labeled.csv" in files and "inputs.csv" in files:
        session_name = os.path.basename(root_path)
        print(f"🔄 Processing Session: {session_name} (Found in {root_path})")
        
        vis_path = os.path.join(root_path, "visual_features.csv")
        labels_path = os.path.join(root_path, "events_labeled.csv")
        inputs_path = os.path.join(root_path, "inputs.csv")

        # --- LOAD ---
        df_vis = pd.read_csv(vis_path)
        df_labels = pd.read_csv(labels_path)
        df_inputs = pd.read_csv(inputs_path, header=None, names=["Timestamp", "Device", "Key", "Window"])

        # Ensure Timestamps are numeric and sorted
        df_vis['Timestamp'] = pd.to_numeric(df_vis['Timestamp'], errors='coerce')
        df_labels['Timestamp'] = pd.to_numeric(df_labels['Timestamp'], errors='coerce')
        df_inputs['Timestamp'] = pd.to_numeric(df_inputs['Timestamp'], errors='coerce')

        df_vis = df_vis.sort_values('Timestamp').reset_index(drop=True)
        df_labels = df_labels.sort_values('Timestamp')
        df_inputs = df_inputs.sort_values('Timestamp')

        # --- SYNC LABELS & WINDOWS ---
        df_master = pd.merge_asof(df_vis, df_labels, on='Timestamp', direction='backward')
        df_master['Label'] = df_master['Label'].fillna('NORMAL') 
        
        df_windows = df_inputs[['Timestamp', 'Window']].dropna()
        df_master = pd.merge_asof(df_master, df_windows, on='Timestamp', direction='backward')

        unauth_list = ['google', 'chatgpt', 'search', 'chrome', 'edge', 'firefox', 'bing', 'brave', 'notepad', 'opera']
        df_master['Unauthorized_Window'] = df_master['Window'].apply(
            lambda x: 1.0 if any(w in str(x).lower() for w in unauth_list) else 0.0
        )

        # --- REBUILD PHYSICS ENGINE ---
        spatial_cols = ['Pitch', 'Yaw', 'Roll', 'Gaze_X', 'Gaze_Y']
        for col in spatial_cols:
            df_master[f'{col}_Vel'] = df_master[col].diff().fillna(0)
            df_master[f'{col}_Var'] = df_master[col].rolling(window=60, min_periods=1).var().fillna(0)

        # --- BUILD INTERACTION ENGINE ---
        window_seconds = 2.0 
        input_timestamps = df_inputs['Timestamp'].values
        interactions_list = []

        for t in df_master['Timestamp']:
            if pd.isna(t):
                interactions_list.append(0)
                continue
            start_idx = np.searchsorted(input_timestamps, t - window_seconds)
            end_idx = np.searchsorted(input_timestamps, t)
            interactions_list.append(end_idx - start_idx)

        df_master['Interactions_Last_2s'] = interactions_list
        df_master['Is_Interacting'] = (df_master['Interactions_Last_2s'] > 0).astype(float)

        # --- FINALIZE SESSION ---
        df_master['Session_ID'] = session_name
        df_master.rename(columns={'Label': 'Target_Label'}, inplace=True)
        
        final_cols = [
            'Session_ID', 'Timestamp', 'Frame', 'Face_Dete', 
            'Pitch', 'Yaw', 'Roll', 'Gaze_X', 'Gaze_Y',
            'Pitch_Vel', 'Yaw_Vel', 'Roll_Vel', 'Gaze_X_Vel', 'Gaze_Y_Vel',
            'Pitch_Var', 'Yaw_Var', 'Roll_Var', 'Gaze_X_Var', 'Gaze_Y_Var',
            'Interactions_Last_2s', 'Is_Interacting', 'Unauthorized_Window', 'Target_Label'
        ]
        
        # Filter to only keep the columns we need
        df_master = df_master[[c for c in final_cols if c in df_master.columns]]
        all_sessions_data.append(df_master)

# --- STITCH AND SAVE ---
if len(all_sessions_data) > 0:
    print("\n🧵 Stitching all sessions together...")
    final_df = pd.concat(all_sessions_data, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Success! Data Merged Flawlessly.")
    print(f"✅ Saved as: {OUTPUT_FILE}")
    print(f"Total Sessions Processed: {len(all_sessions_data)}")
    print(f"Total Video Frames: {len(final_df)}")
    print(f"Frames actively typing/clicking: {final_df['Is_Interacting'].sum()}")
else:
    print("\n❌ Error: Scanned 'Honeypot_Sessions' but found no folders containing the 3 required files.")