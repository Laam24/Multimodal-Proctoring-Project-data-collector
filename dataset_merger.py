import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
BASE_DIR = "Honeypot_Sessions"
OUTPUT_FILE = "FINAL_TRAINING_DATASET.csv"

# Words that indicate a student is cheating digitally
UNAUTHORIZED_WINDOWS = ['google', 'chatgpt', 'search', 'chrome', 'edge', 'firefox', 'bing', 'brave', 'notepad', 'opera']

def is_unauthorized(window_title):
    if pd.isna(window_title) or not isinstance(window_title, str): 
        return 0
    window_lower = window_title.lower()
    for word in UNAUTHORIZED_WINDOWS:
        if word in window_lower:
            return 1
    return 0

def main():
    print("=== GRAND DATASET MERGER (V1.5 - WITH FEATURE ENGINEERING) ===")
    all_session_data = []

    # Get all valid session folders
    sessions = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    for session in sessions:
        session_path = os.path.join(BASE_DIR, session)
        vis_file = os.path.join(session_path, "visual_features.csv")
        labels_file = os.path.join(session_path, "events_labeled.csv")
        inputs_file = os.path.join(session_path, "inputs.csv")
        
        # Check if all required files exist
        if not (os.path.exists(vis_file) and os.path.exists(labels_file) and os.path.exists(inputs_file)):
            print(f"Skipping {session}: Missing one or more required CSV files.")
            continue
            
        print(f"Processing: {session}")
        
        # 1. Load Visual Features (The Master 20 FPS Timeline)
        vis_df = pd.read_csv(vis_file)
        vis_df['Timestamp'] = vis_df['Timestamp'].astype(float)
        vis_df = vis_df.sort_values('Timestamp')
        
        # --- NEW: FEATURE ENGINEERING (Velocity & Variance) ---
        # 1. Velocity (Delta): How fast is the posture/gaze changing per frame?
        # A fast spike means a sudden head turn or eye dart.
        features_to_track = ['Pitch', 'Yaw', 'Roll', 'Gaze_X', 'Gaze_Y']
        for col in features_to_track:
            if col in vis_df.columns:
                vis_df[f'{col}_Velocity'] = vis_df[col].diff().fillna(0)
        
        # 2. Variance (Volatility): How much are they moving over a 3-second (60 frame) window?
        # High Variance = Active typing. Low Variance = Frozen/Reading a cheat sheet.
        for col in features_to_track:
            if col in vis_df.columns:
                vis_df[f'{col}_Variance'] = vis_df[col].rolling(window=60, min_periods=1).var().fillna(0)
        # ------------------------------------------------------
        
        # 2. Load and merge the Ground Truth Labels
        labels_df = pd.read_csv(labels_file)
        labels_df['Timestamp'] = labels_df['Timestamp'].astype(float)
        labels_df = labels_df.sort_values('Timestamp')
        
        # pd.merge_asof assigns the last known label to each visual frame
        vis_df = pd.merge_asof(vis_df, labels_df, on='Timestamp', direction='backward')
        
        # Rename 'Label' to 'Target_Label' to match the PyTorch GRU script
        vis_df = vis_df.rename(columns={'Label': 'Target_Label'})
        
        # 3. Load and merge Inputs (for Active Window tracking)
        inputs_df = pd.read_csv(inputs_file)
        inputs_df['Timestamp'] = inputs_df['Timestamp'].astype(float)
        
        # We only care about the Active_Window status
        window_df = inputs_df[['Timestamp', 'Active_Window']].dropna().sort_values('Timestamp')
        
        # pd.merge_asof assigns the last clicked window to each visual frame
        if not window_df.empty:
            vis_df = pd.merge_asof(vis_df, window_df, on='Timestamp', direction='backward')
            vis_df['Unauthorized_Window'] = vis_df['Active_Window'].apply(is_unauthorized)
            vis_df = vis_df.drop(columns=['Active_Window'])
        else:
            vis_df['Unauthorized_Window'] = 0
            
        # 4. Cleanup and Formatting
        # If there are frames before the very first labeled event, default them to NORMAL
        vis_df['Target_Label'] = vis_df['Target_Label'].fillna("NORMAL")
        
        # Hard Override: If an unauthorized window is actively focused, force DIGITAL CHEATING
        vis_df.loc[vis_df['Unauthorized_Window'] == 1, 'Target_Label'] = 'CHEAT_DIGITAL'
        
        # Insert Session ID so the AI model knows where the data came from
        vis_df.insert(0, 'Session_ID', session)
        
        all_session_data.append(vis_df)

    if all_session_data:
        # Combine all processed sessions into one massive dataset
        final_df = pd.concat(all_session_data, ignore_index=True)
        
        # Save to disk
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n=====================================")
        print(f"✅ SUCCESS! Final feature-engineered dataset saved to: {OUTPUT_FILE}")
        print(f"Total Frames Processed: {len(final_df)}")
        print("\nGround Truth Class Distribution:")
        print(final_df['Target_Label'].value_counts())
    else:
        print("❌ ERROR: No complete sessions found to merge.")

if __name__ == "__main__":
    main()