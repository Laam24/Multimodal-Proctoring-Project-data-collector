import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
BASE_DIR = "Honeypot_Sessions"
LABELS_FILE = "master_labels.csv"
OUTPUT_FILE = "FINAL_TRAINING_DATASET.csv"

# Words that indicate a student is cheating digitally
UNAUTHORIZED_WINDOWS =['google', 'chatgpt', 'search', 'chrome', 'edge', 'firefox', 'bing', 'brave', 'notepad']

def is_unauthorized(window_title):
    if not isinstance(window_title, str): 
        return 0
    window_lower = window_title.lower()
    for word in UNAUTHORIZED_WINDOWS:
        if word in window_lower:
            return 1
    return 0

def main():
    print("=== GRAND DATASET MERGER ===")
    
    # 1. Load Master Labels into a Dictionary
    print(f"Loading {LABELS_FILE}...")
    labels_dict = {}
    try:
        labels_df = pd.read_csv(LABELS_FILE)
        for index, row in labels_df.iterrows():
            session = row['Session_Folder']
            q_num = row['Question']
            label = row['Standard_Label']
            if session not in labels_dict:
                labels_dict[session] = {}
            labels_dict[session][q_num] = label
    except Exception as e:
        print(f"Error reading {LABELS_FILE}: {e}")
        return

    all_session_data =[]

    # 2. Process Every Session Folder
    for session_folder in os.listdir(BASE_DIR):
        session_path = os.path.join(BASE_DIR, session_folder)
        if not os.path.isdir(session_path): continue

        vis_csv = os.path.join(session_path, "visual_features.csv")
        inp_csv = os.path.join(session_path, "inputs.csv")
        ev_csv = os.path.join(session_path, "exam_events.csv")

        if not (os.path.exists(vis_csv) and os.path.exists(inp_csv) and os.path.exists(ev_csv)):
            continue

        print(f"Merging Data for: {session_folder}")

        # Load DataFrames
        vis_df = pd.read_csv(vis_csv)
        inp_df = pd.read_csv(inp_csv)
        ev_df = pd.read_csv(ev_csv)

        # Ensure timestamps are numeric
        vis_df['Timestamp'] = pd.to_numeric(vis_df['Timestamp'])
        inp_df['Timestamp'] = pd.to_numeric(inp_df['Timestamp'])
        ev_df['Timestamp'] = pd.to_numeric(ev_df['Timestamp'])

        # 3. Parse Question Start/End Times
        q_times = {}
        for i, row in ev_df.iterrows():
            if row['Event'] == 'QUESTION_START':
                q_times[row['Details']] = {'start': row['Timestamp'], 'end': 9999999999.0}
            elif row['Event'] == 'QUESTION_END':
                if row['Details'] in q_times:
                    q_times[row['Details']]['end'] = row['Timestamp']

        # 4. Sort timelines for accurate merging
        vis_df = vis_df.sort_values('Timestamp')
        inp_df = inp_df.sort_values('Timestamp')

        # 5. Bring in the "Active_Window" from inputs.csv
        # We drop empty rows and use merge_asof to carry the last known window forward to the video frames
        valid_windows = inp_df.dropna(subset=['Active_Window'])[['Timestamp', 'Active_Window']]
        if not valid_windows.empty:
            vis_df = pd.merge_asof(vis_df, valid_windows, on='Timestamp', direction='backward')
        else:
            vis_df['Active_Window'] = "Unknown"

        # 6. Check for Unauthorized Windows
        vis_df['Unauthorized_Window'] = vis_df['Active_Window'].apply(is_unauthorized)

        # 7. Apply the Ground Truth Labels
        def assign_label(t):
            for q, times in q_times.items():
                if times['start'] <= t <= times['end']:
                    try:
                        return q, labels_dict[session_folder][q]
                    except KeyError:
                        return q, "UNKNOWN"
            return "IDLE", "TRANSITION" # If they are between questions

        results = vis_df['Timestamp'].apply(assign_label)
        vis_df['Question'] = [res[0] for res in results]
        vis_df['Target_Label'] =[res[1] for res in results]

        # 8. HARD OVERRIDE: If an unauthorized window is open during a question, it is DIGITAL CHEATING.
        vis_df.loc[(vis_df['Unauthorized_Window'] == 1) & (vis_df['Question'] != 'IDLE'), 'Target_Label'] = 'CHEAT_DIGITAL'

        # Add the Session ID so the AI knows which user this is
        vis_df.insert(0, 'Session_ID', session_folder)
        all_session_data.append(vis_df)

    # 9. Combine all users into one mega-dataset
    if all_session_data:
        final_df = pd.concat(all_session_data, ignore_index=True)
        # Drop transition/idle frames so we only train on pure exam behavior
        final_df = final_df[final_df['Target_Label'] != 'TRANSITION']
        
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n=====================================")
        print(f"✅ SUCCESS! Final dataset saved to: {OUTPUT_FILE}")
        print(f"Total Training Rows (Frames): {len(final_df)}")
        print(f"=====================================")
    else:
        print("No valid data found to merge.")

if __name__ == "__main__":
    main()