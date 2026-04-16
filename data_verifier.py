import pandas as pd
import cv2
import os

# --- CONFIGURATION ---
DATASET_FILE = "FINAL_TRAINING_DATASET.csv"
SESSIONS_DIR = "Honeypot_Sessions"
FPS = 20 # The FPS of your recorded videos

def verify_session(session_id, session_df):
    print(f"\nVerifying session: {session_id}")
    video_path = os.path.join(SESSIONS_DIR, session_id, "webcam_video.mp4")
    
    if not os.path.exists(video_path):
        print(f"  -> Video not found. Skipping.")
        return

    cap = cv2.VideoCapture(video_path)
    
    for index, row in session_df.iterrows():
        ret, frame = cap.read()
        if not ret:
            break
            
        label = row['Target_Label']
        question = row['Question']
        
        # Determine color based on label
        color = (0, 255, 0) # Green for NORMAL
        if 'CHEAT' in label:
            color = (0, 0, 255) # Red for CHEAT
        elif 'UNKNOWN' in label:
            color = (0, 255, 255) # Yellow for UNKNOWN
        
        # Draw the label and question on the frame
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0,0,0), -1)
        cv2.putText(frame, f"LABEL: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Question: {question}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Data Verifier - Press Q to stop, any other key to advance frame', frame)
        
        key = cv2.waitKey(0) # Wait indefinitely for a key press
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("=== DATASET VISUAL VERIFIER ===")
    
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"Error: {DATASET_FILE} not found. Please run the merger script first.")
        return
        
    # Let the user choose which session to verify
    sessions = df['Session_ID'].unique()
    
    while True:
        print("\nAvailable Sessions to Verify:")
        for i, session in enumerate(sessions):
            print(f"  [{i}] {session}")
        
        try:
            choice = input("Enter the number of the session to verify (or 'q' to quit): ")
            if choice.lower() == 'q':
                break
            
            session_id = sessions[int(choice)]
            session_df = df[df['Session_ID'] == session_id].copy()
            verify_session(session_id, session_df)
            
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid number.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()