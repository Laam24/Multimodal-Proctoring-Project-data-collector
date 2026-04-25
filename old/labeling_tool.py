import pandas as pd
import cv2
import os
import csv
import time

# --- CONFIGURATION ---
SESSIONS_DIR = "Honeypot_Sessions"
JUMP_SECONDS = 5 # How many seconds to jump forward/backward

def get_session_start_time(session_path):
    # ... (unchanged)
    events_file = os.path.join(session_path, "exam_events.csv")
    if not os.path.exists(events_file): return None
    try:
        with open(events_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[1] == "SESSION_START":
                    return float(row[0])
    except: return None
    return None

def run_labeler(session_id):
    session_path = os.path.join(SESSIONS_DIR, session_id)
    webcam_path = os.path.join(session_path, "webcam_video.mp4")
    phone_path = os.path.join(session_path, "phone_video.mp4")
    output_path = os.path.join(session_path, "events_labeled.csv")

    if not (os.path.exists(webcam_path) and os.path.exists(phone_path)):
        print("Error: Video files not found."); return

    cap_webcam = cv2.VideoCapture(webcam_path)
    cap_phone = cv2.VideoCapture(phone_path)
    
    start_time = get_session_start_time(session_path)
    if start_time is None: print("Error: No session start time."); return

    labeled_events = [(start_time, "NORMAL")]
    is_playing = False
    
    # --- THE FIX: TIME-BASED PLAYBACK ---
    current_time_sec = 0.0
    webcam_fps = cap_webcam.get(cv2.CAP_PROP_FPS)
    if webcam_fps == 0: webcam_fps = 20.0 # Default if FPS is not readable

    print("\n--- SYNCHRONIZED LABELING TOOL (v3) ---")
    print("  [SPACE]: Play/Pause | [A/S]: Rewind/Forward")
    print("  [C]: Cheat(Physical) | [D]: Cheat(Digital) | [N]: Normal")
    print("  [Q]: Quit & Save")
    print("-" * 40)

    while cap_webcam.isOpened():
        # If playing, advance the master clock
        if is_playing:
            current_time_sec += 1.0 / webcam_fps

        # Seek both videos to the master clock time
        cap_webcam.set(cv2.CAP_PROP_POS_MSEC, current_time_sec * 1000)
        cap_phone.set(cv2.CAP_PROP_POS_MSEC, current_time_sec * 1000)
        
        ret_web, frame_web = cap_webcam.read()
        ret_phone, frame_phone = cap_phone.read()

        if not ret_web: break
        if not ret_phone: frame_phone = np.zeros((360, 480, 3), dtype=np.uint8)

        # Display Logic
        current_timestamp = start_time + current_time_sec
        current_label = labeled_events[-1][1] if labeled_events else "NORMAL"
        color = (0, 0, 255) if "CHEAT" in current_label else (0, 255, 0)
        time_str = time.strftime('%M:%S', time.gmtime(current_time_sec))
        
        cv2.rectangle(frame_web, (0, 0), (frame_web.shape[1], 40), (0,0,0), -1)
        cv2.putText(frame_web, f"Time: {time_str} | Status: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Webcam View (Control Window)", frame_web)
        cv2.imshow("Phone Workspace View", cv2.resize(frame_phone, (480, 360)))

        # Keyboard Controls
        wait_time = int(1000 / webcam_fps) if is_playing else 0
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'): break
        elif key == ord(' '): is_playing = not is_playing
        elif key == ord('c'): labeled_events.append((current_timestamp, "CHEAT_PHYSICAL")); print(f"Logged CHEAT_PHYSICAL at {time_str}")
        elif key == ord('d'): labeled_events.append((current_timestamp, "CHEAT_DIGITAL")); print(f"Logged CHEAT_DIGITAL at {time_str}")
        elif key == ord('n'): labeled_events.append((current_timestamp, "NORMAL")); print(f"Logged NORMAL at {time_str}")
        elif key == ord('a'): # Rewind
            current_time_sec = max(0, current_time_sec - JUMP_SECONDS)
        elif key == ord('s'): # Forward
            current_time_sec += JUMP_SECONDS

    # Save Logic
    cap_webcam.release(); cap_phone.release(); cv2.destroyAllWindows()
    if labeled_events:
        labeled_events.sort(key=lambda x: x[0])
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Label'])
            for ts, label in labeled_events:
                writer.writerow([f"{ts:.4f}", label])
        print(f"\n✅ Success! Labeled events saved to {output_path}")

def main():
    sessions = [d for d in os.listdir(SESSIONS_DIR) if os.path.isdir(os.path.join(SESSIONS_DIR, d))]
    while True:
        print("\nAvailable Sessions to Label:")
        for i, session in enumerate(sessions): print(f"  [{i}] {session}")
        choice = input("Enter the number of the session to label (or 'q' to quit): ")
        if choice.lower() == 'q': break
        try:
            session_id = sessions[int(choice)]
            run_labeler(session_id)
        except (ValueError, IndexError): print("Invalid input.")

if __name__ == "__main__":
    main()