import os
import cv2
import csv
import numpy as np
import mediapipe as mp

BASE_DIR = "Honeypot_Sessions"
FACE_MODEL_PATH = 'face_landmarker.task'
FPS = 20.0  # The framerate we used to record

def calculate_head_pose(face_landmarks, img_w, img_h):
    """Calculates 3D Head Pitch, Yaw, and Roll from 2D landmarks."""
    image_points = np.array([
        (face_landmarks[1].x * img_w, face_landmarks[1].y * img_h),     # Nose tip
        (face_landmarks[152].x * img_w, face_landmarks[152].y * img_h), # Chin
        (face_landmarks[263].x * img_w, face_landmarks[263].y * img_h), # Left eye left corner
        (face_landmarks[33].x * img_w, face_landmarks[33].y * img_h),   # Right eye right corner
        (face_landmarks[287].x * img_w, face_landmarks[287].y * img_h), # Left Mouth corner
        (face_landmarks[57].x * img_w, face_landmarks[57].y * img_h)    # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])

    camera_matrix = np.array([[img_w, 0, img_w/2],[0, img_w, img_h/2],[0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles (Pitch, Yaw, Roll)
    pitch = np.arcsin(rmat[2, 1])
    yaw = np.arctan2(-rmat[2, 0], rmat[2, 2])
    roll = np.arctan2(-rmat[0, 1], rmat[1, 1])

    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

def get_session_start_time(session_path):
    """Reads exam_events.csv to find the exact Unix timestamp when recording started."""
    events_file = os.path.join(session_path, "exam_events.csv")
    if not os.path.exists(events_file):
        return None
    with open(events_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            if row[1] == "SESSION_START":
                return float(row[0])
    return None

def process_session(session_folder, options):
    session_path = os.path.join(BASE_DIR, session_folder)
    video_path = os.path.join(session_path, "webcam_video.mp4")
    output_csv = os.path.join(session_path, "visual_features.csv")

    if not os.path.exists(video_path):
        return

    print(f"Processing: {session_folder}")
    
    start_time = get_session_start_time(session_path)
    if start_time is None:
        print(f"  -> Skipping. No SESSION_START found in exam_events.csv")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- THE FIX: Create a fresh AI model for EVERY video so the timer resets ---
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Frame', 'Face_Detected', 'Pitch', 'Yaw', 'Roll', 'Gaze_X', 'Gaze_Y'])

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Calculate the exact timestamp for this frame
                frame_timestamp = start_time + (frame_count / FPS)
                
                # Convert for MediaPipe
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # We use fake increasing ms timestamps for the video mode
                result = landmarker.detect_for_video(mp_image, int(frame_count * (1000/FPS)))

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    img_h, img_w, _ = frame.shape
                    
                    # Approximate Gaze using Nose Tip (Landmark 1) normalized coordinates
                    gaze_x, gaze_y = landmarks[1].x, landmarks[1].y
                    
                    pitch, yaw, roll = calculate_head_pose(landmarks, img_w, img_h)
                    
                    writer.writerow([f"{frame_timestamp:.3f}", frame_count, 1, f"{pitch:.2f}", f"{yaw:.2f}", f"{roll:.2f}", f"{gaze_x:.4f}", f"{gaze_y:.4f}"])
                else:
                    # Face lost (e.g. looked totally down or left frame)
                    writer.writerow([f"{frame_timestamp:.3f}", frame_count, 0, "", "", "", "", ""])
                
                frame_count += 1
                if frame_count % 200 == 0:
                    print(f"  -> Extracted {frame_count}/{total_frames} frames...", end='\r')

        cap.release()
        print(f"  -> Done! Extracted {frame_count} frames. Saved to visual_features.csv\n")

def main():
    print("=== VISUAL FEATURE EXTRACTOR ===")
    
    # Initialize Options once
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1)
    
    for folder in os.listdir(BASE_DIR):
        if os.path.isdir(os.path.join(BASE_DIR, folder)):
            process_session(folder, options)

    print("All sessions processed successfully!")

if __name__ == "__main__":
    main()