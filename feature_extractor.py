import os
import cv2
import csv
import numpy as np
import mediapipe as mp

BASE_DIR = "Honeypot_Sessions"
FACE_MODEL_PATH = 'face_landmarker.task'
FPS = 20.0  

def calculate_head_pose(face_landmarks, img_w, img_h):
    """Calculates 3D Head Pitch, Yaw, and Roll from 2D landmarks."""
    image_points = np.array([
        (face_landmarks[1].x * img_w, face_landmarks[1].y * img_h),     
        (face_landmarks[152].x * img_w, face_landmarks[152].y * img_h), 
        (face_landmarks[263].x * img_w, face_landmarks[263].y * img_h), 
        (face_landmarks[33].x * img_w, face_landmarks[33].y * img_h),   
        (face_landmarks[287].x * img_w, face_landmarks[287].y * img_h), 
        (face_landmarks[57].x * img_w, face_landmarks[57].y * img_h)    
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])

    camera_matrix = np.array([[img_w, 0, img_w/2],[0, img_w, img_h/2],[0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rmat, _ = cv2.Rodrigues(rotation_vector)
    pitch = np.arcsin(rmat[2, 1])
    yaw = np.arctan2(-rmat[2, 0], rmat[2, 2])
    roll = np.arctan2(-rmat[0, 1], rmat[1, 1])

    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

def calculate_true_iris_gaze(landmarks):
    """Calculates True Eye Gaze using Iris-to-Sclera ratios."""
    try:
        # Right Eye (viewer's left)
        r_inner = landmarks[362].x
        r_outer = landmarks[263].x
        r_iris = landmarks[473].x
        
        # Left Eye (viewer's right)
        l_inner = landmarks[133].x
        l_outer = landmarks[33].x
        l_iris = landmarks[468].x
        
        # Horizontal Ratios (0.0 to 1.0)
        r_width = abs(r_inner - r_outer) + 1e-6
        r_ratio = abs(r_iris - r_outer) / r_width
        
        l_width = abs(l_inner - l_outer) + 1e-6
        l_ratio = abs(l_iris - l_outer) / l_width
        
        # Vertical Ratios (Using Top/Bottom Eyelids)
        r_top = landmarks[386].y
        r_bottom = landmarks[374].y
        r_iris_y = landmarks[473].y
        r_height = abs(r_bottom - r_top) + 1e-6
        r_y_ratio = abs(r_iris_y - r_top) / r_height
        
        l_top = landmarks[159].y
        l_bottom = landmarks[145].y
        l_iris_y = landmarks[468].y
        l_height = abs(l_bottom - l_top) + 1e-6
        l_y_ratio = abs(l_iris_y - l_top) / l_height
        
        gaze_x = (r_ratio + l_ratio) / 2.0
        gaze_y = (r_y_ratio + l_y_ratio) / 2.0
        
        return gaze_x, gaze_y
    except IndexError:
        # Fallback if Iris tracking fails
        return 0.5, 0.5

def get_session_start_time(session_path):
    events_file = os.path.join(session_path, "exam_events.csv")
    if not os.path.exists(events_file):
        return None
    with open(events_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
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
        print(f"  -> Skipping. No SESSION_START.")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Frame', 'Face_Detected', 'Pitch', 'Yaw', 'Roll', 'Gaze_X', 'Gaze_Y'])

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                frame_timestamp = start_time + (frame_count / FPS)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(mp_image, int(frame_count * (1000/FPS)))

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    img_h, img_w, _ = frame.shape
                    
                    # --- THE IRIS FIX INJECTED HERE ---
                    gaze_x, gaze_y = calculate_true_iris_gaze(landmarks)
                    pitch, yaw, roll = calculate_head_pose(landmarks, img_w, img_h)
                    
                    writer.writerow([f"{frame_timestamp:.3f}", frame_count, 1, f"{pitch:.2f}", f"{yaw:.2f}", f"{roll:.2f}", f"{gaze_x:.4f}", f"{gaze_y:.4f}"])
                else:
                    writer.writerow([f"{frame_timestamp:.3f}", frame_count, 0, "", "", "", "", ""])
                
                frame_count += 1
                if frame_count % 200 == 0:
                    print(f"  -> Extracted {frame_count}/{total_frames} frames...", end='\r')

        cap.release()
        print(f"  -> Done! Extracted {frame_count} frames. Saved to visual_features.csv\n")

def main():
    print("=== TRUE IRIS VISUAL FEATURE EXTRACTOR ===")
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