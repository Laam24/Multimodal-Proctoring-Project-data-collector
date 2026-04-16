import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import wave
import time
import csv
import os
import threading
import ctypes
from pynput import keyboard, mouse

# --- CONFIGURATION ---
OUTPUT_DIR = "Dataset_Raw"
FACE_MODEL_PATH = 'face_landmarker.task'
HAND_MODEL_PATH = 'hand_landmarker.task'

# Audio Settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Windows API
user32 = ctypes.windll.user32

# --- HELPER CLASSES ---

class DataWriter:
    def __init__(self, session_path):
        # Video/Biometric Log - NOW INCLUDES HAND COLUMNS
        self.video_file = open(os.path.join(session_path, 'video_log.csv'), 'w', newline='')
        self.video_writer = csv.writer(self.video_file)
        self.video_writer.writerow([
            'Timestamp', 'Frame_ID', 
            'Gaze_X', 'Gaze_Y', 
            'Head_Yaw', 'Head_Pitch', 'Head_Roll', 
            'Active_Window',
            'Left_Hand_Vis', 'Right_Hand_Vis', # 1 if visible, 0 if not
            'L_Wrist_X', 'L_Wrist_Y',          # Coordinates to track hand position
            'R_Wrist_X', 'R_Wrist_Y'
        ])

        # Interaction Log
        self.input_file = open(os.path.join(session_path, 'interaction_log.csv'), 'w', newline='')
        self.input_writer = csv.writer(self.input_file)
        self.input_writer.writerow(['Timestamp', 'Event_Type', 'Key_Button', 'X', 'Y', 'Action'])

    def log_video(self, timestamp, frame_id, gaze_x, gaze_y, h_yaw, h_pitch, h_roll, window_title, 
                  l_vis, r_vis, l_wx, l_wy, r_wx, r_wy):
        self.video_writer.writerow([
            timestamp, frame_id, gaze_x, gaze_y, h_yaw, h_pitch, h_roll, window_title,
            l_vis, r_vis, l_wx, l_wy, r_wx, r_wy
        ])

    def log_interaction(self, timestamp, event_type, key_button, x, y, action):
        self.input_writer.writerow([timestamp, event_type, key_button, x, y, action])

    def close(self):
        self.video_file.close()
        self.input_file.close()

class AudioRecorder(threading.Thread):
    def __init__(self, session_path):
        super().__init__()
        self.session_path = session_path
        self.recording = True
        self.frames = []
        self.p = pyaudio.PyAudio()

    def run(self):
        try:
            stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            while self.recording:
                data = stream.read(CHUNK)
                self.frames.append(data)
            stream.stop_stream()
            stream.close()
            self.p.terminate()
            wav_path = os.path.join(self.session_path, 'audio.wav')
            wf = wave.open(wav_path, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
        except Exception as e:
            print(f"Audio Error: {e}")

    def stop(self):
        self.recording = False

# --- MAIN LOGIC ---

def get_active_window():
    hwnd = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(hwnd)
    buff = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buff, length + 1)
    return buff.value

def calculate_head_pose(face_landmarks, image_width, image_height):
    # Standard PnP solving
    image_points = np.array([
        (face_landmarks[1].x, face_landmarks[1].y),
        (face_landmarks[152].x, face_landmarks[152].y),
        (face_landmarks[263].x, face_landmarks[263].y),
        (face_landmarks[33].x, face_landmarks[33].y),
        (face_landmarks[287].x, face_landmarks[287].y),
        (face_landmarks[57].x, face_landmarks[57].y)
    ], dtype="double")
    image_points[:, 0] *= image_width
    image_points[:, 1] *= image_height

    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])
    
    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2], rotation_vector, translation_vector, camera_matrix, dist_coeffs

def main():
    print("=== MULTIMODAL DATA COLLECTOR v2 (Face + Hands) ===")
    
    subject_id = input("Enter Subject Name/ID: ").strip()
    exam_mode = input("Enter Exam Mode (keyboard/pen_paper): ").strip()
    task_label = input("Enter Task Label: ").strip()
    
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    session_name = f"{subject_id}_{exam_mode}_{task_label}_{timestamp_str}"
    session_path = os.path.join(OUTPUT_DIR, session_name)
    
    if not os.path.exists(session_path):
        os.makedirs(session_path)
    
    # Initialize Writers
    data_writer = DataWriter(session_path)
    audio_thread = AudioRecorder(session_path)
    
    # Input Listeners
    def on_press(key):
        try: k = key.char
        except: k = str(key)
        data_writer.log_interaction(time.time(), "Keyboard", k, 0, 0, "Press")

    def on_move(x, y):
        data_writer.log_interaction(time.time(), "Mouse", "Move", x, y, "Move")

    def on_click(x, y, button, pressed):
        act = "Pressed" if pressed else "Released"
        data_writer.log_interaction(time.time(), "Mouse", str(button), x, y, act)
        
    k_listener = keyboard.Listener(on_press=on_press)
    m_listener = mouse.Listener(on_move=on_move, on_click=on_click)
    
    # --- MEDIAPIPE SETUP (Dual Models) ---
    mp_tasks = mp.tasks.vision
    BaseOptions = mp.tasks.BaseOptions
    
    # Face Options
    face_options = mp_tasks.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_tasks.RunningMode.VIDEO,
        num_faces=1)
    
    # Hand Options
    hand_options = mp_tasks.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_tasks.RunningMode.VIDEO,
        num_hands=2)
    
    cap = cv2.VideoCapture(0)
    
    print("\nStarting Recording in 3 seconds...")
    time.sleep(3)
    
    audio_thread.start()
    k_listener.start()
    m_listener.start()
    
    frame_idx = 0
    start_time = time.time()
    
    # Create both landmarkers
    with mp_tasks.FaceLandmarker.create_from_options(face_options) as face_landmarker, \
         mp_tasks.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
         
        while True:
            current_time = time.time()
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            mp_timestamp = int((current_time - start_time) * 1000)
            
            # 1. Detect Face
            face_result = face_landmarker.detect_for_video(mp_image, mp_timestamp)
            
            # 2. Detect Hands
            hand_result = hand_landmarker.detect_for_video(mp_image, mp_timestamp)
            
            # --- EXTRACT DATA ---
            # Defaults
            yaw, pitch, roll, gaze_x, gaze_y = 0, 0, 0, 0, 0
            l_vis, r_vis = 0, 0
            l_wx, l_wy, r_wx, r_wy = 0, 0, 0, 0
            
            # Process Face
            if face_result.face_landmarks:
                landmarks = face_result.face_landmarks[0]
                h, w, _ = frame.shape
                gaze_x, gaze_y = landmarks[1].x, landmarks[1].y
                pitch, yaw, roll, rv, tv, cm, dc = calculate_head_pose(landmarks, w, h)
                
                # Draw Axis
                (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rv, tv, cm, dc)
                p1 = (int(landmarks[1].x * w), int(landmarks[1].y * h))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                cv2.line(frame, p1, p2, (255, 0, 0), 2)
            
            # Process Hands
            if hand_result.hand_landmarks:
                for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                    # Check chirality (Left vs Right)
                    # Note: MediaPipe assumes mirrored image by default for chirality
                    handedness = hand_result.handedness[i][0].category_name
                    
                    # Get Wrist Coordinates (Landmark 0)
                    wrist = hand_landmarks[0]
                    
                    if handedness == "Left":
                        l_vis = 1
                        l_wx, l_wy = wrist.x, wrist.y
                        color = (0, 255, 0) # Green for Left
                    else:
                        r_vis = 1
                        r_wx, r_wy = wrist.x, wrist.y
                        color = (0, 0, 255) # Red for Right
                        
                    # Visualization: Draw dots on wrist and finger tips
                    h, w, _ = frame.shape
                    cx, cy = int(wrist.x * w), int(wrist.y * h)
                    cv2.circle(frame, (cx, cy), 5, color, -1)

            # Log Data
            active_window = get_active_window()
            data_writer.log_video(current_time, frame_idx, gaze_x, gaze_y, yaw, pitch, roll, active_window,
                                  l_vis, r_vis, l_wx, l_wy, r_wx, r_wy)
            
            cv2.imshow('Collector v2 (Face+Hands)', frame)
            if cv2.waitKey(1) == ord('q'):
                break
                
    # Cleanup
    audio_thread.stop()
    audio_thread.join()
    k_listener.stop()
    m_listener.stop()
    cap.release()
    cv2.destroyAllWindows()
    data_writer.close()
    
    print(f"\nSession Saved: {session_path}")

if __name__ == "__main__":
    main()