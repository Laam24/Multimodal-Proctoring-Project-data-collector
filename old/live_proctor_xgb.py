import tkinter as tk
from tkinter import scrolledtext, messagebox
import cv2
from PIL import Image, ImageTk
import xgboost as xgb
import numpy as np
import mediapipe as mp
import ctypes
import joblib
from collections import deque
import time
from scipy.stats import mode 

# --- CONFIGURATION ---
MODEL_PATH = "proctor_xgboost_v1.json"
ENCODER_PATH = "xgboost_label_encoder.pkl"
FEATURES_PATH = "feature_names.pkl" 
FACE_MODEL_PATH = 'face_landmarker.task'

user32 = ctypes.windll.user32

# --- SENSOR HELPERS ---
def get_active_window_flag():
    try:
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        window_title = buff.value.lower()
        unauth_list = ['google', 'chatgpt', 'search', 'chrome', 'edge', 'firefox', 'bing', 'brave', 'notepad', 'opera']
        return 1.0 if any(word in window_title for word in unauth_list) else 0.0
    except: return 0.0

def calculate_head_pose(face_landmarks, img_w, img_h):
    try:
        # Landmarks for PnP: Nose, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth
        image_points = np.array([
            (face_landmarks[1].x * img_w, face_landmarks[1].y * img_h),
            (face_landmarks[152].x * img_w, face_landmarks[152].y * img_h),
            (face_landmarks[33].x * img_w, face_landmarks[33].y * img_h),
            (face_landmarks[263].x * img_w, face_landmarks[263].y * img_h),
            (face_landmarks[57].x * img_w, face_landmarks[57].y * img_h),
            (face_landmarks[287].x * img_w, face_landmarks[287].y * img_h)
        ], dtype="double")
        
        model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])
        
        camera_matrix = np.array([[img_w, 0, img_w/2],[0, img_w, img_h/2], [0, 0, 1]], dtype="double")
        _, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = np.degrees(np.arcsin(rmat[2, 1])), np.degrees(np.arctan2(-rmat[2, 0], rmat[2, 2])), np.degrees(np.arctan2(-rmat[0, 1], rmat[1, 1]))
        return pitch, yaw, roll
    except: return 0.0, 0.0, 0.0

class LiveProctorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Calibrated XGBoost Proctoring")
        self.root.geometry("1000x900")
        
        # 1. State & Calibration Buffers
        self.is_running = False
        self.is_calibrating = False
        self.calib_frames = []
        self.offsets = {"p": 0, "y": 0, "r": 0, "gx": 0, "gy": 0}
        
        self.cap = None
        self.start_time = None
        self.frame_counter = 0
        self.prediction_buffer = deque(maxlen=20) # Increased for stability
        self.raw_data_history = deque(maxlen=60) 

        # UI (Same as before)
        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(pady=10)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        self.start_btn = tk.Button(btn_frame, text="Start & Calibrate", command=self.start_session, bg="green", fg="white", font=("Arial", 12, "bold"))
        self.start_btn.pack(side=tk.LEFT, padx=10)
        self.stop_btn = tk.Button(btn_frame, text="Stop Session", command=self.stop_session, bg="orange", fg="white", font=("Arial", 12, "bold"), state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        self.exam_box = scrolledtext.ScrolledText(root, height=10, font=("Arial", 12))
        self.exam_box.pack(padx=20, pady=10, fill=tk.X)

        # Load Models (XGBoost logic)
        self.model = xgb.XGBClassifier()
        self.model.load_model(MODEL_PATH)
        self.le = joblib.load(ENCODER_PATH)
        self.feature_names = joblib.load(FEATURES_PATH)
        
        mp_tasks = mp.tasks.vision
        options = mp_tasks.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_tasks.RunningMode.VIDEO, num_faces=1)
        self.landmarker = mp_tasks.FaceLandmarker.create_from_options(options)

    def start_session(self):
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.is_calibrating = True
        self.calib_frames = []
        self.start_time = time.time()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        print(">>> CALIBRATING: LOOK STRAIGHT AT THE SCREEN <<<")
        self.update_video()
    
    def stop_session(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.video_label.config(image='')
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)


    def update_video(self):
        if not self.is_running or not self.cap.isOpened(): return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # 1. Get Raw Data
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.landmarker.detect_for_video(mp_img, int((time.time() - self.start_time) * 1000))
            unauth = get_active_window_flag()
            
            p, y, r, gx, gy = 0.0, 0.0, 0.0, 0.0, 0.0
            face_det = 0.0
            if result.face_landmarks:
                face_det = 1.0
                lms = result.face_landmarks[0]
                gx, gy = lms[1].x, lms[1].y
                p, y, r = calculate_head_pose(lms, w, h)

            # --- 2. CALIBRATION LOGIC ---
            if self.is_calibrating:
                if len(self.calib_frames) < 50: # Collect 50 frames of "Normal"
                    self.calib_frames.append([p, y, r, gx, gy])
                    cv2.putText(frame, f"CALIBRATING... {len(self.calib_frames)}/50", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    # Calculate offsets to make this pose equal to your "CSV Normal"
                    avg_p, avg_y, avg_r, avg_gx, avg_gy = np.mean(self.calib_frames, axis=0)
                    # We want current_pose - offset to match CSV's "Normal" values (e.g. Pitch 14)
                    self.offsets = {"p": avg_p - 14.0, "y": avg_y - 5.0, "r": avg_r - 176.0, "gx": avg_gx - 0.6, "gy": avg_gy - 0.6}
                    self.is_calibrating = False
                    print("✅ CALIBRATION COMPLETE")
            
            # --- 3. APPLY OFFSETS ---
            p -= self.offsets["p"]
            y -= self.offsets["y"]
            r -= self.offsets["r"]
            gx -= self.offsets["gx"]
            gy -= self.offsets["gy"]

            # 4. Math & Prediction (Velocity/Variance)
            self.raw_data_history.append([p, y, r, gx, gy])
            hist = np.array(self.raw_data_history)
            p_v, y_v, r_v, gx_v, gy_v = (hist[-1]-hist[-2]) if len(hist)>1 else (0,0,0,0,0)
            p_var, y_var, r_var, gx_var, gy_var = np.var(hist, axis=0) if len(hist)>1 else (0,0,0,0,0)

            if not self.is_calibrating:
                current_row = np.array([[self.frame_counter, face_det, p, y, r, gx, gy, p_v, y_v, r_v, gx_v, gy_v, p_var, y_var, r_var, gx_var, gy_var, unauth]])
                probs = self.model.predict_proba(current_row)[0]
                pred_idx = np.argmax(probs)
                conf = probs[pred_idx] * 100
                
                # SENSITIVITY FIX: Only accept a "Cheat" if the AI is 95%+ sure
                if pred_idx != 2 and conf < 95.0: # If it's not Normal but confidence is low
                    self.prediction_buffer.append(2) # Default back to NORMAL
                else:
                    self.prediction_buffer.append(pred_idx)
                
                smoothed_idx = mode(self.prediction_buffer, keepdims=True)[0][0]
                label = self.le.inverse_transform([smoothed_idx])[0]

                # HUD
                color = (0, 255, 0) if label == "NORMAL" else (0, 0, 255)
                cv2.rectangle(frame, (10, 10), (550, 100), (0,0,0), -1)
                cv2.putText(frame, f"STATUS: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(frame, f"CONFIDENCE: {conf:.1f}%", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Display
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        self.root.after(15, self.update_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveProctorApp(root)
    root.mainloop()