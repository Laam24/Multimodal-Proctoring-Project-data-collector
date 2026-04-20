import tkinter as tk
from tkinter import scrolledtext, messagebox
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import ctypes
import joblib
from collections import deque
import time
from scipy.stats import mode
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_PATH = "behavioral_engine_v1.pth"
SCALER_PATH = "behavioral_scaler.pkl"
FACE_MODEL_PATH = 'face_landmarker.task'
LABEL_MAP = {0: 'NORMAL', 1: 'CHEAT_PHYSICAL', 2: 'CHEAT_DIGITAL'}

user32 = ctypes.windll.user32

# --- PILLAR 3: THE BEHAVIORAL ENGINE ---
class BehavioralEngine(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, num_layers=2):
        super(BehavioralEngine, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        return self.fc(lstm_out[:, -1, :])

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

# --- MAIN APP ---
class LiveProctorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Proctoring AI (YOLO + BiLSTM)")
        self.root.geometry("1000x900")
        
        # 1. State & Calibration Buffers
        self.is_running = False
        self.is_calibrating = False
        self.calib_frames = []
        self.offsets = {"p": 0, "y": 0, "r": 0, "gx": 0, "gy": 0}
        
        self.cap = None
        self.start_time = None
        self.frame_counter = 0
        
        self.raw_data_history = deque(maxlen=60)
        self.sequence_buffer = deque(maxlen=60)
        self.prediction_buffer = deque(maxlen=15)

        # 2. UI Layout
        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(pady=10)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        self.start_btn = tk.Button(btn_frame, text="Start Session", command=self.start_session, bg="green", fg="white", font=("Arial", 12, "bold"), width=15)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        self.stop_btn = tk.Button(btn_frame, text="Stop Session", command=self.stop_session, bg="orange", fg="white", font=("Arial", 12, "bold"), state=tk.DISABLED, width=15)
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        tk.Label(root, text="Exam Answer Sheet:", font=("Arial", 10)).pack(pady=(10,0))
        self.exam_box = scrolledtext.ScrolledText(root, height=10, font=("Arial", 12))
        self.exam_box.pack(padx=20, pady=10, fill=tk.X)

        # 3. Load All 3 Pillars
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Pillar 2: YOLO Sentinel
            print("Loading YOLOv8n Object Sentinel...")
            self.yolo_model = YOLO('yolov8n.pt')
            self.yolo_model.to(self.device)
            
            # Pillar 3: Behavioral Engine
            print("Loading 1D-CNN+BiLSTM Behavioral Engine...")
            self.scaler = joblib.load(SCALER_PATH)
            self.model = BehavioralEngine(input_size=17).to(self.device)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval()
            
            # Vision Extractor
            mp_tasks = mp.tasks.vision
            options = mp_tasks.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH),
                running_mode=mp_tasks.RunningMode.VIDEO, num_faces=1)
            self.landmarker = mp_tasks.FaceLandmarker.create_from_options(options)
            print("✅ Hybrid System Online and Ready.")
        except Exception as e:
            messagebox.showerror("Error", f"Initialization failed: {e}")
            self.root.destroy()

    def start_session(self):
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.is_calibrating = True
        self.calib_frames = []
        self.start_time = time.time()
        self.frame_counter = 0
        
        self.raw_data_history.clear()
        self.sequence_buffer.clear()
        self.prediction_buffer.clear()
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_video()

    def stop_session(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.video_label.config(image='')
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def update_video(self):
        if not self.is_running or not self.cap.isOpened(): return

        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Setup HUD Base
                cv2.rectangle(frame, (10, 10), (550, 100), (0,0,0), -1)

                # --- PILLAR 1: DIGITAL AUDITOR (Absolute Priority) ---
                unauth = get_active_window_flag()
                if unauth == 1.0:
                    cv2.putText(frame, "STATUS: CHEAT_DIGITAL", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(frame, "CONFIDENCE: 100.0% (OS LEVEL)", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, "UNAUTHORIZED WINDOW DETECTED!", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    for _ in range(5): self.prediction_buffer.append(2) # Flood buffer
                    self._render_frame(frame)
                    self.root.after(15, self.update_video)
                    return # Skip AI completely

                # --- PILLAR 2: YOLO SENTINEL ---
                # Class 67 is cell phone in COCO dataset
                results = self.yolo_model(frame, classes=[67], verbose=False)
                phone_detected = False
                for r in results:
                    if len(r.boxes) > 0:
                        phone_detected = True
                        # Draw bounding box for proof
                        x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        break
                
                if phone_detected:
                    cv2.putText(frame, "STATUS: CHEAT_PHYSICAL", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(frame, "CONFIDENCE: 99.0% (YOLO OBJECT)", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    for _ in range(5): self.prediction_buffer.append(1)
                    self._render_frame(frame)
                    self.root.after(15, self.update_video)
                    return # Skip Behavioral AI

                # --- PILLAR 3: BEHAVIORAL ENGINE ---
                # 1. Feature Extraction
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp = int((time.time() - self.start_time) * 1000)
                result = self.landmarker.detect_for_video(mp_img, timestamp)
                
                face_det, p, y, r, gx, gy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                if result.face_landmarks:
                    face_det = 1.0
                    lms = result.face_landmarks[0]
                    gx, gy = lms[1].x, lms[1].y
                    p, y, r = calculate_head_pose(lms, w, h)

                # 2. CALIBRATION PHASE (Fixing Covariate Shift)
                if self.is_calibrating:
                    if len(self.calib_frames) < 60:
                        self.calib_frames.append([p, y, r, gx, gy])
                        cv2.putText(frame, f"CALIBRATING BASELINE... {len(self.calib_frames)}/60", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                        cv2.putText(frame, "SIT STILL AND LOOK AT SCREEN", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        # Calculate personal offsets
                        avg_p, avg_y, avg_r, avg_gx, avg_gy = np.mean(self.calib_frames, axis=0)
                        self.offsets = {"p": avg_p, "y": avg_y, "r": avg_r, "gx": avg_gx, "gy": avg_gy}
                        self.is_calibrating = False
                        print(f"✅ Baseline Set - P:{avg_p:.1f}, Y:{avg_y:.1f}")
                    
                    self._render_frame(frame)
                    self.root.after(15, self.update_video)
                    return
                
                # 3. Apply Relative Offsets
                p -= self.offsets["p"]
                y -= self.offsets["y"]
                r -= self.offsets["r"]
                gx -= self.offsets["gx"]
                gy -= self.offsets["gy"]

                # 4. Physics Engine (Velocity & Variance)
                self.raw_data_history.append([p, y, r, gx, gy])
                hist = np.array(self.raw_data_history)
                
                p_v, y_v, r_v, gx_v, gy_v = 0.0, 0.0, 0.0, 0.0, 0.0
                p_var, y_var, r_var, gx_var, gy_var = 0.0, 0.0, 0.0, 0.0, 0.0

                if len(hist) > 1:
                    p_v, y_v, r_v, gx_v, gy_v = hist[-1] - hist[-2]
                    p_var, y_var, r_var, gx_var, gy_var = np.var(hist, axis=0)

                # 5. Assemble the 17-Feature Vector
                current_features = [
                    face_det, p, y, r, gx, gy,
                    p_v, y_v, r_v, gx_v, gy_v,
                    p_var, y_var, r_var, gx_var, gy_var, unauth
                ]
                self.sequence_buffer.append(current_features)

                # 6. BiLSTM Inference
                if len(self.sequence_buffer) == 60:
                    seq_array = np.array(self.sequence_buffer)
                    seq_scaled = self.scaler.transform(seq_array)
                    tensor_in = torch.FloatTensor(seq_scaled).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(tensor_in)
                        probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
                        
                    pred_idx = np.argmax(probs)
                    self.prediction_buffer.append(pred_idx)
                    
                    # UI Smoothing
                    mode_result = mode(self.prediction_buffer, keepdims=True)
                    smoothed_idx = int(mode_result.mode[0])
                    
                    label = LABEL_MAP[smoothed_idx]
                    conf = probs[smoothed_idx] * 100

                    color = (0, 255, 0) if label == "NORMAL" else (0, 165, 255) # Orange for behavior cheat
                    cv2.putText(frame, f"STATUS: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    cv2.putText(frame, f"CONFIDENCE: {conf:.1f}% (BEHAVIOR)", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    loading_text = f"LOADING AI MEMORY: {len(self.sequence_buffer)}/60"
                    cv2.putText(frame, loading_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                self._render_frame(frame)

        except Exception as e:
            print(f"CRITICAL ERROR IN VIDEO LOOP: {e}")
            
        self.root.after(15, self.update_video)

    def _render_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveProctorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_session)
    root.mainloop()