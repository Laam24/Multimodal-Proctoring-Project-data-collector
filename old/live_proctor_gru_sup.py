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

# --- CONFIGURATION ---
MODEL_PATH = "super_gru_v1.pth"
SCALER_PATH = "super_gru_scaler.pkl"
FACE_MODEL_PATH = 'face_landmarker.task'
LABEL_MAP = {0: 'NORMAL', 1: 'CHEAT_PHYSICAL', 2: 'CHEAT_DIGITAL'}

user32 = ctypes.windll.user32

# --- THE SUPER-GRU ARCHITECTURE ---
# Must match the training script EXACTLY
class SuperProctorGRU(nn.Module):
    def __init__(self, input_size=18, hidden_size=128, num_layers=2):
        super(SuperProctorGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

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
        self.root.title("Live Super-GRU Proctoring AI")
        self.root.geometry("1000x900")
        
        # 1. State Buffers
        self.is_running = False
        self.cap = None
        self.start_time = None
        self.frame_counter = 0
        
        self.raw_data_history = deque(maxlen=60)    # Math buffer for variance/velocity
        self.sequence_buffer = deque(maxlen=60)     # Tensor input buffer for the GRU
        self.prediction_buffer = deque(maxlen=15)   # UI Smoothing buffer

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

        # 3. Load Models
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.scaler = joblib.load(SCALER_PATH)
            
            self.model = SuperProctorGRU(input_size=18).to(self.device)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval() # CRITICAL: Sets dropout layers to testing mode
            
            mp_tasks = mp.tasks.vision
            options = mp_tasks.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH),
                running_mode=mp_tasks.RunningMode.VIDEO, num_faces=1)
            self.landmarker = mp_tasks.FaceLandmarker.create_from_options(options)
            print("✅ Super-GRU Intelligence Online.")
        except Exception as e:
            messagebox.showerror("Error", f"Initialization failed: {e}")
            self.root.destroy()

    def start_session(self):
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
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
                self.frame_counter += 1
                
                # 1. Feature Extraction
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp = int((time.time() - self.start_time) * 1000)
                result = self.landmarker.detect_for_video(mp_img, timestamp)
                unauth = get_active_window_flag()
                
                face_det, p, y, r, gx, gy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                if result.face_landmarks:
                    face_det = 1.0
                    lms = result.face_landmarks[0]
                    gx, gy = lms[1].x, lms[1].y
                    p, y, r = calculate_head_pose(lms, w, h)

                # 2. Physics Engine (Velocity & Variance)
                self.raw_data_history.append([p, y, r, gx, gy])
                hist = np.array(self.raw_data_history)
                
                p_v, y_v, r_v, gx_v, gy_v = 0.0, 0.0, 0.0, 0.0, 0.0
                p_var, y_var, r_var, gx_var, gy_var = 0.0, 0.0, 0.0, 0.0, 0.0

                if len(hist) > 1:
                    p_v, y_v, r_v, gx_v, gy_v = hist[-1] - hist[-2]
                    p_var, y_var, r_var, gx_var, gy_var = np.var(hist, axis=0)

                # 3. Assemble the 18-Feature Vector (Must match CSV column order)
                current_features = [
                    self.frame_counter, face_det, p, y, r, gx, gy,
                    p_v, y_v, r_v, gx_v, gy_v,
                    p_var, y_var, r_var, gx_var, gy_var, unauth
                ]
                self.sequence_buffer.append(current_features)

                # 4. Super-GRU Inference
                cv2.rectangle(frame, (10, 10), (550, 100), (0,0,0), -1)
                
                if len(self.sequence_buffer) == 60:
                    # Scale the 60-frame sequence
                    seq_array = np.array(self.sequence_buffer)
                    seq_scaled = self.scaler.transform(seq_array)
                    
                    # Convert to PyTorch Tensor: shape [1, 60, 18]
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

                    # Draw Active HUD
                    color = (0, 255, 0) if label == "NORMAL" else (0, 0, 255)
                    cv2.putText(frame, f"STATUS: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    cv2.putText(frame, f"CONFIDENCE: {conf:.1f}%", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    if unauth == 1.0 and label == "CHEAT_DIGITAL":
                        cv2.putText(frame, "UNAUTHORIZED WINDOW!", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # Draw Loading HUD
                    loading_text = f"LOADING CONTEXT: {len(self.sequence_buffer)}/60"
                    cv2.putText(frame, loading_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                # 5. Render to Screen
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                
                self.video_label.imgtk = img_tk
                self.video_label.configure(image=img_tk)

        except Exception as e:
            print(f"CRITICAL ERROR IN VIDEO LOOP: {e}")
            
        self.root.after(15, self.update_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveProctorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_session)
    root.mainloop()