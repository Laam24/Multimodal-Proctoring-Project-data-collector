import tkinter as tk
from tkinter import scrolledtext
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
MODEL_PATH = "best_gru_model_weighted.pth" 
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
FACE_MODEL_PATH = 'face_landmarker.task'

SEQUENCE_LENGTH = 60
FEATURES =['Face_Detected', 'Pitch', 'Yaw', 'Roll', 'Gaze_X', 'Gaze_Y', 'Unauthorized_Window']
UNAUTHORIZED_WINDOWS =['google', 'chatgpt', 'search', 'chrome', 'edge', 'firefox', 'bing', 'brave', 'notepad', 'opera']

user32 = ctypes.windll.user32

# --- GRU MODEL ARCHITECTURE (Unchanged) ---
class ProctoringGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ProctoringGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc1(out[:, -1, :])
        out = self.dropout(self.relu(out))
        return self.fc2(out)

# --- HELPER FUNCTIONS (Unchanged) ---
def get_active_window_flag():
    try:
        hwnd = user32.GetForegroundWindow(); length = user32.GetWindowTextLengthW(hwnd); buff = ctypes.create_unicode_buffer(length + 1); user32.GetWindowTextW(hwnd, buff, length + 1); window_title = buff.value.lower()
        return 1.0 if any(word in window_title for word in UNAUTHORIZED_WINDOWS) else 0.0
    except: return 0.0

def calculate_head_pose(face_landmarks, img_w, img_h):
    try:
        image_points = np.array([(lm.x * img_w, lm.y * img_h) for lm in [face_landmarks[1], face_landmarks[152], face_landmarks[263], face_landmarks[33], face_landmarks[287], face_landmarks[57]]], dtype="double")
        model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
        camera_matrix = np.array([[img_w, 0, img_w/2],[0, img_w, img_h/2], [0, 0, 1]], dtype="double")
        success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = np.degrees(np.arcsin(rmat[2, 1])), np.degrees(np.arctan2(-rmat[2, 0], rmat[2, 2])), np.degrees(np.arctan2(-rmat[0, 1], rmat[1, 1]))
        return pitch, yaw, roll
    except: return 0.0, 0.0, 0.0


# --- TKINTER GUI WRAPPER ---
class LiveProctorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live GRU Proctoring AI - Interactive Exam")
        self.root.geometry("800x850") # Set size for standard webcams
        
        # 1. State Variables
        self.is_running = False
        self.cap = None
        self.start_time = None
        self.memory_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_buffer = deque(maxlen=15)
        
        # 2. UI Layout
        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(pady=10)
        
        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)
        
        self.start_btn = tk.Button(control_frame, text="Start Session", command=self.start_session, bg="green", fg="white", font=("Arial", 12, "bold"))
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = tk.Button(control_frame, text="Stop Session", command=self.stop_session, bg="orange", fg="white", font=("Arial", 12, "bold"), state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        tk.Label(root, text="Exam Answer Sheet:", font=("Arial", 12, "bold")).pack(pady=(10, 0))
        self.exam_box = scrolledtext.ScrolledText(root, height=8, font=("Arial", 12))
        self.exam_box.pack(padx=20, pady=10, fill=tk.X)

        # 3. Load AI and Vision Models Once
        print("Loading AI Model and Scalers...")
        self.device = torch.device('cpu')
        self.scaler = joblib.load(SCALER_PATH)
        self.label_encoder = joblib.load(ENCODER_PATH)
        num_classes = len(self.label_encoder.classes_)
        
        self.model = ProctoringGRU(input_size=len(FEATURES), hidden_size=64, num_layers=2, num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        
        mp_tasks = mp.tasks.vision
        options = mp_tasks.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH), running_mode=mp_tasks.RunningMode.VIDEO, num_faces=1)
        self.landmarker = mp_tasks.FaceLandmarker.create_from_options(options)
        print("AI Ready.")

    def start_session(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            self.start_time = time.time()
            self.memory_buffer.clear()
            self.prediction_buffer.clear()
            
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            print("\n>>> LIVE GRU PROCTORING (SMOOTHED) STARTED <<<")
            self.update_video()

    def stop_session(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        print("\n>>> SESSION STOPPED <<<")

    def on_close(self):
        self.stop_session()
        self.root.destroy()

    def update_video(self):
        if self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1) # Mirror the frame for natural view
                img_h, img_w, _ = frame.shape
                
                # Format for MediaPipe Tasks API
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((time.time() - self.start_time) * 1000)
                
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                unauth_window = get_active_window_flag()
                
                # --- FEATURE EXTRACTION (Unchanged) ---
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    gaze_x, gaze_y = landmarks[1].x, landmarks[1].y
                    pitch, yaw, roll = calculate_head_pose(landmarks, img_w, img_h)
                    current_features = [1.0, pitch, yaw, roll, gaze_x, gaze_y, unauth_window]
                else:
                    current_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, unauth_window]
                    
                self.memory_buffer.append(current_features)
                
                # --- AI INFERENCE & DRAWING (Unchanged) ---
                if len(self.memory_buffer) == SEQUENCE_LENGTH:
                    scaled_memory = self.scaler.transform(self.memory_buffer)
                    tensor_memory = torch.tensor(np.array([scaled_memory]), dtype=torch.float32).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(tensor_memory)
                        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                        predicted_class_idx = torch.argmax(probabilities).item()
                        
                        self.prediction_buffer.append(predicted_class_idx)
                        
                        # Smoothing
                        smoothed_prediction_idx = mode(self.prediction_buffer, keepdims=True)[0][0]
                        predicted_label = self.label_encoder.inverse_transform([smoothed_prediction_idx])[0]
                        confidence = probabilities[smoothed_prediction_idx].item() * 100

                    # Draw the custom HUD
                    color = (0, 255, 0) if predicted_label == "NORMAL" else (0, 0, 255)
                    
                    cv2.rectangle(frame, (10, 10), (600, 100), (0,0,0), -1)
                    cv2.putText(frame, f"STATUS: {predicted_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    cv2.putText(frame, f"CONFIDENCE: {confidence:.1f}%", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    if unauth_window == 1.0 and predicted_label == "CHEAT_DIGITAL":
                        cv2.putText(frame, "UNAUTHORIZED WINDOW DETECTED!", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, f"Calibrating Memory... {len(self.memory_buffer)}/{SEQUENCE_LENGTH}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # --- SEND TO TKINTER UI ---
                # Convert BGR back to RGB for Tkinter to display it correctly
                final_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(final_rgb)
                tk_img = ImageTk.PhotoImage(image=pil_img)
                
                self.video_label.image = tk_img
                self.video_label.config(image=tk_img)
            
            # Request next frame update in 30ms (~30fps)
            self.root.after(30, self.update_video)

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveProctorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()