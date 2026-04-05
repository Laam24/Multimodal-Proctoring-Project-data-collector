import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import ctypes
import joblib
from collections import deque
import time
from scipy.stats import mode # New import for finding the most common prediction

# --- CONFIGURATION ---
MODEL_PATH = "best_gru_model_weighted.pth" # Using the weighted model
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

# --- MAIN LIVE PROCTORING LOOP ---
def main():
    print("Loading AI Model and Scalers...")
    device = torch.device('cpu')
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    num_classes = len(label_encoder.classes_)
    model = ProctoringGRU(input_size=len(FEATURES), hidden_size=64, num_layers=2, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("AI Ready.")

    mp_tasks = mp.tasks.vision
    options = mp_tasks.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH), running_mode=mp_tasks.RunningMode.VIDEO, num_faces=1)
    landmarker = mp_tasks.FaceLandmarker.create_from_options(options)
    
    memory_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # --- NEW: PREDICTION SMOOTHING BUFFER ---
    # It will store the last 15 predictions (about 1.5-2 seconds)
    prediction_buffer = deque(maxlen=15)
    
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    print("\n>>> LIVE GRU PROCTORING (SMOOTHED) STARTED <<<")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        img_h, img_w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        unauth_window = get_active_window_flag()
        
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            gaze_x, gaze_y = landmarks[1].x, landmarks[1].y
            pitch, yaw, roll = calculate_head_pose(landmarks, img_w, img_h)
            current_features = [1.0, pitch, yaw, roll, gaze_x, gaze_y, unauth_window]
        else:
            current_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, unauth_window]
            
        memory_buffer.append(current_features)
        
        if len(memory_buffer) == SEQUENCE_LENGTH:
            scaled_memory = scaler.transform(memory_buffer)
            tensor_memory = torch.tensor(np.array([scaled_memory]), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                output = model(tensor_memory)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                predicted_class_idx = torch.argmax(probabilities).item()
                
                # Add the latest prediction to the smoothing buffer
                prediction_buffer.append(predicted_class_idx)
                
                # --- NEW: SMOOTHING LOGIC ---
                # Find the most common prediction in the buffer
                smoothed_prediction_idx = mode(prediction_buffer)[0]
                
                predicted_label = label_encoder.inverse_transform([smoothed_prediction_idx])[0]
                confidence = probabilities[smoothed_prediction_idx].item() * 100

            color = (0, 255, 0) if predicted_label == "NORMAL" else (0, 0, 255)
            
            cv2.rectangle(frame, (10, 10), (600, 100), (0,0,0), -1)
            cv2.putText(frame, f"STATUS: {predicted_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"CONFIDENCE: {confidence:.1f}%", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if unauth_window == 1.0 and predicted_label == "CHEAT_DIGITAL":
                cv2.putText(frame, "UNAUTHORIZED WINDOW DETECTED!", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, f"Calibrating Memory... {len(memory_buffer)}/{SEQUENCE_LENGTH}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Live GRU Proctoring AI (Smoothed)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()