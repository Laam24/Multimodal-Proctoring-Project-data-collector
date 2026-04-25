import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import ctypes
import joblib
from collections import deque
import time

# --- CONFIGURATION ---
MODEL_PATH = "best_proctoring_model.pth"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
FACE_MODEL_PATH = 'face_landmarker.task'

SEQUENCE_LENGTH = 60  # 3 seconds of memory
FEATURES =['Face_Detected', 'Pitch', 'Yaw', 'Roll', 'Gaze_X', 'Gaze_Y', 'Unauthorized_Window']
UNAUTHORIZED_WINDOWS =['google', 'chatgpt', 'search', 'chrome', 'edge', 'firefox', 'bing', 'brave', 'notepad', 'opera']

user32 = ctypes.windll.user32

# --- 1. PYTORCH MODEL ARCHITECTURE (Must match training exactly) ---
class ProctoringLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ProctoringLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.dropout(self.relu(out))
        return self.fc2(out)

# --- 2. HELPER FUNCTIONS ---
def get_active_window_flag():
    try:
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        window_title = buff.value.lower()
        
        for word in UNAUTHORIZED_WINDOWS:
            if word in window_title:
                return 1.0 # Unauthorized
        return 0.0 # Authorized
    except:
        return 0.0

def calculate_head_pose(face_landmarks, img_w, img_h):
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
    camera_matrix = np.array([[img_w, 0, img_w/2],[0, img_w, img_h/2], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rotation_vector)
    pitch = np.degrees(np.arcsin(rmat[2, 1]))
    yaw = np.degrees(np.arctan2(-rmat[2, 0], rmat[2, 2]))
    roll = np.degrees(np.arctan2(-rmat[0, 1], rmat[1, 1]))
    return pitch, yaw, roll

# --- 3. MAIN LIVE PROCTORING LOOP ---
def main():
    print("Loading AI Model and Scalers...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Scaler and Encoder
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    num_classes = len(label_encoder.classes_)
    
    # Load PyTorch Model
    model = ProctoringLSTM(input_size=len(FEATURES), hidden_size=64, num_layers=2, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval() # Set to evaluation mode
    print("AI Ready.")

    # Initialize MediaPipe
    mp_tasks = mp.tasks.vision
    options = mp_tasks.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_tasks.RunningMode.VIDEO,
        num_faces=1)
    
    landmarker = mp_tasks.FaceLandmarker.create_from_options(options)
    
    # Initialize Memory Buffer (Deletes oldest frame automatically when full)
    memory_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    cap = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time()
    
    print("\n>>> LIVE PROCTORING STARTED <<<")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        img_h, img_w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # 1. Extract Current Frame Features
        unauth_window = get_active_window_flag()
        
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            gaze_x, gaze_y = landmarks[1].x, landmarks[1].y
            pitch, yaw, roll = calculate_head_pose(landmarks, img_w, img_h)
            current_features =[1.0, pitch, yaw, roll, gaze_x, gaze_y, unauth_window]
            
            # --- THE FIX: Pure OpenCV Visual Feedback ---
            # Draw green dots on key facial landmarks (Nose, Chin, Eyes, Mouth)
            for lm_index in[1, 152, 263, 33, 287, 57]:
                lm = landmarks[lm_index]
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        else:
            # Face lost
            current_features =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, unauth_window]
            
        # 2. Add to Rolling Memory
        memory_buffer.append(current_features)
        
        # 3. Predict if we have enough memory (3 seconds)
        if len(memory_buffer) == SEQUENCE_LENGTH:
            # Scale the entire 60-frame memory block
            scaled_memory = scaler.transform(memory_buffer)
            
            # Convert to PyTorch Tensor: Shape (Batch=1, Seq=60, Features=7)
            tensor_memory = torch.tensor(np.array([scaled_memory]), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                output = model(tensor_memory)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                
                # Get the highest probability class
                predicted_class_idx = torch.argmax(probabilities).item()
                predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
                confidence = probabilities[predicted_class_idx].item() * 100

            # 4. Display Results on Screen
            color = (0, 255, 0) if predicted_label == "NORMAL" else (0, 0, 255)
            
            # Draw UI Box
            cv2.rectangle(frame, (10, 10), (600, 100), (0,0,0), -1)
            cv2.putText(frame, f"STATUS: {predicted_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"CONFIDENCE: {confidence:.1f}%", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # If digital cheating, flash a specific warning
            if unauth_window == 1.0:
                cv2.putText(frame, "UNAUTHORIZED WINDOW DETECTED!", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        else:
            # Still gathering the first 3 seconds of data
            cv2.putText(frame, f"Calibrating Memory... {len(memory_buffer)}/{SEQUENCE_LENGTH}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Live Proctoring AI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()