import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import pyaudio
import wave
import time
import csv
import os
import threading
import ctypes
import numpy as np
import mediapipe as mp
from pynput import keyboard, mouse
from PIL import Image, ImageTk
from datetime import datetime

# --- All configurations, helpers, and definitions remain the same ---
BASE_OUTPUT_DIR = "Dataset_Structured"
FRAME_RATE = 20.0
RESOLUTION = (640, 480)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
user32 = ctypes.windll.user32
FACE_MODEL_PATH = 'face_landmarker.task'
HAND_MODEL_PATH = 'hand_landmarker.task'
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]
def draw_head_pose_axis(image, face_landmarks_list):
    if not face_landmarks_list: return
    face_landmarks = face_landmarks_list[0]
    img_h, img_w, _ = image.shape
    try:
        image_points = np.array([(lm.x * img_w, lm.y * img_h) for lm in [face_landmarks[1], face_landmarks[152], face_landmarks[263], face_landmarks[33], face_landmarks[287], face_landmarks[57]]], dtype="double")
        model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
        camera_matrix = np.array([[img_w, 0, img_w/2], [0, img_w, img_h/2], [0, 0, 1]], dtype="double")
        (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, np.zeros((4, 1)))
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(image, p1, p2, (255, 0, 0), 2)
    except: pass
def draw_hand_skeleton(image, hand_landmarks_list):
    if not hand_landmarks_list: return
    img_h, img_w, _ = image.shape
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS: cv2.line(image, points[start_idx], points[end_idx], (255, 255, 255), 2)
        for point in points: cv2.circle(image, point, 3, (0, 255, 0), -1)

STATIC_ACTIONS = {
    "Calibration": { "Look Center": ("CALIB_CENTER", 10, "Look at the CENTER dot."), "Look Corners": ("CALIB_CORNERS", 15, "Look: Top-Left -> Top-Right -> ..."), "Head Turn": ("CALIB_YAW", 15, "Slowly turn head LEFT, then RIGHT."), },
    "Normal Behavior": { "Reading": ("NORM_READ", 30, "Read text on screen silently."), "Typing": ("NORM_TYPE", 60, "Type a paragraph normally."), "Thinking": ("NORM_THINK", 15, "Look up/away to think."), "Writing": ("NORM_WRITE", 60, "Look down at desk to write."), },
    "Cheating Scenarios": { "Phone in Lap": ("CHEAT_PHONE_LAP", 30, "Hide phone in lap. Text."), "Side Glances": ("CHEAT_SIDE", 20, "Look repeatedly at a cheat sheet."), "Whispering": ("CHEAT_VOICE", 15, "Turn and whisper."), "Absence": ("CHEAT_ABSENCE", 10, "Leave the chair."), "Alt-Tab (Cheat)": ("CHEAT_DIGITAL", 30, "Switch windows to search for answers."), },
    "Pedagogy / Attention": { "Bored/Drowsy": ("ATTN_BORED", 30, "Slouch, stare blankly."), "Distracted (Fun)": ("ATTN_DISTRACTED", 30, "Watch YouTube / use social media."), }
}
DYNAMIC_SCENARIOS = {
    "Fair Exam Flow": [ ("NORM_READ", 20, "Read Question 1."), ("TRANSITION", 5, "GET READY to type..."), ("NORM_TYPE", 30, "Type the answer."), ("NORM_THINK", 10, "Pause and think."), ("NORM_READ", 20, "Read Question 2."), ],
    "Digital Cheater Flow": [ ("NORM_TYPE", 20, "Start typing."), ("TRANSITION", 5, "GET READY to cheat..."), ("CHEAT_DIGITAL", 30, "Alt-Tab to Google/ChatGPT."), ("TRANSITION", 5, "GET READY to return..."), ("NORM_TYPE", 20, "Return and type the answer."), ],
    "Physical Cheater Flow": [ ("NORM_WRITE", 30, "Start writing on paper."), ("TRANSITION", 5, "GET READY to use phone..."), ("CHEAT_PHONE_LAP", 30, "Check your phone in your lap."), ("TRANSITION", 5, "GET READY to look at notes..."), ("CHEAT_SIDE", 15, "Glance at a side note."), ("NORM_WRITE", 30, "Go back to writing."), ]
}
class DataRecorder:
    # This class remains identical
    def __init__(self, subject_id, action_label):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(BASE_OUTPUT_DIR, subject_id, action_label, ts)
        os.makedirs(self.session_dir, exist_ok=True)
        self.is_recording = True
        self.input_file = open(os.path.join(self.session_dir, 'inputs.csv'), 'w', newline='', encoding='utf-8')
        self.input_writer = csv.writer(self.input_file)
        self.input_writer.writerow(['Timestamp', 'Event_Type', 'Key_Button', 'Active_Window'])
        self.meta_file = open(os.path.join(self.session_dir, 'labels.csv'), 'w', newline='', encoding='utf-8')
        self.meta_writer = csv.writer(self.meta_file)
        self.meta_writer.writerow(['Start_Time', 'End_Time', 'Label', 'Instruction'])
        self.audio_frames = []
        self.audio_thread = threading.Thread(target=self._record_audio)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_path = os.path.join(self.session_dir, 'video.mp4')
        self.video_out = cv2.VideoWriter(self.video_path, fourcc, FRAME_RATE, RESOLUTION)
        self.audio_thread.start()
        self.k_listener = keyboard.Listener(on_press=self._on_key)
        self.m_listener = mouse.Listener(on_click=self._on_click)
        self.k_listener.start()
        self.m_listener.start()
    def write_video_frame(self, frame):
        if self.is_recording and self.video_out: self.video_out.write(frame)
    def log_meta(self, start, end, label, instruction): self.meta_writer.writerow([start, end, label, instruction]); self.meta_file.flush()
    def _record_audio(self):
        p = pyaudio.PyAudio(); stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        while self.is_recording: self.audio_frames.append(stream.read(CHUNK))
        stream.close(); p.terminate()
        with wave.open(os.path.join(self.session_dir, 'audio.wav'), 'wb') as wf:
            wf.setnchannels(CHANNELS); wf.setsampwidth(p.get_sample_size(FORMAT)); wf.setframerate(RATE); wf.writeframes(b''.join(self.audio_frames))
    def _get_active_window(self):
        try:
            hwnd = user32.GetForegroundWindow(); length = user32.GetWindowTextLengthW(hwnd); buff = ctypes.create_unicode_buffer(length + 1); user32.GetWindowTextW(hwnd, buff, length + 1); return buff.value
        except: return ""
    def _on_key(self, key):
        if self.is_recording:
            try: k = key.char
            except: k = str(key)
            self.input_writer.writerow([time.time(), "Keyboard", k, self._get_active_window()])
    def _on_click(self, x, y, button, pressed):
        if self.is_recording and pressed: self.input_writer.writerow([time.time(), "Mouse", str(button), self._get_active_window()])
    def stop(self):
        self.is_recording = False; self.k_listener.stop(); self.m_listener.stop(); self.audio_thread.join()
        if self.video_out: self.video_out.release()
        self.input_file.close(); self.meta_file.close(); return self.session_dir

class DashboardApp:
    def __init__(self, root):
        self.root = root; self.root.title("Director's Dashboard v7.8 (Final)"); self.root.geometry("1100x800")
        self.lock = threading.Lock(); self.latest_face_result = None; self.latest_hand_result = None
        
        mp_tasks = mp.tasks.vision; BaseOptions = mp.tasks.BaseOptions
        face_options = mp_tasks.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH), running_mode=mp_tasks.RunningMode.LIVE_STREAM, result_callback=self._face_result_callback, num_faces=1)
        self.face_landmarker = mp_tasks.FaceLandmarker.create_from_options(face_options)
        hand_options = mp_tasks.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH), running_mode=mp_tasks.RunningMode.LIVE_STREAM, result_callback=self._hand_result_callback, num_hands=2)
        self.hand_landmarker = mp_tasks.HandLandmarker.create_from_options(hand_options)
        
        self.cap = cv2.VideoCapture(0); self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0]); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        
        self.recorder = None; self.recording_active = False; self.countdown_val = 0; self.action_end_time = 0; self.dynamic_scenario = None; self.dynamic_step = 0
        
        left_panel = tk.Frame(root, width=300, bg="#f0f0f0"); left_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas = tk.Canvas(left_panel, width=300, bg="#f0f0f0"); self.scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#f0f0f0")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw"); self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True); self.scrollbar.pack(side="right", fill="y")
        
        tk.Label(self.scrollable_frame, text="Subject ID:", font=("Arial", 12)).pack(pady=5, padx=10)
        self.entry_id = tk.Entry(self.scrollable_frame, font=("Arial", 12)); self.entry_id.insert(0, "User_01"); self.entry_id.pack(pady=5, padx=10, fill=tk.X)
        
        self.notebook = ttk.Notebook(self.scrollable_frame); self.notebook.pack(expand=True, fill=tk.BOTH, pady=10, padx=5)
        
        static_tab = tk.Frame(self.notebook); self.notebook.add(static_tab, text="Static")
        self.static_buttons = []
        for category, actions in STATIC_ACTIONS.items():
            lbl_frame = tk.LabelFrame(static_tab, text=category, padx=5, pady=5); lbl_frame.pack(fill=tk.X, padx=5, pady=5)
            for btn_text, (label, duration, instr) in actions.items():
                btn = tk.Button(lbl_frame, text=btn_text, anchor="w", command=lambda l=label, d=duration, i=instr: self.prep_recording(l, d, i)); btn.pack(fill=tk.X, pady=2)
                self.static_buttons.append(btn)
        
        dynamic_tab = tk.Frame(self.notebook); self.notebook.add(dynamic_tab, text="Dynamic")
        self.dynamic_buttons = []
        for name, playlist in DYNAMIC_SCENARIOS.items():
            btn = tk.Button(dynamic_tab, text=name, height=2, command=lambda n=name, p=playlist: self.prep_dynamic_scenario(n, p)); btn.pack(fill=tk.X, pady=5, padx=5)
            self.dynamic_buttons.append(btn)

        right_panel = tk.Frame(root, bg="#333"); right_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.lbl_instruction = tk.Label(right_panel, text="Select an action.", font=("Arial", 16, "bold"), bg="#333", fg="white", wraplength=600); self.lbl_instruction.pack(pady=20)
        self.lbl_timer = tk.Label(right_panel, text="00:00", font=("Consolas", 30, "bold"), bg="#333", fg="gray"); self.lbl_timer.pack(pady=5)
        self.lbl_video = tk.Label(right_panel); self.lbl_video.pack()
        self.lbl_status = tk.Label(right_panel, text="IDLE", font=("Arial", 14), bg="gray", fg="black"); self.lbl_status.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.update_feed(); self.update_timer()

    def _face_result_callback(self, result, _, __):
        with self.lock: self.latest_face_result = result
    def _hand_result_callback(self, result, _, __):
        with self.lock: self.latest_hand_result = result

    def set_controls_state(self, state):
        self.entry_id.config(state=state)
        for btn in self.static_buttons + self.dynamic_buttons: btn.config(state=state)

    def prep_recording(self, label, duration, instruction):
        if self.recording_active: return
        self.dynamic_scenario = None; self.prep_countdown(label, duration, instruction)
    def prep_dynamic_scenario(self, name, playlist):
        if self.recording_active: return
        self.dynamic_scenario = playlist; self.dynamic_step = 0
        self.prep_countdown(name.replace(" ", "_"), 0, f"Scenario: {name}")
    def prep_countdown(self, label, duration, instruction):
        subject_id = self.entry_id.get()
        if not subject_id: messagebox.showerror("Error", "Please enter Subject ID"); return
        self.target_label, self.target_duration, self.target_instruction = label, duration, instruction
        self.lbl_instruction.config(text=f"TASK: {instruction}")
        self.set_controls_state('disabled')
        self.countdown_val = 3; self.run_countdown()

    def run_countdown(self):
        if self.countdown_val > 0:
            self.lbl_status.config(text=f"STARTING IN {self.countdown_val}...", bg="orange")
            self.countdown_val -= 1; self.root.after(1000, self.run_countdown)
        else: self.start_recording()

    def start_recording(self):
        self.recording_active = True
        self.recorder = DataRecorder(self.entry_id.get(), self.target_label)
        if self.dynamic_scenario:
            self.run_dynamic_step()
        else:
            self.lbl_status.config(text=f"RECORDING: {self.target_label}", bg="red", fg="white")
            self.lbl_timer.config(fg="red"); self.action_end_time = time.time() + self.target_duration
            
            # --- THE FIX for Static Scenario Logging ---
            def log_static_action():
                self.recorder.log_meta(self.action_end_time - self.target_duration, self.action_end_time, self.target_label, self.target_instruction)
                self.stop_recording()
            
            self.root.after(int(self.target_duration * 1000), log_static_action)
            
    def run_dynamic_step(self):
        if not self.recording_active or self.dynamic_step >= len(self.dynamic_scenario):
            self.stop_recording(); return
            
        label, duration, instruction = self.dynamic_scenario[self.dynamic_step]
        self.lbl_status.config(text=f"REC: {label}", bg="red", fg="white")
        self.lbl_instruction.config(text=f"ACTION: {instruction}")
        self.lbl_timer.config(fg="red")
        
        start_time = time.time()
        self.action_end_time = start_time + duration
        
        # --- THE FIX for Dynamic Scenario Logging ---
        def create_log_callback(s, l, i):
            def callback():
                if self.recorder: self.recorder.log_meta(s, time.time(), l, i)
            return callback

        self.root.after(int(duration*1000), create_log_callback(start_time, label, instruction))
        self.root.after(int(duration*1000), self.run_dynamic_step)
        
        self.dynamic_step += 1
        
    def stop_recording(self):
        if self.recorder: saved_path = self.recorder.stop(); self.recorder = None
        self.recording_active = False
        self.lbl_status.config(text=f"SAVED TO: {saved_path}", bg="green", fg="white")
        self.lbl_timer.config(text="00:00", fg="gray"); self.lbl_instruction.config(text="Select next action.")
        self.set_controls_state('normal')

    def update_timer(self):
        if self.recording_active:
            remaining = max(0, int(self.action_end_time - time.time()))
            self.lbl_timer.config(text=f"00:{remaining:02d}")
        self.root.after(100, self.update_timer)
    
    def update_feed(self):
        ret, frame = self.cap.read()
        if ret:
            clean_frame = frame.copy()
            timestamp_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            self.face_landmarker.detect_async(mp_image, timestamp_ms)
            self.hand_landmarker.detect_async(mp_image, timestamp_ms)
            with self.lock:
                if self.latest_face_result and self.latest_face_result.face_landmarks:
                    draw_head_pose_axis(frame, self.latest_face_result.face_landmarks)
                if self.latest_hand_result and self.latest_hand_result.hand_landmarks:
                    draw_hand_skeleton(frame, self.latest_hand_result.hand_landmarks)
            if self.recording_active:
                self.recorder.write_video_frame(clean_frame)
                cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_video.imgtk = imgtk; self.lbl_video.configure(image=imgtk)
        self.root.after(20, self.update_feed)
    
    def on_closing(self):
        if self.cap.isOpened(): self.cap.release()
        self.face_landmarker.close(); self.hand_landmarker.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()