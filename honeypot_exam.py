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
import qrcode
import requests
from pynput import keyboard, mouse
from PIL import Image, ImageTk
from datetime import datetime

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = "Honeypot_Sessions"
FRAME_RATE = 20.0
RESOLUTION = (640, 480)
AUDIO_CHUNK = 1024
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 44100
user32 = ctypes.windll.user32
EXAM_DURATION_MINUTES = 10

# --- URL SETUP ---
# 1. Update this with your current ngrok link so the QR code works
NGROK_URL = "https://unnodding-hwa-unepigrammatically.ngrok-free.dev"

# 2. Local communication (Do not change this)
LOCAL_API_URL = "http://127.0.0.1:5000"

QUESTIONS =[
    "1. What is a Digital Image?",
    "2. What is Inheritence?",
    "3. Explain the difference between FIFO and LIFO.",
    "4. Write the definition of the following according to database : DBMS, Entity, Relation, Attribute.",
    "5. What is a For-Loop?"
]

class DataRecorder:
    def __init__(self, session_dir):
        self.session_dir = session_dir
        os.makedirs(self.session_dir, exist_ok=True)
        self.is_recording = True
        self.event_file = open(os.path.join(self.session_dir, 'exam_events.csv'), 'w', newline='', encoding='utf-8')
        self.event_writer = csv.writer(self.event_file)
        self.event_writer.writerow(['Timestamp', 'Event', 'Details'])
        self.input_file = open(os.path.join(self.session_dir, 'inputs.csv'), 'w', newline='', encoding='utf-8')
        self.input_writer = csv.writer(self.input_file)
        self.input_writer.writerow(['Timestamp', 'Event_Type', 'Key_Button', 'Active_Window'])
        self.audio_frames =[]
        self.audio_thread = threading.Thread(target=self._record_audio)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_out = cv2.VideoWriter(os.path.join(self.session_dir, 'webcam_video.mp4'), fourcc, FRAME_RATE, RESOLUTION)
        self.audio_thread.start()
        self.k_listener = keyboard.Listener(on_press=self._on_key)
        self.m_listener = mouse.Listener(on_click=self._on_click)
        self.k_listener.start()
        self.m_listener.start()
        self.log_event("SESSION_START", "")

    def write_video_frame(self, frame):
        if self.is_recording and self.video_out: self.video_out.write(frame)
    def log_event(self, event, details):
        if self.is_recording: self.event_writer.writerow([time.time(), event, details]); self.event_file.flush()
    def _record_audio(self):
        p = pyaudio.PyAudio(); stream = p.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True, frames_per_buffer=AUDIO_CHUNK)
        while self.is_recording: self.audio_frames.append(stream.read(AUDIO_CHUNK))
        stream.close(); p.terminate()
        with wave.open(os.path.join(self.session_dir, 'audio.wav'), 'wb') as wf:
            wf.setnchannels(AUDIO_CHANNELS); wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT)); wf.setframerate(AUDIO_RATE); wf.writeframes(b''.join(self.audio_frames))
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
        self.input_file.close(); self.event_file.close(); return self.session_dir

class ExamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Honeypot Exam Collector")
        self.root.geometry("1000x800")
        self.recorder = None
        self.exam_active = False
        self.current_question = -1
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        
        # --- UI FRAMES ---
        self.setup_frame = tk.Frame(root)
        self.exam_frame = tk.Frame(root)
        self.setup_frame.pack(expand=True, fill=tk.BOTH)
        
        # --- SETUP SCREEN ---
        tk.Label(self.setup_frame, text="Honeypot Exam Protocol", font=("Arial", 24, "bold")).pack(pady=20)
        tk.Label(self.setup_frame, text="Subject ID:", font=("Arial", 12)).pack()
        self.entry_id = tk.Entry(self.setup_frame, font=("Arial", 12)); self.entry_id.insert(0, "User_01"); self.entry_id.pack(pady=5)
        
        qr_url = f"{NGROK_URL}/"
        try:
            qr_img = qrcode.make(qr_url).resize((200, 200))
            self.qr_photo = ImageTk.PhotoImage(qr_img)
            tk.Label(self.setup_frame, text="Scan this with your phone to start the workspace camera:", font=("Arial", 12)).pack(pady=10)
            tk.Label(self.setup_frame, image=self.qr_photo).pack()
        except Exception as e:
            print(f"DEBUG: QR Code generation failed: {e}")
        
        tk.Button(self.setup_frame, text="START EXAM", font=("Arial", 16), bg="green", fg="white", command=self.start_exam).pack(pady=30)
        
        # --- EXAM SCREEN ---
        top_bar = tk.Frame(self.exam_frame)
        top_bar.pack(fill=tk.X, pady=10, padx=20)
        
        self.lbl_timer = tk.Label(top_bar, text="10:00", font=("Consolas", 24), fg="red")
        self.lbl_timer.pack(side=tk.LEFT)
        
        # Mini video preview to ensure camera is running
        self.lbl_mini_vid = tk.Label(top_bar, text="Camera Preview Loading...")
        self.lbl_mini_vid.pack(side=tk.RIGHT)

        self.lbl_question = tk.Label(self.exam_frame, text="", font=("Arial", 16, "bold"), wraplength=800, justify="left")
        self.lbl_question.pack(pady=30, padx=20, fill=tk.X)
        
        self.answer_text = tk.Text(self.exam_frame, height=12, width=80, font=("Arial", 12))
        self.answer_text.pack(pady=10)
        
        self.btn_next = tk.Button(self.exam_frame, text="Next Question", font=("Arial", 14), bg="blue", fg="white", command=self.next_question)
        self.btn_next.pack(pady=20)
        
        self.update_feed()

    def start_exam(self):
        print("DEBUG: Start Exam Button Clicked.")
        subject_id = self.entry_id.get()
        if not subject_id: messagebox.showerror("Error", "Please enter Subject ID"); return
        
        session_dir = os.path.join(BASE_OUTPUT_DIR, f"{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"DEBUG: Session folder will be: {session_dir}")

        # Send start command to server
        try:
            print(f"DEBUG: Sending command to {LOCAL_API_URL}...")
            res = requests.post(f"{LOCAL_API_URL}/start_session", json={"session_dir": session_dir}, timeout=3)
            print(f"DEBUG: Server responded with status: {res.status_code}")
        except Exception as e:
            print(f"DEBUG: Failed to contact server: {e}")
            messagebox.showerror("Error", f"Could not contact server. Is server.py running?\nError: {e}")
            return

        # Start background recording
        print("DEBUG: Initializing DataRecorder...")
        try:
            self.recorder = DataRecorder(session_dir)
        except Exception as e:
            print(f"DEBUG: DataRecorder crashed! {e}")
            messagebox.showerror("Error", f"Failed to start webcam/microphones.\n{e}")
            return
            
        self.exam_active = True
        
        # UI SWITCH (The part that was failing)
        print("DEBUG: Switching UI Frames...")
        self.setup_frame.pack_forget()
        self.exam_frame.pack(expand=True, fill=tk.BOTH)
        self.root.update() # FORCE REDRAW OF THE SCREEN
        
        self.exam_end_time = time.time() + (EXAM_DURATION_MINUTES * 60)
        self.update_exam_timer()
        
        print("DEBUG: Loading first question...")
        self.next_question()

    def next_question(self):
        if self.current_question >= 0:
            answer = self.answer_text.get("1.0", tk.END).strip()
            with open(os.path.join(self.recorder.session_dir, f"answer_q{self.current_question+1}.txt"), 'w', encoding='utf-8') as f:
                f.write(answer)
            self.recorder.log_event("QUESTION_END", f"Q{self.current_question+1}")
        
        self.current_question += 1
        self.answer_text.delete("1.0", tk.END)
        
        if self.current_question >= len(QUESTIONS): 
            self.finish_exam()
            return
            
        self.lbl_question.config(text=QUESTIONS[self.current_question])
        self.recorder.log_event("QUESTION_START", f"Q{self.current_question+1}")
        
        if self.current_question == len(QUESTIONS) - 1: 
            self.btn_next.config(text="FINISH EXAM", bg="red")

    def update_exam_timer(self):
        if self.exam_active:
            remaining = max(0, int(self.exam_end_time - time.time()))
            mins, secs = divmod(remaining, 60)
            self.lbl_timer.config(text=f"{mins:02d}:{secs:02d}")
            if remaining == 0: self.finish_exam()
            self.root.after(1000, self.update_exam_timer)

    def finish_exam(self):
        if not self.exam_active: return
        self.exam_active = False
        print("DEBUG: Finishing Exam...")
        
        # 1. STOP THE DESKTOP RECORDER FIRST
        if self.recorder:
            self.recorder.log_event("SESSION_END", "")
            saved_path = self.recorder.stop()
            self.recorder = None
        
        # 2. TELL THE SERVER TO STOP
        try:
            print("DEBUG: Sending STOP to server...")
            requests.post(f"{LOCAL_API_URL}/stop_session", timeout=5)
        except Exception as e:
            print(f"DEBUG: Failed to stop server: {e}")

        messagebox.showinfo("Exam Finished", f"Session saved to:\n{saved_path}")
        self.root.destroy()
        
    def update_feed(self):
        ret, frame = self.cap.read()
        if ret:
            if self.exam_active and self.recorder:
                self.recorder.write_video_frame(frame)
                
                # Show a tiny preview in the top right to confirm it's working
                preview = cv2.resize(frame, (160, 120))
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(preview_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.lbl_mini_vid.imgtk = imgtk
                self.lbl_mini_vid.configure(image=imgtk)
                
        self.root.after(int(1000/FRAME_RATE), self.update_feed)

    def on_closing(self):
        if self.exam_active:
            if messagebox.askokcancel("Quit", "Quit exam? Progress will be lost."):
                if self.recorder: self.recorder.stop()
                try: requests.post(f"{LOCAL_API_URL}/stop_session", timeout=2)
                except: pass
                self.root.destroy()
        else:
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()