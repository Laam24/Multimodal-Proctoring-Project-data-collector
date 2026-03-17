from flask import Flask, render_template, request, jsonify
import os
import base64
import threading
import time
from datetime import datetime
from pyngrok import ngrok
import cv2
import numpy as np

app = Flask(__name__)
print("Flask App Initialized.")

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = "Honeypot_Sessions"

# This will store the video frames for the current session
current_session_data = {
    "frames": [],
    "is_recording": False,
    "output_path": "",
    "start_time": 0.0,
    "lock": threading.Lock() 
}
print(f"Initial server state: is_recording = {current_session_data['is_recording']}")

@app.route('/')
def index():
    """Serves the main mobile camera page."""
    return render_template('camera.html')

@app.route('/status', methods=['GET'])
def get_status():
    """Tells the phone if the desktop app has triggered the recording."""
    with current_session_data["lock"]:
        is_rec = current_session_data["is_recording"]
    return jsonify({"is_recording": is_rec})

@app.route('/ping', methods=['GET'])
def ping():
    print("DEBUG: Received /ping request. Server is responsive.")
    return jsonify({"status": "pong"})

@app.route('/start_session', methods=['POST'])
def start_session():
    """
    Triggered by the Desktop App.
    Creates a folder for the incoming phone video stream.
    """
    print("\n" + "="*30)
    print("DEBUG: Received /start_session request.")
    data = request.get_json()
    session_dir = data.get("session_dir")

    if not session_dir:
        print("ERROR: /start_session failed - no session_dir provided.")
        return jsonify({"status": "error", "message": "session_dir not provided"}), 400
    
    with current_session_data["lock"]:
        current_session_data["frames"] = []
        current_session_data["output_path"] = os.path.join(session_dir, "phone_video.mp4")
        current_session_data["is_recording"] = True
        current_session_data["start_time"] = time.time()
    
    print(f"INFO: Phone camera recording started for session: {session_dir}")
    print(f"INFO: is_recording state is now TRUE.")
    print("="*30 + "\n")
    return jsonify({"status": "success", "message": "Recording started"})

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """Receives a single video frame from the phone's webpage."""
    with current_session_data["lock"]:
        is_rec = current_session_data["is_recording"]
    
    if not is_rec:
        return jsonify({"status": "stopped"}), 200

    data = request.get_json()
    image_data = data.get('image')
    
    if image_data:
        try:
            header, encoded = image_data.split(",", 1)
            frame_bytes = base64.b64decode(encoded)
            with current_session_data["lock"]:
                current_session_data["frames"].append(frame_bytes)
                # ADD THIS LINE TO DEBUG:
                print(f"Frame Received. Total in memory: {len(current_session_data['frames'])}", end='\r')
        except Exception as e:
            print(f"ERROR: Could not decode base64 frame: {e}")

    return jsonify({"status": "received"})

def save_file_in_background(frames, path, session_duration):
    """
    This function runs in a separate thread to save the video
    without freezing the server.
    """
    print("\n" + "="*30)
    print(f"BACKGROUND SAVE THREAD: Starting to process {len(frames)} frames...")
    try:
        if len(frames) == 0:
            print("WARNING: No frames received from phone. Video will not be saved.")
            return

        first_frame_img = None
        for f in frames:
            nparr = np.frombuffer(f, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                first_frame_img = img
                break
        
        if first_frame_img is None:
            print("ERROR in BACKGROUND: All frames were corrupted. Cannot save video.")
            return

        height, width, _ = first_frame_img.shape
        if session_duration <= 0: session_duration = 1.0 
        
        actual_fps = max(1.0, len(frames) / session_duration)
        
        print(f"BACKGROUND SAVE THREAD: Video properties: {width}x{height} @ {actual_fps:.2f} FPS")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, actual_fps, (width, height))

        valid_frames = 0
        for frame_bytes in frames:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                out.write(img)
                valid_frames += 1
        
        out.release()
        print(f"✅ SUCCESS: BACKGROUND SAVE THREAD: Video saved with {valid_frames} frames.")
    except Exception as e:
        print(f"❌ ERROR in BACKGROUND SAVE THREAD: {e}")
    print("="*30 + "\n")

@app.route('/stop_session', methods=['POST'])
def stop_session():
    """Stops recording and saves video in the background."""
    print("\n" + "="*30)
    print("DEBUG: Received /stop_session request.")
    
    with current_session_data["lock"]:
        if not current_session_data["is_recording"]:
            print("INFO: /stop_session called but already stopped.")
            return jsonify({"status": "already stopped"}), 200

        current_session_data["is_recording"] = False
        print("INFO: is_recording state is now FALSE. Waiting for in-flight frames...")

    time.sleep(1.5) # Grace period for frames in transit

    with current_session_data["lock"]:
        duration = time.time() - current_session_data["start_time"]
        frames_to_save = list(current_session_data["frames"])
        save_path = current_session_data["output_path"]
        print(f"INFO: Grace period over. Total frames captured: {len(frames_to_save)}")
    
    threading.Thread(target=save_file_in_background, args=(frames_to_save, save_path, duration)).start()

    with current_session_data["lock"]:
        current_session_data["frames"] = []
        current_session_data["output_path"] = ""
    
    print("DEBUG: /stop_session request finished.")
    print("="*30 + "\n")
    return jsonify({"status": "success", "message": "Stop command received."})

if __name__ == '__main__':
    try:
        # --- IMPORTANT: PASTE YOUR NGROK AUTHTOKEN HERE ---
        ngrok.set_auth_token("3B1usqCMqogFMZ5B1Gf8FSWi8Ja_2GzPpkSiCgGSEUEhgUo8d")
        # ----------------------------------------------------
    except Exception as e:
        print(f"Could not set ngrok authtoken: {e}")
        
    for tunnel in ngrok.get_tunnels():
        if '5000' in tunnel.config['addr']:
            print(f"Disconnecting old tunnel: {tunnel.public_url}")
            ngrok.disconnect(tunnel.public_url)
            
    try:
        public_url = ngrok.connect(5000)
        print(f" * Ngrok Tunnel is active at: {public_url}")
        print(" * This is the URL you must put into the 'NGROK_URL' variable in your desktop app.")
        app.run(host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        print(f"!!! NGROK FAILED TO START: {e}")
    finally:
        print("\nShutting down ngrok tunnels...")
        ngrok.kill()


