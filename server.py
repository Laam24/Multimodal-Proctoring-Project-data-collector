from flask import Flask, render_template, request, jsonify
import os
import threading
import time
from pyngrok import ngrok

app = Flask(__name__)
print("Flask App Initialized.")

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = "Honeypot_Sessions"

# Notice we removed the "frames" array because we no longer process individual JPEGs
current_session_data = {
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
    return jsonify({"status": "pong"})

@app.route('/start_session', methods=['POST'])
def start_session():
    """Triggered by the Desktop App to initialize the session."""
    print("\n" + "="*30)
    print("DEBUG: Received /start_session request.")
    data = request.get_json()
    session_dir = data.get("session_dir")

    if not session_dir:
        return jsonify({"status": "error", "message": "session_dir not provided"}), 400
    
    with current_session_data["lock"]:
        # Set to save the direct video blob from the phone
        current_session_data["output_path"] = os.path.join(session_dir, "phone_video.mp4")
        current_session_data["is_recording"] = True
        current_session_data["start_time"] = time.time()
    
    print(f"INFO: Phone camera recording started for session: {session_dir}")
    print("="*30 + "\n")
    return jsonify({"status": "success", "message": "Recording started"})

@app.route('/stop_session', methods=['POST'])
def stop_session():
    """Tells the phone to stop recording and upload the video."""
    print("\n" + "="*30)
    print("DEBUG: Received /stop_session request.")
    
    with current_session_data["lock"]:
        current_session_data["is_recording"] = False
        
    print("INFO: is_recording set to FALSE. Waiting for phone to upload video...")
    print("="*30 + "\n")
    return jsonify({"status": "success", "message": "Stop command received."})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Receives the final, complete video file directly from the phone's memory."""
    print("\n" + "="*30)
    print("DEBUG: Receiving high-quality video from phone...")
    
    with current_session_data["lock"]:
        save_path = current_session_data["output_path"]
    
    if not save_path:
        print("ERROR: No active session path found.")
        return jsonify({"error": "No active session"}), 400
        
    if 'video' not in request.files:
        print("ERROR: No video file received in the request.")
        return jsonify({"error": "No video file part"}), 400
        
    file = request.files['video']
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
        print(f"✅ SUCCESS: Flawless client-side video saved to:\n{save_path}")
    except Exception as e:
        print(f"❌ ERROR saving video: {e}")
        
    print("="*30 + "\n")
    return jsonify({"status": "success"})

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