import time
import csv
import os
from pynput import keyboard

# --- CONFIGURATION ---
OUTPUT_FILE = "researcher_event_log.csv"

class EventLogger:
    def __init__(self):
        # Create CSV if not exists, else append
        file_exists = os.path.isfile(OUTPUT_FILE)
        self.file = open(OUTPUT_FILE, 'a', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        
        if not file_exists:
            self.writer.writerow(['Timestamp', 'Event_Label', 'Notes'])
        
        print(f"Logging events to: {OUTPUT_FILE}")
        print("------------------------------------------------")
        print("CONTROLS:")
        print("[SPACE] : SYNC CLAP (Start of Session)")
        print("[ 1 ]   : Start NORMAL_KEYBOARD")
        print("[ 2 ]   : Start NORMAL_PAPER")
        print("[ 3 ]   : Start SUSPICIOUS_PHONE")
        print("[ 4 ]   : Start SUSPICIOUS_SIDE_GLANCE")
        print("[ 5 ]   : Start DISTRACTED_SCREEN")
        print("[ 0 ]   : END Current Action / Return to Neutral")
        print("[ ESC ] : Exit Logger")
        print("------------------------------------------------")

    def log(self, label):
        t = time.time()
        print(f"{t} -> LOGGED: {label}")
        self.writer.writerow([t, label, ""])
        self.file.flush() # Ensure it saves immediately

    def start(self):
        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self.log("SYNC_CLAP")
                elif key == keyboard.Key.esc:
                    print("Exiting...")
                    return False
                elif key.char == '1':
                    self.log("START_NORMAL_KEYBOARD")
                elif key.char == '2':
                    self.log("START_NORMAL_PAPER")
                elif key.char == '3':
                    self.log("START_SUSPICIOUS_PHONE")
                elif key.char == '4':
                    self.log("START_SUSPICIOUS_SIDE")
                elif key.char == '5':
                    self.log("START_DISTRACTED_SCREEN")
                elif key.char == '0':
                    self.log("END_ACTION")
            except AttributeError:
                pass # Ignore special keys other than space/esc

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        
        self.file.close()

if __name__ == "__main__":
    logger = EventLogger()
    logger.start()