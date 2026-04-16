import ctypes
import time

# Load required Windows libraries
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

print("---------------------------------------")
print("ACTIVE WINDOW MONITORING STARTED")
print("Switch between different apps to see their titles.")
print("Press CTRL+C in the command prompt to stop.")
print("---------------------------------------")

def get_active_window_title():
    # Get the handle (ID) of the foreground window
    hwnd = user32.GetForegroundWindow()
    
    # Get the length of the window title
    length = user32.GetWindowTextLengthW(hwnd)
    
    # Create a buffer to store the title
    buff = ctypes.create_unicode_buffer(length + 1)
    
    # Get the title text
    user32.GetWindowTextW(hwnd, buff, length + 1)
    
    return buff.value

last_window = ""

try:
    while True:
        current_window = get_active_window_title()
        
        # Only print if the window has changed
        if current_window != last_window:
            print(f"Active Window: {current_window}")
            last_window = current_window
        
        # Check every 0.5 seconds
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopping script...")