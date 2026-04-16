from pynput import keyboard, mouse
import time

print("---------------------------------------------------------")
print("INTERACTION MONITORING ACTIVE")
print("1. Move your mouse around.")
print("2. Click somewhere.")
print("3. Type some keys.")
print("PRESS 'ESC' ON YOUR KEYBOARD TO STOP THE SCRIPT.")
print("---------------------------------------------------------")

# --- Mouse Callbacks ---
def on_move(x, y):
    # We print only every 20th move to avoid flooding the console too fast
    # In the real data collector, we will save ALL of them.
    if int(time.time() * 100) % 20 == 0:
        print(f"Mouse moved to ({x}, {y})")

def on_click(x, y, button, pressed):
    action = "Pressed" if pressed else "Released"
    print(f"Mouse {action} at ({x}, {y}) with {button}")

def on_scroll(x, y, dx, dy):
    print(f"Mouse Scrolled at ({x}, {y}) ({dx}, {dy})")

# --- Keyboard Callbacks ---
def on_press(key):
    try:
        print(f"Key Pressed: {key.char}")
    except AttributeError:
        print(f"Special Key Pressed: {key}")

def on_release(key):
    # Stop the script if ESC is pressed
    if key == keyboard.Key.esc:
        print("ESC pressed. Stopping listeners...")
        # Returning False stops the listener
        return False

# --- Start Listeners ---
# We start the mouse listener in a non-blocking way
mouse_listener = mouse.Listener(
    on_move=on_move,
    on_click=on_click,
    on_scroll=on_scroll)
mouse_listener.start()

# We start the keyboard listener. Since this is the last one,
# we can let it block the main thread until ESC is pressed.
with keyboard.Listener(on_press=on_press, on_release=on_release) as keyboard_listener:
    keyboard_listener.join()

print("Script finished successfully.")