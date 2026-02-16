import cv2

# Initialize the webcam. The '0' is usually the default built-in webcam.
# If you have multiple webcams, you might need to try '1', '2', etc.
cap = cv2.VideoCapture(0)

# Check if the webcam was opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop forever until the user presses the 'q' key
while True:
    # Read one frame from the webcam
    # 'ret' is a boolean that is True if the frame was read successfully
    # 'frame' is the actual image data
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    # Display the resulting frame in a window titled "Webcam Feed"
    cv2.imshow('Webcam Feed', frame)

    # Wait for 1 millisecond. If the 'q' key is pressed during this time,
    # cv2.waitKey(1) will return the ASCII value of 'q', and we break the loop.
    if cv2.waitKey(1) == ord('q'):
        break

# When the loop is finished, release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

print("Webcam feed closed successfully.")