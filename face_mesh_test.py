import cv2
import mediapipe as mp
import numpy as np

# Define the new way to use MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create the face landmarker instance with the video mode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1) # We only care about one person

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# We need a timestamp to process the video frame
frame_timestamp_ms = 0

# The main loop, using the 'with' statement for proper resource management
with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        # Increment timestamp
        frame_timestamp_ms += 1 # A simple increment is fine for real-time display

        # Convert the frame to a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Detect face landmarks from the input image.
        face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Process the detection results.
        if face_landmarker_result.face_landmarks:
            for face_landmarks in face_landmarker_result.face_landmarks:
                # Draw the landmarks
                for landmark in face_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1) # Draw green dots

        # Display the resulting frame
        cv2.imshow('Facial Landmarks', frame)

        if cv2.waitKey(1) == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("Script finished successfully.")