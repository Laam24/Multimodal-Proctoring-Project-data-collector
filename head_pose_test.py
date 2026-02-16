import cv2
import mediapipe as mp
import numpy as np

# --- Boilerplate MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1)

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_timestamp_ms = 0

# --- Main Loop ---
with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        frame_timestamp_ms += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if face_landmarker_result.face_landmarks:
            # Get the first face's landmarks
            face_landmarks = face_landmarker_result.face_landmarks[0]

            # --- Head Pose Estimation Logic ---
            # 1. Get required 2D landmarks for PnP algorithm
            image_points = np.array([
                (face_landmarks[1].x, face_landmarks[1].y),     # Nose tip
                (face_landmarks[152].x, face_landmarks[152].y),   # Chin
                (face_landmarks[263].x, face_landmarks[263].y),   # Left eye left corner
                (face_landmarks[33].x, face_landmarks[33].y),    # Right eye right corner
                (face_landmarks[287].x, face_landmarks[287].y),   # Left Mouth corner
                (face_landmarks[57].x, face_landmarks[57].y)     # Right mouth corner
            ], dtype="double")

            # 2. Define a generic 3D model of a face
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left Mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])

            # 3. Approximate camera internals
            image_height, image_width, _ = frame.shape
            focal_length = image_width
            center = (image_width / 2, image_height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            
            # Assume no lens distortion
            dist_coeffs = np.zeros((4, 1))

            # Convert image points to pixel coordinates
            image_points[:, 0] *= image_width
            image_points[:, 1] *= image_height

            # 4. Use OpenCV's solvePnP to find head pose
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            # 5. Project a 3D axis onto the 2D image to visualize the pose
            (axis_end_points, _) = cv2.projectPoints(
                np.array([(200.0, 0.0, 0.0), (0.0, 200.0, 0.0), (0.0, 0.0, 200.0)]),
                rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # Draw the 3D axis on the image
            nose_tip_2d = (int(image_points[0, 0]), int(image_points[0, 1]))
            p1 = (int(axis_end_points[0][0][0]), int(axis_end_points[0][0][1])) # X-axis (Red)
            p2 = (int(axis_end_points[1][0][0]), int(axis_end_points[1][0][1])) # Y-axis (Green)
            p3 = (int(axis_end_points[2][0][0]), int(axis_end_points[2][0][1])) # Z-axis (Blue)

            cv2.line(frame, nose_tip_2d, p1, (0, 0, 255), 2) # Red line for X-axis
            cv2.line(frame, nose_tip_2d, p2, (0, 255, 0), 2) # Green line for Y-axis
            cv2.line(frame, nose_tip_2d, p3, (255, 0, 0), 2) # Blue line for Z-axis

        # Display the resulting frame
        cv2.imshow('Head Pose', frame)

        if cv2.waitKey(1) == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Script finished successfully.")