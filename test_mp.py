import mediapipe as mp
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing

print("✅ MediaPipe Version:", mp.__version__)
print("✅ Face Mesh Loaded:", mp_face_mesh)
print("✅ Drawing Utils Loaded:", mp_drawing)
print("All systems go!")