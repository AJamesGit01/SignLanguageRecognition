# Real-Time ASL Phrase Prediction (Two-Hand Version) using TensorFlow

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# === Paths ===
MODEL_PATH = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\models\ASLDatasetModels\ASL_Dataset.keras"
LABEL_PATH = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\dataset\ASL\ASL_Dataset_Classes.npy"

# === Load model and labels ===
print("📦 Loading TensorFlow model...")
model = tf.keras.models.load_model(MODEL_PATH)
label_classes = np.load(LABEL_PATH, allow_pickle=True)
print(f"✅ Loaded model with {len(label_classes)} output labels")

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,                 # ✅ Allow both hands
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# === Helper: Extract both-hand keypoints (126 features = 2 × 21 × 3) ===
def extract_two_hand_keypoints(results):
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[hand_idx].classification[0].label
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            if label.lower() == 'left':
                left_hand = np.array(coords)
            else:
                right_hand = np.array(coords)

    # Always return fixed-length 126-dim vector
    return np.concatenate([left_hand, right_hand])

# === Prediction Smoothing ===
predictions_queue = deque(maxlen=10)

# === Start Webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("🚫 Cannot access webcam.")
    exit()

print("🎥 Starting webcam... Press 'q' to quit.")
print("🖐 Show both hands clearly to the camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed, skipping...")
        continue

    # Flip and preprocess
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # === Extract hand features (always 126 features) ===
    features = extract_two_hand_keypoints(results).reshape(1, -1)

    # === Predict if at least one hand is detected ===
    if np.any(features):
        probs = model.predict(features, verbose=0)
        pred_idx = np.argmax(probs)
        pred_label = label_classes[pred_idx]
        confidence = probs[0][pred_idx]

        predictions_queue.append(pred_label)
        stable_prediction = max(set(predictions_queue), key=predictions_queue.count)

        # Display text
        cv2.putText(frame,
                    f"{stable_prediction} ({confidence*100:.1f}%)",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Hands Detected",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)

    # === Draw landmarks for both hands ===
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0),
                                       thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0),
                                       thickness=2)
            )

    # Show live frame
    cv2.imshow("ASL Phrase Recognition (TensorFlow Two-Hand)", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
hands.close()
print("🛑 Webcam closed.")
