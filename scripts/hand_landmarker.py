# Prediction Script with Dominance-Based Stable Prediction


import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =============================================
# CONFIG
# =============================================
SEQ_LEN = 50
PREDICT_EVERY = 2
CONF_THRESHOLD = 0.25
COOLDOWN_FRAMES = 12
DOMINANCE_THRESHOLD = 12

# =============================================
# PATHS
# =============================================
BASE = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\models"

MODEL_PATH   = f"{BASE}\\Sign_Model.keras"
CLASSES_PATH = f"{BASE}\\classes.npy"
MEAN_PATH    = f"{BASE}\\norm_mean.npy"
STD_PATH     = f"{BASE}\\norm_std.npy"
TASK_PATH    = f"{BASE}\\hand_landmarker.task"

# =============================================
# LOAD MODEL
# =============================================
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)
mean = np.load(MEAN_PATH)
std = np.load(STD_PATH)

# =============================================
# MEDIAPIPE HAND LANDMARKER
# =============================================
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_landmarker = HandLandmarker.create_from_options(options)

# =============================================
# BUFFERS
# =============================================
sequence = deque(maxlen=SEQ_LEN)
prev_frame = None
recognized_sentence = []
cooldown = 0
frame_count = 0
dominance_counter = {}

cap = cv2.VideoCapture(0)

# =============================================
# PREDICT FUNCTION
# =============================================
def predict_label(window):
    arr = np.array(window, dtype=np.float32).reshape(1, SEQ_LEN, 252)
    arr = (arr - mean) / std
    probs = model.predict(arr, verbose=0)[0]
    idx = np.argmax(probs)
    return classes[idx], float(probs[idx])

# =============================================
# REAL-TIME LOOP
# =============================================
print("🎥 Dominance-Based Stable Prediction Started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    left = np.zeros(63)
    right = np.zeros(63)

    if result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name.lower()
            coords = []
            for lm in hand_landmarks:
                coords.extend([lm.x, lm.y, lm.z])

            if handedness == "left":
                left = coords
            else:
                right = coords

    else:
        cooldown = max(0, cooldown - 1)
        cv2.putText(frame, "No Hands Detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # ---------------- FEATURE BUILD ----------------
    pos = np.concatenate([left, right])

    vel = np.zeros_like(pos) if prev_frame is None else pos - prev_frame
    prev_frame = pos.copy()

    full_feat = np.concatenate([pos, vel])
    sequence.append(full_feat)

    frame_count += 1
    cooldown = max(0, cooldown - 1)

    # ---------------- PREDICTION ----------------
    if len(sequence) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:
        label, conf = predict_label(sequence)

        if conf >= CONF_THRESHOLD:
            dominance_counter[label] = dominance_counter.get(label, 0) + 1
            dominant_label = max(dominance_counter, key=dominance_counter.get)

            if dominance_counter[dominant_label] >= DOMINANCE_THRESHOLD and cooldown == 0:
                if not recognized_sentence or dominant_label != recognized_sentence[-1]:
                    recognized_sentence.append(dominant_label)
                    print(f"✔ ACCEPTED: {dominant_label}")

                dominance_counter.clear()
                cooldown = COOLDOWN_FRAMES

    # ---------------- DISPLAY ----------------
    cv2.putText(frame, " ".join(recognized_sentence[-10:]),
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 0), 2)

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFINAL GLOSS SEQUENCE:")
print(recognized_sentence)
