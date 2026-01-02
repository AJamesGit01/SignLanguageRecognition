import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

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

TFLITE_PATH  = f"{BASE}\\Sign_Model.tflite"
CLASSES_PATH = f"{BASE}\\classes.npy"
MEAN_PATH    = f"{BASE}\\norm_mean.npy"
STD_PATH     = f"{BASE}\\norm_std.npy"

# =============================================
# LOAD METADATA
# =============================================
classes = np.load(CLASSES_PATH, allow_pickle=True)
mean = np.load(MEAN_PATH).astype(np.float32)
std  = np.load(STD_PATH).astype(np.float32)

# =============================================
# LOAD TFLITE MODEL (ONCE)
# =============================================
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ Input shape:", input_details[0]["shape"])
print("✅ Input dtype:", input_details[0]["dtype"])

# =============================================
# MEDIAPIPE HANDS
# =============================================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =============================================
# BUFFERS
# =============================================
sequence = deque(maxlen=SEQ_LEN)
prev_pos = None

recognized_sentence = []
dominance_counter = {}
cooldown = 0
frame_count = 0

cap = cv2.VideoCapture(0)

# =============================================
# TFLITE PREDICTION FUNCTION
# =============================================
def predict_label(window):
    arr = np.array(window, dtype=np.float32).reshape(1, SEQ_LEN, 252)
    arr = (arr - mean) / std

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    idx = np.argmax(probs)
    return classes[idx], float(probs[idx])

# =============================================
# REAL-TIME LOOP
# =============================================
print("🎥 TFLite Dominance-Based Prediction Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if not results.multi_hand_landmarks:
        prev_pos = None
        cooldown = max(0, cooldown - 1)

        cv2.putText(frame, "No Hands", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    for i, hand in enumerate(results.multi_hand_landmarks):
        label = results.multi_handedness[i].classification[0].label.lower()
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()

        if label == "left":
            left[:] = coords
        else:
            right[:] = coords

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    pos = np.concatenate([left, right])

    vel = np.zeros_like(pos) if prev_pos is None else pos - prev_pos
    prev_pos = pos.copy()

    sequence.append(np.concatenate([pos, vel]))

    frame_count += 1
    cooldown = max(0, cooldown - 1)

    if len(sequence) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:
        label, conf = predict_label(sequence)

        if conf >= CONF_THRESHOLD:
            dominance_counter[label] = dominance_counter.get(label, 0) + 1
            dominant = max(dominance_counter, key=dominance_counter.get)

            if dominance_counter[dominant] >= DOMINANCE_THRESHOLD and cooldown == 0:
                if not recognized_sentence or dominant != recognized_sentence[-1]:
                    recognized_sentence.append(dominant)
                    print("✔ ACCEPTED:", dominant)

                dominance_counter.clear()
                cooldown = COOLDOWN_FRAMES

    cv2.putText(frame, " ".join(recognized_sentence[-10:]),
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255,255,0), 2)

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFINAL GLOSS SEQUENCE:")
print(recognized_sentence)
