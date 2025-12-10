import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# =============================================
#              CONFIG
# =============================================
SEQ_LEN = 50
PREDICT_EVERY = 2
CONF_THRESHOLD = 0.25
COOLDOWN_FRAMES = 12

DOMINANCE_THRESHOLD = 12   # NEW – required dominant appearances

# =============================================
#              PATHS
# =============================================
MODEL_PATH = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\models\Sign_Model.keras"
CLASSES_PATH = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\models\classes.npy"

# =============================================
#              LOAD MODEL + LABELS
# =============================================
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

# =============================================
#              MEDIAPIPE
# =============================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# =============================================
#              BUFFERS
# =============================================
sequence = deque(maxlen=SEQ_LEN)
recognized_sentence = []
cooldown = 0
frame_count = 0

dominance_counter = {}   # NEW — counts label dominance

cap = cv2.VideoCapture(0)

# =============================================
#              PREDICT FUNCTION
# =============================================
def predict_label(window):
    arr = np.array(window, dtype=np.float32).reshape(1, SEQ_LEN, 126)
    probs = model.predict(arr, verbose=0)[0]
    idx = np.argmax(probs)
    return classes[idx], float(probs[idx]), probs

# =============================================
#        🔵 REAL-TIME LOOP (DOMINANCE VERSION)
# =============================================
print("🎥 Dominance-Based Stable Prediction Started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left = np.zeros(63)
    right = np.zeros(63)

    # If NO HANDS detected
    if not results.multi_hand_landmarks:
        cooldown = max(0, cooldown - 1)

        cv2.putText(frame, "No Hands Detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        cv2.putText(frame, " ".join(recognized_sentence[-10:]),
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255,255,0), 2)

        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Extract hand landmarks
    for i, hand in enumerate(results.multi_hand_landmarks):
        hl = results.multi_handedness[i].classification[0].label.lower()
        coords = []
        for lm in hand.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        if hl == "left":
            left = coords
        else:
            right = coords

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    sequence.append(np.concatenate([left, right]))
    frame_count += 1

    if cooldown > 0:
        cooldown -= 1

    # ---------------- PREDICTION ----------------
    if len(sequence) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:

        raw_label, raw_conf, full_probs = predict_label(sequence)
        idx = np.argmax(full_probs)
        strongest_label = classes[idx]
        strongest_conf = full_probs[idx]

        # Low confidence → skip
        if strongest_conf < CONF_THRESHOLD:
            # Do NOT break loop — just skip prediction
            pass
        else:
            # Count dominance
            if strongest_label not in dominance_counter:
                dominance_counter[strongest_label] = 1
            else:
                dominance_counter[strongest_label] += 1

            # Pick the most dominant label so far
            dominant_label = max(dominance_counter,
                                 key=dominance_counter.get)
            dominant_count = dominance_counter[dominant_label]

            # Accept dominant gloss
            if dominant_count >= DOMINANCE_THRESHOLD and cooldown == 0:

                # avoid duplicates
                if len(recognized_sentence) == 0 or dominant_label != recognized_sentence[-1]:
                    recognized_sentence.append(dominant_label)
                    print(f"✔ ACCEPTED: {dominant_label}")

                # Reset for next gloss
                dominance_counter.clear()
                cooldown = COOLDOWN_FRAMES

    # ---------------- DISPLAY ONLY ACCEPTED ----------------
    cv2.putText(frame, " ".join(recognized_sentence[-10:]),
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255,255,0), 2)

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFINAL GLOSS SEQUENCE:")
print(recognized_sentence)
