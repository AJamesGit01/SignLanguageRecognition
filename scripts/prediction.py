import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter

# =============================================
#              CONFIG
# =============================================
SEQ_LEN = 50
PREDICT_EVERY = 2
CONF_THRESHOLD = 0.25
STABILITY_REQUIRED = 4        # NEW — must be stable for N predictions
COOLDOWN_FRAMES = 12          # avoid duplicates

# =============================================
#              PATHS
# =============================================
MODEL_PATH = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\models\Sign_Model.keras"
CLASSES_PATH = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\models\classes.npy"

# =============================================
#              LOAD MODEL + LABELS
# =============================================
print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)
print("Loaded classes:", classes)

# =============================================
#              MEDIAPIPE SETUP
# =============================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# =============================================
#              BUFFERS
# =============================================
sequence = deque(maxlen=SEQ_LEN)
confidence_history = deque(maxlen=10)   # NEW — for stability
label_history = deque(maxlen=10)
recognized_sentence = []
cooldown = 0
frame_count = 0

cap = cv2.VideoCapture(0)

# =============================================
#              PREDICTION FUNCTION
# =============================================
def predict_label(window):
    arr = np.array(window, dtype=np.float32).reshape(1, SEQ_LEN, 126)
    probs = model.predict(arr, verbose=0)[0]
    idx = np.argmax(probs)
    return classes[idx], float(probs[idx]), probs


# =============================================
#           🔵 REAL-TIME LOOP (UPDATED)
# =============================================
print("🎥 ASL+FSL+SHARED Stable Prediction Started...")

stable_candidate = None
stability_counter = 0  # NEW

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left = np.zeros(63)
    right = np.zeros(63)

    if not (results.multi_hand_landmarks and results.multi_handedness):
        cv2.putText(frame, "No Hands Detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cooldown = max(0, cooldown - 1)
        cv2.imshow("ASL+FSL+SHARED Recognition", frame)
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
            left = np.array(coords)
        else:
            right = np.array(coords)

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    sequence.append(np.concatenate([left, right]))
    frame_count += 1

    if cooldown > 0:
        cooldown -= 1

    # ========== MAIN PREDICTION ==========
    if len(sequence) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:

        label, conf, full_probs = predict_label(sequence)

        # store history
        label_history.append(label)
        confidence_history.append(conf)

        # Find the most CONFIDENT label over last history
        max_conf_idx = np.argmax(full_probs)
        strongest_label = classes[max_conf_idx]
        strongest_conf = full_probs[max_conf_idx]

        print(f"Predicted: {label} | Strongest: {strongest_label} ({strongest_conf:.2f})")

        # IGNORE LOW CONFIDENCE
        if strongest_conf < CONF_THRESHOLD:
            stable_candidate = None
            stability_counter = 0
        else:
            # Check stability over time
            if stable_candidate == strongest_label:
                stability_counter += 1
            else:
                stable_candidate = strongest_label
                stability_counter = 1

        # Only accept gloss after being stable enough
        if stability_counter >= STABILITY_REQUIRED and cooldown == 0:
            if len(recognized_sentence) == 0 or stable_candidate != recognized_sentence[-1]:
                recognized_sentence.append(stable_candidate)
                cooldown = COOLDOWN_FRAMES
                print(f"✔ ACCEPTED: {stable_candidate}")

    # Display sentence
    cv2.putText(frame, " ".join(recognized_sentence[-10:]),
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display stable prediction
    cv2.putText(frame, f"Stable: {stable_candidate}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)

    cv2.imshow("ASL+FSL+SHARED Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFINAL GLOSS SEQUENCE:")
print(recognized_sentence)
