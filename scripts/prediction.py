import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter
import time

# =============================================
#              CONFIG
# =============================================
SEQ_LEN = 50
PREDICT_EVERY = 2
SMOOTH_WINDOW = 8
CONF_THRESHOLD = 0.70        # NEW — minimum confidence
COOLDOWN_FRAMES = 12         # NEW — avoid repeating same gloss

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
num_classes = len(classes)

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
predictions_smooth = deque(maxlen=SMOOTH_WINDOW)
recognized_sentence = []
frame_count = 0
cooldown = 0     # NEW — cooldown timer

cap = cv2.VideoCapture(0)

# =============================================
#              PREDICTION FUNCTION
# =============================================
def predict_label(window):
    array = np.array(window, dtype=np.float32).reshape(1, SEQ_LEN, 126)
    probs = model.predict(array, verbose=0)
    idx = np.argmax(probs)
    return classes[idx], float(probs[0][idx])


# =============================================
#         🔵 REAL-TIME LOOP
# =============================================
print("🎥 ASL+FSL Mixed Prediction Started...")

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

        cooldown = max(0, cooldown - 1)  # cooldown still counts
        cv2.imshow("ASL+FSL+SHARED Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for i, hand in enumerate(results.multi_hand_landmarks):
        hand_label = results.multi_handedness[i].classification[0].label.lower()
        coords = []

        for lm in hand.landmark:
            coords.extend([lm.x, lm.y, lm.z])

        if hand_label == "left":
            left = np.array(coords)
        else:
            right = np.array(coords)

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    sequence.append(np.concatenate([left, right]))
    frame_count += 1

    if cooldown > 0:
        cooldown -= 1

    if len(sequence) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:

        label, conf = predict_label(sequence)
        predictions_smooth.append(label)

        stable_label = Counter(predictions_smooth).most_common(1)[0][0]

        # 💡 Confidence filter (NEW)
        if conf < CONF_THRESHOLD:
            stable_label = "..."

        # Print to terminal (debug)
        print(f"Predicted: {label} | Conf: {conf:.2f} | Stable: {stable_label} | Cooldown: {cooldown}")

        # 💡 ADD TO SENTENCE IF:
        #    - not in cooldown
        #    - not "..."
        #    - different from last added gloss
        if cooldown == 0 and stable_label != "..." \
           and (len(recognized_sentence) == 0 or stable_label != recognized_sentence[-1]):
            
            recognized_sentence.append(stable_label)
            cooldown = COOLDOWN_FRAMES  # reset cooldown

    # Display current recognized gloss sequence
    cv2.putText(frame, " ".join(recognized_sentence[-10:]),
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 0), 2)

    # Display current stable label
    cv2.putText(frame, f"Stable: {stable_label} ({conf*100:.0f}%)",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)

    cv2.imshow("ASL+FSL+SHARED Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFinal Recognized Gloss Sequence:")
print(recognized_sentence)
