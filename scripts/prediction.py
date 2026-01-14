import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import base64

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

class Recognizer:
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence = deque(maxlen=SEQ_LEN)
        self.prev_pos = None
        self.recognized_sentence = []
        self.dominance_counter = {}
        self.cooldown = 0
        self.frame_count = 0

    def predict_label(self, window):
        arr = np.array(window, dtype=np.float32).reshape(1, SEQ_LEN, 252)
        arr = (arr - mean) / std
        interpreter.set_tensor(input_details[0]["index"], arr)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]["index"])[0]
        idx = np.argmax(probs)
        return classes[idx], float(probs[idx])

    def process_frame(self, frame_bgr):
        frame = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        left = np.zeros(63, dtype=np.float32)
        right = np.zeros(63, dtype=np.float32)
        if not results.multi_hand_landmarks:
            self.prev_pos = None
            self.cooldown = max(0, self.cooldown - 1)
            self.frame_count += 1
            return None
        for i, hand in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label.lower()
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
            if label == "left":
                left[:] = coords
            else:
                right[:] = coords
        pos = np.concatenate([left, right])
        vel = np.zeros_like(pos) if self.prev_pos is None else pos - self.prev_pos
        self.prev_pos = pos.copy()
        self.sequence.append(np.concatenate([pos, vel]))
        self.frame_count += 1
        self.cooldown = max(0, self.cooldown - 1)
        accepted = None
        if len(self.sequence) == SEQ_LEN and self.frame_count % PREDICT_EVERY == 0:
            label, conf = self.predict_label(self.sequence)
            if conf >= CONF_THRESHOLD:
                self.dominance_counter[label] = self.dominance_counter.get(label, 0) + 1
                dominant = max(self.dominance_counter, key=self.dominance_counter.get)
                if self.dominance_counter[dominant] >= DOMINANCE_THRESHOLD and self.cooldown == 0:
                    if not self.recognized_sentence or dominant != self.recognized_sentence[-1]:
                        self.recognized_sentence.append(dominant)
                        accepted = dominant
                    self.dominance_counter.clear()
                    self.cooldown = COOLDOWN_FRAMES
        return accepted

    def get_sentence(self):
        return list(self.recognized_sentence)

    def reset(self):
        self.sequence.clear()
        self.prev_pos = None
        self.recognized_sentence = []
        self.dominance_counter = {}
        self.cooldown = 0
        self.frame_count = 0

def _run_webcam():
    recog = Recognizer()
    cap = cv2.VideoCapture(0)
    print("🎥 TFLite Dominance-Based Prediction Running...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        accepted = recog.process_frame(frame)
        if accepted is not None:
            print("✔ ACCEPTED:", accepted)
        cv2.putText(frame, " ".join(recog.get_sentence()[-10:]), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("\nFINAL GLOSS SEQUENCE:")
    print(recog.get_sentence())

if __name__ == "__main__":
    _run_webcam()
