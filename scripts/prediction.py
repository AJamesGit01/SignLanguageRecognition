import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
from pathlib import Path
from collections import deque
from queue import Queue, Empty
from flask import Flask, Response, jsonify, make_response
import threading

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
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

TFLITE_PATH  = MODELS_DIR / "Sign_Model.tflite"
CLASSES_PATH = MODELS_DIR / "classes.npy"
MEAN_PATH    = MODELS_DIR / "norm_mean.npy"
STD_PATH     = MODELS_DIR / "norm_std.npy"

# =============================================
# FLASK MJPEG STREAM
# =============================================
app = Flask(__name__)
jpeg_frame = None
jpeg_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()
preview_frame = None
preview_lock = threading.Lock()
frame_queue = Queue(maxsize=1)
accepted_label = ""
accepted_lock = threading.Lock()


def mjpeg_stream():
    boundary = b"--frame\r\n"
    while True:
        with jpeg_lock:
            frame = jpeg_frame
        if frame is not None:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/accepted_label")
def accepted_label_endpoint():
    with accepted_lock:
        label = accepted_label
    resp = make_response(jsonify({"label": label}))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# =============================================
# LOAD METADATA
# =============================================
classes = np.load(str(CLASSES_PATH), allow_pickle=True)
mean = np.load(str(MEAN_PATH)).astype(np.float32)
std  = np.load(str(STD_PATH)).astype(np.float32)

# =============================================
# LOAD TFLITE MODEL (ONCE)
# =============================================
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
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

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

# Start MJPEG server in a background thread
threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False),
    daemon=True
).start()


def capture_loop():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        with frame_lock:
            latest_frame = frame
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        try:
            frame_queue.put_nowait(frame)
        except Exception:
            pass


def stream_loop():
    global jpeg_frame
    while True:
        with preview_lock:
            frame = None if preview_frame is None else preview_frame.copy()
        if frame is None:
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        stream_frame = cv2.resize(frame, (640, 480))
        ret, jpeg = cv2.imencode(
            ".jpg",
            stream_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
        )
        if ret:
            with jpeg_lock:
                jpeg_frame = jpeg.tobytes()
        time.sleep(0.01)


threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=stream_loop, daemon=True).start()

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
    try:
        frame = frame_queue.get_nowait()
    except Empty:
        time.sleep(0.01)
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if not results.multi_hand_landmarks:
        prev_pos = None
        cooldown = max(0, cooldown - 1)

        cv2.putText(frame, "No Hands", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Update preview frame for MJPEG stream even when no hands are detected
        with preview_lock:
            preview_frame = frame.copy()

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
                    with accepted_lock:
                        accepted_label = dominant

                dominance_counter.clear()
                cooldown = COOLDOWN_FRAMES

    cv2.putText(frame, " ".join(recognized_sentence[-10:]),
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255,255,0), 2)

    # Update preview frame for MJPEG stream (includes landmarks + text)
    with preview_lock:
        preview_frame = frame.copy()

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFINAL GLOSS SEQUENCE:")
print(recognized_sentence)
