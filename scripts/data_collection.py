import cv2
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque
import pandas as pd

# === CONFIG ===
DATA_DIR = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\dataset\FSL"
os.makedirs(DATA_DIR, exist_ok=True)

label = input("Enter label: ").strip().lower()
SAVE_PATH = os.path.join(DATA_DIR, f"{label}.csv")

SEQ_LEN = 50                  # frames per sequence
MAX_SEQUENCES = 500           # NEW — auto-stop when reached

# === MediaPipe Hands ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

sequence = deque(maxlen=SEQ_LEN)
collected_sequences = []

cap = cv2.VideoCapture(0)
print("\n📷 Starting capture in 3 seconds...")
time.sleep(3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Extract 126 features per frame
    left = np.zeros(63)
    right = np.zeros(63)

    # === NO HANDS DETECTED HANDLING (NEW) ===
    if not (results.multi_hand_landmarks and results.multi_handedness):
        cv2.putText(frame, "No hands detected!", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.putText(frame, f"Sequences: {len(collected_sequences)}/{MAX_SEQUENCES}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        cv2.imshow("Dynamic Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        continue  # Skip to next frame (NO saving)

    # If hands detected
    for i, hand in enumerate(results.multi_hand_landmarks):
        handed = results.multi_handedness[i].classification[0].label
        lm_list = []

        for lm in hand.landmark:
            lm_list.extend([lm.x, lm.y, lm.z])

        if handed.lower() == "left":
            left = np.array(lm_list)
        else:
            right = np.array(lm_list)

    # Combine features
    frame_features = np.concatenate([left, right])
    sequence.append(frame_features)

    # Draw
    for landmark in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, landmark, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Label: {label}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Frames: {len(sequence)}/{SEQ_LEN}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"Sequences: {len(collected_sequences)}/{MAX_SEQUENCES}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Dynamic Capture", frame)

    # === Save when full sequence ===
    if len(sequence) == SEQ_LEN:
        collected_sequences.append(np.array(sequence).flatten())
        print(f"📦 Saved sequence #{len(collected_sequences)}")

    # === AUTO CLOSE WHEN 500 REACHED (NEW) ===
    if len(collected_sequences) >= MAX_SEQUENCES:
        print("\n🎉 500 sequences collected! Auto-saving and closing...")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === SAVE DATA ===
if collected_sequences:
    df = pd.DataFrame(collected_sequences)
    df["label"] = label
    df.to_csv(SAVE_PATH, index=False)

    print(f"\n✅ Saved dataset to {SAVE_PATH}")
    print(f"Total sequences collected: {len(collected_sequences)}")
else:
    print("🚫 No sequences collected.")
