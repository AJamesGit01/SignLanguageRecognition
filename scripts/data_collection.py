import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# === Configuration ===
DATA_DIR = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\dataset\FSL"
os.makedirs(DATA_DIR, exist_ok=True)

label = input("Book").strip().lower()
SAVE_PATH = os.path.join(DATA_DIR, f"{label}.csv")

# === MediaPipe setup ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, 
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)
data = []
frame_count = 0
saved_count = 0

print("📷 Starting capture in 3 seconds...")
time.sleep(3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    frame_count += 1

    # Detect hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # === If no hand detected ===
    if not results.multi_hand_landmarks:
        cv2.putText(frame, "No hands detected!", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Two-Hand Capture", frame)

        # Skip saving this frame (no data collected)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # === If hand(s) detected, record landmarks ===
    row = []
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])

    # Pad second hand if only one hand detected
    if len(results.multi_hand_landmarks) == 1:
        row.extend([0] * (21 * 3))

    row.append(label)
    data.append(row)
    saved_count += 1

    # Draw hands on frame
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Collecting: {label}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Samples: {saved_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Two-Hand Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Save dataset ===
columns = []
for hand in ["L1_", "L2_"]:
    for i in range(21):
        columns += [f"{hand}x{i}", f"{hand}y{i}", f"{hand}z{i}"]
columns.append("label")

df = pd.DataFrame(data, columns=columns)

if not df.empty:
    df.to_csv(SAVE_PATH, index=False)
    print(f"\n✅ Dataset saved to {SAVE_PATH}")
    print(f"🧮 Frames processed: {frame_count}")
    print(f"💾 Valid samples collected: {saved_count}")
else:
    print("🚫 No valid samples collected (no hands detected). File not saved.")