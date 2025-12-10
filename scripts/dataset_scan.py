import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

# ===============================
# CONFIG
# ===============================
BASE_DIR = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\dataset"
ASL_DIR = os.path.join(BASE_DIR, "ASL")
FSL_DIR = os.path.join(BASE_DIR, "FSL")
SHARED_DIR = os.path.join(BASE_DIR, "SHARED")

EXPECTED_COLS = 6301        # 6300 features + label
SEQ_LEN = 50
FEATURES = 126


# ===============================
# LOAD CSV FILES
# ===============================
def get_csv_files(folder):
    return glob.glob(os.path.join(folder, "*.csv"))


asl_files = get_csv_files(ASL_DIR)
fsl_files = get_csv_files(FSL_DIR)
shared_files = get_csv_files(SHARED_DIR)

print("===== SCANNING DATASETS =====")
print(f"📁 ASL files found: {len(asl_files)}")
print(f"📁 FSL files found: {len(fsl_files)}\n")
print(f"📁 SHARED files found: {len(shared_files)}\n")

all_files = asl_files + fsl_files

# Reports
shape_errors = []
missing_hand_sequences = []
labels_list = []


# ===============================
# CHECK EACH FILE
# ===============================
for file in all_files:
    df = pd.read_csv(file)

    filename = os.path.basename(file)

    # -------- A. SHAPE VALIDATION --------
    if df.shape[1] != EXPECTED_COLS:
        shape_errors.append((filename, df.shape[1]))

    # -------- COLLECT LABELS FOR C --------
    labels_list.extend(df["label"].tolist())

    # -------- D. ZERO-HAND CHECK --------
    # left hand = first 63 * 50 = 3150 features
    # right hand = next 3150 features
    left_columns = list(range(0, 63 * SEQ_LEN))
    right_columns = list(range(63 * SEQ_LEN, 63 * SEQ_LEN * 2))

    for idx, row in df.iterrows():
        left_sum = row[left_columns].sum()
        right_sum = row[right_columns].sum()

        if left_sum == 0 and right_sum == 0:
            missing_hand_sequences.append((filename, idx))


# ===============================
# RESULTS
# ===============================

print("===== A. SHAPE VALIDATION =====")
if shape_errors:
    for file, cols in shape_errors:
        print(f"❌ {file}: {cols} columns (expected 6301)")
else:
    print("✅ All files have correct 6301 columns.")


print("\n===== C. CLASS DISTRIBUTION =====")
class_counts = Counter(labels_list)
for label, count in class_counts.items():
    print(f"{label}: {count} samples")

print("\n⚠️ IMBALANCE WARNING:")
for label, count in class_counts.items():
    if count < 50:
        print(f"❗ {label}: very low samples ({count})")
    elif count > max(class_counts.values()) * 0.5:
        print(f"❗ {label}: dominating class ({count})")


print("\n===== D. MISSING HAND SEQUENCES =====")
if missing_hand_sequences:
    print(f"❌ Found {len(missing_hand_sequences)} sequences with BOTH hands missing:")
    for file, seq_id in missing_hand_sequences[:20]:  # limit output
        print(f"- {file}, sequence #{seq_id}")
else:
    print("✅ No sequences with both hands missing.")


print("\n===== SCAN COMPLETE =====")
