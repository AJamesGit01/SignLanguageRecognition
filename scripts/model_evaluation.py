import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# CONFIG
# ===============================
BASE_DIR = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\dataset"
ASL_DIR = os.path.join(BASE_DIR, "ASL")
FSL_DIR = os.path.join(BASE_DIR, "FSL")
SHARED_DIR = os.path.join(BASE_DIR, "SHARED")  

MODEL_PATH = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\models\Sign_Model.keras"
CLASSES_PATH = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\models\classes.npy"

SEQ_LEN = 50
FEATURES = 126


# ===============================
# LOAD MODEL + LABELS
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)
class_to_index = {c: i for i, c in enumerate(classes)}

print("Loaded classes:", classes)


# ===============================
# HELPERS
# ===============================
def load_sequences(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    X, y = [], []

    for f in files:
        df = pd.read_csv(f)

        # --- CHECK COLUMN COUNT ---
        if df.shape[1] != (SEQ_LEN * FEATURES + 1):
            print(f"❌ ERROR: {f} has {df.shape[1]} columns (expected {SEQ_LEN*FEATURES+1})")
            continue

        for idx, row in df.iterrows():
            values = row.values

            # --- LAST COLUMN MUST BE LABEL ---
            label = values[-1]
            if label not in class_to_index:
                print(f"❌ Invalid label '{label}' in file {f}, row {idx}")
                continue

            feature_values = values[:-1]

            # --- CHECK IF NUMERIC ---
            try:
                feature_values = feature_values.astype(np.float32)
            except:
                print(f"❌ Non-numeric values in {f}, row {idx}")
                continue

            # --- CHECK LENGTH ---
            if len(feature_values) != SEQ_LEN * FEATURES:
                print(f"❌ Invalid length in {f}, row {idx}")
                continue

            # --- RESHAPE ---
            try:
                features = feature_values.reshape(SEQ_LEN, FEATURES)
            except:
                print(f"❌ FAILED RESHAPE in {f}, row {idx}")
                continue

            X.append(features)
            y.append(class_to_index[label])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)



# ===============================
# LOAD ALL DATA (ASL + FSL + SHARED)
# ===============================
X_asl, y_asl = load_sequences(ASL_DIR)
X_fsl, y_fsl = load_sequences(FSL_DIR)

print(f"\nASL Loaded: {len(X_asl)} samples")
print(f"FSL Loaded: {len(X_fsl)} samples")

# OPTIONAL folder
if os.path.exists(SHARED_DIR):
    X_shared, y_shared = load_sequences(SHARED_DIR)
    print(f"Shared Loaded: {len(X_shared)}")
else:
    X_shared = np.array([])
    y_shared = np.array([])


# ===============================
# COMBINE ALL
# ===============================
X_all = np.concatenate([X_asl, X_fsl] + ([X_shared] if len(X_shared) > 0 else []))
y_all = np.concatenate([y_asl, y_fsl] + ([y_shared] if len(y_shared) > 0 else []))

print(f"\nTOTAL SAMPLES LOADED FOR TESTING: {len(X_all)}")


# ===============================
# RUN MODEL ON ALL DATA
# ===============================
preds = model.predict(X_all)
pred_classes = np.argmax(preds, axis=1)


# ===============================
# REPORT
# ===============================
print("\n===== MODEL ACCURACY ON DATASET =====")
print("Accuracy:", accuracy_score(y_all, pred_classes))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_all, pred_classes, target_names=classes))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_all, pred_classes))
