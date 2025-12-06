import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

# === 1️⃣ Dataset Loading & Combining ===
DATA_DIR = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition\dataset\FSL"
all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

print(f"📁 Found {len(all_files)} dataset files")

# Load ALL .csv files inside the ASL/FSL folder
all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

print(f"📁 Found {len(all_files)} dataset files")
print("Files:", [os.path.basename(f) for f in all_files])

df_list = []
for file in all_files:
    try:
        df = pd.read_csv(file)
        if df.empty:
            print(f"⚠️ Skipped empty file: {os.path.basename(file)}")
            continue

        # Add label automatically based on filename
        df['label'] = os.path.splitext(os.path.basename(file))[0]

        df_list.append(df)
        print(f"✅ Loaded {os.path.basename(file)} ({df.shape[0]} samples)")
    except Exception as e:
        print(f"❌ Error reading {os.path.basename(file)}: {e}")

# Combine all
if not df_list:
    raise ValueError("🚫 No valid datasets found. Make sure CSV files are present.")

final_df = pd.concat(df_list, ignore_index=True)

combined_path = os.path.join(DATA_DIR, "FSLdataset.csv")
final_df.to_csv(combined_path, index=False)

print("\n✅ Combined dataset created successfully!")
print(f"📄 Saved to: {combined_path}")
print("🧮 Total samples:", final_df.shape[0])
print("🏷️ Unique labels:", final_df['label'].unique())


# === 2️⃣ Preprocessing ===
print("\n🔧 Cleaning and preparing data...")

# Drop NaNs and ensure numeric
final_df = final_df.dropna()
X = final_df.drop('label', axis=1)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0).values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(final_df['label'])

# Save label encoder for later decoding
np.save(os.path.join(DATA_DIR, "ObjectThings_classes.npy"), label_encoder.classes_)
print("💾 Saved label classes for later decoding")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 3️⃣ Build TensorFlow Model ===
print("\n🧠 Building TensorFlow Model...")

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === 4️⃣ Train Model ===
print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    verbose=1
)

# === 5️⃣ Evaluate ===
print("\n📊 Evaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"📉 Test Loss: {test_loss:.4f}")

# === 6️⃣ Save Models (Keras, SavedModel, TFLite) ===
TFMODELS_DIR = os.path.join(DATA_DIR, "FSLDatasetModels")
os.makedirs(TFMODELS_DIR, exist_ok=True)

KERAS_PATH = os.path.join(TFMODELS_DIR, "FSL_Dataset.keras")
SAVEDMODEL_PATH = os.path.join(TFMODELS_DIR, "FSL_Dataset_SavedModel")
TFLITE_PATH = os.path.join(TFMODELS_DIR, "FSL_Dataset_Model.tflite")

# Save .keras
model.save(KERAS_PATH)
print(f"💾 Saved Keras model → {KERAS_PATH}")

# Save as TensorFlow SavedModel
model.export(SAVEDMODEL_PATH)
print(f"💾 Saved TensorFlow SavedModel → {SAVEDMODEL_PATH}")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"💾 Saved TFLite model → {TFLITE_PATH}")

print("\n✅ All models exported successfully!")
