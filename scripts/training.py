import numpy as np
import pandas as pd
import glob
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =============================================
#        📁 BASE PATHS
# =============================================
BASE_DIR = r"C:\Users\JamJayDatuin\Documents\Machine Learning Projects\SignLanguageRecognition"
ASL_DIR = os.path.join(BASE_DIR, "dataset", "ASL")
FSL_DIR = os.path.join(BASE_DIR, "dataset", "FSL")
SHARED_DIR = os.path.join(BASE_DIR, "dataset", "SHARED")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================
#        🔄 LOAD ASL + FSL + SHARED
# =============================================
def load_dataset(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    return dfs

asl_dfs = load_dataset(ASL_DIR)
fsl_dfs = load_dataset(FSL_DIR)
shared_dfs = load_dataset(SHARED_DIR)

df = pd.concat(asl_dfs + fsl_dfs + shared_dfs, ignore_index=True)

print(f"📌 Loaded ASL files: {len(asl_dfs)}")
print(f"📌 Loaded FSL files: {len(fsl_dfs)}")
print(f"📌 Loaded SHARED files: {len(shared_dfs)}")
print(f"📌 Total merged samples: {len(df)}")

# =============================================
#        🎯 SEPARATE LABELS + FEATURES
# =============================================
labels = df['label']
df = df.drop('label', axis=1)

SEQ_LEN = 50
FEATURES = 126

# Convert to float32 and reshape
X = df.values.astype(np.float32).reshape(len(df), SEQ_LEN, FEATURES)

# Compute velocity (frame-to-frame difference)
vel = X[:, 1:, :] - X[:, :-1, :]
vel = np.pad(vel, ((0,0),(1,0),(0,0)), mode='constant')
X = np.concatenate([X, vel], axis=2)

FEATURES = 252  # updated after concatenation

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

print("\n📌 Combined Classes:", list(le.classes_))
print("📌 Total Classes:", len(le.classes_))

# Save the merged classes list
np.save(os.path.join(MODELS_DIR, "classes.npy"), le.classes_)

# =============================================
#        ✂️ TRAIN / TEST SPLIT
# =============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# =============================================
#        📏 NORMALIZATION (TRAIN-ONLY STATS)
# =============================================
mean = X_train.mean(axis=(0,1), keepdims=True).astype(np.float32)
std = X_train.std(axis=(0,1), keepdims=True).astype(np.float32) + 1e-6

X_train = ((X_train - mean) / std).astype(np.float32)
X_test  = ((X_test - mean) / std).astype(np.float32)

np.save(os.path.join(MODELS_DIR, "norm_mean.npy"), mean)
np.save(os.path.join(MODELS_DIR, "norm_std.npy"), std)

# =============================================
#        💾 SAVE PROCESSED DATASETS
# =============================================
def save_processed(folder, X_train, X_test, y_train, y_test, classes):
    processed = os.path.join(folder, "processed")
    os.makedirs(processed, exist_ok=True)

    np.save(os.path.join(processed, "X_train.npy"), X_train)
    np.save(os.path.join(processed, "X_test.npy"), X_test)
    np.save(os.path.join(processed, "y_train.npy"), y_train)
    np.save(os.path.join(processed, "y_test.npy"), y_test)
    np.save(os.path.join(processed, "classes.npy"), classes)

    print(f"💾 Saved processed dataset → {processed}")

save_processed(ASL_DIR, X_train, X_test, y_train, y_test, le.classes_)
save_processed(FSL_DIR, X_train, X_test, y_train, y_test, le.classes_)
save_processed(SHARED_DIR, X_train, X_test, y_train, y_test, le.classes_)



# =============================================
#     🚀 TFLITE-FRIENDLY GRU MODEL
# =============================================

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu',
                            input_shape=(SEQ_LEN, FEATURES)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.DepthwiseConv1D(3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.GRU(64),

    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# =============================================
#        🧠 TRAIN MODEL
# =============================================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=32
)


# =============================================
#        📊 EVALUATE MODEL
# =============================================
loss, acc = model.evaluate(X_test, y_test)
print("\n📊 Test Accuracy:", acc)


# =============================================
#        💾 SAVE MODELS
# =============================================
keras_path = os.path.join(MODELS_DIR, "Sign_Model.keras")
model.save(keras_path)
print("\n💾 Saved Keras model:", keras_path)


# =============================================
#        🚀 TFLITE CONVERSION
# =============================================
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

tflite_path = os.path.join(MODELS_DIR, "Sign_Model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("\n💾 Saved TFLite model:", tflite_path)
