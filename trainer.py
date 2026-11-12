
# --------------------------------------------------IMPORTS--------------------------------------------------
import os
import pickle
import time
import numpy as npy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from pathlib import Path
# --------------------------------------------------END OF IMPORTS--------------------------------------------------


# List (empty)
sequences = []
labels = []
timestamp=int(time.time())

# Relative path functions for compatibility
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SEQUENCES_DIR = os.path.join(DATA_DIR, "sequences")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# Name of trained model and encoder files with destination folder as defined above
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "TrainedModel.keras")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, f"Encoder.pkl")


# Check if sequences exist in sequences directory
if not os.path.exists(SEQUENCES_DIR) or not os.listdir(SEQUENCES_DIR):
    raise FileNotFoundError(f"No .npy files found in {SEQUENCES_DIR}. Please run vid_processor.py first to generate sequence data.")


# os.walk from sequences as datasets for training LSTM for inference loops
for root, dirs, files in os.walk(SEQUENCES_DIR):
    dirs[:]=[d for d in dirs if not d.startswith(('.', '$'))]
    for file in files:
        if file.endswith('.npy'):
            file_path = os.path.join(root, file)
            data = npy.load(file_path)
            label = os.path.basename(root)
            sequences.append(data)
            labels.append(label)
if len(sequences) == 0:
    raise ValueError(f"No sequences found in {SEQUENCES_DIR}. Please ensure that .npy files are present.")
X = npy.array(sequences)
print("X shape:", X.shape)
print("Labels:", set(labels))



# --------------------------------------------------TRAIN LOGIC--------------------------------------------------
# Encoder for labels to display in camera feed window instead of indexes
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(labels)
y_categorical = to_categorical(y_encoded)
print("Y shape:", y_categorical.shape)

num_classes = max(y_categorical.shape[1], 2)
model = Sequential([
    tf.keras.Input(shape=(45, 126)),
    LSTM(64),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation= "softmax")
])

# MODEL COMPILE
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# LSTM LOGIC
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2)
model.fit(X_train, y_train, epochs=25, batch_size=8, validation_data=(X_test, y_test))  # Ignore could not be resolved errors
print("Test accuracy:", model.evaluate(X_test, y_test))

# Saves the trained model to specified artifacts directory
model.save(MODEL_PATH)
with open (ENCODER_PATH, "wb") as f:
    pickle.dump(encoder, f)
print(f"Saved model and encoder to {ARTIFACTS_DIR} {timestamp}")

# --------------------------------------------------END OF TRAIN LOGIC--------------------------------------------------

