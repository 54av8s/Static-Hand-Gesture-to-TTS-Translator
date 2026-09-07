import os
import numpy as npy
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
sequences = []
labels = []

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SEQUENCES_DIR = os.path.join(DATA_DIR, "sequences")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "gesture_lstm_model.keras")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

for filename in os.listdir(SEQUENCES_DIR):
    if filename.endswith(".npy"):
        data = npy.load(os.path.join(SEQUENCES_DIR, filename))
        sequences.append(data)
        label = filename.split("_")[0]
        labels.append(label)
X = npy.array(sequences)
print("X shape:", X.shape)
print("Labels found:", labels)

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical  # Ignore could not be resolved errors

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(labels)
y_categorical = to_categorical(y_encoded)
print("Y shape:", y_categorical.shape)

from tensorflow.keras.models import Sequential # Ignore could not be resolved errors
from tensorflow.keras.layers import LSTM, Dense # Ignore could not be resolved errors

model = Sequential([
    LSTM(64, input_shape=(45,126)),
    Dense(y_categorical.shape[1], activation= "softmax")
])

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X, y_categorical, epochs=10)  # Ignore could not be resolved errors

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2)
model.fit(X_train, y_train, epochs=20)
print("Test accuracy:", model.evaluate(X_test, y_test))

model.save(MODEL_PATH)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(labels)
y_categorical = to_categorical(y_encoded)

with open (ENCODER_PATH, "wb") as f:
    pickle.dump(encoder, f)

