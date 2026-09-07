import os

import cv2 as ov
import mediapipe as mp
import numpy as npy
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import warnings
print(f"CURRENT MP VERSION: {mp.__version__}")
print(tf.__version__)
print(npy.__version__)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "gesture_lstm_model.keras")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")


model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)
    

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def normalize(landmarks):
    coords = []
    for lm in landmarks.landmark:
        coords.append((lm.x, lm.y, lm.z))
    coords = npy.array(coords)
    wrist = coords[0]
    coords = coords - wrist
    ref_distance = npy.linalg.norm(coords[9])
    coords = coords / ref_distance
    return coords.flatten()

def normalize_two_hands(results):
    hand_data = {"Left": npy.zeros(63), "Right": npy.zeros(63)}
    if results.multi_hand_landmarks:
        for hand_index, landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_index].classification[0].label
            hand_data[handedness] = normalize(landmarks)
    return npy.concatenate([hand_data["Left"], hand_data["Right"]])

frame_buffer = deque(maxlen=45)

cap = ov.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = ov.flip(frame, 1)
    resized_frame = ov.resize(frame, (640, 480))
    rgb_frame = ov.cvtColor(resized_frame, ov.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    combined = normalize_two_hands(results)
    frame_buffer.append(combined)

    if len(frame_buffer) == 45:
        sequence = npy.array(frame_buffer)
        sequence = npy.expand_dims(sequence, axis=0) 
        prediction = model.predict(sequence, verbose=0)
        predicted_index = npy.argmax(prediction)
        predicted_label = encoder.inverse_transform([predicted_index])[0]  # needs the encoder
        ov.putText(resized_frame, f"Class: {predicted_label}", (10, 30), ov.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    ov.imshow("LSTM Live Test", resized_frame)

    if ov.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
ov.destroyAllWindows()