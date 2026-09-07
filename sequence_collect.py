import cv2 as ov
import mediapipe as mp
import numpy as npy
from collections import deque
import time
import os



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# label list only visible in Feed 
joints = [
         (4, "Thumb tip"),
         (8, "Index tip"),
         (12, "middle tip"),
         (16, "Ring tip"),
         (20, "Pinky tip"),
         (0,  "Wrist"),
         (2,  "Thumb base")
] 

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
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_index, landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_index].classification[0].label
            hand_data[handedness] = normalize(landmarks)
    combined = npy.concatenate([hand_data["Left"], hand_data["Right"]])
    return combined

sequence_labels = {
    'z': "Wave",
    'c': "Come here",
    'v': "More",
    'b': "Help"
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SEQUENCES_DIR = os.path.join(DATA_DIR, "sequences")
os.makedirs(SEQUENCES_DIR, exist_ok=True)
frame_buffer = deque(maxlen=45)  # Buffer to store last 45 frames
def save_sequence(buffer, label):
    sequence_array = npy.array(buffer)
    filename = os.path.join(SEQUENCES_DIR, f"{label}_{int(time.time())}.npy")
    npy.save(filename, sequence_array)
    print(f"Saved sequence for: '{label}' - shape {sequence_array.shape}")
    print(f"File saved as: {filename}")
    print(sequence_array.shape)

cap = ov.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = ov.flip(frame, 1)
    resized_frame = ov.resize(frame, (640,480))
    rgb_frame = ov.cvtColor(resized_frame, ov.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        combined = normalize_two_hands(results)
        frame_buffer.append(combined)

    ov.imshow("Feed", resized_frame)
    key = ov.waitKey(1) & 0xFF
    if key == ord('x'):
        break
    elif chr(key) in sequence_labels:
        print(f"Key pressed: {chr(key)}, buffer length: {len(frame_buffer)}")
        if len(frame_buffer) == 45:
            save_sequence(frame_buffer, sequence_labels[chr(key)])
cap.release()
ov.destroyAllWindows()


