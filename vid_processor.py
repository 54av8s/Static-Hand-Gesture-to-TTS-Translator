# Script to process videos but also feeds into \sequences

import os
import cv2 as ov
import mediapipe as mp
import numpy as npy
import time
from utils.lstm_labels import sequence_labels

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SEQUENCES_DIR = os.path.join(DATA_DIR, "sequences")
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, "raw_videos") 
os.makedirs(SEQUENCES_DIR, exist_ok=True)
os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Def thread below
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

def process_video_file(video_path):
    cap = ov.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = ov.resize(frame, (640, 480))
        rgb_frame = ov.cvtColor(resized, ov.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        combined = normalize_two_hands(results)
        all_frames.append(combined)
    cap.release()
    return npy.array(all_frames)


def resample_sequence(frames, target_length=45):
    if len(frames)==0:
        return None
    indices = npy.linspace(0, len(frames) - 1, target_length).astype(int)
    return frames [indices]

for video_file in os.listdir(RAW_VIDEOS_DIR):
    if video_file.endswith((".mp4", ".mov")):
        label = video_file.split("_")[0]
        accepted_labels = set(sequence_labels.values())

        if label not in accepted_labels:
            print(f"Skipped, unknown label: {label} in file {video_file}")
            continue
        
        video_path = os.path.join(RAW_VIDEOS_DIR, video_file)

        frames = process_video_file(video_path)
        resample = resample_sequence(frames)

        if resample is None or len(resample) < 45:
            print(f"Skipped; no hands detected or too short: {video_file}")
            continue
        
        out_path = os.path.join(SEQUENCES_DIR, f"{label}_{int(time.time())}.npy")
        npy.save(out_path, resample)
        print(f"Processed file {video_file} to {out_path}")
        

