# Imports
import os
import cv2 as ov
import mediapipe as mp
import numpy as npy
import time
import argparse
from utils.lstm_labels import sequence_labels
from pathlib import Path


# --------------Relative paths for argparse--------------
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_INPUT_DIR = SCRIPT_DIR / "data" / "raw_videos"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "sequences"
# --------------------END OF Relative paths for argparse--------------------

# ----------------------MEDIAPIPE INITIALIZATION AND PARAMETERS FOR ACCURACY-------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# ----------------------END OF MediaPipe Parameters-------------------------


# ----------------------------------------------------Helper functions for video processing----------------------------------------------------
def walk_pruner(dirs: list[str]) -> None:
    dirs[:] = [
        d for d in dirs
        if not d.startswith(('.', '$')) and 'shortcut' not in d.lower()
    ]

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
    cap = ov.VideoCapture(str(video_path))
    total_frames = int(cap.get(ov.CAP_PROP_FRAME_COUNT))
    all_frames = []
    frame_idx = 0
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
       while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 2 !=0:
            continue

        resized = ov.resize(frame, (640, 480))
        rgb_frame = ov.cvtColor(resized, ov.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        combined = normalize_two_hands(results)
        all_frames.append(combined)
        if frame_idx % 30 ==0:
            print(f"Frame {frame_idx}/total {total_frames} processed for {os.path.basename(video_path)}")
    cap.release()
    print(f"Finished {os.path.basename(video_path)} with {len(all_frames)} frames")
    return npy.array(all_frames)



def resample_sequence(frames, target_length=45):
    if len(frames)==0:
        return None
    indices = npy.linspace(0, len(frames) - 1, target_length).astype(int)
    return frames [indices]
# ----------------------------------------------------END OF HELPER FUNCTIONS----------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and save as .npy sequences")
    parser.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing video files to process"
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save processed sequence files"
    )
    args = parser.parse_args()

    INPUT_DIR = os.path.abspath(args.input_dir)
    OUTPUT_DIR = os.path.abspath(args.output_dir)

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    accepted_labels = {label.lower() for label in sequence_labels}

    for root, dirs, files in os.walk(INPUT_DIR):
        walk_pruner(dirs)
        valid_videos= [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        for file in valid_videos:
            full_path = os.path.join(root, file)
            filename = os.path.basename(full_path)
            label = filename.split('_')[0].lower()

            if label not in accepted_labels:
                print(f"Skipped, not known label '{label}' of {filename}")
                continue

            stem = Path(full_path).stem
            label_output_dir = os.path.join(OUTPUT_DIR, label)
            os.makedirs(label_output_dir, exist_ok=True)
            out_path=os.path.join(label_output_dir, f"{stem}.npy")

            if os.path.exists(out_path):
                print(f"Skipped, already exists: {out_path}")
                continue
            frames = process_video_file(full_path)
            resampled = resample_sequence(frames)
            if resampled is None or len(resampled) <45:
                print(f"Skipped, not enough frames after resampling for {filename}")
                continue
            npy.save(out_path, resampled)
            print(f"Processed and saved {filename} to {out_path}")



