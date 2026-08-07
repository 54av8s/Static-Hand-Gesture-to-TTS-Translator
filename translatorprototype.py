# Versions
import sys # Shows current Python version (3.12 as an isolated environment)
import mediapipe as mp # Importing MediaPipe and assigning it the variable "mp" 
import cv2 as ov  # Importing OpenCV/CV2 as ov, also giving it a variable to be used in the code
import numpy as npy # Importing NumPy as npy, also assigning it a variable that can easily be understood in the code
import csv # Importing CSV to save data in a CSV file
import joblib # From scikitknnmodel.py, importing joblib to load the trained model for gesture recognition
import win32com.client # TTS function in place for pyttsx3 
import queue   
import pythoncom
import threading
import warnings
import time
import statistics

# Version checks for the libraries used in this project :3
warnings.filterwarnings("ignore", category=UserWarning)
print(sys.executable)
print(f"CURRENT MP VERSION: {mp.__version__}")
print(f"CURRENT PYTHON VERSION AS .VENV: {sys.version}")    # Checking versions of our installed libraries, results return as print in the terminal
print(f"CURRENT OPENCV VERSION: {ov.__version__}")
print("RUNNING CORRECT FILE - VERSION CHECK 1")

# Importing MediaPipe and assigning standard parameters
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.6,
    min_tracking_confidence = 0.6
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


# Normalization function — defined ONCE, called every frame
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


from gesturedictionary import gesture_labels  # Importing the gesture labels from the gesture dictionary file

# Save function for data collection
def save_sample(normalized_row, label):
    with open("gesture_data.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label]+ list(normalized_row))
  



# setting up the camera
model = joblib.load("gesture_knn_model.pkl")  # Load the trained model
cap = ov.VideoCapture(0)
last_spoken = None  # Initialize last_spoken variable
speech_queue = queue.Queue()  # Queue for speech synthesis

def _tts_worker():
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    while True:
        text = speech_queue.get()
        speaker.Speak(text)

threading.Thread(target=_tts_worker, daemon=True).start()
        
def say(text):
    while not speech_queue.empty():
         try:
             speech_queue.get_nowait()
         except queue.Empty:
             break
    speech_queue.put(text)


# Buffer to store last 30 frames refer to function 
from collections import deque
frame_buffer = deque(maxlen=30)  # Buffer to store the last 30 frames

# Elapsed time tracking between TTS callouts
last_spoken_time = time.time()  # Initialize last spoken time

# Empty intervals list to store the time intervals between TTS callouts
intervals = []


while True:
    ret, frame = cap.read()
    frame = ov.flip(frame, 1) # Mirrors video
    resized_frame = ov.resize(frame, (1100, 720))
    rgb_frame = ov.cvtColor(resized_frame, ov.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    normalized = None  # Initialize normalized variable

    if results.multi_hand_landmarks:
        for hand_index, landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(resized_frame, landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = resized_frame.shape
          
            normalized = normalize(landmarks)
            frame_buffer.append(normalized)  # Add the normalized data to the buffer
            prediction = model.predict([normalized])[0]

            # Script to count intervals between speech
            if prediction != last_spoken:
                now = time.time()
                gap = now - last_spoken_time
                intervals.append(gap)  # Store the interval
                print(f"Time since last callout: {gap: .2f} seconds")
                print(f"TTS about to speak: '{prediction}'")
                last_spoken_time = now  # Update last spoken time
                say(prediction)
            last_spoken = prediction  # Update last_spoken variable

            # Stats for interval analysis
            if intervals:
                print(f"Average: {statistics.mean(intervals):.2f} seconds")
                print(f"Median: {statistics.median(intervals):.2f} seconds")
                if len (intervals) >=2:
                    print(f"Standard deviation: {statistics.stdev(intervals):.2f} seconds")
                else: 
                    print("Standard deviation: N/A (not enough data points)")
            else:
                print("No intervals yet")
    

            handedness = results.multi_handedness[hand_index].classification[0].label
            for landmark_id, label in joints:
                tip = landmarks.landmark[landmark_id]
                x = int(tip.x * w)
                y = int(tip.y * h)
                labelfont = ov.FONT_HERSHEY_PLAIN
                ov.circle(resized_frame, (x, y), 1, (0, 0, 255), 2)
                ov.putText(resized_frame, label, (x+5, y), labelfont, 1, (0, 255, 0), 2)
            y_offset = 30 + (hand_index * 30)
            ov.putText(resized_frame, f"{handedness}: {prediction}", (10, y_offset), labelfont, 1, (255, 255, 255), 2)
                
    ov.imshow("Feed", resized_frame)
    key = ov.waitKey(1) & 0xFF
    if key == ord('x'):
        break
    elif normalized is not None and chr(key) in gesture_labels:
        save_sample(normalized, gesture_labels[chr(key)])
        print(f"Saved sample for: {gesture_labels[chr(key)]}")

cap.release()
ov.destroyAllWindows()
