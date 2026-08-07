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
import pyttsx3 

warnings.filterwarnings("ignore", category=UserWarning)
print(sys.executable)
print(f"CURRENT MP VERSION: {mp.__version__}")
print(f"CURRENT PYTHON VERSION AS .VENV: {sys.version}")    # Checking versions of our installed libraries, results return as print in the terminal
print(f"CURRENT OPENCV VERSION: {ov.__version__}")
print("RUNNING CORRECT FILE - VERSION CHECK 1")

# Importing MediaPipe and assigning standard parameters for accuracy and reliability
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.6,
    min_tracking_confidence = 0.6
)
mp_draw = mp.solutions.drawing_utils

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
  

# setting up TTS
model = joblib.load("gesture_knn_model.pkl")  # Load the trained model
cap = ov.VideoCapture(0)
last_spoken = None  # Initialize last_spoken variable
speech_queue = queue.Queue()  # Queue for speech synthesis

# TTS worker thread
def _tts_worker():
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    while True:
        text = speech_queue.get()
        speaker.Speak(text)

threading.Thread(target=_tts_worker, daemon=True).start()

# TTS queueing function
def say(text):
    while not speech_queue.empty():
         try:
             speech_queue.get_nowait()
         except queue.Empty:
             break
    speech_queue.put(text)

# Camera loop
try:
    while True:
        ret, frame = cap.read()
        frame = ov.flip(frame, 1) # Mirrors video
        resized_frame = ov.resize(frame, (640, 480))
        rgb_frame = ov.cvtColor(resized_frame, ov.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        normalized = None  # Initialize normalized variable

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                normalized = normalize(landmarks) 
                prediction = model.predict([normalized])[0]

            if prediction != last_spoken:
                print(f"TTS about to speak: '{prediction}'")
                say(prediction)
                last_spoken = prediction  # Update last_spoken variable
                
# ctrl + c to exit
except KeyboardInterrupt: 
    print("KeyboardInterrupt received. Exiting...")
cap.release()

# Shortened code for easy readability and understanding of the code. The code is now more organized and easier to follow, with clear comments explaining each section.