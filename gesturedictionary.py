# New gesture labels to be added here for main translator script readability and maintainability. 
gesture_labels= {
    'a': "Hello",
    'b': "Quiet",
    'c': "Okay",
    'd': "Pinky",
    'e': "Thumbs up",
    'f': "Peace"
}

# Counter for saved gestures in gesture_data.csv
import csv
from collections import Counter
with open("gesture_data.csv", "r") as f:
    labels = [row[0] for row in csv.reader(f) if row]
print(Counter(labels))