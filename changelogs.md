# Local changelogs file (only until oldest log related to this branch)
format in MM//DD
### (8/22)
1. Fixed TTS still speaking after closing feed window, changes also applied to the other branch

### (8/15)
1. Added script to stop TTS when feed window is closed (still needs debugging)
2. Added callout "unknown" when prediction does not meet confidence level 

### (8/10)
1. Added gitignore parameters 

### (8/08)
1. Added dedicated changelogs for this branch (global changelogs will be kept)

### (8/07)
1. Added frame buffer for future LSTM research and TTS timing interval with average, median, and stdev calculations for trial-and-error 
2. Fixed stdev causing feed crashes
3. Added eSpeak NG TTS (also synced with headlessmode branch)
4. Fixed TTS delay
5. Uninstalled pyttsx3 from .venv
6. Updated file requirements 
7. Synced changes from translatorprototype.py with headlessmode.py except all functions related to labelling and interval calculations (avg, median, stdev)

### (8/05)
1. Added version checks to scikit model file
2. Added GitHub repo for this project

### (7/21)
1. Temporarily reverted to win32com TTS worker due to bugs where it would only speak once 

### (7/20)
1. Added TTS functionality with debug prediction print
2. Replaced pyttsx3 with win32com TTS
3. Added three new gestures: “Thumbs up”, “Pinky”, “Peace” (refer to gesturedictionary.py in source codes folder to gesture list)
4. Fixed TTS delay 

### (7/19)
1. Exported gesture_labels into gesturedictionary.py as separate file for easier main file readability 
2. Added gesturedictionary.py in source codes folder
3. Added gesture saves counter in gesturedictionary.py
4. New Scikit ML file in source codes 
5. Updated source codes to reflect newest changes
