# NOTE: Applicable to all branches

# (8/07)
Added frame buffer for future LSTM research and TTS timing interval with average, median, and stdev calculations for trial-and-error 
Fixed stdev causing feed crashes
Added eSpeak NG TTS to both repository branches
Fixed TTS delay
Uninstalled pyttsx3 from .venv
Updated file requirements for headlessmode.py and translatorprototype.py
Synced changes from translatorprototype.py with headlessmode.py except all functions related to labelling and interval calculations (avg, median, stdev)

# (8/06) 
Added headless mode as separate file
Removed feed window and all related labelling 
Changed camera loop exit keybinds
Added headlessmode branch in GitHub repository, will commit to a pull request before hardware migration
Removed “TTS about to speak” print from headlessmode.py

# (8/05)
Added version checks to scikit model file
Added GitHub repo for this project

# (7/21)
Temporarily reverted to win32com TTS worker due to bugs where it would only speak once 
(7/20)
Added TTS functionality with debug prediction print
Replaced pyttsx3 with win32com TTS
Added three new gestures: “Thumbs up”, “Pinky”, “Peace” (refer to gesturedictionary.py in source codes folder to gesture list)
Fixed TTS delay 


# (7/19)
Exported gesture_labels into gesturedictionary.py as separate file for easier main file readability 
Added gesturedictionary.py in source codes folder
Added gesture saves counter in gesturedictionary.py
New Scikit ML file in source codes 
Updated source codes to reflect newest changes
