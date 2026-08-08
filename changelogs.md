# This changelogs.md file is an extension of the changelogs from the main branch, i think i'll organize change logs based on file instead of a general one :3

# (8/08)
1. Added changelogs.md file and corrected filing errors

# (8/07)
1. Added frame buffer for future LSTM research and TTS timing interval with average, median, and stdev calculations for trial-and-error 
2. Fixed stdev causing feed crashes
3. Added eSpeak NG TTS to both repository branches
4. Fixed TTS delay
5. Uninstalled pyttsx3 from .venv
6. Updated file requirements for headlessmode.py and translatorprototype.py
7. Synced changes from translatorprototype.py with headlessmode.py except all functions related to labelling and interval calculations (avg, median, stdev)

# (8/06) 
1. Added headless mode as separate file
2. Removed feed window and all related labelling 
3. Changed camera loop exit keybinds
4. Added headlessmode branch in GitHub repository, will commit to a pull request before hardware migration
5. Removed “TTS about to speak” print from headlessmode.py

# (8/05)
1. Added version checks to scikit model file
2. Added GitHub repo for this project

# (7/21)
1. Temporarily reverted to win32com TTS worker due to bugs where it would only speak once 

# (7/20)
1. Added TTS functionality with debug prediction print
2. Replaced pyttsx3 with win32com TTS
3. Added three new gestures: “Thumbs up”, “Pinky”, “Peace” (refer to gesturedictionary.py in source codes folder to gesture list)
4. Fixed TTS delay 

# (7/19)
1. Exported gesture_labels into gesturedictionary.py as separate file for easier main file readability 
2. Added gesturedictionary.py in source codes folder
3. Added gesture saves counter in gesturedictionary.py
4. New Scikit ML file in source codes 
5. Updated source codes to reflect newest changes
