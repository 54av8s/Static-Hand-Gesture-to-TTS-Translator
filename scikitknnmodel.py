import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

print(f"pandas version: {pd.__version__}")  # Print the version of pandas
print(f"scikit-learn version: {__import__('sklearn').__version__}")  # Print the version of scikit-learn
data = pd.read_csv("gesture_data.csv", header=None)
X = data.iloc[:, 1:]  # Features (normalized coordinates)
Y = data.iloc[:, 0]   # Labels (gestures)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, Y_train)

print("Model accuracy:", model.score(X_test, Y_test))
joblib.dump(model, "gesture_knn_model.pkl")  # Save the trained model
print("Model saved/updated")