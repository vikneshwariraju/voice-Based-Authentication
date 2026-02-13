import os
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from app.core.config import VOICE_SAMPLES_DIR, MODEL_PATH
import joblib

DATA_DIR = "voice_samples"
MODEL_PATH = "voice_recognition_model.pkl"
LABELS_PATH = "labels.npy"

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

X = []
y = []

for user_folder in os.listdir(DATA_DIR):
    user_path = os.path.join(DATA_DIR, user_folder)
    if os.path.isdir(user_path):
        for audio_file in os.listdir(user_path):
            file_path = os.path.join(user_path, audio_file)
            features = extract_features(file_path)
            X.append(features)
            y.append(user_folder)

X = np.array(X)
y = np.array(y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

joblib.dump(model, MODEL_PATH)
np.save(LABELS_PATH, y)

print("Training complete. Model saved!")
