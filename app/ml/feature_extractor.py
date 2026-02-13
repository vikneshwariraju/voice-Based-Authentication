import librosa
import numpy as np

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)
