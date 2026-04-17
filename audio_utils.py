import librosa
import numpy as np

def extract_mfcc(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=16000)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
     # Fix length (1 second)
    y = librosa.util.fix_length(y, size=16000)

    # Delta (velocity)
    delta = librosa.feature.delta(mfcc)

    # Delta-Delta (acceleration)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Combine all
    combined = np.vstack((mfcc, delta, delta2))

     # Take mean + std (very important)
    mean = np.mean(combined, axis=1)
    std = np.std(combined, axis=1)

    features = np.concatenate((mean, std))

    return features