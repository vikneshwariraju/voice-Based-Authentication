import os
import numpy as np
import pickle
from audio_utils import extract_mfcc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def train_model():
    X = []
    y = []

    data_path = "data"

    for user in os.listdir(data_path):
        user_folder = os.path.join(data_path, user)
        
        if not os.path.isdir(user_folder):
            continue

        for file in os.listdir(user_folder):
            file_path = os.path.join(user_folder, file)

            mfcc = extract_mfcc(file_path)

            X.append(mfcc)
            y.append(user)

    X = np.array(X)
    Y=np.array(y)
    print("Training data shape:", X.shape)

# 🔥 Normalize data (VERY IMPORTANT)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# 🔥 Train SVM
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_scaled, y)

 # Save model + scaler
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("✅ SVM Model trained & saved!")

if __name__ == "__main__":
    train_model()