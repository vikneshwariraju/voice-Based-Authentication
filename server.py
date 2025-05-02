from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import numpy as np
import librosa
import uvicorn
import joblib
import os

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = joblib.load("voice_recognition_model.pkl")  # Replace with your actual model file

# Feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features = np.mean(mfccs.T, axis=0)
    return features

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        with open("temp.webm", "wb") as f:
            f.write(await file.read())

        # Convert to wav using pydub
        sound = AudioSegment.from_file("temp.webm")
        sound.export("temp.wav", format="wav")

        # Extract features
        features = extract_features("temp.wav").reshape(1, -1)

        # Predict
        prediction = model.predict(features)
        return {"prediction": prediction[0]}

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Clean up
        if os.path.exists("temp.webm"):
            os.remove("temp.webm")
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)


