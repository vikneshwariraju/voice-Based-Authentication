from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import numpy as np
import librosa
import joblib
import os

AudioSegment.converter = "ffmpeg"

app = FastAPI()

origins = [
    "https://voice-based-authentication.vercel.app",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "voice_recognition_model.pkl")
model = joblib.load(MODEL_PATH)

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with open("temp.webm", "wb") as f:
            f.write(await file.read())

        sound = AudioSegment.from_file("temp.webm")
        sound.export("temp.wav", format="wav")

        features = extract_features("temp.wav").reshape(1, -1)
        prediction = model.predict(features)

        return {"prediction": str(prediction[0])}

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists("temp.webm"):
            os.remove("temp.webm")
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")




