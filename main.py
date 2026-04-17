from fastapi import FastAPI, File , UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import shutil
import os 
import uuid
import pickle
from train_model import train_model
from audio_utils import extract_mfcc


app = FastAPI()

# Load model + scaler
model = None
scalar = None

def load_model():
    global model, scaler
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

#Load once at startup
load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER ="audio"
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

@app.get("/")
def home():
    return {"message": "Backend is running 🚀"}

@app.post("/register/")
async def register(user: str, file: UploadFile = File(...)):
    user_folder = os.path.join("data", user)
    os.makedirs(user_folder, exist_ok=True)
    
    file_path = os.path.join(user_folder, f"{uuid.uuid4()}.wav")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"Saved: {file_path}. Starting Retraining...")

# 1. 🔥 AUTO-TRAIN: This calls your function from train_model.py
    return {"message": f"{user} registered✅"} 

@app.post("/login/")
async def login(file: UploadFile = File(...)):
    temp_path = "temp.wav"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract MFCC
    mfcc = extract_mfcc(temp_path)
    mfcc_scaled = scaler.transform([mfcc])

    # Predict
    prediction = model.predict(mfcc_scaled)[0]
    confidence = max(model.predict_proba(mfcc_scaled)[0]) * 100

    print("Prediction:", prediction)
    print("Confidence:", confidence)

    # Reject logic
    if confidence < 70:
        return {
            "status": "REJECTED",
            "confidence": round(confidence, 2)
        }

    return {
        "status": "AUTHENTICATED",
        "user": prediction,
        "confidence": round(confidence, 2),
        
    }


