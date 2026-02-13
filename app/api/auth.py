from fastapi import APIRouter, UploadFile, File
from pydub import AudioSegment
import uuid
import os
import numpy as np

from app.ml.feature_extractor import extract_features
from app.services.auth_service import authenticate

router = APIRouter()

AudioSegment.converter = "ffmpeg"

@router.post("/authenticate")
async def authenticate_voice(file: UploadFile = File(...)):
    webm_name = f"temp_{uuid.uuid4()}.webm"
    wav_name = webm_name.replace(".webm", ".wav")

    try:
        with open(webm_name, "wb") as f:
            f.write(await file.read())

        sound = AudioSegment.from_file(webm_name)
        sound.export(wav_name, format="wav")

        features = extract_features(wav_name).reshape(1, -1)

        user, confidence, status = authenticate(features)

        return {
            "user": user,
            "confidence": round(confidence * 100, 2),
            "status": status
        }

    finally:
        if os.path.exists(webm_name):
            os.remove(webm_name)
        if os.path.exists(wav_name):
            os.remove(wav_name)
