import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(DATA_DIR, "voice_recognition_model.pkl")
LABELS_PATH = os.path.join(DATA_DIR, "labels.npy")
VOICE_SAMPLES_DIR = os.path.join(DATA_DIR, "voice_samples")
