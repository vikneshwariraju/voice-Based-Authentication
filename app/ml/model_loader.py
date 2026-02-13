import joblib
from app.core.config import MODEL_PATH

model = joblib.load(MODEL_PATH)
