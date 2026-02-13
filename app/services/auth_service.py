import numpy as np
from app.ml.model_loader import model

THRESHOLD = 0.6

def authenticate(features):
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)

    confidence = float(np.max(probs))

    if confidence < THRESHOLD:
        status = "ACCESS DENIED"
    else:
        status = "AUTHENTICATED"

    return prediction, confidence, status
