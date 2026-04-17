# Voice-Based Authentication System  
A proof-of-concept voice authentication system using machine-learning, browser audio capture and a web backend.

## 🚀 Overview  
This project implements a **voice‐based user authentication system**.  
Users speak into their browser; the audio is sent to a backend service which extracts features and runs a trained ML model to verify the speaker’s identity.  
It integrates:  
- Frontend: audio capture via browser.  
- Backend: a Python server that loads a trained model (voice recognition) and returns an authentication result.  
- ML pipeline: feature extraction + model training (e.g., MFCCs + KNN or another classifier) for voice classification.

## 🧩 Architecture  
1. **Data collection & preprocessing**  
   - Record audio samples for each user/speaker.  
   - Extract features (e.g., Mel-frequency cepstral coefficients) from the raw audio.  
2. **Model training (`train_model.py`)**  
   - Use the extracted features to train an ML classifier.  
   - Save the model (e.g., `voice_recognition_model.pkl`) and label mapping (e.g., `labels.npy`).  
3. **Backend service (`server.py`)**  
   - Loads the model and label mapping at startup.  
   - Exposes an API endpoint (e.g., POST audio) that:  
     - Receives raw audio (browser upload).  
     - Preprocesses: feature extraction → classification.  
     - Returns authentication outcome (speaker identity or “unauthenticated”).  
4. **Frontend (`index.html`)**  
   - Captures user audio via browser microphone.  
   - Sends audio to backend endpoint.  
   - Displays result to the user (authenticated / rejected).  

## 🛠 Technologies Used  
- Python (ML & backend)  
- Audio feature extraction (e.g., `librosa`)  
- Machine-learning classifier (e.g., K-Nearest Neighbors)  
- Browser API (getUserMedia) for audio capture  
- HTTP server (Flask/FastAPI or similar)  
- Model serialization (pickle)  
- Static HTML/JS for front-end  

