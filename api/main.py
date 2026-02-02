from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import io
import os
import gc  # Added for memory management
from pydub import AudioSegment
import speech_recognition as sr
from src.config import *
from src.preprocessing import extract_features
import librosa

app = FastAPI(title="SurakshaVaani Pro", description="Dual-Engine Distress Detection")

# --- GLOBAL VARIABLES ---
# We do NOT load the model here anymore to save startup RAM
model = None
recognizer = sr.Recognizer()

def get_model():
    """
    Lazy Loader: Only loads the model when strictly necessary.
    """
    global model
    if model is None:
        print("⏳ Loading AI Model for the first time...")
        model_path = "models/surakshavaani_final.h5"
        
        if not os.path.exists(model_path):
            print(f"❌ Critical: Model not found at {model_path}")
            return None
            
        try:
            # Force garbage collection before loading to free up space
            gc.collect()
            
            # Load model without compilation (Saves Memory & Fixes Mac/Linux issues)
            model = load_model(model_path, compile=False)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Model Load Failed: {e}")
            return None
    return model

# --- LIGHTWEIGHT HOME PAGE ---
@app.get("/")
async def home():
    # This will load INSTANTLY because it doesn't touch TensorFlow
    return {
        "status": "online",
        "message": "SurakshaVaani Server is Running.",
        "memory_saver_mode": True
    }

# --- API ENDPOINT ---
@app.post("/analyze-threat")
async def analyze_threat(file: UploadFile = File(...)):
    # 1. Load Model NOW (On Demand)
    ai_model = get_model()
    
    if ai_model is None:
        raise HTTPException(status_code=503, detail="AI Model failed to load. Check server logs.")

    if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Audio format not supported")

    try:
        # --- STEP 1: SAVE & CONVERT AUDIO ---
        audio_bytes = await file.read()
        file_ext = file.filename.split('.')[-1]
        temp_original = f"temp_{file.filename}"
        temp_wav = f"temp_clean_{file.filename}.wav" # Unique name to avoid conflicts
        
        with open(temp_original, "wb") as f:
            f.write(audio_bytes)
            
        try:
            audio = AudioSegment.from_file(temp_original)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(temp_wav, format="wav")
        except Exception:
            # Fallback
            with open(temp_wav, "wb") as f:
                f.write(audio_bytes)

        # --- STEP 2: KEYWORD DETECTION ---
        detected_text = ""
        keyword_risk = False
        threat_keywords = ["help", "help me", "bachao", "stop", "police", "call 100", "kill", "gun"]
        
        try:
            with sr.AudioFile(temp_wav) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                detected_text = recognizer.recognize_google(audio_data).lower()
                print(f"🗣️ Heard: {detected_text}")
                if any(k in detected_text for k in threat_keywords):
                    keyword_risk = True
        except Exception:
            detected_text = "unintelligible"

        # --- STEP 3: TONE ANALYSIS ---
        y, sr_rate = librosa.load(temp_wav, sr=SAMPLE_RATE)
        
        # Volume Check
        rms = librosa.feature.rms(y=y)[0]
        avg_volume = np.mean(rms)
        is_loud = avg_volume > 0.05 

        # AI Prediction
        if len(y) > SAMPLES_PER_TRACK: y_model = y[:SAMPLES_PER_TRACK]
        else: y_model = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), 'constant')

        features = extract_features(y_model)
        features = np.expand_dims(features, axis=0)
        
        preds = ai_model.predict(features)
        emotion = INT_TO_EMOTION[np.argmax(preds)]
        confidence = float(np.max(preds))

        # --- STEP 4: DECISION ---
        sos_triggered = False
        trigger_reason = []

        if emotion in ["Fear", "Panic", "Anger"] and confidence > 0.5:
            sos_triggered = True
            trigger_reason.append(f"Detected {emotion}")

        if is_loud and emotion in ["Fear", "Surprise", "Sad"]:
            sos_triggered = True
            trigger_reason.append("High Volume Scream")

        if keyword_risk:
            sos_triggered = True
            trigger_reason.append(f"Keyword: {detected_text}")

        # --- MEMORY CLEANUP (CRITICAL FOR FREE SERVER) ---
        if os.path.exists(temp_original): os.remove(temp_original)
        if os.path.exists(temp_wav): os.remove(temp_wav)
        
        # Explicitly delete heavy variables
        del y, y_model, features, preds, audio_bytes
        gc.collect() # Force RAM cleanup

        return {
            "sos_activated": sos_triggered,
            "reasons": trigger_reason,
            "transcription": detected_text,
            "tone_analysis": {"emotion": emotion, "confidence": round(confidence, 2)}
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))