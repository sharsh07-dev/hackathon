from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
from tensorflow.keras.models import load_model  # <--- FIXED: ADDED THIS IMPORT
import numpy as np
import io
import os
from pydub import AudioSegment
import soundfile as sf
import librosa
import speech_recognition as sr
from src.config import *
from src.preprocessing import extract_features

app = FastAPI(title="SurakshaVaani Pro", description="Dual-Engine Distress Detection")

# --- GLOBAL VARIABLES ---
model = None
recognizer = sr.Recognizer()

# --- LOAD MODEL AT STARTUP ---
@app.on_event("startup")
async def load_ai_model():
    global model
    model_path = "models/surakshavaani_final.h5" 
    
    # 1. Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ CRITICAL ERROR: Model file not found at {model_path}")
        return

    # 2. Check file size (Fix for Git LFS pointer issue)
    file_size = os.path.getsize(model_path)
    print(f"🔍 Found model file. Size: {file_size / (1024 * 1024):.2f} MB")
    
    if file_size < 10000: # If less than 10KB, it's definitely corrupt/pointer
        print("❌ ERROR: Model file is too small! It might be a Git LFS pointer.")
        return

    try:
        # 3. Load Model (compile=False fixes Mac-to-Linux compatibility)
        model = load_model(model_path, compile=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

# --- API ENDPOINT ---
@app.post("/analyze-threat")
async def analyze_threat(file: UploadFile = File(...)):
    # 0. Safety Check: Is model loaded?
    if model is None:
        raise HTTPException(status_code=503, detail="AI Model is not loaded. Check server logs.")

    if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Audio format not supported")

    try:
        # --- STEP 1: SAVE & CONVERT AUDIO ---
        audio_bytes = await file.read()
        
        # Save original file temporarily
        file_ext = file.filename.split('.')[-1]
        temp_original = f"temp_upload.{file_ext}"
        temp_wav = "temp_clean.wav"
        
        with open(temp_original, "wb") as f:
            f.write(audio_bytes)
            
        # Convert ANYTHING to WAV using Pydub
        try:
            audio = AudioSegment.from_file(temp_original)
            # Set to mono and 16000Hz (Perfect for AI)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(temp_wav, format="wav")
        except Exception as e:
            print(f"Conversion Error: {e}")
            # Fallback: Try treating as raw WAV
            with open(temp_wav, "wb") as f:
                f.write(audio_bytes)

        # --- STEP 2: KEYWORD DETECTION (The Verbal Ear) ---
        detected_text = ""
        keyword_risk = False
        
        # THREAT DICTIONARY
        threat_keywords = [
            "help", "help me", "please help", "save me", "emergency", "urgent", "sos",
            "bachao", "bachao mujhe", "madad", "police", "call 100", "call 911",
            "stop", "don't touch me", "leave me alone", "let me go",
            "kidnapped", "abducted", "gun", "knife", "weapon", "shoot", "kill",
            "ambulance", "doctor", "hospital", "heart attack", "can't breathe"
        ]
        
        try:
            with sr.AudioFile(temp_wav) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                
                # Recognize
                detected_text = recognizer.recognize_google(audio_data).lower()
                print(f"🗣️ Heard: {detected_text}")
                
                if any(k in detected_text for k in threat_keywords):
                    keyword_risk = True
                    
        except Exception as e:
            print(f"Speech Error: {e}")
            detected_text = "[Unintelligible / Noise]"

        # --- STEP 3: TONE ANALYSIS (Using the Clean WAV) ---
        # Load the CLEAN WAV file
        y, sr_rate = librosa.load(temp_wav, sr=SAMPLE_RATE)
        
        # A. Volume Check
        rms = librosa.feature.rms(y=y)[0]
        avg_volume = np.mean(rms)
        is_loud = avg_volume > 0.05 

        # B. AI Prediction
        if len(y) > SAMPLES_PER_TRACK: y_model = y[:SAMPLES_PER_TRACK]
        else: y_model = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), 'constant')

        features = extract_features(y_model)
        features = np.expand_dims(features, axis=0)
        
        preds = model.predict(features)
        emotion = INT_TO_EMOTION[np.argmax(preds)]
        confidence = float(np.max(preds))

        # --- STEP 4: FINAL DECISION ---
        sos_triggered = False
        trigger_reason = []

        if emotion in ["Fear", "Panic", "Anger"] and confidence > 0.5:
            sos_triggered = True
            trigger_reason.append(f"Detected {emotion} Tone")

        if is_loud and emotion in ["Fear", "Surprise", "Sad"]:
            sos_triggered = True
            trigger_reason.append("High Volume Scream")

        if keyword_risk:
            sos_triggered = True
            trigger_reason.append(f"Keyword Heard: '{detected_text}'")

        # Cleanup
        if os.path.exists(temp_original): os.remove(temp_original)
        if os.path.exists(temp_wav): os.remove(temp_wav)

        return {
            "sos_activated": sos_triggered,
            "reasons": trigger_reason,
            "transcription": detected_text,
            "tone_analysis": {
                "emotion": emotion,
                "confidence": round(confidence, 2),
                "loudness": round(float(avg_volume), 4)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))#JE