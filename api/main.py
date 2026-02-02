from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
import io
import os
from pydub import AudioSegment
import soundfile as sf
import librosa
import speech_recognition as sr  # <--- NEW: The Verbal Ear
from src.config import *
from src.preprocessing import extract_features

app = FastAPI(title="SurakshaVaani Pro", description="Dual-Engine Distress Detection")

# Load Emotion Model
model = None
recognizer = sr.Recognizer() # Initialize Speech Recognizer

@app.on_event("startup")
async def load_model():
    global model
    try:
        model_path = "models/surakshavaani_final.h5" 
        if not os.path.exists(model_path):
            model_path = "models/saved_models/best_model.h5"
        model = tf.keras.models.load_model(model_path)
        print("✅ SurakshaVaani Brain Loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
@app.post("/analyze-threat")
async def analyze_threat(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Audio format not supported")

    try:
        # --- STEP 1: SAVE & CONVERT AUDIO ---
        audio_bytes = await file.read()
        
        # 1. Save original file temporarily
        file_ext = file.filename.split('.')[-1]
        temp_original = f"temp_upload.{file_ext}"
        temp_wav = "temp_clean.wav"
        
        with open(temp_original, "wb") as f:
            f.write(audio_bytes)
            
        # 2. Convert ANYTHING to WAV using Pydub
        try:
            audio = AudioSegment.from_file(temp_original)
            # Set to mono and 16000Hz (Perfect for AI)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(temp_wav, format="wav")
        except Exception as e:
            print(f"Conversion Error: {e}")
            # Fallback: Try treating as raw WAV if conversion fails
            with open(temp_wav, "wb") as f:
                f.write(audio_bytes)

        # --- STEP 2: KEYWORD DETECTION ---
        detected_text = ""
        keyword_risk = False
        
        # (Paste the BIG list of 60+ keywords here from the previous message)
        threat_keywords = [
            "help", "help me", "please help", "save me", "emergency", "urgent", "sos",
            "bachao", "bachao mujhe", "madad", "police", "call 100", "call 911",
            "stop", "don't touch me", "leave me alone", "let me go",
            "kidnapped", "abducted", "gun", "knife", "weapon", "shoot", "kill",
            "ambulance", "doctor", "hospital", "heart attack", "can't breathe"
            # ... add the rest of the list here
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
        # Load the CLEAN WAV file we just made
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
        raise HTTPException(status_code=500, detail=str(e))