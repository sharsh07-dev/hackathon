import os
#changes #changes
# Audio Configuration
SAMPLE_RATE = 16000
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Mel-Spectrogram Parameters
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# Dataset Paths
RAVDESS_PATH = "data/raw_audio/ravdess"
TESS_PATH = "data/raw_audio/tess"      # <--- NEW
CREMA_PATH = "data/raw_audio/crema"    # <--- NEW

# NEW 8-CLASS MAPPING (Standard SER Emotions)
# We map all datasets to these 8 standard IDs
EMOTION_MAP = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fear': 5,
    'disgust': 6,
    'surprise': 7
}

# Reverse Mapping for API Response
INT_TO_EMOTION = {v: k for k, v in EMOTION_MAP.items()}

# Updated Priority for Emergency App
PRIORITY_MAP = {
    "fear": "HIGH",
    "disgust": "HIGH",    # Often confused with extreme screaming
    "angry": "MEDIUM",    # Anger can be a fight/argument (Potential Danger)
    "surprise": "MEDIUM",
    "sad": "MEDIUM",      # UPGRADED from LOW (Crying check happens later)
    "happy": "LOW",
    "calm": "LOW",
    "neutral": "LOW"
}