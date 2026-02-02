import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.config import *
from src.preprocessing import load_and_fix_length, extract_features, augment_audio
from src.model import create_surakshavaani_cnn

# --- LOADER 1: RAVDESS ---
def load_ravdess():
    print("📂 Loading RAVDESS...")
    x, y = [], []
    for file in glob.glob(os.path.join(RAVDESS_PATH, "Actor_*", "*.wav")):
        fname = os.path.basename(file)
        parts = fname.split("-")
        try:
            emotion_code = int(parts[2])
            # Map RAVDESS codes to our 8 classes
            # 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprise
            code_to_str = {
                1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
            }
            label_str = code_to_str.get(emotion_code)
            if label_str in EMOTION_MAP:
                audio = load_and_fix_length(file)
                if audio is not None:
                    x.append(extract_features(audio))
                    y.append(EMOTION_MAP[label_str])
        except: continue
    return x, y

# --- LOADER 2: TESS ---
def load_tess():
    print("📂 Loading TESS...")
    x, y = [], []
    # TESS folder structure: 'OAF_angry', 'YAF_fear', etc.
    for folder in glob.glob(os.path.join(TESS_PATH, "*")):
        folder_name = os.path.basename(folder) # e.g., "OAF_angry"
        
        # Extract emotion from folder name
        emotion = folder_name.split("_")[-1].lower() # "angry"
        if emotion == "ps": emotion = "surprise" # Fix TESS specific naming
        
        if emotion in EMOTION_MAP:
            for file in glob.glob(os.path.join(folder, "*.wav")):
                audio = load_and_fix_length(file)
                if audio is not None:
                    x.append(extract_features(audio))
                    y.append(EMOTION_MAP[emotion])
    return x, y

# --- LOADER 3: CREMA-D ---
def load_crema():
    print("📂 Loading CREMA-D...")
    x, y = [], []
    # CREMA Filename: 1001_DFA_ANG_XX.wav
    mapping = {
        'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad', 
        'ANG': 'angry', 'FEA': 'fear', 'DIS': 'disgust'
    }
    for file in glob.glob(os.path.join(CREMA_PATH, "*.wav")):
        fname = os.path.basename(file)
        part = fname.split("_")[2] # "ANG"
        
        if part in mapping:
            label_str = mapping[part]
            if label_str in EMOTION_MAP:
                audio = load_and_fix_length(file)
                if audio is not None:
                    x.append(extract_features(audio))
                    y.append(EMOTION_MAP[label_str])
    return x, y

if __name__ == "__main__":
    # 1. Load All Datasets
    x1, y1 = load_ravdess()
    x2, y2 = load_tess()
    x3, y3 = load_crema()
    
    # Combine
    X_data = np.array(x1 + x2 + x3)
    y_data = np.array(y1 + y2 + y3)
    
    if len(X_data) == 0:
        print("❌ No data found! Please check TESS/CREMA paths.")
        exit()
        
    print(f"✅ Total Samples Loaded: {len(X_data)}") # Should be > 10,000

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
    
    # 3. Class Weights (Crucial for 8 classes)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # 4. Build & Train
    model = create_surakshavaani_cnn(input_shape=X_train[0].shape)
    
    checkpoint = ModelCheckpoint("models/saved_models/best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    
    print("🚀 Starting Mega-Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=40,
        batch_size=64, # Increased batch size for more data
        class_weight=class_weight_dict,
        callbacks=[checkpoint, early_stop]
    )
    
    model.save("models/surakshavaani_final.h5")
    print("✅ Model Trained on All Emotions!")