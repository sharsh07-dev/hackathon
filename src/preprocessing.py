import librosa
import numpy as np
import soundfile as sf
from .config import *

def load_and_fix_length(file_path):
    """Loads audio, resamples to 16kHz, and fixes duration to 3s."""
    try:
        # Load audio with native sampling rate first, then resample
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Trim silence (Top-dB 20 is standard for speech)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Fix Length (Pad or Truncate)
        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        else:
            padding = SAMPLES_PER_TRACK - len(y)
            offset = padding // 2
            y = np.pad(y, (offset, SAMPLES_PER_TRACK - len(y) - offset), 'constant')
            
        return y
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_features(y):
    """Generates Log-Mel Spectrogram from audio signal."""
    # Compute Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, 
        sr=SAMPLE_RATE, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        n_mels=N_MELS
    )
    
    # Convert to Log-scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Add channel dimension for CNN input (Height, Width, Channel)
    return log_mel_spectrogram[..., np.newaxis]

def augment_audio(y):
    """Applies random augmentations for robustness."""
    aug_choice = np.random.choice(['noise', 'pitch', 'speed', 'none'], p=[0.3, 0.3, 0.2, 0.2])
    
    if aug_choice == 'noise':
        noise = np.random.randn(len(y))
        y = y + 0.005 * noise
    elif aug_choice == 'pitch':
        steps = np.random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=steps)
    elif aug_choice == 'speed':
        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)
        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        else:
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), 'constant')
            
    return y