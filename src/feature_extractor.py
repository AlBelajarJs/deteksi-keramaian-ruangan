import numpy as np
import librosa
import soundfile as sf

# 32 Feature Names in exact order
FEATURE_NAMES = [
    "rms_mean", "rms_std",
    "centroid_mean", "centroid_std",
    "zcr_mean", "zcr_std",
] + [f"mfcc_{i}_mean" for i in range(1, 14)] + [f"mfcc_{i}_std" for i in range(1, 14)]

def extract_32_features_from_signal(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Mengekstrak 32 fitur akustik dari array sinyal 1D (mono):
    - RMS (mean, std) = 2 fitur
    - Spectral Centroid (mean, std) = 2 fitur
    - Zero Crossing Rate (mean, std) = 2 fitur
    - MFCC 1-13 (mean, std) = 26 fitur
    Total = 32 fitur numerik.
    """
    if y is None or len(y) == 0:
        return np.zeros(32, dtype=np.float64)

    # Pastikan sinyal 1D float
    y_proc = np.asarray(y, dtype=np.float32).flatten()

    # 1. Antisipasi sinyal hening mutlak (digital zero) agar tidak memicu NaN pada MFCC/Centroid
    if np.max(np.abs(y_proc)) < 1e-6:
        y_proc = y_proc + (1e-6 * np.random.randn(len(y_proc)).astype(np.float32))

    # 2. Ekstraksi fitur
    # - RMS Energy
    rms = librosa.feature.rms(y=y_proc)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # - Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y_proc, sr=sr)[0]
    cent_mean = float(np.mean(cent))
    cent_std = float(np.std(cent))

    # - Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y_proc)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # - MFCC 13 Koefisien
    mfcc = librosa.feature.mfcc(y=y_proc, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # shape (13,)
    mfcc_std = np.std(mfcc, axis=1)    # shape (13,)

    # 3. Gabungkan menjadi vektor 1D berukuran (32,)
    features = np.hstack([
        rms_mean, rms_std,
        cent_mean, cent_std,
        zcr_mean, zcr_std,
        mfcc_mean,
        mfcc_std
    ]).astype(np.float64)

    # Pengaman akhir terhadap NaN atau Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    assert len(features) == 32, f"Ekstraksi menghasilkan {len(features)} fitur, seharusnya 32"
    return features

def extract_32_features_from_file(file_path: str, sr: int = 16000) -> np.ndarray:
    """
    Fungsi utilitas membaca file audio (wav/mp3/dll) dan mengekstrak 32 fitur langsung.
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return extract_32_features_from_signal(y, sr=sr)
