import os
import sys
import glob
import time
import soundfile as sf
import numpy as np
import pandas as pd

# Pastikan path root terdaftar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.feature_extractor import extract_32_features_from_signal, FEATURE_NAMES
except ImportError:
    from feature_extractor import extract_32_features_from_signal, FEATURE_NAMES

def extract_features_pipeline(
    chunk_base_dir="data/02_chunks",
    output_csv="data/03_processed/audio_features_32.csv",
    sr=16000
):
    """
    Mengekstrak 32 fitur akustik dari seluruh file potongan (.wav) di data/02_chunks/
    dan menyimpannya ke tabel CSV data/03_processed/audio_features_32.csv
    lengkap dengan metadata parent_file untuk Group-Aware Validation.
    """
    classes = [
        {"folder": "tidak_ramai", "label": 0, "class_name": "tidak_ramai"},
        {"folder": "ramai", "label": 1, "class_name": "ramai"}
    ]

    metadata_cols = ["chunk_filename", "parent_file", "label", "class_name"]
    expected_cols = metadata_cols + FEATURE_NAMES

    rows = []
    start_time = time.time()

    print("=" * 60)
    print(" MEMULAI EKSTRAKSI 32 FITUR AUDIO DARI CHUNKS")
    print("=" * 60)

    for cls_info in classes:
        folder_path = os.path.join(chunk_base_dir, cls_info["folder"])
        if not os.path.exists(folder_path):
            print(f"[PERINGATAN] Folder tidak ditemukan: {folder_path}")
            continue

        wav_files = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
        print(f"\nMengekstrak kelas '{cls_info['class_name']}' ({len(wav_files)} chunks)...")

        for idx, fpath in enumerate(wav_files, 1):
            fname = os.path.basename(fpath)
            parent_file = fname.split("__")[0] if "__" in fname else os.path.splitext(fname)[0]

            try:
                # Baca sinyal audio langsung dengan soundfile (cepat untuk .wav)
                y, file_sr = sf.read(fpath)
                feats = extract_32_features_from_signal(y, sr=file_sr)

                row_data = {
                    "chunk_filename": fname,
                    "parent_file": parent_file,
                    "label": cls_info["label"],
                    "class_name": cls_info["class_name"]
                }
                for f_name, f_val in zip(FEATURE_NAMES, feats):
                    row_data[f_name] = f_val

                rows.append(row_data)

            except Exception as e:
                print(f"  [ERROR] Gagal mengekstrak {fname}: {e}")

            if idx % 200 == 0 or idx == len(wav_files):
                print(f"  Progress: {idx}/{len(wav_files)} ({idx/len(wav_files)*100:.1f}%)")

    # Susun DataFrame
    df = pd.DataFrame(rows)
    # Urutkan kolom sesuai spesifikasi
    df = df[expected_cols]

    # Simpan ke CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    elapsed = time.time() - start_time

    # ================= VALIDASI OTOMATIS =================
    print("\n" + "=" * 60)
    print(" VALIDASI HASIL EKSTRAKSI FITUR (CHECKPOINT TAHAP 4)")
    print("=" * 60)
    print(f"1. File CSV tersimpan di : {output_csv}")
    print(f"2. Dimensi DataFrame     : {df.shape} (Baris, Kolom)")
    print(f"   - Jumlah Kolom        : {len(df.columns)} (Wajib 36: 4 metadata + 32 fitur)")
    print(f"3. Distribusi Kelas:\n{df['class_name'].value_counts().to_string()}")
    print(f"4. Jumlah Parent File Unik : {df['parent_file'].nunique()} grup rekaman asli")

    # Cek missing values (NaN/Null/Inf)
    nan_count = df[FEATURE_NAMES].isna().sum().sum()
    null_count = df[FEATURE_NAMES].isnull().sum().sum()
    inf_count = np.isinf(df[FEATURE_NAMES].values).sum()
    print(f"5. Pengecekan Kualitas Data:")
    print(f"   - NaN Values : {nan_count}")
    print(f"   - Null Values: {null_count}")
    print(f"   - Inf Values : {inf_count}")

    # Statistik Fitur Utama
    print("\n6. Statistik Deskriptif Fitur Akustik Utama (Mean per Kelas):")
    summary = df.groupby("class_name")[["rms_mean", "rms_std", "centroid_mean", "centroid_std", "zcr_mean"]].mean()
    print(summary.to_string())

    print(f"\nWaktu komputasi total: {elapsed:.2f} detik.")
    print("=" * 60)

    assert df.shape[1] == 36, f"Jumlah kolom salah! Ditemukan {df.shape[1]}, harus 36."
    assert nan_count == 0, f"Ditemukan {nan_count} nilai NaN!"

    return df

if __name__ == "__main__":
    extract_features_pipeline()
