#!/usr/bin/env python3
"""
Sistem Deteksi Tingkat Keramaian Ruangan Berbasis Sinyal Akustik
Modul Sensor Pemantau Kontinu (Continuous Live Stream Monitor)
"""

import os
import sys
import time
import argparse
import csv
from datetime import datetime
import joblib
import numpy as np

# Path setup agar import src/feature_extractor selalu berhasil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.feature_extractor import extract_32_features_from_signal, FEATURE_NAMES
except ImportError:
    from feature_extractor import extract_32_features_from_signal, FEATURE_NAMES

def make_volume_meter(rms: float, max_expected: float = 0.12, bar_len: int = 10) -> str:
    """Membuat visualisasi text-based meter untuk energi sinyal RMS."""
    ratio = min(max(rms / max_expected, 0.0), 1.0)
    filled_len = int(round(ratio * bar_len))
    bar = "█" * filled_len + "░" * (bar_len - filled_len)
    return f"[{bar}] RMS: {rms:.4f}"

def load_system_artifacts(models_dir: str = "models"):
    """Memuat model Random Forest dan scaler."""
    rf_path = os.path.join(models_dir, "rf_audio_classifier.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Model file tidak ditemukan di '{rf_path}'. Jalankan 'python src/step3_train_model.py' terlebih dahulu.")

    rf_model = joblib.load(rf_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return rf_model, scaler

def append_to_log_csv(log_path: str, row_dict: dict):
    """Menyimpan riwayat deteksi ke file CSV."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)
    headers = ["timestamp", "rms_energy", "predicted_label", "status", "confidence", "inference_ms"]

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def run_continuous_monitor(
    duration: float = 5.0,
    sr: int = 16000,
    gain: float = 1.0,
    device=None,
    max_chunks: int = None,
    log_csv: str = "reports/live_detection_log.csv",
    simulate_file: str = None
):
    """Menjalankan loop pemantauan suara secara terus menerus."""
    print("=" * 75)
    print(" 🎙️ CROWDSENSE AUDIO: CONTINUOUS LIVE STREAM MONITOR (TERMINAL ENGINE)")
    print("=" * 75)
    print(f"Konfigurasi:")
    print(f" - Jendela Waktu      : {duration} detik ({int(duration * sr)} sampel)")
    print(f" - Sampling Rate      : {sr} Hz Mono")
    print(f" - Penguatan Mic (Gain: {gain:.1f}x")
    print(f" - Log Deteksi        : {log_csv}")
    if max_chunks:
        print(f" - Batas Siklus       : {max_chunks} siklus (Testing Mode)")
    print("=" * 75)

    # 1. Muat Model
    try:
        rf_model, scaler = load_system_artifacts()
        rf_model.n_jobs = 1  # Hindari multithreading process spawning overhead pada inferensi 1 sampel
        print("✅ Artefak Model Random Forest & Scaler berhasil dimuat.")
        # Lakukan warmup kompilasi librosa & model
        warmup_sig = np.zeros(sr, dtype=np.float32)
        _w_feat = extract_32_features_from_signal(warmup_sig, sr=sr)
        _ = rf_model.predict(_w_feat.reshape(1, -1))
        _ = rf_model.predict_proba(_w_feat.reshape(1, -1))
        print("⚡ Warmup DSP & Inference Engine selesai (Ready for real-time).")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        sys.exit(1)

    # 2. Cek Perangkat Suara
    if simulate_file is None:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            default_in = sd.default.device[0]
            if default_in < 0 and device is None:
                print("⚠️ Tidak ada mikrofon input default terdeteksi!")
                sys.exit(1)
            in_info = sd.query_devices(device if device is not None else default_in, "input")
            print(f"✅ Mikrofon Aktif: {in_info['name']}")
        except Exception as e:
            print(f"❌ Error inisialisasi sounddevice / mikrofon: {e}")
            print("   Jika berjalan di server tanpa audio card, gunakan argumen --simulate-file <path_audio>.")
            sys.exit(1)
    else:
        print(f"ℹ️ Menggunakan Mode Simulasi File: {simulate_file}")
        import librosa
        if not os.path.exists(simulate_file):
            print(f"❌ File simulasi tidak ditemukan: {simulate_file}")
            sys.exit(1)
        y_sim_full, _ = librosa.load(simulate_file, sr=sr, mono=True)
        sim_pos = 0

    chunk_samples = int(duration * sr)
    cycle_count = 0
    total_ramai = 0
    total_tenang = 0
    total_infer_time = 0.0

    print("\n🚀 Sensor aktif dan mulai memantau ruangan... (Tekan Ctrl+C untuk berhenti)\n")
    print(f"{'TIMESTAMP':<10} | {'METER ENERGI SUARA':<25} | {'STATUS RUANGAN':<22} | {'CONFIDENCE':<11} | {'LATENSI'}")
    print("-" * 88)

    try:
        while True:
            cycle_count += 1
            t_rec_start = time.time()

            # Tangkap Audio Buffer 5 Detik
            if simulate_file is None:
                rec_data = sd.rec(chunk_samples, samplerate=sr, channels=1, dtype="float32", device=device)
                sd.wait()
                y_chunk = rec_data.flatten()
            else:
                # Simulasi potongan
                if sim_pos + chunk_samples <= len(y_sim_full):
                    y_chunk = y_sim_full[sim_pos : sim_pos + chunk_samples]
                    sim_pos += chunk_samples
                else:
                    y_chunk = y_sim_full[sim_pos:]
                    y_chunk = np.pad(y_chunk, (0, chunk_samples - len(y_chunk)), mode="constant")
                    sim_pos = 0  # Loop kembali
                time.sleep(0.05)  # simulasi jeda

            # Terapkan Mic Gain Multiplier jika ada
            if gain != 1.0:
                y_chunk = np.clip(y_chunk * gain, -1.0, 1.0)

            # Hitung RMS energi dasar untuk meter visual
            rms_val = float(np.sqrt(np.mean(y_chunk ** 2)))
            meter_str = make_volume_meter(rms_val)

            # Ekstraksi 32 Fitur & Inferensi Cepat
            t0 = time.perf_counter()
            features_32 = extract_32_features_from_signal(y_chunk, sr=sr)
            X_in = features_32.reshape(1, -1)
            pred_class = int(rf_model.predict(X_in)[0])
            pred_proba = rf_model.predict_proba(X_in)[0]
            infer_ms = (time.perf_counter() - t0) * 1000.0

            total_infer_time += infer_ms
            now_str = datetime.now().strftime("%H:%M:%S")
            conf_pct = float(pred_proba[pred_class] * 100)

            if pred_class == 1:
                status_str = "🚨 RAMAI / BERISIK"
                total_ramai += 1
                # Bunyikan alarm suara terminal di setiap deteksi RAMAI
                try:
                    if sys.platform == "win32":
                        import winsound
                        winsound.Beep(880, 200)
                    else:
                        print("\a", end="", flush=True)
                except Exception:
                    pass
            else:
                status_str = "🟢 TENANG / KONDUSIF"
                total_tenang += 1

            # Cetak baris pemantauan
            print(f"[{now_str}] | {meter_str:<25} | {status_str:<22} | {conf_pct:6.2f}%     | {infer_ms:5.1f} ms")

            # Catat ke file log CSV
            log_row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rms_energy": round(rms_val, 6),
                "predicted_label": pred_class,
                "status": "Ramai" if pred_class == 1 else "Tidak Ramai",
                "confidence": round(conf_pct, 2),
                "inference_ms": round(infer_ms, 2)
            }
            append_to_log_csv(log_csv, log_row)

            # Periksa apakah sudah mencapai batas pengujian
            if max_chunks is not None and cycle_count >= max_chunks:
                print(f"\n[INFO] Batas pengujian ({max_chunks} siklus) tercapai.")
                break

    except KeyboardInterrupt:
        print("\n\n⏹️ Pemantauan dihentikan oleh pengguna.")

    # Ringkasan Akhir
    avg_latency = (total_infer_time / cycle_count) if cycle_count > 0 else 0
    print("\n" + "=" * 75)
    print(" 📊 RINGKASAN SESI PEMANTAUAN RUANGAN")
    print("=" * 75)
    print(f" - Total Siklus Terpantau : {cycle_count} jendela (durasi total: {cycle_count * duration:.1f} detik)")
    print(f" - Frekuensi Ramai (1)    : {total_ramai} kali ({total_ramai/max(cycle_count,1)*100:.1f}%)")
    print(f" - Frekuensi Tenang (0)   : {total_tenang} kali ({total_tenang/max(cycle_count,1)*100:.1f}%)")
    print(f" - Rata-rata Latensi AI   : {avg_latency:.2f} ms per siklus")
    print(f" - File Log Tersimpan     : {os.path.abspath(log_csv)}")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(description="CrowdSense Audio: Continuous Live Stream Monitor")
    parser.add_argument("--duration", type=float, default=5.0, help="Durasi perekaman per jendela (detik, default: 5.0)")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate audio (default: 16000 Hz)")
    parser.add_argument("--gain", type=float, default=1.0, help="Faktor penguat mikrofon (default: 1.0x)")
    parser.add_argument("--device", type=int, default=None, help="Index perangkat audio input sounddevice (default: None)")
    parser.add_argument("--max-chunks", type=int, default=None, help="Batas jumlah siklus pemantauan sebelum berhenti otomatis")
    parser.add_argument("--log-csv", type=str, default="reports/live_detection_log.csv", help="Lokasi file output log CSV")
    parser.add_argument("--simulate-file", type=str, default=None, help="Path ke file audio untuk simulasi tanpa mikrofon")

    args = parser.parse_args()
    run_continuous_monitor(
        duration=args.duration,
        sr=args.sr,
        gain=args.gain,
        device=args.device,
        max_chunks=args.max_chunks,
        log_csv=args.log_csv,
        simulate_file=args.simulate_file
    )

if __name__ == "__main__":
    main()
