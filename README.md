# 🎙️ CrowdSense Audio: Sistem Deteksi Tingkat Keramaian Ruangan Berbasis Sinyal Akustik

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Ready-FF4B4B.svg)](https://streamlit.io/)
[![CRISP-DM Standard](https://img.shields.io/badge/Methodology-CRISP--DM-success.svg)](#metodologi-crisp-dm)

Proyek Sains Data *End-to-End* untuk mengklasifikasikan tingkat keramaian ruangan secara otomatis (**"Ramai / Kelas 1"** vs **"Tidak Ramai / Kelas 0"**) menggunakan ekstraksi **32 parameter fitur statistik audio** (Librosa) dan algoritma **Random Forest Classifier** yang ringan, akurat, dan **menjaga privasi (*Privacy-Preserving*)**.

---

## 📌 Ringkasan Eksekutif & Keunggulan Solusi
1. **Perlindungan Privasi 100% (*Privacy-Preserving*):** Berbeda dengan kamera CCTV atau mikrofon konvensional yang merekam isi percakapan, sistem ini langsung mengonversi sinyal audio menjadi representasi numerik statistik (RMS, Spectral Centroid, ZCR, dan MFCC) tanpa menyimpan audio percakapan pribadi pengunjung.
2. **Desain Anti-Kebocoran Data (*Group-Aware Anti-Leakage*):** Pemisahan data latih dan uji menggunakan `GroupShuffleSplit` berbasis file induk asli (`parent_file`), menjamin 100% tidak ada sampel audio dari ruangan yang sama yang bocor antara *Train* dan *Test*.
3. **Inferensi Super Cepat:** Waktu inferensi rata-rata hanya **`13.19 milidetik`** per sampel ($75\times$ lebih cepat dari batas toleransi 1 detik).
4. **Performa Tangguh:** Mencapai **Akurasi 80.08%** pada data uji bebas kebocoran dan **Mean F1-Score 0.8806 (88.06%)** pada 5-Fold GroupKFold Cross-Validation.

---

## 🏗️ Struktur Direktori Proyek

```text
PSD-AUDIO/
│
├── data/
│   ├── 01_raw/                         # Rekaman audio mentah asli
│   │   ├── ramai/                      # 101 file (97 valid diproses)
│   │   ├── tidak_ramai/                # 94 file (89 valid diproses)
│   │   └── file_mapping.csv            # Pemetaan nama file asli ke nama standar
│   │
│   ├── 02_chunks/                      # Potongan audio berdurasi tetap 5 detik (.wav @16kHz)
│   │   ├── ramai/                      # 742 file chunk .wav
│   │   └── tidak_ramai/                # 543 file chunk .wav
│   │
│   └── 03_processed/                   # Tabel fitur numerik hasil ekstraksi
│       └── audio_features_32.csv       # 1.285 baris x 36 kolom (4 metadata + 32 fitur)
│
├── notebooks/                                              # Rangkaian 5 Notebook CRISP-DM Terhubung (Pre-Executed)
│   ├── 00_Index_dan_Business_Understanding.ipynb           # Fase 1: Master TOC, Problem Statement & Env Check
│   ├── 01_Data_Understanding_dan_EDA.ipynb                 # Fase 2: Eksplorasi Waveform, Mel-Spectrogram & Sinyal
│   ├── 02_Data_Preparation_Chunking_dan_Ekstraksi_32Fitur.ipynb # Fase 3: Preprocessing, 32 Fitur, Boxplot & Korelasi
│   ├── 03_Modeling_dan_Benchmarking.ipynb                  # Fase 4: Anti-Leakage Split, 5-Fold CV, & 4 Model Benchmark
│   └── 04_Evaluation_dan_Simulasi_Deployment.ipynb         # Fase 5 & 6: Confusion Matrix, ROC, FI & Live Simulation
│
├── src/                                # Modul pipa data dan inferensi Python
│   ├── __init__.py
│   ├── feature_extractor.py            # Single source of truth ekstraksi 32 fitur audio
│   ├── step1_chunk_audio.py            # Pipa pemotongan audio 5 detik anti-leakage
│   ├── step2_extract_features.py       # Pipa ekstraksi 32 fitur ke CSV
│   ├── step3_train_model.py            # Pipa training, benchmarking & evaluasi 4 model
│   └── live_monitor.py                 # Engine pemantau suara live tanpa henti via terminal
│
├── models/                             # Artefak model serialisasi
│   ├── rf_audio_classifier.pkl         # Model Random Forest Classifier utama
│   ├── best_model.pkl                  # Model performa terbaik
│   ├── scaler.pkl                      # StandardScaler yang di-fit pada data latih
│   └── model_metadata.json             # Metadata metrik pengujian dan konfigurasi fitur
│
├── reports/                            # Laporan evaluasi dan visualisasi
│   ├── model_benchmark_results.csv     # Tabel perbandingan lengkap 4 algoritma
│   ├── live_detection_log.csv          # Riwayat log pemantauan live real-time
│   └── figures/                        # Grafik publikasi resolusi tinggi (300 DPI)
│       ├── 01_distribusi_kelas.png
│       ├── 02_waveform_comparison.png
│       ├── 03_mel_spectrogram_comparison.png
│       ├── 04_boxplot_fitur_utama.png
│       ├── 05_correlation_heatmap.png
│       ├── 06_model_comparison_bar.png
│       ├── 07_confusion_matrix_rf.png
│       ├── 08_roc_curves_comparison.png
│       └── 09_top10_feature_importance.png
│
├── app.py                              # Deployment Aplikasi Web Interaktif (Streamlit)
├── requirements.txt                    # Daftar spesifikasi pustaka Python
├── .gitignore                          # Konfigurasi proteksi repositori GitHub
├── dokumentasi-alur.md                 # Blueprint teknis proyek
├── LAPORAN_CRISPDM.md                  # Laporan lengkap 6 fase CRISP-DM & Panduan 10 Slide Sidang
└── README.md                           # Dokumentasi utama proyek
```

---

## 📓 Rangkaian 5 Notebook CRISP-DM Terhubung (*Pre-Executed*)
Seluruh tahapan siklus CRISP-DM telah dieksekusi secara tuntas di dalam 5 Jupyter Notebook yang saling terhubung melalui *header* dan *footer* navigasi interaktif (siap dibaca di GitHub Viewer tanpa perlu mengeksekusi ulang):

1. [🏠 **Notebook 00: Index & Business Understanding**](notebooks/00_Index_dan_Business_Understanding.ipynb)
   - Latar belakang masalah akustik vs CCTV, perumusan sasaran bisnis vs teknis, arsitektur sistem, dan verifikasi dataset mentah.
2. [📊 **Notebook 01: Data Understanding & EDA**](notebooks/01_Data_Understanding_dan_EDA.ipynb)
   - Karakteristik data audio mentah, plot *Waveform* (domain waktu), dan *Mel-Spectrogram* (domain frekuensi vokal manusia vs deru AC).
3. [⚙️ **Notebook 02: Data Preparation, Chunking & Ekstraksi 32 Fitur**](notebooks/02_Data_Preparation_Chunking_dan_Ekstraksi_32Fitur.ipynb)
   - Pipeline chunking 5 detik terisolasi, safe silence trimming, ekstraksi 32 parameter DSP Librosa, boxplot, dan Pearson correlation heatmap.
4. [🧠 **Notebook 03: Modeling & Benchmarking**](notebooks/03_Modeling_dan_Benchmarking.ipynb)
   - Pemisahan data anti-kebocoran (`GroupShuffleSplit` disjoint groups), 5-Fold GroupKFold CV, dan benchmarking 4 algoritma klasifikasi.
5. [🚀 **Notebook 04: Evaluation & Simulasi Deployment**](notebooks/04_Evaluation_dan_Simulasi_Deployment.ipynb)
   - Evaluasi mendalam Confusion Matrix, kurva ROC-AUC, analisis ilmiah Top 10 Feature Importance, simulasi inferensi live per sampel, dan panduan deployment Streamlit.

---

## 📊 Hasil Eksperimen & Benchmarking Nyata

### 1. Komparasi 4 Algoritma Klasifikasi (Data Uji 236 Chunks)

| Model / Algoritma | Akurasi | Presisi | Recall | F1-Score (Binary) | F1-Score (Macro) | ROC-AUC | Waktu Inferensi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Random Forest (Tuned Balanced)** | **80.08%** | **75.00%** | **77.32%** | **0.7614** | **0.7953** | **0.8463** | **13.19 ms/sampel** |
| 🥈 **Logistic Regression (Scaled)** | 79.24% | 74.00% | 76.29% | 0.7513 | 0.7865 | 0.8589 | 0.31 ms/sampel |
| 🥉 **Gradient Boosting Classifier** | 77.12% | 69.37% | 79.38% | 0.7404 | 0.7679 | 0.8596 | 0.61 ms/sampel |
| 4 **Support Vector Machine (RBF)** | 77.12% | 73.12% | 70.10% | 0.7158 | 0.7622 | 0.8605 | 0.35 ms/sampel |

> **Catatan Validasi Silang (5-Fold GroupKFold CV):**
> Random Forest menghasilkan skor F1 per fold: `[0.9072, 0.8739, 0.9244, 0.8708, 0.8265]` dengan rata-rata **`0.8806 (+/- 0.0337)`**, membuktikan kemampuan generalisasi yang sangat stabil di berbagai ruangan berbeda.

---

### 2. Confusion Matrix Random Forest pada Test Set

```text
                      PREDIKSI MODEL
                  Tidak Ramai (0)   Ramai (1)
AKTUAL  
Tidak Ramai (0)        114 (TN)       25 (FP)   -> Spesifisitas: 82.01%
Ramai (1)               22 (FN)       75 (TP)   -> Recall/Sensitivitas: 77.32%
```

---

### 3. Top 10 Feature Importance (Interpretasi Akustik)

| Peringkat | Fitur | Bobot Importance (%) | Alasan Fisis / Akustik |
| :---: | :--- | :---: | :--- |
| **#1** | `rms_mean` | **9.49%** | Tingkat energi suara rata-rata kerumunan jauh lebih tinggi dari ruangan sepi. |
| **#2** | `rms_std` | **6.39%** | Fluktuasi energi vokal dinamis membedakan keramaian dari dengung AC konstan. |
| **#3** | `mfcc_1_mean` | **5.60%** | Energi pada spektrum dasar pita suara manusia (*vocal tract filter*). |
| **#4** | `mfcc_12_std` | **4.74%** | Dinamika artikulasi frekuensi tinggi percakapan kelompok. |
| **#5** | `mfcc_2_mean` | **4.06%** | Bentuk formant vokal frekuensi rendah-menengah. |
| **#6** | `mfcc_5_mean` | **3.92%** | Resonansi khas intonasi suara manusia. |
| **#7** | `mfcc_13_std` | **3.84%** | Variasi komponen spektral atas khas ucapan manusia. |
| **#8** | `mfcc_3_std` | **3.81%** | Dinamika transisi vokal-konsonan. |
| **#9** | `zcr_mean` | **3.68%** | Konsonan frikatif desis manusia ('s', 'sy', 'f', 't', 'k') menaikkan lintasan nol. |
| **#10** | `centroid_mean` | **3.67%** | Pergeseran titik berat spektral ke frekuensi menengah ($1.500 - 2.500\text{ Hz}$). |

---

## 🚀 Panduan Instalasi & Menjalankan Proyek

### 1. Kloning Direktori & Instalasi Dependensi
Pastikan Python versi 3.9 s/d 3.13 telah terpasang. Buka terminal di direktori proyek dan jalankan:

```bash
pip install -r requirements.txt
```

### 2. Menjalankan Seluruh Pipa Data (End-to-End Pipeline)

```bash
# Tahap 1: Pemotongan file audio mentah menjadi potongan 5 detik (.wav @16kHz)
python src/step1_chunk_audio.py

# Tahap 2: Ekstraksi 32 fitur akustik dari seluruh potongan ke CSV
python src/step2_extract_features.py

# Tahap 3: Pelatihan model, benchmarking 4 algoritma, dan penyimpanan model .pkl
python src/step3_train_model.py
```

### 3. Menjalankan Aplikasi Web Interaktif (Deployment Streamlit)

Jalankan perintah Streamlit berikut:

```bash
streamlit run app.py
```

Browser akan otomatis membuka alamat lokal: `http://localhost:8501`.

### 4. Menjalankan Sensor Pemantau Suara Real-Time Kontinu (CLI IoT Engine)

Jalankan skrip pemantau ruangan tanpa henti langsung dari terminal:

```bash
# Mode pemantauan kontinu default 5 detik (Tekan Ctrl+C untuk berhenti)
python src/live_monitor.py

# Mode pengujian cepat (misal 3 siklus dengan durasi per siklus 2 detik)
python src/live_monitor.py --max-chunks 3 --duration 2.0

# Opsi kalibrasi penguat mikrofon (jika suara mic laptop terlalu pelan)
python src/live_monitor.py --gain 1.5
```

---

## 🖥️ Fitur Lengkap Aplikasi Web (`app.py`)
1. **Tab 1 — 🎙️ Live Microphone & Radar Real-Time:**
   - **Mode Rekam Cepat Browser (`st.audio_input`):** Rekam suara sesaat langsung dari mikrofon browser (bicara atau buat suara bising untuk menguji) dan dapatkan vonis status instan.
   - **Mode Radar Otomatis (Continuous 5s Loop):** Pemantauan otomatis tiap 5 detik dari mikrofon laptop dengan pembaruan grafik garis fluktuasi keramaian real-time.
   - **Slider Mic Gain Booster:** Kalibrasi kepekaan mikrofon laptop jika sinyal input terlalu kecil.
2. **Tab 2 — 📁 Deteksi File & Sampel Demo:**
   - **Mode Demo Cepat (Quick Test):** Tersedia dropdown sampel audio bawaan (suasana ramai dan sepi) sehingga Anda bisa mendemokan sistem saat presentasi tanpa harus mencari file manual di komputer.
   - **Unggah File Audio:** Mendukung format `.wav`, `.mp3`, `.mpeg`, `.ogg`, atau `.flac` dengan analisis timeline 5 detik dinamis.
   - **Visualisasi Sinyal Langsung:** Menampilkan grafik *Waveform* dan *Mel-Spectrogram* secara berdampingan.
3. **Tab 3 — 📊 Dashboard Metodologi CRISP-DM:**
   - Memuat tabel benchmark 4 model, confusion matrix, kurva ROC-AUC, dan feature importance resolusi tinggi langsung di dalam dashboard web sebagai sarana presentasi sidang/kelas.

---

## 📜 Lisensi
Proyek ini dikembangkan untuk keperluan akademik dan penelitian Sains Data di bawah lisensi MIT.
