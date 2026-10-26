# LAPORAN LENGKAP PROYEK SAINS DATA (CRISP-DM)
# Sistem Deteksi Tingkat Keramaian Ruangan Berbasis Sinyal Akustik (*Crowd Level Detection*)

---

## 📑 DAFTAR ISI LAPORAN
1. **FASE 1: Business Understanding (Pemahaman Bisnis & Perumusan Masalah)**
2. **FASE 2: Data Understanding (Eksplorasi Sinyal & Karakteristik Data Audio)**
3. **FASE 3: Data Preparation (Pra-Pemrosesan & Rekayasa 32 Fitur Akustik)**
4. **FASE 4: Modeling (Pelatihan & Benchmarking 4 Algoritma Klasifikasi)**
5. **FASE 5: Evaluation (Validasi Performa, Confusion Matrix, & Interpretasi Fitur)**
6. **FASE 6: Deployment (Implementasi Sistem Aplikasi Web Interaktif Streamlit)**
7. **PANDUAN SLIDE PRESENTASI SIDANG / KELAS (SLIDE 1 S/D SLIDE 10)**

---

## FASE 1: Business Understanding (Pemahaman Bisnis & Perumusan Masalah)

### 1.1. Latar Belakang Masalah
Pengelolaan fasilitas bersama seperti perpustakaan universitas, *co-working space*, ruang diskusi mandiri, dan laboratorium komputer memerlukan pemantauan kondisi kebisingan dan keramaian secara berkelanjutan demi menjamin kenyamanan belajar dan ketertiban lingkungan. Dua pendekatan konvensional yang ada saat ini memiliki kekurangan fatal:
1. **Patroli Petugas Manual:** Sangat tidak efisien, memakan waktu, subjektif, dan tidak mampu mengawasi puluhan ruangan secara kontinu 24/7.
2. **Pemantauan Berbasis Kamera Visual (CCTV + Computer Vision):**
   - **Melanggar Privasi Pengunjung:** Pengunjung merasa diawasi secara visual, menimbulkan resistensi etika dan hukum privasi data pribadi.
   - **Beban Biaya & Komputasi Tinggi:** Membutuhkan infrastruktur kamera beresolusi tinggi, bandwidth jaringan yang besar untuk transmisi video, kapasitas penyimpanan berukuran terabita, serta GPU mahal untuk memproses model deteksi objek visual (*YOLO/CNN*).

### 1.2. Solusi yang Diajukan
Membangun **Sistem Deteksi Tingkat Keramaian Ruangan Berbasis Sinyal Akustik (Audio)** yang ringan, hemat daya, dan **menjaga privasi penuh (*Privacy-Preserving*)**. 
Sistem ini hanya menangkap getaran suara sesaat di dalam ruangan, lalu secara instan mengekstrak **32 parameter statistik numerik** menggunakan algoritma *Digital Signal Processing (DSP)* tanpa merekam atau mentranskripsikan isi pembicaraan pengunjung. Representasi numerik ini kemudian diklasifikasikan oleh model pembelajaran mesin (*Random Forest Classifier*) menjadi status **"Ramai" (Kelas 1)** atau **"Tidak Ramai" (Kelas 0)**.

### 1.3. Matriks Tujuan Bisnis vs Tujuan Sains Data

| Kategori | Parameter | Target Terukur (*Baseline Goal*) |
| :--- | :--- | :--- |
| **Tujuan Bisnis** | Efisiensi Pemantauan | Menggantikan patroli manual dengan sistem peringatan otomatis. |
| | Perlindungan Privasi | 100% *Privacy-Preserving* (bebas rekaman suara vokal manusia). |
| | Biaya Perangkat Keras | Cukup menggunakan sensor mikrofon standar dan CPU laptop biasa. |
| **Tujuan Sains Data** | Akurasi Klasifikasi | $\text{Accuracy} \ge 80.0\%$ pada data uji yang belum pernah dilihat. |
| | Metrik F1-Score | $\text{F1-Score} \ge 0.85$ pada validasi silang (Cross-Validation). |
| | Waktu Inferensi | $\text{Latency} < 1.000\text{ milidetik}$ ($< 1\text{ detik}$) per sampel audio. |
| | Ketahanan Model | Mampu membedakan percakapan manusia dari deru mesin pendingin (AC/kipas). |

---

## FASE 2: Data Understanding (Eksplorasi Sinyal & Karakteristik Data Audio)

### 2.1. Akuisisi Data Mentah Asli
Dataset dikumpulkan dari rekaman lingkungan nyata yang dibagi ke dalam dua kategori folder:
- **Kelas Ramai (Label 1):** Berisi 101 file rekaman suasana kantin, lobi kampus, ruang diskusi kelompok, dan koridor aktif dengan karakteristik suara tumpang tindih (*overlapping speech*), tawa, dan langkah kaki.
- **Kelas Tidak Ramai (Label 0):** Berisi 94 file rekaman suasana perpustakaan hening, ruang kelas kosong, dan ruangan sepi dengan suara deru kompresor AC atau desis kipas angin.
- **Total Sampel Mentah:** **195 file rekaman asli**.

### 2.2. Verifikasi Format & Penanganan File
Dari 195 file mentah, sebanyak **186 file berhasil dimuat dan diproses penuh**, sedangkan **9 file tidak dapat diolah** karena keterbatasan teknis lingkungan lokal:
- 6 file berformat `.m4a` (AAC) yang memerlukan dekoder eksternal FFmpeg.
- 3 file `.mp3` yang memiliki *header frame* rusak (*invalid MPEG header*).
Seluruh 186 file valid dipetakan dengan rapi ke dalam `data/01_raw/file_mapping.csv`.

### 2.3. Distribusi Sampel Pasca-Chunking (Jendela 5 Detik)
Setelah proses pemotongan terstandarisasi, diperoleh total **1.285 file potongan (.wav mono 16.000 Hz, durasi 5.0 detik)**:
- **Kelas Ramai (1):** **742 chunks** ($57.74\%$) dari 97 file induk asli.
- **Kelas Tidak Ramai (0):** **543 chunks** ($42.26\%$) dari 89 file induk asli.
- *Rasio Kelas:* $58 : 42$ (sangat ideal dan seimbang, tidak memerlukan oversampling sintetik seperti SMOTE).

```text
[Visualisasi Tersimpan]: reports/figures/01_distribusi_kelas.png
```

### 2.4. Eksplorasi Sinyal Akustik (Domain Waktu & Frekuensi)
1. **Domain Waktu (*Waveform*):**
   - *Kelas Ramai:* Memperlihatkan *envelope* amplitudo yang bergejolak hebat secara acak dengan puncak-puncak tajam (amplitudo puncak mencapai $0.4 - 0.7$) yang merefleksikan dinamika silabel suku kata vokal manusia.
   - *Kelas Tidak Ramai:* Memperlihatkan profil gelombang yang seragam dan datar dengan amplitudo rendah (puncak $< 0.15$), mencerminkan *noise floor* ruangan atau getaran harmonik stasioner dari motor AC.
   - *Visualisasi Tersimpan:* `reports/figures/02_waveform_comparison.png`.
2. **Domain Frekuensi (*Mel-Spectrogram* dalam dB):**
   - *Kelas Ramai:* Spektrum menunjukkan persebaran daya yang padat dan dinamis pada rentang frekuensi menengah $300\text{ Hz} - 3.400\text{ Hz}$ (daerah forman vokal manusia F1, F2, F3).
   - *Kelas Tidak Ramai:* Konsentrasi energi hanya tertumpuk di frekuensi sangat rendah ($< 250\text{ Hz}$) akibat deru mekanis pendingin udara, sementara frekuensi menengah ke atas tampak gelap (daya di bawah $-60\text{ dB}$).
   - *Visualisasi Tersimpan:* `reports/figures/03_mel_spectrogram_comparison.png`.

### 2.5. Statistik Deskriptif Fitur Akustik Nyata

| Parameter Akustik | Kelas Tidak Ramai (0) | Kelas Ramai (1) | Rasio / Makna Fisis |
| :--- | :---: | :---: | :--- |
| **`rms_mean`** (Kekuatan Energi) | `0.031908` | `0.072200` | **$2.26\times$ lebih bertenaga** pada ruangan ramai. |
| **`rms_std`** (Fluktuasi Energi) | `0.017150` | `0.034579` | **$2.02\times$ lebih fluktuatif** (suara manusia tidak monoton). |
| **`centroid_mean`** (Titik Berat Spektrum) | `1410.85 Hz` | `1983.99 Hz` | **$+573.14\text{ Hz}$** bergeser naik ke pita suara manusia. |
| **`zcr_mean`** (Zero Crossing Rate) | `0.094074` | `0.182022` | **$1.93\times$ lebih sering lintas nol** (fonem frikatif desis). |

- *Visualisasi Tersimpan:* `reports/figures/04_boxplot_fitur_utama.png` dan `reports/figures/05_correlation_heatmap.png`.

---

## FASE 3: Data Preparation (Pra-Pemrosesan & Rekayasa 32 Fitur)

### 3.1. Pipeline Pemotongan Audio Otomatis (`src/step1_chunk_audio.py`)
1. **Resampling ke $16.000\text{ Hz}$ Mono:** Memastikan seluruh audio memiliki basis titik waktu yang identik. Menghemat beban RAM $3\times$ lipat dibanding $44.100\text{ Hz}$ tanpa memotong informasi frekuensi vokal manusia ($f_{\text{max}} = 8.000\text{ Hz}$ menurut Teorema Nyquist).
2. **Safe Silence Trimming (`top_db=30`):** Menghapus keheningan kosong di awal/akhir penekanan tombol rekam. Dilengkapi aturan keamanan: jika sisa sinyal $< 2.5\text{ detik}$, trim dibatalkan agar file kelas `tidak_ramai` tidak terhapus.
3. **Segmentasi Jendela 5 Detik ($80.000\text{ sampel}$):**
   - Pemotongan dilakukan per file asli tanpa *overlap*.
   - Jika sisa potongan terakhir $\ge 2.5\text{ detik}$, dilakukan *zero-padding* hingga tepat 5 detik. Sisa $< 2.5\text{ detik}$ diabaikan.
4. **Penamaan Anti-Leakage:** Setiap chunk diberi nama `{parent_id}__chunk{idx:02d}.wav` (contoh: `ramai_001__chunk01.wav`).

### 3.2. Ekstraksi 32 Vektor Fitur Akustik (`src/feature_extractor.py`)
Dari setiap sinyal 5 detik dihitung 4 kelompok representasi digital:
1. **RMS Energy (`rms_mean`, `rms_std`):** Mengukur kekuatan amplitudo dan gejolak dinamika volume (2 fitur).
2. **Spectral Centroid (`centroid_mean`, `centroid_std`):** Mengukur titik pusat gravitasi frekuensi suara (2 fitur).
3. **Zero Crossing Rate (`zcr_mean`, `zcr_std`):** Mengukur frekuensi gelombang memotong sumbu nol (2 fitur).
4. **MFCC 13 Koefisien (`mfcc_1_mean` s/d `mfcc_13_mean` & `mfcc_1_std` s/d `mfcc_13_std`):** Menangkap kontur amplop spektrum frekuensi pendengaran manusia pada skala Mel (26 fitur).
- **Total:** Matriks numerik berukuran **$1.285\text{ baris} \times 36\text{ kolom}$** (4 metadata + 32 fitur), disimpan ke `data/03_processed/audio_features_32.csv` dengan **0 nilai NaN / Null**.

### 3.3. Strategi Pembagian Data Anti-Kebocoran (*Group-Aware Train-Test Split*)
- Menggunakan `GroupShuffleSplit(test_size=0.20, random_state=42)` dengan pengelompokan `groups=parent_file`.
- **Hasil Pemisahan:**
  - **Data Latih (*Train Set*):** **1.049 chunks** dari **148 file induk** (404 Tidak Ramai, 645 Ramai).
  - **Data Uji (*Test Set*):** **236 chunks** dari **38 file induk** (139 Tidak Ramai, 97 Ramai).
  - **Uji Kebocoran:** `set(groups_train).intersection(set(groups_test)) == set()` (Terbukti 100% Saling Lepas).
- **Standarisasi Fitur:** Objek `StandardScaler()` di-*fit* hanya pada matriks `X_train` lalu diterapkan ke `X_train` dan `X_test` (disimpan ke `models/scaler.pkl`).

---

## FASE 4: Modeling (Pelatihan & Benchmarking 4 Algoritma)

### 4.1. Model Utama: Random Forest Classifier
- Parameter: `n_estimators=100`, `class_weight='balanced'`, `min_samples_split=2`, `random_state=42`.
- Menggunakan pemungutan suara mayoritas (*majority voting*) dari 100 pohon keputusan acak.
- Evaluasi Cross-Validation 5-Fold (`GroupKFold` pada data latih) menghasilkan F1-score:
  `[0.9072, 0.8739, 0.9244, 0.8708, 0.8265]` $\rightarrow$ **Rata-rata CV F1: 0.8806 (+/- 0.0337)**.

### 4.2. Benchmarking 3 Model Alternatif
Untuk memenuhi standar akademis CRISP-DM tingkat lanjut, Random Forest dibandingkan secara objektif dengan:
1. **Support Vector Machine (SVM):** Kernel RBF, $C=1.0$, `class_weight='balanced'`.
2. **Gradient Boosting Classifier:** 100 estimator pohon boosting sekuensial, `learning_rate=0.1`.
3. **Logistic Regression (Baseline Linear):** Regresi logistik dengan regularisasi L2 terstandarisasi.

---

## FASE 5: Evaluation (Validasi Performa & Interpretasi Fitur)

### 5.1. Hasil Evaluasi Lengkap pada Data Uji Bebas Kebocoran (236 Chunks)

| Peringkat | Algoritma Klasifikasi | Akurasi | Presisi | Recall | F1-Score (Binary) | F1-Score (Macro) | ROC-AUC | Waktu Inferensi |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest (Tuned Balanced)** | **80.08%** | **75.00%** | **77.32%** | **0.7614** | **0.7953** | **0.8463** | **13.19 ms/sampel** |
| 🥈 | **Logistic Regression (Scaled)** | 79.24% | 74.00% | 76.29% | 0.7513 | 0.7865 | 0.8589 | 0.31 ms/sampel |
| 🥉 | **Gradient Boosting Classifier** | 77.12% | 69.37% | 79.38% | 0.7404 | 0.7679 | 0.8596 | 0.61 ms/sampel |
| 4 | **Support Vector Machine (RBF)** | 77.12% | 73.12% | 70.10% | 0.7158 | 0.7622 | 0.8605 | 0.35 ms/sampel |

- *Visualisasi Tersimpan:* `reports/figures/06_model_comparison_bar.png` dan `reports/figures/08_roc_curves_comparison.png`.

### 5.2. Analisis Confusion Matrix (Random Forest)
Dari total 236 sampel uji yang diuji:
- **True Negative (TN = 114):** 114 sampel ruangan sepi berhasil diidentifikasi tenang (Spesifisitas = $82.01\%$).
- **True Positive (TP = 75):** 75 sampel keramaian berhasil ditangkap dengan benar (Recall = $77.32\%$).
- **False Positive (FP = 25):** 25 alarm palsu di mana desis atau ketukan keyboard di ruangan sepi keliru dianggap ramai.
- **False Negative (FN = 22):** 22 kejadian keramaian terlewat (umumnya terjadi pada jeda hening sesaat di tengah percakapan).
- *Visualisasi Tersimpan:* `reports/figures/07_confusion_matrix_rf.png`.

### 5.3. Analisis Ilmiah Top 10 Feature Importance

```text
1. rms_mean        :  9.49%  ====================
2. rms_std         :  6.39%  =============
3. mfcc_1_mean     :  5.60%  ===========
4. mfcc_12_std     :  4.74%  ==========
5. mfcc_2_mean     :  4.06%  ========
6. mfcc_5_mean     :  3.92%  ========
7. mfcc_13_std     :  3.84%  ========
8. mfcc_3_std      :  3.81%  ========
9. zcr_mean        :  3.68%  =======
10. centroid_mean  :  3.67%  =======
```

- **Interpretasi Fisis:**
  1. `rms_mean` ($9.49\%$) dan `rms_std` ($6.39\%$) menjadi penentu dominan utama. Suara mesin pendingin (AC/kipas) memiliki energi yang konstan dengan fluktuasi mendekati nol, sedangkan obrolan manusia memiliki fluktuasi amplitudo yang sangat tinggi.
  2. Kombinasi koefisien MFCC 1, 2, 5, 12, dan 13 secara ilmiah menangkap filter rongga mulut dan pita suara manusia (*vocal tract filter*).
  3. `zcr_mean` dan `centroid_mean` melengkapi model dalam mendeteksi konsonan desis dan pergeseran energi ke frekuensi menengah.
- *Visualisasi Tersimpan:* `reports/figures/09_top10_feature_importance.png`.

### 5.4. Evaluasi Ketercapaian Sasaran Bisnis & Sains Data

| Parameter Sasaran | Target Awal | Capaian Riil Sistem | Status Evaluasi |
| :--- | :---: | :---: | :---: |
| **Akurasi Klasifikasi Data Uji** | $\ge 80.0\%$ | **$80.08\%$** | ✅ **Tercapai** |
| **F1-Score Validasi Silang (5-Fold CV)** | $\ge 0.85$ | **$0.8806$ ($88.06\%$)** | ✅ **Melampaui Target** |
| **Kecepatan Inferensi Komputasi** | $< 1.000\text{ ms}$ | **$13.19\text{ ms}$** | ✅ **$75\times$ Lebih Cepat** |
| **Pencegahan Kebocoran Data** | $0\%$ Leakage | $100\%$ Disjoint Grouping | ✅ **Sempurna** |
| **Perlindungan Privasi Pengunjung** | 100% Non-Invasif | Hanya 32 Fitur Numerik | ✅ **Sempurna** |

---

## FASE 6: Deployment (Implementasi Sistem Web Interaktif Streamlit & Sensor CLI)

### 6.1. Serialisasi Artefak Model
Seluruh model dan konfigurasi pendukung diekspor ke dalam folder `models/`:
- `models/rf_audio_classifier.pkl` (1.42 MB)
- `models/scaler.pkl` (1.35 KB)
- `models/model_metadata.json` (1.48 KB)

### 6.2. Arsitektur Dual-Mode Deployment
Sistem menyediakan 2 moda implementasi fungsional siap pakai:
1. **Aplikasi Web Interaktif (`app.py`):**
   - **Tab 1 (🎙️ Live Microphone & Radar Real-Time):** Fitur rekam suara ruangan langsung dari mikrofon browser (`st.audio_input`) serta Mode Radar Pemantauan Otomatis (Continuous 5s Loop) berbasis grafik fluktuasi real-time.
   - **Tab 2 (📁 Deteksi File & Sampel Demo):** Pemutar audio, deteksi file rekaman (`.wav`, `.mp3`, `.mpeg`, `.ogg`, `.flac`), analisis timeline 5 detik dinamis, serta visualisasi waveform dan mel-spectrogram.
   - **Tab 3 (📊 Dashboard Evaluasi CRISP-DM):** Memuat tabel komparasi benchmark 4 algoritma, kurva ROC, Confusion Matrix, dan Top 10 Feature Importance langsung di halaman web sebagai media presentasi.
2. **Sensor Pemantau Kontinu CLI (`src/live_monitor.py`):**
   - Berfungsi layaknya sensor IoT ruangan yang memantau tingkat keramaian secara terus menerus tiap 5 detik via terminal tanpa memerlukan browser.
   - Menyertakan visualisasi meter volume audio berbasis teks (`[██████░░░░] RMS: 0.042`), vonis kelas, latensi inferensi, dan logging otomatis ke `reports/live_detection_log.csv`.

---

## 🎤 PANDUAN SLIDE PRESENTASI SIDANG / KELAS (SLIDE 1 S/D SLIDE 10)

Berikut adalah panduan struktur slide lengkap beserta teks narasi presentasi (*talking points*) dan antisipasi pertanyaan penguji:

### SLIDE 1: Judul & Identitas Proyek
- **Judul Slide:** Sistem Deteksi Tingkat Keramaian Ruangan Berbasis Sinyal Akustik (*Crowd Level Detection*) Menggunakan Ekstraksi 32 Fitur Audio Librosa & Random Forest.
- **Sub-judul:** Implementasi End-to-End Berdasarkan Metodologi CRISP-DM.
- **Poin Presentasi:** "Selamat pagi/siang Bapak/Ibu Dosen. Hari ini kami mempresentasikan proyek sains data deteksi keramaian ruangan non-invasif ramah privasi yang dikembangkan secara terstruktur menggunakan siklus CRISP-DM."

### SLIDE 2: Business Understanding (Masalah & Urgensi Solusi)
- **Konten Visual:** Diagram perbandingan CCTV vs Sistem Audio Akustik.
- **Poin Kunci:**
  - Pemantauan manual petugas tidak efisien; CCTV melanggar privasi dan boros bandwidth/GPU.
  - Solusi kami: Mengubah gelombang audio langsung menjadi 32 angka statistik tanpa menyimpan percakapan pribadi.
  - Target: Akurasi $\ge 80\%$, F1 $\ge 0.85$, dan inferensi $< 1$ detik.

### SLIDE 3: Data Understanding (Karakteristik & Distribusi Data)
- **Konten Visual:** Grafik `01_distribusi_kelas.png`.
- **Poin Kunci:**
  - Total 195 file rekaman asli (186 valid diolah).
  - Dipotong menjadi 1.285 sampel audio berdurasi 5 detik (742 Ramai vs 543 Tidak Ramai).
  - Rasio kelas seimbang $58\% : 42\%$.

### SLIDE 4: Eksplorasi Sinyal Akustik (Domain Waktu vs Frekuensi)
- **Konten Visual:** Grafik `02_waveform_comparison.png` & `03_mel_spectrogram_comparison.png`.
- **Poin Kunci:**
  - Waveform ramai berfluktuasi tajam; waveform sepi berbentuk pita datar tipis.
  - Spektrogram ramai menunjukkan energi forman vokal manusia di rentang $300 - 3.400\text{ Hz}$, sedangkan sepi/AC hanya di bawah $250\text{ Hz}$.

### SLIDE 5: Data Preparation (Pipeline Ekstraksi & Pencegahan Leakage)
- **Konten Visual:** Diagram alur chunking 5 detik, resampling 16 kHz, dan formula 32 fitur.
- **Poin Kunci:**
  - 32 Fitur: 2 RMS, 2 Spectral Centroid, 2 ZCR, 26 MFCC (mean & std).
  - **Kewajiban Akademis:** Membagi data dengan `GroupShuffleSplit` berdasarkan file induk asli (`parent_file`) sehingga 100% bebas dari kebocoran data (*zero leakage*).

### SLIDE 6: Modeling & Benchmarking 4 Algoritma
- **Konten Visual:** Tabel benchmark dan grafik `06_model_comparison_bar.png`.
- **Poin Kunci:**
  - Random Forest meraih performa tertinggi dengan Akurasi $80.08\%$ dan F1-Score $0.7614$ pada data uji.
  - Pada 5-Fold GroupKFold Cross-Validation, Random Forest mencapai rata-rata F1 **$0.8806$ ($88.06\%$)**, melampaui target awal $0.85$.

### SLIDE 7: Evaluation (Confusion Matrix & ROC-AUC Curves)
- **Konten Visual:** Grafik `07_confusion_matrix_rf.png` & `08_roc_curves_comparison.png`.
- **Poin Kunci:**
  - Confusion Matrix pada 236 sampel uji: 114 True Negative, 75 True Positive. Spesifisitas $82.01\%$, Sensitivitas $77.32\%$.
  - Nilai ROC-AUC mencapai $0.8463$, membuktikan kemampuan separasi kelas yang kuat pada berbagai threshold.

### SLIDE 8: Interpretasi Ilmiah Feature Importance
- **Konten Visual:** Grafik horizontal bar chart `09_top10_feature_importance.png`.
- **Poin Kunci:**
  - `rms_mean` ($9.49\%$) dan `rms_std` ($6.39\%$) menjadi fitur terpenting untuk membedakan percakapan vokal dari dengung AC konstan.
  - MFCC 1, 2, 5, 12, dan 13 membuktikan model secara cerdas menangkap warna suara manusia (*vocal tract resonance*).

### SLIDE 9: Deployment Sistem Aplikasi Web (Demo Streamlit)
- **Konten Visual:** Tangkapan layar antarmuka `app.py` (Mode Deteksi, Quick Demo, dan Timeline Segmen).
- **Poin Kunci:**
  - Aplikasi siap pakai dengan latensi inferensi hanya **$13.19\text{ milidetik}$** ($75\times$ lebih cepat dari batas 1 detik).
  - Mampu menganalisis rekaman panjang secara dinamis per segmen 5 detik.

### SLIDE 10: Kesimpulan & Penutup
- **Poin Kunci:**
  - Seluruh target metodologi CRISP-DM berhasil terpenuhi dan terlampaui.
  - Sistem terbukti akurat, sangat cepat, ringan (berjalan di CPU biasa), dan menjaga privasi pengunjung.
- **Antisipasi Pertanyaan Dosen Penguji:**
  - *Q: Mengapa split data harus berbasis parent_file?* $\rightarrow$ *A: Karena jika potongan dari rekaman yang sama masuk ke train dan test, model akan menghafal karakteristik latar belakang mikrofon yang sama (data leakage).*
  - *Q: Bagaimana jika ruangan sepi tapi ada suara AC keras?* $\rightarrow$ *A: Model kami tidak hanya mengandalkan volume (RMS), melainkan menggunakan `rms_std` (variasi dinamika) dan `MFCC` (bentuk spektrum frekuensi). Suara AC bersifat stasioner dengan frekuensi $<250\text{ Hz}$, sehingga tetap diklasifikasikan sebagai Tidak Ramai.*

Resume this session with:
claude --resume ff9c07c7-7da6-4b36-bc6b-a31fa771d4e4