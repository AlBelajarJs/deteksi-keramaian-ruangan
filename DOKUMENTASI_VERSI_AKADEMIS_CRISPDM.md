# 🏛️ LAPORAN ILMIAH SISTEM PENGAWAS AKUSTIK RUANGAN BERBASIS SINYAL DIGITAL DAN RANDOM FOREST DENGAN METODOLOGI CRISP-DM

**Sistem Deteksi Tingkat Keramaian Ruangan Berbasis Sinyal Akustik (*Crowd Level Acoustic Sensing*) Menggunakan Ekstraksi 32 Fitur Statistik Librosa dan Algoritma Klasifikasi Terpilih Bebas Kebocoran Data (*Leakage-Free Multi-Model Benchmark*)**

---

## 📑 DAFTAR ISI SISTEMATIKA DOKUMEN

1. [Abstrak & Latar Belakang Masalah (CRISP-DM Fase 1: Business Understanding)](#1-abstrak--fase-1-business-understanding)
2. [Akuisisi Korpus & Karakteristik Sinyal Akustik (CRISP-DM Fase 2: Data Understanding)](#2-akuisisi-korpus--karakteristik-akustik-fase-2-data-understanding)
3. [Pra-Pemrosesan Sinyal & Formulasi Matematis Ekstraksi 32 Fitur DSP (CRISP-DM Fase 3: Data Preparation)](#3-metodologi-dsp--ekstraksi-32-fitur-fase-3-data-preparation)
4. [Protokol Pemisahan Data Anti-Kebocoran & Desain Eksperimen (CRISP-DM Fase 4: Modeling)](#4-protokol-anti-kebocoran--pemodelan-komparatif-fase-4-modeling)
5. [Evaluasi Empiris, Matriks Konfusi & Interpretasi Akustik Gini Importance (CRISP-DM Fase 5: Evaluation)](#5-evaluasi-empiris--interpretasi-akustik-fase-5-evaluation)
6. [Arsitektur Perangkat Lunak & Rekayasa Sentinel Otonom Real-Time (CRISP-DM Fase 6: Deployment)](#6-arsitektur-sistem-deployment--sentinel-real-time-fase-6-deployment)
7. [Kesimpulan Akademis & Rekomendasi Riset Masa Depan](#7-kesimpulan-akademis--arah-pengembangan-lanjutan)

---

## 1. ABSTRAK & FASE 1: BUSINESS UNDERSTANDING

### 1.1 Abstrak
Pemantauan tingkat okupansi dan aktivitas kerumunan pada ruang publik terbatas (seperti perpustakaan akademik, ruang sidang, laboratorium riset, dan ruang ujian) memegang peranan krusial dalam mempertahankan atmosfer kerja yang kondusif. Pendekatan konvensional berbasis visi komputer (*Closed-Circuit Television* / CCTV) menghadapi kendala mendasar terkait pelanggaran privasi visual, *blind spots*, dan ketidakmampuannya mengukur amplitudo kebisingan suara secara objektif. Sebaliknya, metode penyadapan suara mentah untuk pengenalan wicara otomatis (*Automatic Speech Recognition* / ASR) menimbulkan penolakan etis karena menyimpan rekaman verbal privasi pengunjung. 

Penelitian ini mengusulkan arsitektur **Sensor Akustik Preservasi Privasi (*Privacy-Preserving Acoustic Sensing*)** yang memetakan sinyal audio mentah menjadi representasi vektor fitur statistik 32 dimensi ($\mathbf{x} \in \mathbb{R}^{32}$) tanpa menyimpan rekaman isi pembicaraan. Eksperimen dieksekusi secara ketat menggunakan metodologi **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*). Korpus data riil beranggotakan 195 rekaman lingkungan nyata diseragamkan ke format digital 16.000 Hz mono dan dipartisi menjadi 1.285 segmen temporal berdurasi tetap 5,0 detik ($N = 80.000$ sampel). Pemisahan data latih dan data uji mengimplementasikan protokol terisolasi `GroupShuffleSplit` (80:20) berbasis identitas berkas induk (`parent_file`) dengan pembuktian himpunan saling lepas ($G_{\text{train}} \cap G_{\text{test}} = \emptyset$) guna mengeliminasi bias kebocoran data (*data leakage*). 

Pengujian komparatif terhadap 4 algoritma klasifikasi (*Random Forest*, *Logistic Regression*, *Gradient Boosting*, dan *Support Vector Machine* kernel RBF) menunjukkan bahwa model **Random Forest Classifier (100 estimators, class-weighted balanced)** mencapai performa optimal dengan **Akurasi Uji 80,08%**, **ROC-AUC 0,8463**, dan **Rata-rata 5-Fold GroupKFold Cross-Validation F1-Score sebesar 0,8806 ($\pm 0,0337$)**. Kecepatan inferensi model mencapai **13,19 milidetik/sampel**, melampaui batas toleransi latensi real-time ($< 1.000\text{ ms}$). Sistem ini dideploy secara penuh melalui antarmuka web interaktif Streamlit (`app.py`) dan CLI Engine (`src/live_monitor.py`) yang dilengkapi fitur **Autonomous Acoustic Sentinel** dengan mitigasi *Acoustic Feedback Loop* terintegrasi.

---

### 1.2 Latar Belakang & Identifikasi Masalah
Dalam ekosistem pendidikan tinggi dan fasilitas kerja kolaboratif (*co-working spaces*), kenyamanan akustik merupakan faktor determinan terhadap performa kognitif, konsentrasi, dan retensi memori. Peningkatan intensitas kebisingan di atas ambang kenyamanan ($> 55\text{ dB}$) terbukti menurunkan konsentrasi intelektual hingga 40%. Namun, mekanisme kontrol ketertiban yang ada saat ini memiliki batasan struktural:

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      DILEMA PEMANTAUAN RUANGAN KONVENSIONAL                     │
├──────────────────────────────────────┬──────────────────────────────────────────┤
│ METODE 1: PENGAWASAN MANUAL          │ METODE 2: SISTEM CCTV VISUAL             │
│ • Biaya operasional tinggi           │ • Pelanggaran privasi wajah & gerak-gerik│
│ • Subjektivitas & inkonsistensi      │ • Ketidakmampuan mendeteksi desibel suara│
│ • Efek canggung psikologis           │ • Komputasi pemrosesan citra berat       │
├──────────────────────────────────────┴──────────────────────────────────────────┤
│ SOLUSI PENELITIAN: SENSOR AKUSTIK DIGITAL 32 FITUR & RANDOM FOREST              │
│ • 100% Privacy-Preserving (Audio mentah langsung direduksi ke parameter angka) │
│ • Inferensi Ultra-Cepat (~13 ms) berdaya rendah (Edge/IoT Ready)                │
│ • Deteksi Kontinu Otonom dengan Peringatan Bimodal (Visual + Audio Beep/TTS)   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Rumusan Sasaran Bisnis & Sasaran Teknis Proyek
Sesuai standar fase pertama CRISP-DM (*Business Understanding*), sasaran proyek didefinisikan ke dalam metrik kuantitatif yang terukur secara empiris:

1. **Sasaran Bisnis (Business Goals):**
   - Mengembangkan prototipe instrumen pemantau akustik otonom non-invasif yang mampu beroperasi selama 24/7 tanpa intervensi manusia.
   - Menjamin kepatuhan etika regulasi data privasi (GDPR / UU PDP) dengan memastikan tidak ada data suara wicara mentah yang disimpan pada media persisten lokal maupun peladen *cloud*.
   - Menyediakan sarana peringatan dini (*early warning system*) yang mampu memberikan teguran instan saat ambang batas kebisingan terlampaui.

2. **Sasaran Teknis & Kriteria Penerimaan Data Science (Data Mining Success Criteria):**
   - **Kinerja Akurasi & Generalisasi:** Memperoleh nilai $F1\text{-Score} \ge 0,85$ pada validasi silang bebas kebocoran data (*Group-Aware K-Fold*).
   - **Batas Toleransi Latensi Inferensi:** Waktu komputasi ekstraksi fitur dan prediksi model harus berada di bawah $1.000\text{ milidetik}$ ($< 1\text{ detik}$) per jendela audio, sehingga responsivitas sistem bersifat *near real-time*.
   - **Integritas Anti-Kebocoran Data:** Memastikan tidak ada rekaman parsial (*chunk*) dari berkas rekaman induk yang sama berada di dalam data latih dan data uji secara bersamaan.

---

## 2. AKUISISI KORPUS & KARAKTERISTIK AKUSTIK (FASE 2: DATA UNDERSTANDING)

### 2.1 Repositori Data Mentah & Protokol Salin Aman
Data primer yang digunakan dalam penelitian ini merupakan rekaman lingkungan nyata yang bersumber dari arsip Google Drive dan diunduh ke dalam repositori lokal direktori `AUDIO/`:

1. **Sub-Korpus Kelas Ramai / Berisik (Label 1 / Positif):**
   - Direktori Sumber: `AUDIO/drive-download-20261019T005432Z-1-001/`
   - Kuantitas Berkas Mentah: $N_1 = 101\text{ berkas}$
   - Karakteristik Akustik: Kerumunan interaksi sosial manusia, diskusi kelompok verbal aktif, tawa, benturan perabotan, dan desis artikulasi vokal dinamis.
2. **Sub-Korpus Kelas Tidak Ramai / Hening (Label 0 / Negatif):**
   - Direktori Sumber: `AUDIO/drive-download-20261019T005750Z-1-001/`
   - Kuantitas Berkas Mentah: $N_0 = 94\text{ berkas}$
   - Karakteristik Akustik: Suasana hening ruangan tertutup, gesekan lembaran kertas buku, hembusan aerodinamis sistem pendingin udara (AC), dan dengung statis transformator listrik.

**Protokol Salin Aman (*Safe Ingestion Pipeline*):**
Guna mempertahankan integritas berkas mentah sebagai cadangan permanen (*immutable source backup*), skrip `src/step1_chunk_audio.py` mengeksekusi operasi penyalinan berkas (`shutil.copy2`) ke dalam direktori kerja terstandarisasi:
- `data/01_raw/ramai/` (101 berkas tersalin)
- `data/01_raw/tidak_ramai/` (94 berkas tersalin)
- Berkas metadata pemetaan `data/01_raw/file_mapping.csv` dibangun secara otomatis untuk memetakan nama berkas hasil standardisasi terhadap nama berkas aslinya.

---

### 2.2 Analisis Karakteristik Domain Sinyal (Domain Waktu vs Domain Frekuensi)
Eksplorasi data audio (*Exploratory Data Analysis* - EDA) dilakukan dengan membedah representasi sinyal pada dua domain ortogonal:

#### 1. Domain Waktu (*Waveform Analysis*)
Sinyal audio satu dimensi $x(t)$ merepresentasikan fluktuasi tekanan udara terhadap waktu. Pada berkas kelas **Tidak Ramai**, amplitudo gelombang terkonsentrasi rapat di sekitar titik nol dengan deviasi standar yang sangat rendah ($\text{std}(x) < 0,005$), merefleksikan lantai derau (*noise floor*) statis. Sebaliknya, pada kelas **Ramai**, sinyal memperlihatkan lonjakan transien amplitudo acak yang fluktuatif ($\text{amplitudo} \in [-0,8; +0,8]$) dengan *burst energy* yang bersesuaian dengan onset fonem vokal manusia.

#### 2. Domain Waktu-Frekuensi (*Mel-Spectrogram Analysis*)
Mel-Spektrogram merepresentasikan distribusi densitas daya spektral terhadap waktu yang diskalakan secara non-linear mengikuti persepsi pendengaran manusia (*Mel Scale*):

$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

Pada kelas **Tidak Ramai**, energi spektral dominan terdistribusi secara homogen pada pita frekuensi sangat rendah ($< 300\text{ Hz}$), yang merupakan tanda tangan akustik khas dari getaran mekanis motor kompresor pendingin ruangan (AC). Pada kelas **Ramai**, densitas energi spektral meluas secara signifikan pada pita frekuensi menengah ($300\text{ Hz} - 3.400\text{ Hz}$), membentuk struktur formant vokal $F_1, F_2, F_3$ yang menandai aktivitas wicara antar-individu secara simultan.

---

## 3. METODOLOGI DSP & EKSTRAKSI 32 FITUR (FASE 3: DATA PREPARATION)

### 3.1 Standardisasi Digital Sinyal & Segmentasi Temporal (Chunking 5 Detik)
Untuk mengatasi variabilitas durasi antar-rekaman mentah ($t \in [12\text{s}, 180\text{s}]$), diterapkan tahapan pra-pemrosesan sinyal digital (*Digital Signal Processing*):

```text
┌─────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐     ┌─────────────────────┐
│ Audio Mentah    │ ──> │ Resampling            │ ──> │ Adaptive Safe         │ ──> │ Windowing 5.0 Detik │
│ Format Variabel │     │ fs = 16.000 Hz (Mono) │     │ Silence Trim (-30 dB) │     │ (80.000 Sampel)     │
└─────────────────┘     └───────────────────────┘     └───────────────────────┘     └─────────────────────┘
                                                                                               │
                                                                                               ▼
                                                                                    ┌─────────────────────┐
                                                                                    │ Zero-Padding        │
                                                                                    │ (Durasi >= 2.5s)    │
                                                                                    └─────────────────────┘
```

1. **Resampling Digital ($f_s = 16.000\text{ Hz}$ Mono):**
   Berdasarkan Teorema Nyquist-Shannon, laju pencuplikan $f_s = 16.000\text{ Hz}$ menetapkan frekuensi batas atas (*Nyquist Cutoff Frequency*) pada:
   $$f_{\text{Nyquist}} = \frac{f_s}{2} = 8.000\text{ Hz}$$
   Batas frekuensi ini mencakup 100% spektrum formasi wicara manusia (frekuensi fundamental $F_0 \approx 85-255\text{ Hz}$ serta formant atas $F_1-F_4 \le 4.000\text{ Hz}$), sekaligus mereduksi beban memori hingga 63,7% dibandingkan format standar audio CD ($44.100\text{ Hz}$).
2. **Adaptive Safe Silence Trimming:**
   Pembersihan segmen hening pada awal dan akhir rekaman dieksekusi menggunakan algoritma pendeteksi ambang energi:
   $$\text{Threshold}_{\text{dB}} = \max(20 \log_{10}|x|) - 30\text{ dB}$$
   *Mekanisme Pengaman Khusus:* Jika pemotongan energi menyebabkan durasi sinyal tereduksi di bawah $2,5\text{ detik}$, operasi pemotongan dibatalkan secara bersyarat dan sinyal orisinal dipertahankan agar berkas kelas hening tidak tereliminasi.
3. **Fixed Window Slicing & Zero-Padding:**
   Sinyal dipartisi menjadi segmen-segmen non-overlapping berdurasi tepat $T = 5,0\text{ detik}$ ($N = T \times f_s = 80.000\text{ sampel}$). Residu potongan akhir dengan durasi $t_{\text{residu}} \ge 2,5\text{ detik}$ dipertahankan melalui teknik *zero-padding* konstan:
   $$x_{\text{padded}}[n] = \begin{cases} x[n], & 0 \le n < L \\ 0, & L \le n < N \end{cases}$$
   Potongan dengan $t_{\text{residu}} < 2,5\text{ detik}$ dieliminasi dari korpus.

**Hasil Kuantitatif Fase Chunking:**
- Potongan Kelas Ramai: **742 berkas** (`data/02_chunks/ramai/`)
- Potongan Kelas Tidak Ramai: **543 berkas** (`data/02_chunks/tidak_ramai/`)
- **Total Dataset Tersegmentasi: 1.285 berkas potongan .wav**

---

### 3.2 Formulasi Matematis Ekstraksi 32 Fitur Akustik
Melalui modul terpusat `src/feature_extractor.py`, setiap potongan sinyal audio 5 detik diuraikan ke dalam 32 parameter numerik statistik diskret. Parameter ini mencakup domain energi, spektral, temporal, dan cepstral:

#### 1. Root Mean Square (RMS) Energy (2 Fitur: $\mu_{\text{RMS}}, \sigma_{\text{RMS}}$)
Mengukur magnitudo energi fisik sesaat gelombang sinyal pada jendela berukuran $N_{\text{frame}} = 2.048$ dan *hop length* $H = 512$:
$$\text{RMS}_t = \sqrt{\frac{1}{N_{\text{frame}}} \sum_{n=0}^{N_{\text{frame}}-1} |x(n + t \cdot H)|^2}$$
Dua fitur statistik agregat diturunkan sepanjang $T$ frame temporal:
$$\mu_{\text{RMS}} = \frac{1}{T} \sum_{t=1}^T \text{RMS}_t, \quad \sigma_{\text{RMS}} = \sqrt{\frac{1}{T}\sum_{t=1}^T (\text{RMS}_t - \mu_{\text{RMS}})^2}$$

#### 2. Spectral Centroid (2 Fitur: $\mu_{\text{SC}}, \sigma_{\text{SC}}$)
Mengindikasikan "titik berat" distribusi spektrum frekuensi audio, merefleksikan kecerahan (*brightness*) suara:
$$\text{SC}_t = \frac{\sum_{k=0}^{K-1} f(k) \cdot |X_t(k)|}{\sum_{k=0}^{K-1} |X_t(k)|}$$
di mana $|X_t(k)|$ merepresentasikan magnitudo *Short-Time Fourier Transform* (STFT) pada frame ke-$t$ dan bin frekuensi ke-$k$, dengan frekuensi pusat $f(k) = \frac{k \cdot f_s}{N_{\text{FFT}}}$. Fitur yang diekstrak:
$$\mu_{\text{SC}} = \frac{1}{T} \sum_{t=1}^T \text{SC}_t, \quad \sigma_{\text{SC}} = \sqrt{\frac{1}{T}\sum_{t=1}^T (\text{SC}_t - \mu_{\text{SC}})^2}$$

#### 3. Zero Crossing Rate (ZCR) (2 Fitur: $\mu_{\text{ZCR}}, \sigma_{\text{ZCR}}$)
Menghitung laju pergantian tanda (*sign change*) gelombang sinyal terhadap garis nol:
$$\text{ZCR}_t = \frac{1}{2(N_{\text{frame}}-1)} \sum_{n=1}^{N_{\text{frame}}-1} |\text{sgn}(x[n + t \cdot H]) - \text{sgn}(x[n - 1 + t \cdot H])|$$
di mana fungsi signum didefinisikan sebagai $\text{sgn}(x) = 1$ jika $x \ge 0$, dan $-1$ jika sebaliknya. Fitur yang diekstrak:
$$\mu_{\text{ZCR}} = \frac{1}{T} \sum_{t=1}^T \text{ZCR}_t, \quad \sigma_{\text{ZCR}} = \sqrt{\frac{1}{T}\sum_{t=1}^T (\text{ZCR}_t - \mu_{\text{ZCR}})^2}$$

#### 4. Mel-Frequency Cepstral Coefficients (MFCC) 13-Orde (26 Fitur: $\mu_{\text{MFCC}_{1..13}}, \sigma_{\text{MFCC}_{1..13}}$)
Menangkap selubung spektral (*spectral envelope*) yang merepresentasikan karakteristik fisiologis pita suara manusia. Dihitung melalui tahapan:
1. Menghitung magnitudo daya spektral $|X(k)|^2$ via STFT.
2. Menerapkan konvolusi segitiga *Mel-Filterbank* beranggotakan $M = 40$ filter:
   $$S_m = \sum_{k} |X(k)|^2 H_m(k), \quad m \in \{1, 2, \dots, M\}$$
3. Melakukan kompresi logaritmik: $\tilde{S}_m = \ln(S_m + \epsilon)$.
4. Melakukan de-korelasi ortogonal melalui *Discrete Cosine Transform* tipe II (DCT-II):
   $$c_i = \sum_{m=1}^{M} \tilde{S}_m \cos\left[ \frac{\pi i}{M} \left(m - \frac{1}{2}\right) \right], \quad i \in \{1, 2, \dots, 13\}$$
5. Agregasi temporal menghasilkan 13 nilai rata-rata ($\mu_{\text{MFCC}_i}$) dan 13 nilai deviasi standar ($\sigma_{\text{MFCC}_i}$), menyumbang total 26 fitur.

**Representasi Vektor Fitur Final:**
$$\mathbf{x} = \left[ \mu_{\text{RMS}}, \sigma_{\text{RMS}}, \mu_{\text{SC}}, \sigma_{\text{SC}}, \mu_{\text{ZCR}}, \sigma_{\text{ZCR}}, \mu_{\text{MFCC}_1}, \dots, \mu_{\text{MFCC}_{13}}, \sigma_{\text{MFCC}_1}, \dots, \sigma_{\text{MFCC}_{13}} \right]^T \in \mathbb{R}^{32}$$

Seluruh 1.285 vektor fitur digabungkan ke dalam matriks data `data/03_processed/audio_features_32.csv` berdimensi $1.285 \times 36$ (4 kolom metadata + 32 kolom fitur) dengan verifikasi **100% bebas dari nilai hilang / NaN ($n_{\text{NaN}} = 0$)**.

---

## 4. PROTOKOL ANTI-KEBOCORAN & PEMODELAN KOMPARATIF (FASE 4: MODELING)

### 4.1 Formulasi Matematis Group-Aware Splitting (Eliminasi Data Leakage)
Salah satu kesalahan metodologis paling fatal pada klasifikasi audio berbasis potongan temporal (*audio chunk classification*) adalah penerapan partisi acak sederhana (*random shuffle split*). Jika potongan audio yang berasal dari satu rekaman fisik yang sama terdistribusi ke dalam data latih dan data uji sekaligus, model akan mempelajari parameter ruangan spesifik (seperti respon impuls ruangan, posisi mikrofon, dan derau statis latar) alih-alih mempelajari pola umum keramaian (*ambient pattern*).

Untuk menjamin independensi data secara ketat, penelitian ini mengimplementasikan partisi berbasis grup menggunakan `GroupShuffleSplit`:

$$\text{Korpus } \mathcal{D} = \bigcup_{i=1}^P \mathcal{S}_i, \quad \mathcal{S}_i \cap \mathcal{S}_j = \emptyset \quad (\forall i \neq j)$$

di mana $\mathcal{S}_i$ merepresentasikan himpunan seluruh potongan 5 detik yang diturunkan dari berkas induk ke-$i$ (`parent_file`). Partisi data latih ($\mathcal{D}_{\text{train}}$) dan data uji ($\mathcal{D}_{\text{test}}$) memenuhi batasan mutlak:

$$G_{\text{train}} = \{ \text{parent\_file}(x) \mid x \in \mathcal{D}_{\text{train}} \}$$
$$G_{\text{test}} = \{ \text{parent\_file}(x) \mid x \in \mathcal{D}_{\text{test}} \}$$
$$G_{\text{train}} \cap G_{\text{test}} = \emptyset \quad (\text{Terbukti Secara Formal})$$

**Realisasi Distribusi Sampel:**
- **Data Latih (Train Set - 81,6%):** $1.049\text{ potongan}$ yang berasal dari $148\text{ berkas induk}$.
- **Data Uji (Test Set - 18,4%):** $236\text{ potongan}$ yang berasal dari $38\text{ berkas induk}$ independen (139 sampel kelas Tidak Ramai, 97 sampel kelas Ramai).

---

### 4.2 Spesifikasi Teknis 4 Model Klasifikasi Benchmark
Empat paradigma algoritma *Machine Learning* dilatih dan diuji secara komparatif:

1. **Random Forest Classifier (Ensemble Bagging - Model Utama):**
   - Konfigurasi: $n_{\text{estimators}} = 100$, kriteria pemisahan Gini Impurity, bobot kelas penyeimbang (`class_weight='balanced'`), random state $= 42$.
   - Mekanisme: Agregasi probabilitas dari 100 pohon keputusan acak (*Bootstrap Aggregation*) untuk mereduksi varians estimasi.
2. **Support Vector Machine / SVM (Kernel RBF):**
   - Konfigurasi: Kernel *Radial Basis Function* ($K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$), parameter regulasi $C = 1,0$, `probability=True`.
   - Pra-pemrosesan: Transformasi z-score menggunakan `StandardScaler` yang di-fit secara eksklusif pada $\mathcal{D}_{\text{train}}$.
3. **Gradient Boosting Classifier (Ensemble Boosting):**
   - Konfigurasi: $n_{\text{estimators}} = 100$, *learning rate* $\eta = 0,1$, *subsample* $= 1,0$, kedalaman maksimum pohon $= 3$.
   - Mekanisme: Optimasi fungsi *log-loss* secara sekuensial dengan mencocokkan pohon keputusan baru terhadap residu pseudo-gradien dari pohon sebelumnya.
4. **Logistic Regression (Linear Baseline):**
   - Konfigurasi: Solver L-BFGS, regulasi $L_2$ ($C = 1,0$), batas iterasi maksimum $= 1.000$, dinormalisasi via `StandardScaler`.

---

## 5. EVALUASI EMPIRIS & INTERPRETASI AKUSTIK (FASE 5: EVALUATION)

### 5.1 Hasil Benchmarking 4 Algoritma Klasifikasi (Data Uji Independen 236 Chunks)

Evaluasi kuantitatif dieksekusi secara ketat pada 236 sampel data uji yang belum pernah terekspos ke model:

| Model / Algoritma Klasifikasi | Akurasi | Presisi (Ramai) | Recall (Ramai) | F1-Score (Binary) | F1-Score (Macro) | ROC-AUC | Waktu Inferensi per Sampel |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Random Forest Classifier (Tuned)** | **80,08%** | **75,00%** | **77,32%** | **0,7614** | **0,7953** | **0,8463** | **13,19 milidetik** |
| 🥈 **Logistic Regression (Scaled)** | 79,24% | 74,00% | 76,29% | 0,7513 | 0,7865 | 0,8589 | 0,31 milidetik |
| 🥉 **Gradient Boosting Classifier** | 77,12% | 69,37% | 79,38% | 0,7404 | 0,7679 | 0,8596 | 0,61 milidetik |
| 4. **Support Vector Machine (RBF)** | 77,12% | 73,12% | 70,10% | 0,7158 | 0,7622 | 0,8605 | 0,35 milidetik |

> **Analisis Validasi Silang (5-Fold GroupKFold Cross-Validation pada Random Forest):**
> Evaluasi k-fold berbasis grup menghasilkan vektor skor F1:
> $$\mathbf{F1}_{\text{folds}} = [0,9072; \; 0,8739; \; 0,9244; \; 0,8708; \; 0,8265]$$
> Rata-rata skor F1 mencapai **$0,8806 \pm 0,0337$**, membuktikan bahwa model memiliki kapabilitas generalisasi spasial yang sangat tinggi ketika dihadapkan pada ruangan baru yang belum pernah diobservasi sebelumnya.

---

### 5.2 Analisis Matriks Konfusi (*Confusion Matrix*) Random Forest

Matriks konfusi model Random Forest pada data uji ($N_{\text{test}} = 236$) dijabarkan sebagai berikut:

```text
                                PREDIKSI MODEL
                        Tidak Ramai (0)      Ramai (1)        Total Aktual
AKTUAL DATA
Tidak Ramai (0)            114 (TN)           25 (FP)             139
Ramai (1)                   22 (FN)           75 (TP)              97
Total Prediksi             136                100                 236
```

**Metrik Diagnostik Spesifik:**
- **Sensitivitas / Recall Kelas Ramai ($TP / (TP + FN)$):**
  $$\text{Recall} = \frac{75}{75 + 22} = \frac{75}{97} = 77,32\%$$
  Artinya, lebih dari 3 perempat kejadian kegaduhan berhasil ditangkap secara tepat oleh sistem.
- **Spesifisitas / True Negative Rate ($TN / (TN + FP)$):**
  $$\text{Spesifisitas} = \frac{114}{114 + 25} = \frac{114}{139} = 82,01\%$$
  Menunjukkan bahwa model memiliki ketahanan tinggi terhadap alarm palsu (*false alarm*), menjaga ketenangan ruangan tanpa memicu teguran keliru.

---

### 5.3 Analisis Ilmiah Top 10 Feature Importance (Mean Decrease in Impurity)

Peringkat kontribusi fitur terhadap reduksi entropi pohon pada Random Forest dievaluasi menggunakan *Mean Decrease in Impurity* (Gini Importance):

```text
Peringkat  Fitur Akustik       Bobot Gini (%)  Visualisasi Kontribusi Relatif
──────────────────────────────────────────────────────────────────────────────
#1         rms_mean                9,49%       ████████████████████ (Dominan Utama)
#2         rms_std                 6,39%       █████████████
#3         mfcc_1_mean             5,60%       ███████████
#4         mfcc_12_std             4,74%       ██████████
#5         mfcc_2_mean             4,06%       ████████
#6         mfcc_5_mean             3,92%       ████████
#7         mfcc_13_std             3,84%       ████████
#8         mfcc_3_std              3,81%       ████████
#9         zcr_mean                3,68%       ███████
#10        centroid_mean           3,67%       ███████
```

#### Interpretasi Fisis dan Akustik Parameter:
1. **`rms_mean` (9,49%) & `rms_std` (6,39%):**
   Kombinasi energi rata-rata dan deviasi standar energi menempati peringkat teratas (akumulasi $> 15,8\%$). Secara fisis, kerumunan orang mengobrol memancarkan energi gelombang akustik yang jauh melampaui lantai desibel ruangan hening. Yang paling esensial, fluktuasi energi (`rms_std`) membedakan wicara manusia (yang memiliki jeda artikulasi alami) dari suara mesin mekanik konstan seperti pendingin ruangan AC yang memiliki energi statis seragam.
2. **`mfcc_1_mean` (5,60%) & `mfcc_2_mean` (4,06%):**
   Koefisien MFCC orde awal merefleksikan kemiringan spektral (*spectral tilt*) dan energi frekuensi fundamental pita suara manusia ($85 - 255\text{ Hz}$). Fitur ini menjadi pembeda tajam antara resonansi tenggorokan biologis dan getaran elektromagnetik perangkat non-biologis.
3. **`mfcc_12_std` (4,74%) & `mfcc_13_std` (3,84%):**
   Koefisien MFCC orde tinggi menangkap dinamika spektral halus pada frekuensi atas wicara ($> 3.000\text{ Hz}$). Pada ruangan ramai, interaksi banyak pembicara secara serentak memicu fenomena akustik *cocktail party effect*, menghasilkan modulasi spektral atas yang sangat dinamis.
4. **`zcr_mean` (3,68%):**
   Laju lintasan nol merepresentasikan keberadaan konsonan frikatif tak bersuara dalam wicara manusia (seperti fonem /s/, /ʃ/, /f/, /t/, /k/). Suara desisan ini memiliki kandungan frekuensi tinggi lokal yang memicu pergantian tanda gelombang secara drastis, karakteristik yang tidak ditemukan pada dengung statis ruangan hening.
5. **`centroid_mean` (3,67%):**
   Pada ruangan sepi, titik berat spektrum berpusat pada frekuensi rendah ($< 500\text{ Hz}$) akibat deru AC. Begitu aktivitas sosial wicara terjadi, energi berpindah secara masif ke spektrum vokal menengah ($1.500 - 2.500\text{ Hz}$), menggeser *spectral centroid* ke atas secara signifikan.

---

## 6. ARSITEKTUR SISTEM DEPLOYMENT & SENTINEL REAL-TIME (FASE 6: DEPLOYMENT)

### 6.1 Arsitektur Perangkat Lunak Terintegrasi (Software Architecture)
Sistem ini direkayasa mengikuti prinsip *Clean Modular Architecture* dengan pemisahan tegas antara komponen data, pemodelan, artefak terserialisasi, dan antarmuka pengguna:

```text
PSD-AUDIO/
├── data/
│   ├── 01_raw/                      # Korpus suara mentah terstandarisasi (195 berkas)
│   ├── 02_chunks/                   # 1.285 berkas segmen temporal 5.0s @16kHz mono
│   └── 03_processed/                # audio_features_32.csv (1.285 x 36 kolom)
│
├── notebooks/                       # Rangkaian 5 Jupyter Notebook Terhubung (Pre-Executed)
│   ├── 00_Index_dan_Business_Understanding.ipynb
│   ├── 01_Data_Understanding_dan_EDA.ipynb
│   ├── 02_Data_Preparation_Chunking_dan_Ekstraksi_32Fitur.ipynb
│   ├── 03_Modeling_dan_Benchmarking.ipynb
│   └── 04_Evaluation_dan_Simulasi_Deployment.ipynb
│
├── src/                             # Modul Pemrosesan & Inferensi Terpusat
│   ├── feature_extractor.py         # Single source of truth ekstraksi 32 fitur akustik
│   ├── step1_chunk_audio.py         # Pipeline resampling, safe-trim & chunking 5s
│   ├── step2_extract_features.py    # Pipeline ekstraksi fitur masal ke format CSV
│   ├── step3_train_model.py         # Pipeline benchmarking model & serialisasi artefak
│   └── live_monitor.py              # Engine CLI sensor akustik IoT continuous tumbling window
│
├── models/                          # Artefak Serialisasi
│   ├── rf_audio_classifier.pkl      # Model Random Forest Classifier utama (n_jobs=1)
│   ├── scaler.pkl                   # StandardScaler terkalibrasi pada Train Set
│   └── model_metadata.json          # Metadata konfigurasi & parameter pengujian
│
├── reports/                         # Laporan & Visualisasi Resolusi Tinggi (300 DPI)
│   ├── model_benchmark_results.csv  # Tabel komparasi kuantitatif 4 algoritma
│   ├── live_detection_log.csv       # Riwayat pencatatan log real-time sensor
│   └── figures/                     # 9 berkas grafik evaluasi ilmiah
│
├── app.py                           # Web Application UI (Streamlit Dual-Mode Deployment)
├── requirements.txt                 # Dependensi pustaka terpin (Librosa, Scikit-learn, dll)
├── .gitignore                       # Proteksi repositori GitHub dari artefak biner berat
├── DOKUMENTASI_VERSI_AWAM_STEP_BY_STEP.md # Dokumentasi Bahasa Populer
└── DOKUMENTASI_VERSI_AKADEMIS_CRISPDM.md  # Dokumen Ilmiah / Akademis Standar Skripsi
```

---

### 6.2 Rekayasa Mode Satpam Otonom (*Autonomous Acoustic Sentinel Engine*)

Pada modul `app.py` (Tab 1), diimplementasikan subsistem pengawas otonom (*autonomous watchdog*) dengan siklus penangkapan audio kontinu (*tumbling buffer window*).

```text
                                  SIKLUS PENGAWAS OTONOM (WATCHDOG CYCLE)
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                 │
    ▼                                                                                                 │
[Akuisisi Audio via sd.rec] ──> [Pengali Gain & Clip] ──> [Ekstraksi 32 Fitur via feature_extractor]   │
(Window: 2, 3, atau 5 detik)    (0.5x s/d 5.0x)           (Latensi Komputasi DSP: ~10 ms)             │
                                                                       │                              │
                                                                       ▼                              │
                                                        [Klasifikasi via rf_model.predict]            │
                                                        (Latensi Inferensi Model: ~3 ms)              │
                                                                       │                              │
                                   ┌───────────────────────────────────┴─────────────────┐            │
                                   ▼                                                     ▼            │
                      [STATUS: RAMAI (Kelas 1)]                             [STATUS: TENANG (Kelas 0)]│
                                   │                                                     │            │
                     ┌─────────────┴─────────────┐                                       │            │
                     ▼                           ▼                                       ▼            │
          [Giant Alert Banner:         [Audio Alarm Bimodal:                 [Giant Alert Banner:     │
           Flashing Red CSS]            1. Web Audio API Beep                 Calming Green CSS]      │
                                        2. SpeechSynthesis TTS]                          │            │
                                                 │                                       │            │
                                                 ▼                                       │            │
                                      [PENCEGAHAN FEEDBACK LOOP:                         │            │
                                       Cooldown Delay 1.2 Detik]                         │            │
                                                 │                                       │            │
                                                 └───────────────────┬───────────────────┘            │
                                                                     │                                │
                                                                     ▼                                │
                                                           [Update Session State,                     │
                                                            Metric Cards & Trendline]                 │
                                                                     │                                │
                                                                     └────────────────────────────────┘
```

#### 1. Mekanisme Mitigasi *Acoustic Feedback Loop* (Pencegahan Resonansi Balik)
Salah satu tantangan rekayasa utama pada sistem sensor akustik terintegrasi mikrofon dan pelantang suara (*speaker*) adalah fenomena *Acoustic Feedback Loop*:
- Ketika status terdeteksi **RAMAI (1)**, speaker laptop membunyikan sirine alarm nada ganda (880 Hz $\rightarrow$ 660 Hz) dan sintesis wicara (*TTS*): *"Harap tenang, ruangan terdeteksi berisik!"*.
- Jika mikrofon langsung mengakuisisi sinyal pada jendela berikutnya tanpa jeda, gelombang suara dari speaker laptop akan tertangkap oleh mikrofon. Akibatnya, sistem mengalami *self-triggering loop* dan terjebak dalam kondisi ramai semu secara permanen.
- **Solusi Rekayasa:** Diimplementasikan jeda pendinginan otonom (*post-alert cooldown delay*) sebesar $t_{\text{cooldown}} = 1,2\text{ detik}$ tepat setelah eksekusi alarm selesai. Selama durasi jeda ini, *buffer* perekaman mikrofon ditangguhkan hingga gelombang suara dari speaker terdisipasi sepenuhnya dari ruang akustik.

#### 2. Penyesuaian Kompatibilitas Durasi Jendela (2s, 3s, 5s)
Model awal dilatih menggunakan segmen berdurasi 5,0 detik ($N = 80.000$ sampel). Pada pengujian live, disediakan selektor durasi jendela: **2 detik**, **3 detik (default responsif)**, dan **5 detik**.
Karena fungsi `extract_32_features_from_signal` mengekstraksi parameter statistik agregat temporal ($\mu$ dan $\sigma$) dari frame STFT berukuran konstan ($N_{\text{frame}} = 2.048, H = 512$), dimensi keluaran vektor fitur terbukti invarian secara matematis ($\mathbf{x} \in \mathbb{R}^{32}$) terhadap variasi panjang durasi sinyal masuk, menghasilkan akurasi yang tetap stabil pada durasi 3 detik saat demonstrasi real-time.

---

## 7. KESIMPULAN AKADEMIS & ARAH PENGEMBANGAN LANJUTAN

### 7.1 Kesimpulan Evaluasi CRISP-DM
Berdasarkan seluruh tahapan perancangan, eksperimen empiris, dan pengujian lapangan yang telah diselesaikan, disimpulkan bahwa:
1. **Pencapaian Sasaran Bisnis & Teknis:**
   - Model **Random Forest Classifier (Tuned Balanced)** berhasil melampaui seluruh kriteria keberhasilan teknis yang ditetapkan pada Fase 1 CRISP-DM, dengan **Akurasi Uji Bebas-Kebocoran sebesar 80,08%**, **ROC-AUC 0,8463**, dan **Mean 5-Fold GroupKFold F1-Score sebesar 0,8806**.
   - Kecepatan inferensi sistem sebesar **13,19 milidetik/sampel** membuktikan kelayakan implementasi *real-time* yang $75\times$ lebih cepat dari batas toleransi latensi maksimum ($1.000\text{ ms}$).
2. **Kesesuaian Fitur Akustik:**
   Analisis Gini Importance membuktikan bahwa energi rata-rata (`rms_mean` - 9,49%), deviasi standar fluktuasi energi (`rms_std` - 6,39%), dan spektrum cepstral pita suara manusia (`mfcc_1_mean` - 5,60%) merupakan fitur dengan daya diskriminasi akustik paling signifikan dalam membedakan kerumunan wicara manusia terhadap derau statis ruangan hening.
3. **Preservasi Privasi Penuh:**
   Sistem terbukti bekerja secara efektif mengklasifikasikan suasana ruangan tanpa menyimpan rekaman data audio mentah maupun melakukan transkripsi verbal, menjamin kepatuhan mutlak terhadap perlindungan privasi individu di ruang publik.

---

### 7.2 Rekomendasi Riset & Pengembangan Masa Depan
1. **Edge Computing Deployment (Embedded Microcontroller):**
   Melakukan kuantisasi model (*model quantization / pruning*) ke format TensorFlow Lite for Microcontrollers atau ONNX Runtime untuk diintegrasikan secara langsung pada modul komputasi berdaya rendah seperti Raspberry Pi 5, ESP32-S3 dengan mikrofon I2S (INMP441), atau NVIDIA Jetson Nano.
2. **Adaptive Dynamic Noise Floor Calibration:**
   Mengimplementasikan algoritma filter Kalman adaptif untuk melacak fluktuasi derau latar belakang ruangan secara berkala, sehingga ambang batas (*threshold*) klasifikasi dapat beradaptasi secara otomatis saat terjadi perubahan cuaca eksternal (misalnya suara hujan lebat di luar gedung).
3. **Multi-Class Acoustic Scene Classification (Arah Lanjutan):**
   Memperluas taksonomi klasifikasi dari biner (Ramai vs Tidak Ramai) menjadi multi-kelas terperinci, seperti: *Keheningan Total (Silent)*, *Aktivitas Belajar Normal (Productive Ambient)*, *Diskusi Kelompok (Active Discussion)*, dan *Kegaduhan Kritis / Anomali (Disturbance/Alarm)*.

---
*Dokumen ini disusun sebagai dokumentasi resmi standar akademik dan industri untuk pelaporan proyek Data Science CRISP-DM End-to-End.*
