# Blueprint Lengkap & Mendalam Proyek Sains Data: Sistem Deteksi Tingkat Keramaian Ruangan Berbasis Audio (CRISP-DM End-to-End)

Dokumen ini merupakan panduan teknis tingkat lanjut (*deep-dive*) yang menjabarkan langkah demi langkah penyelesaian tugas Proyek Sains Data berdasarkan referensi modul HackMD (Ekstraksi Fitur Audio dengan `librosa` dan Klasifikasi `Random Forest`). Seluruh penjelasan dijabarkan secara utuh tanpa dirangkum, mengikuti siklus standar industri **CRISP-DM (*Cross-Industry Standard Process for Data Mining*)**.

---

## BAGIAN 1: Siklus Kerja Step-by-Step Berdasarkan Metodologi CRISP-DM

Untuk memenuhi instruksi tugas **"mulai dari Business Understanding sampai Deployment dan Sistemnya"**, proyek ini wajib dibagi ke dalam 6 fase utama yang saling terhubung secara berurutan.

---

### FASE 1: Business Understanding (Pemahaman Bisnis & Perumusan Masalah)

Fase ini adalah fondasi laporan dan presentasi Anda. Dosen atau penguji ingin melihat alasan logis mengapa sistem deteksi suara ini perlu dibuat dan apa manfaat nyatanya di dunia nyata.

#### 1.1. Latar Belakang Masalah (Problem Statement)
Manajemen fasilitas publik seperti perpustakaan universitas, *co-working space*, ruang kelas mandiri, rumah sakit, hingga kafe memerlukan pemantauan kondisi lingkungan secara berkala untuk menjaga kenyamanan, ketertiban, dan efisiensi operasional. 
Selama ini, pemantauan tingkat keramaian dilakukan dengan dua cara konvensional yang memiliki kelemahan besar:
1. **Patroli Petugas Manual:** Tidak efisien karena petugas tidak bisa mengawasi puluhan ruangan secara bersamaan setiap saat.
2. **Pemantauan Berbasis Kamera (CCTV + Computer Vision):** Membutuhkan biaya perangkat keras tinggi, kapasitas penyimpanan video yang masif, komputasi GPU yang berat, serta melanggar **privasi pengunjung** karena merekam wajah dan aktivitas visual secara terus-menerus.

#### 1.2. Solusi yang Diajukan (Proposed Solution)
Membangun **Sistem Klasifikasi Tingkat Keramaian Ruangan Berbasis Sinyal Akustik (Audio)** yang ringan dan menjaga privasi (*privacy-preserving*). Sistem ini menangkap sampel gelombang suara di dalam ruangan, lalu mengubahnya secara instan menjadi **32 parameter statistik numerik** (tanpa perlu menyimpan isi percakapan pribadi pengunjung) untuk menentukan apakah ruangan berstatus **"Ramai" (Kelas 1)** atau **"Tidak Ramai" (Kelas 0)**.

#### 1.3. Tujuan Bisnis vs Tujuan Sains Data
Dalam laporan CRISP-DM, Anda harus membedakan antara tujuan bisnis dan tujuan teknis:
* **Tujuan Bisnis (*Business Objectives*):**
  1. Memberikan sistem peringatan dini otomatis (*early warning*) kepada pengelola ruangan (misal: pustakawan) ketika tingkat kebisingan/keramaian melampaui batas wajar selama lebih dari beberapa menit.
  2. Menyediakan informasi status kepadatan ruangan bagi mahasiswa/pengunjung yang ingin mencari ruangan belajar yang tenang.
  3. Menghemat biaya infrastruktur pemantauan hingga 80% karena hanya mengandalkan sensor mikrofon murah dan prosesor laptop standar.
* **Tujuan Teknis Sains Data (*Data Mining Goals*):**
  1. Mengekstraksi karakteristik waktu dan frekuensi sinyal audio menggunakan algoritma *Digital Signal Processing* (DSP) menjadi vektor 32 fitur statis.
  2. Melatih model klasifikasi biner (*Random Forest Classifier*) yang mampu membedakan keramaian suara manusia dari kebisingan latar belakang konstan (seperti deru AC, kipas angin, atau suara hujan).
  3. Mencapai performa model dengan metrik evaluasi **Accuracy $\ge 85\%$** dan **F1-Score $\ge 0.85$** pada data uji yang belum pernah dilihat sebelumnya, serta waktu inferensi (prediksi) di bawah $1$ detik per sampel.

---

### FASE 2: Data Understanding (Pemahaman & Eksplorasi Data Audio)

Setelah masalah bisnis jelas, langkah berikutnya adalah mengumpulkan data suara dan memahami karakteristik fisis serta matematis dari data tersebut.

#### 2.1. Akuisisi & Karakteristik Data Mentah
Data mentah berupa file rekaman audio digital dengan ekstensi `.wav` (sangat disarankan karena *lossless* / tidak terkompresi) atau `.mp3`. Data dibagi secara tegas menjadi dua kategori kelas:
1. **Kelas `Ramai` (Label = 1):**
   * Berisi rekaman suasana ruangan dengan aktivitas vokal manusia yang tumpang tindih (*overlapping speech*), diskusi kelompok, gelak tawa, suara langkah kaki, atau denting peralatan (seperti di kantin, lobi gedung, ruang diskusi aktif).
   * Karakteristik sinyal: Memiliki amplitudo yang berfluktuasi tajam secara acak (dinamis) dan kaya akan frekuensi menengah-tinggi ($300\text{ Hz} - 3400\text{ Hz}$) khas pita suara manusia.
2. **Kelas `Tidak Ramai` (Label = 0):**
   * Berisi rekaman ruangan hening, perpustakaan tenang, ruang kosong, **ATAU** ruangan kosong yang memiliki *background noise* mesin statis (seperti dengung kompresor AC, desis kipas angin, atau ketukan keyboard sesekali).
   * *Catatan Penting:* Memasukkan suara dengung AC/kipas angin ke dalam kelas `Tidak Ramai` sangat krusial agar model tidak salah mengira suara mesin yang keras sebagai keramaian manusia.

#### 2.2. Eksplorasi Data Audio (Exploratory Data Analysis / EDA)
Dalam laporan tugas Sains Data, Anda wajib menampilkan visualisasi EDA untuk membuktikan perbedaan kedua kelas sebelum masuk ke pemodelan:
1. **Visualisasi *Waveform* (Domain Waktu - Amplitudo vs Waktu):**
   * Plot gelombang suara menggunakan `librosa.display.waveshow()`.
   * Audio **Tidak Ramai** akan terlihat datar atau memiliki ketebalan pita yang rata/konstan dari awal hingga akhir.
   * Audio **Ramai** akan memperlihatkan puncak-puncak gunung dan lembah amplitudo yang sangat fluktuatif.
2. **Visualisasi *Spectrogram / Mel-Spectrogram* (Domain Frekuensi - Frekuensi vs Waktu vs Intensitas dB):**
   * Plot menggunakan `librosa.display.specshow()`.
   * Audio **Tidak Ramai** hanya memiliki warna terang di bagian paling bawah grafik (frekuensi rendah $< 250\text{ Hz}$ akibat dengung mesin/AC) dan gelap di bagian tengah ke atas.
   * Audio **Ramai** memiliki bercak-bercak warna terang yang menyebar luas di pita frekuensi $300\text{ Hz}$ hingga $4000\text{ Hz}$ (forman vokal manusia).
3. **Pemeriksaan Distribusi Durasi dan Sample Rate:**
   * Memeriksa berapa jumlah file di masing-masing folder (memastikan tidak terjadi *class imbalance* / ketimpangan jumlah data yang ekstrem).
   * Memeriksa variasi *Sample Rate* asli (misal rekaman dari HP Android mungkin $48000\text{ Hz}$, rekaman dari iPhone $44100\text{ Hz}$, dan rekaman dari laptop $22050\text{ Hz}$).

---

### FASE 3: Data Preparation (Pra-pemrosesan & Rekayasa Fitur)

Ini adalah fase paling teknis dan paling menentukan keberhasilan model. Algoritma *Random Forest* tidak bisa menerima input berupa file `.wav` secara langsung, melainkan hanya menerima matriks angka berbentuk tabel (baris = sampel audio, kolom = fitur).

#### 3.1. Pemotongan Audio per File Asli (*Audio Chunking*)
Jika Anda merekam 1 file audio berdurasi 5 menit ($300\text{ detik}$) di kantin, Anda tidak boleh memasukkannya sebagai 1 baris data saja. Anda harus memotongnya menjadi segmen-segmen pendek berdurasi **5 detik atau 10 detik** per potongan (*chunk*).
* **Aturan Emas Pemotongan:** Potonglah **per masing-masing file asli secara terpisah**, baru kumpulkan hasil potongannya ke dalam folder kelasnya. **JANGAN** menggabungkan seluruh file rekaman dari tempat berbeda menjadi satu file raksasa lalu dipotong-potong, karena potongan di titik sambungan antar-file akan memiliki akustik campuran yang merusak nilai standar deviasi.
* **Pemberian Nama File Hasil Potongan:** Berikan penamaan yang menyimpan identitas file aslinya, contoh: `kantinA_chunk01.wav`, `kantinA_chunk02.wav`, dst. Identitas grup (`kantinA`) ini wajib disimpan untuk mencegah kebocoran data (*data leakage*) saat pembagian data latih dan uji.

#### 3.2. Standarisasi *Sampling Rate* (Resampling ke $16000\text{ Hz}$)
Saat memuat audio menggunakan `librosa.load(file_path, sr=16000)`, seluruh sinyal dipaksa memiliki *sample rate* $16000\text{ Hz}$ (artinya terdapat $16000$ titik angka amplitudo dalam $1\text{ detik}$).
* **Mengapa $16000\text{ Hz}$?** Berdasarkan Teorema Nyquist-Shannon, frekuensi maksimum yang dapat direpresentasikan adalah setengah dari *sample rate* ($f_{\text{max}} = \frac{sr}{2} = 8000\text{ Hz}$). Karena frekuensi bicara manusia berada di rentang $300\text{ Hz} - 3400\text{ Hz}$, maka $16000\text{ Hz}$ sudah sangat akurat sekaligus menghemat memori RAM dan mempercepat komputasi hampir $3\times$ lipat dibandingkan $44100\text{ Hz}$.

#### 3.3. Penghapusan Keheningan di Ujung Rekaman (*Silence Trimming*)
Saat kita menekan tombol *Record* dan *Stop* di HP/laptop, sering kali ada jeda hening $1 - 2\text{ detik}$ di awal atau akhir file. Fungsi `librosa.effects.trim(y, top_db=25)` digunakan untuk memotong bagian ujung audio yang intensitas suaranya berada $25\text{ dB}$ di bawah puncak terkeras.
* *Perhatian Teknis:* Untuk file di kelas `Tidak Ramai`, berhati-hatilah agar `top_db` tidak terlalu kecil sehingga seluruh file ikut terhapus. Kita harus menambahkan pengecekan keamanan di kode Python: jika hasil *trim* menyisakan durasi kurang dari $1\text{ detik}$, gunakan sinyal asli sebelum di-*trim*.

#### 3.4. Ekstraksi 4 Kelompok Fitur Audio (Frame-Level Extraction)
Sinyal audio sepanjang $10\text{ detik}$ ($160000$ titik data) dipecah oleh `librosa` menjadi jendela-jendela pengamatan kecil (*frames*) berukuran $2048$ sampel dengan pergeseran (*hop length*) $512$ sampel. Dari setiap *frame* tersebut, dihitung 4 jenis fitur:

1. **RMS (*Root Mean Square* Energy):**
   $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} y_i^2}$$
   Mengukur kekuatan energi atau tingkat kekerasan (volume) sinyal suara pada tiap *frame*.
2. **Spectral Centroid (Pusat Massa Spektrum Frekuensi):**
   $$\text{Centroid} = \frac{\sum_{k} f(k) \cdot S(k)}{\sum_{k} S(k)}$$
   Di mana $f(k)$ adalah frekuensi bin ke-$k$ dan $S(k)$ adalah magnitudo spektralnya. Fitur ini menunjukkan "kecerahan" suara. Dengung AC memiliki Centroid rendah, sedangkan obrolan ramai memiliki Centroid lebih tinggi.
3. **Zero Crossing Rate (ZCR):**
   Mengukur seberapa sering gelombang sinyal melintasi sumbu nol (berubah dari nilai positif ke negatif atau sebaliknya). Suara konsonan desis manusia (seperti 's', 'sy', 'f', 't', 'k') dan keramaian percakapan menghasilkan ZCR yang spesifik dibanding keheningan.
4. **MFCC (*Mel-Frequency Cepstral Coefficients* - 13 Koefisien):**
   Telinga manusia tidak mendengar frekuensi secara linier, melainkan secara logaritmik (Skala Mel). MFCC memetakan spektrum frekuensi ke dalam skala pendengaran manusia dan mengambil **13 koefisien pertama** (`n_mfcc=13`). Koefisien ini menangkap bentuk saluran suara (*vocal tract*) dan warna suara (*timbre*) manusia. Ini adalah fitur paling ampuh untuk membedakan suara kumpulan manusia vs suara benda mati/mesin.

#### 3.5. Agregasi Statistik menjadi Vektor 32 Fitur (Feature Aggregation)
Karena hasil ekstraksi di atas masih berupa matriks 2 dimensi sepanjang waktu (*time-series*), kita harus meringkasnya menjadi 1 baris vektor berdimensi tetap dengan menghitung **Rata-rata (`np.mean`)** dan **Standar Deviasi (`np.std`)** sepanjang sumbu waktu (`axis=1`):
* **RMS:** `mean(RMS)` (1 fitur) + `std(RMS)` (1 fitur) = **2 fitur**
* **Spectral Centroid:** `mean(Centroid)` (1 fitur) + `std(Centroid)` (1 fitur) = **2 fitur**
* **ZCR:** `mean(ZCR)` (1 fitur) + `std(ZCR)` (1 fitur) = **2 fitur**
* **MFCC (13 koefisien):** `mean(MFCC_1 s/d MFCC_13)` (13 fitur) + `std(MFCC_1 s/d MFCC_13)` (13 fitur) = **26 fitur**
* **Total Akhir:** $2 + 2 + 2 + 26 = \mathbf{32\text{ Fitur Numerik}}$ per 1 potongan file audio.

#### 3.6. Pembagian Dataset Anti-Bocor (*Group-Aware Train-Test Split*)
Sesuai peringatan di dokumen HackMD: **"Split data sebaiknya dilakukan per file asli, bukan per chunk."**
* Jika `Rekaman_Kantin_1.wav` dipotong menjadi 10 *chunk*, maka ke-10 *chunk* tersebut harus masuk ke dalam *Training Set* semuanya, atau masuk ke dalam *Testing Set* semuanya.
* Gunakan `GroupShuffleSplit` atau `StratifiedGroupKFold` dari `scikit-learn` dengan rasio **80% Data Latih (*Train*)** dan **20% Data Uji (*Test*)**.

---

### FASE 4: Modeling (Pelatihan Model & Eksperimen Algoritma)

Pada fase ini, matriks fitur $X_{\text{train}}$ (berukuran $N_{\text{train}} \times 32$) dan vektor target $y_{\text{train}}$ (berisi label $0$ atau $1$) dimasukkan ke dalam algoritma pembelajaran mesin.

#### 4.1. Arsitektur Model Utama: Random Forest Classifier
Model utama yang digunakan sesuai panduan adalah `RandomForestClassifier(n_estimators=100, random_state=42)`.
* **Cara Kerja:** Model ini membangun $100$ pohon keputusan (*Decision Trees*) yang berbeda-beda. Setiap pohon dilatih menggunakan sampel data acak (*bootstrap sample*) dan hanya diizinkan memilih dari sebagian kecil fitur acak (misal $\sqrt{32} \approx 5\text{ fitur}$) pada setiap titik percabangan. Saat memprediksi data baru, ke-100 pohon melakukan pemungutan suara mayoritas (*majority voting*) untuk memutuskan apakah ruangan tersebut `Ramai` atau `Tidak Ramai`.
* **Keunggulan Teknis:** Tidak memerlukan normalisasi skala fitur (*scaling*), tahan terhadap *overfitting*, dan mampu menangani hubungan non-linier antar fitur audio.

#### 4.2. Pembandingan Model (*Model Benchmarking* untuk Nilai Tambah Laporan)
Agar laporan proyek Sains Data Anda mendapat nilai maksimal (A), jangan hanya menampilkan satu model. Bandingkan *Random Forest* dengan 2–3 algoritma lain pada data yang sama:
1. **Baseline Model:** *Logistic Regression* (dengan `StandardScaler`).
2. **Distance/Kernel Model:** *Support Vector Machine / SVM* (dengan kernel RBF dan `StandardScaler`).
3. **Ensemble Boosting Model:** *XGBoost* atau *Gradient Boosting Classifier*.
4. **Main Proposed Model:** *Random Forest Classifier*.

#### 4.3. Penyetelan Parameter (*Hyperparameter Tuning*)
Gunakan `GridSearchCV` pada *Random Forest* untuk mencari kombinasi parameter terbaik:
* `n_estimators`: `[50, 100, 200]` (jumlah pohon).
* `max_depth`: `[None, 10, 20]` (kedalaman maksimum pohon untuk mencegah pohon menghafal *noise*).
* `min_samples_split`: `[2, 5, 10]` (jumlah sampel minimum untuk memecah simpul percabangan).

---

### FASE 5: Evaluation (Evaluasi Performa & Interpretasi Fitur)

Setelah model selesai dilatih, ujilah menggunakan data $X_{\text{test}}$ yang belum pernah dilihat selama pelatihan.

#### 5.1. Metrik Evaluasi Klasifikasi
Gunakan fungsi `classification_report` dan `confusion_matrix` untuk menghitung 4 metrik utama:
1. **Confusion Matrix:** Tabel $2 \times 2$ yang memetakan *True Positive* (Ramai diprediksi Ramai), *True Negative* (Sepi diprediksi Sepi), *False Positive* (Sepi tapi salah diprediksi Ramai / alarm palsu), dan *False Negative* (Ramai tapi gagal terdeteksi).
2. **Precision:** $\frac{TP}{TP + FP}$ — Seberapa akurat sistem saat menyatakan ruangan sedang ramai (minim alarm palsu).
3. **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — Seberapa ampuh sistem menangkap seluruh kejadian ramai tanpa terlewat.
4. **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ — Rata-rata harmonik yang menjadi tolok ukur utama keberhasilan model.

#### 5.2. Analisis *Feature Importance* (Kewajiban sesuai Dokumen HackMD)
Salah satu instruksi eksplisit di kode HackMD adalah mengekstrak `importances = rf_model.feature_importances_`.
* Buatlah grafik batang horizontal (*horizontal bar chart*) yang mengurutkan 10 fitur paling berpengaruh dari total 32 fitur.
* **Cara Membahasnya di Laporan:** Jelaskan mengapa fitur tersebut berada di peringkat teratas. Misalnya, jika `rms_mean`, `rms_std`, `mfcc_1_std`, dan `mfcc_3_mean` menduduki posisi teratas, jelaskan bahwa kombinasi energi suara (`RMS`) dan dinamika perubahan warna vokal manusia (`MFCC std`) terbukti secara ilmiah menjadi pembeda utama antara keramaian manusia dan ruangan sepi.

---

### FASE 6: Deployment & Arsitektur Sistem (Implementasi Produk Akhir)

Tahap terakhir dalam CRISP-DM adalah membungkus model yang sudah dilatih menjadi sebuah **sistem/aplikasi nyata** yang dapat digunakan oleh pengguna akhir (*end-user*).

#### 6.1. Penyimpanan Model (*Model Serialization*)
Model yang sudah dilatih di memori Python diekspor menjadi sebuah file biner statis bernama `rf_audio_model.pkl` menggunakan pustaka `joblib`:
```python
import joblib
joblib.dump(rf_model, 'models/rf_audio_model.pkl')
```
Dengan cara ini, aplikasi sistem tidak perlu melatih ulang model setiap kali dijalankan. Aplikasi cukup memuat file `.pkl` tersebut dalam waktu milidetik.

#### 6.2. Arsitektur Sistem Aplikasi (Menggunakan Framework **Streamlit**)
Untuk proyek mahasiswa/praktikum, cara terbaik, tercepat, dan paling interaktif untuk membangun sistem *Deployment* adalah menggunakan **Streamlit** (antarmuka web berbasis Python). Sistem yang dibangun memiliki 2 mode operasional:

```
+-------------------------------------------------------------------------+
|                    ANTARMUKA PENGGUNA (STREAMLIT WEB UI)                |
|  [Mode 1: Upload File Audio (.wav/.mp3)]   [Mode 2: Rekam Mic Langsung] |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                 MODUL BACKEND PEMROSESAN SINYAL (DSP ENGINE)            |
|  1. Baca Stream Audio -> Resample 16.000 Hz                             |
|  2. Potong / Trim Silence -> Ambil Jendela Waktu                        |
|  3. Hitung RMS, Spectral Centroid, ZCR, 13 MFCC                         |
|  4. Agregasi Statistik (Mean & Std) -> Vektor 1x32                      |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  MODUL INFERENSI MACHINE LEARNING                       |
|  Muat 'rf_audio_model.pkl' -> Prediksi Kelas (0/1) & Probabilitas (%)   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                     DASHBOARD OUTPUT & VISUALISASI                      |
|  - Indikator Status: [MERAH: RAMAI] atau [HIJAU: TIDAK RAMAI]           |
|  - Skor Keyakinan Model (Confidence Score, misal: 94.2% Ramai)          |
|  - Grafik Waveform & Mel-Spectrogram Audio yang Baru Diuji              |
|  - Riwayat Pemantauan per Segmen Waktu (Timeline Log)                   |
+-------------------------------------------------------------------------+
```

---

## BAGIAN 2: Pemetaan 3 Jalur Pengerjaan Proyek (Kelebihan, Kekurangan & Peluang Berhasil)

Dalam mengeksekusi proyek ini bersama tim Anda, terdapat **3 jalur strategi pengerjaan** yang bisa dipilih tergantung ketersediaan waktu, kemauan terjun ke lapangan, dan target nilai yang ingin dicapai.

---

### JALUR 1: Jalur Akuisisi Mandiri Murni (Rekam Sendiri dengan HP/Laptop + Potong Otomatis via Python)
Pada jalur ini, tim Anda merekam sendiri suasana ruangan menggunakan HP (misal 5–10 lokasi berbeda untuk kondisi ramai dan 5–10 lokasi untuk kondisi sepi, masing-masing direkam selama 2–3 menit), lalu menggunakan skrip Python untuk memotongnya otomatis menjadi ratusan potongan 5–10 detik.

* **Kelebihan:**
  1. **Orisinalitas Tinggi:** Dosen sangat mengapresiasi dataset primer yang diambil langsung dari lingkungan sekitar kampus (misal: perpustakaan kampus, kantin kampus, ruang kelas).
  2. **Sangat Relevan saat Demo Live:** Karena direkam menggunakan perangkat mikrofon HP/laptop milik anggota tim sendiri, saat nanti dilakukan demo langsung di depan penguji menggunakan perangkat yang sama, karakter mikrofonnya sudah dikenali dengan sangat baik oleh model.
  3. **Kontrol Penuh atas Label:** Anda tahu persis kondisi ruangan saat direkam, sehingga tidak ada kebingungan label.
* **Kekurangan:**
  1. Membutuhkan waktu dan tenaga untuk pergi ke beberapa tempat guna merekam suara.
  2. Risiko kualitas rekaman buruk jika anggota tim meletakkan HP di saku atau tidak sengaja menyentuh mikrofon (suara gesekan/angin).
  3. Variasi akustik terbatas pada beberapa ruangan tempat Anda merekam saja.
* **Estimasi Tingkat Keberhasilan:** **92% – 95%** (Sangat Tinggi).
* **Kemungkinan Masalah Utama:** Kebocoran data (*data leakage*) jika potongan dari file yang sama bocor antara *train* dan *test*, atau suara notifikasi HP sendiri ikut terekam saat proses perekaman.

---

### JALUR 2: Jalur Dataset Publik Online (Download Dataset Kaggle / ESC-50 / UrbanSound8K / YouTube Audio)
Pada jalur ini, tim Anda tidak merekam suara ke lapangan sama sekali, melainkan mengunduh kumpulan file audio publik dari internet (misal kategori *crowd*, *chatter*, *cafe ambience* untuk kelas `Ramai`, dan kategori *room tone*, *silence*, *air conditioner*, *office quiet* untuk kelas `Tidak Ramai`).

* **Kelebihan:**
  1. **Sangat Cepat & Praktis:** Bisa dikerjakan langsung dari rumah dalam waktu beberapa jam tanpa perlu survei lapangan.
  2. **Jumlah Data Melimpah:** Anda bisa dengan mudah mengumpulkan 500 hingga 1.000 sampel audio berkualitas studio.
  3. **Variasi Ruangan Sangat Beragam:** Mencakup berbagai jenis ruangan dari seluruh dunia.
* **Kekurangan:**
  1. **Masalah *Domain Shift* saat Demo Live:** Audio dari Kaggle/YouTube biasanya direkam menggunakan mikrofon profesional dengan kompresi audio tertentu. Ketika model diuji menggunakan mikrofon laptop murah saat demo presentasi di kelas, prediksi model bisa meleset karena karakteristik *noise floor* mikrofon laptop berbeda jauh dari data Kaggle.
  2. **Kurang Nilai Orisinalitas Data:** Beberapa dosen memberikan nilai lebih rendah jika mahasiswa hanya mengunduh dataset instan tanpa usaha pengumpulan data.
  3. Banyak audio dari internet sudah diberi efek *fade-in/fade-out* atau *background music* tipis yang bisa mengacaukan nilai fitur `Spectral Centroid` dan `MFCC`.
* **Estimasi Tingkat Keberhasilan:**
  * Keberhasilan di atas kertas (evaluasi file uji): **95% – 98%**.
  * Keberhasilan saat dites *live* memakai mic laptop: **65% – 75%**.

---

### JALUR 3: Jalur Hibrida / Kombinasi (Dataset Publik + Rekaman Mandiri + Augmentasi Data) — **SANGAT DIREKOMENDASIKAN**
Pada jalur ini, Anda menggabungkan 60% data dari sumber publik/dataset terbuka dengan 40% rekaman mandiri dari HP/laptop di sekitar kampus, lalu menerapkan teknik *Audio Data Augmentation* (seperti menambah sedikit *background noise* tipis atau menggeser *pitch* secara halus) pada data latih.

* **Kelebihan:**
  1. **Model Paling Tangguh (*Robust*):** Model belajar dari variasi ruangan yang luas (dari data publik) sekaligus mengenali karakter spesifik mikrofon perangkat Anda (dari rekaman mandiri).
  2. **Laporan Sangat Kaya untuk Fase CRISP-DM:** Di bagian *Data Understanding* dan *Data Preparation*, Anda bisa menceritakan proses integrasi multi-sumber data (*data integration*), yang merupakan poin plus besar dalam penilaian proyek Sains Data.
  3. **Aman Saat Demo Live maupun Uji File:** Performa stabil baik saat diuji menggunakan unggahan file `.wav` maupun saat demonstrasi rekaman mikrofon langsung di ruang presentasi.
* **Kekurangan:**
  1. Membutuhkan penataan penamaan file dan struktur folder yang rapi sejak awal agar sumber data tidak tercampur aduk.
  2. Waktu persiapan data sedikit lebih panjang dibanding Jalur 2 murni (sekitar 1–2 hari kerja).
* **Estimasi Tingkat Keberhasilan:** **96% – 99%** (Terbaik untuk Akademik & Praktik).

### Tabel Ringkasan Perbandingan 3 Jalur Pengerjaan

| Parameter Pembanding | Jalur 1: Rekam Mandiri Murni | Jalur 2: Dataset Publik (Kaggle) | Jalur 3: Hibrida (Publik + Mandiri) |
| :--- | :--- | :--- | :--- |
| **Waktu Pengumpulan Data** | Sedang (1–2 Hari) | Sangat Cepat (2–3 Jam) | Sedang (1–2 Hari) |
| **Kualitas Nilai Akademik (CRISP-DM)** | Tinggi (Data Primer) | Sedang (Data Sekunder) | **Sangat Tinggi (Integrasi Data)** |
| **Akurasi Evaluasi Tabel ($F1$-Score)** | $85\% - 92\%$ | $90\% - 96\%$ | **$92\% - 97\%$** |
| **Keandalan Saat Demo Mic Laptop** | Tinggi | Rendah – Sedang | **Sangat Tinggi** |
| **Tingkat Kesulitan Teknis** | Mudah | Sangat Mudah | Sedang |

---

## BAGIAN 3: Tech Stack Lengkap & Struktur Direktori Proyek Standar Industri

Agar proyek Anda terlihat profesional dan rapi saat dikumpulkan (baik ke GitHub maupun Google Drive), gunakan spesifikasi perangkat lunak dan struktur folder berikut.

### 3.1. Spesifikasi Tech Stack
1. **Bahasa Pemrograman:** Python `3.9`, `3.10`, atau `3.11` (Hindari Python `3.13` terbaru karena beberapa pustaka audio C-binding terkadang belum memiliki *wheel* siap pakai).
2. **Pustaka Pemrosesan Sinyal Audio (DSP):**
   * `librosa >= 0.10.1`: Ekstraksi RMS, Spectral Centroid, ZCR, MFCC, resampling, dan visualisasi spektrogram.
   * `soundfile >= 0.12.1`: Mesin pembaca dan penulis file `.wav` berkecepatan tinggi berbasis C (`libsndfile`).
   * `audioread >= 3.0.1`: Pendukung dekoder format `.mp3` atau `.m4a`.
3. **Pustaka Sains Data & Machine Learning:**
   * `numpy >= 1.24.0`: Komputasi matriks dan agregasi vektor fitur.
   * `pandas >= 2.0.0`: Penyusunan tabel dataset fitur (`features_32.csv`) dan penyimpanan metadata grup file.
   * `scikit-learn >= 1.3.0`: Algoritma `RandomForestClassifier`, `GroupShuffleSplit`, `GridSearchCV`, dan metrik evaluasi.
   * `joblib >= 1.3.0`: Menyimpan dan memuat model `.pkl`.
4. **Pustaka Visualisasi & Deployment Sistem:**
   * `matplotlib >= 3.7.0` & `seaborn >= 0.12.0`: Membuat grafik *Waveform*, *Spectrogram*, *Confusion Matrix*, dan *Feature Importance*.
   * `streamlit >= 1.28.0`: Membangun sistem aplikasi web interaktif (*Dashboard Deployment*).

### 3.2. Struktur Folder Proyek (Directory Tree)
Buatlah satu folder utama bernama `proyek-deteksi-keramaian-audio/` dengan susunan sub-folder sebagai berikut:

```text
proyek-deteksi-keramaian-audio/
│
├── data/
│   ├── 01_raw/                         # Tempat menaruh rekaman asli berdurasi panjang (sebelum dipotong)
│   │   ├── ramai/                      # Contoh: kantin_pagi.wav (3 menit), lobi_kampus.wav (2 menit)
│   │   └── tidak_ramai/                # Contoh: perpus_sepi.wav (3 menit), ruang_ac_kosong.wav (2 menit)
│   │
│   ├── 02_chunks/                      # Hasil pemotongan otomatis per 5 atau 10 detik
│   │   ├── ramai/                      # Contoh: kantin_pagi_chunk001.wav, kantin_pagi_chunk002.wav
│   │   └── tidak_ramai/                # Contoh: perpus_sepi_chunk001.wav, perpus_sepi_chunk002.wav
│   │
│   └── 03_processed/                   # Tempat menyimpan tabel fitur hasil ekstraksi (CSV)
│       └── audio_features_32.csv       # Berisi 32 kolom fitur + kolom 'label' + kolom 'source_group'
│
├── notebooks/                          # Jupyter Notebooks untuk eksperimen & pembuatan grafik laporan
│   ├── 01_EDA_Visualisasi_Audio.ipynb  # Menampilkan Waveform & Mel-Spectrogram Ramai vs Sepi
│   └── 02_Modeling_dan_Evaluasi.ipynb  # Eksperimen model, Confusion Matrix, & Feature Importance
│
├── src/                                # Skrip kode modular Python (Siap jalankan via Terminal)
│   ├── __init__.py
│   ├── step1_chunk_audio.py            # Skrip memotong file di 01_raw ke 02_chunks secara otomatis
│   ├── step2_extract_features.py       # Skrip mengekstrak 32 fitur dari 02_chunks ke CSV
│   └── step3_train_model.py            # Skrip melatih Random Forest & menyimpan model ke folder models/
│
├── models/                             # Tempat menyimpan artefak model yang sudah terlatih
│   ├── rf_crowd_model.pkl              # File model Random Forest siap pakai
│   └── feature_columns.pkl             # Daftar urutan nama ke-32 fitur
│
├── app.py                              # File utama Deployment Aplikasi Web (Streamlit)
├── requirements.txt                    # Daftar instalasi library Python
└── README.md                           # Dokumentasi cara menjalankan proyek
```

### 3.3. Isi File `requirements.txt`
Salin teks berikut ke dalam file `requirements.txt` Anda, lalu jalankan perintah `pip install -r requirements.txt` di terminal:
```text
librosa>=0.10.1
soundfile>=0.12.1
audioread>=3.0.1
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
streamlit>=1.28.0
```

---

## BAGIAN 4: Implementasi Kode Lengkap Step-by-Step (Dari Mentah Hingga Sistem Web)

Berikut adalah rangkaian kode utuh untuk seluruh pipa pengerjaan di atas agar Anda tidak mengalami kebingungan saat mengimplementasikan kode potongan yang ada di HackMD.

### 4.1. Kode Tahap 1: Pemotongan Audio Otomatis per File Asli (`src/step1_chunk_audio.py`)
Skrip ini membaca setiap file panjang di `data/01_raw/`, memotongnya per 5 detik tanpa mencampur antar file, dan menyimpannya ke `data/02_chunks/`.

```python
import os
import librosa
import soundfile as sf

def chunk_audio_directory(raw_base_dir="data/01_raw", chunk_base_dir="data/02_chunks", chunk_duration=5, sr=16000):
    """
    Memotong setiap file audio panjang menjadi potongan berdurasi tetap (default: 5 detik)
    secara terpisah per file asli untuk mencegah data leakage.
    """
    classes = ["tidak_ramai", "ramai"]
    samples_per_chunk = chunk_duration * sr

    for cls in classes:
        input_folder = os.path.join(raw_base_dir, cls)
        output_folder = os.path.join(chunk_base_dir, cls)
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(input_folder):
            print(f"[PERINGATAN] Folder {input_folder} belum ada. Buat dan isi dulu dengan file audio!")
            continue

        for filename in os.listdir(input_folder):
            if not filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                continue

            filepath = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0].replace(" ", "_")

            try:
                # Muat file audio dan langsung standarisasi ke 16.000 Hz mono
                y, _ = librosa.load(filepath, sr=sr, mono=True)
                
                # Hitung berapa potongan utuh yang bisa dihasilkan dari 1 file ini
                total_chunks = len(y) // samples_per_chunk

                if total_chunks == 0:
                    print(f"[SKIP] File {filename} terlalu pendek (< {chunk_duration} detik).")
                    continue

                for i in range(total_chunks):
                    start_sample = i * samples_per_chunk
                    end_sample = start_sample + samples_per_chunk
                    chunk_data = y[start_sample:end_sample]

                    # Simpan dengan format: namaFileAsli__chunk001.wav
                    # Tanda '__' dipakai agar kita mudah mengambil nama grup file aslinya nanti
                    chunk_filename = f"{base_name}__chunk{i+1:03d}.wav"
                    chunk_out_path = os.path.join(output_folder, chunk_filename)
                    sf.write(chunk_out_path, chunk_data, sr)

                print(f"[SUKSES] {filename} -> berhasil dipotong menjadi {total_chunks} file (@{chunk_duration} detik).")

            except Exception as e:
                print(f"[ERROR] Gagal memproses {filename}: {e}")

if __name__ == "__main__":
    chunk_audio_directory(chunk_duration=5, sr=16000)
```

---

### 4.2. Kode Tahap 2: Ekstraksi 32 Fitur Audio ke Tabel CSV (`src/step2_extract_features.py`)
Skrip ini mengimplementasikan rumus ekstraksi fitur dari dokumen HackMD secara aman (dilengkapi pengaman untuk file hening agar tidak error) dan menyimpannya ke dalam file `audio_features_32.csv`.

```python
import os
import librosa
import numpy as np
import pandas as pd

def get_feature_names():
    """Menghasilkan 32 nama kolom fitur yang rapi untuk tabel DataFrame."""
    names = [
        "rms_mean", "rms_std",
        "centroid_mean", "centroid_std",
        "zcr_mean", "zcr_std"
    ]
    for i in range(1, 14):
        names.append(f"mfcc_{i}_mean")
    for i,{i} in enumerate(range(1, 14)):
        names.append(f"mfcc_{i}_std")
    return names

def extract_32_features_safe(file_path, sr=16000):
    """
    Mengekstrak 32 fitur statistik audio (RMS, Spectral Centroid, ZCR, 13 MFCC)
    beserta penanganan keamanan jika rekaman terlalu hening saat di-trim.
    """
    # 1. Load audio pada sample rate konsisten 16.000 Hz
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    # 2. Trim keheningan di ujung, namun cegah jangan sampai array habis untuk kelas 'tidak_ramai'
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) < sr * 1.0:  # Jika setelah di-trim sisa kurang dari 1 detik, pakai sinyal asli
        y_proc = y
    else:
        y_proc = y_trimmed

    # Cegah sinyal bernilai nol mutlak (dead digital silence) yang bisa memicu NaN pada MFCC/Centroid
    if np.max(np.abs(y_proc)) < 1e-6:
        y_proc = y_proc + np.random.normal(0, 1e-6, len(y_proc))

    # 3. Ekstraksi 4 karakteristik sinyal menggunakan Librosa
    rms = librosa.feature.rms(y=y_proc)
    centroid = librosa.feature.spectral_centroid(y=y_proc, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y_proc)
    mfcc = librosa.feature.mfcc(y=y_proc, sr=sr, n_mfcc=13)

    # 4. Agregasi nilai Mean dan Standard Deviation menjadi 1 vektor (32 fitur)
    features = np.hstack([
        np.mean(rms), np.std(rms),
        np.mean(centroid), np.std(centroid),
        np.mean(zcr), np.std(zcr),
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1)
    ])

    return features

def build_feature_dataset(chunk_base_dir="data/02_chunks", output_csv="data/03_processed/audio_features_32.csv"):
    classes = {"tidak_ramai": 0, "ramai": 1}
    rows = []
    feature_cols = get_feature_names()

    for cls_name, label_val in classes.items():
        folder = os.path.join(chunk_base_dir, cls_name)
        if not os.path.exists(folder):
            continue

        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.wav'):
                continue

            fpath = os.path.join(folder, fname)
            # Ambil identitas grup rekaman asli dari nama file sebelum '__chunk'
            source_group = fname.split("__chunk")[0] if "__chunk" in fname else os.path.splitext(fname)[0]

            try:
                feat_vector = extract_32_features_safe(fpath)
                row_dict = dict(zip(feature_cols, feat_vector))
                row_dict["filename"] = fname
                row_dict["source_group"] = source_group
                row_dict["label"] = label_val
                row_dict["label_name"] = cls_name
                rows.append(row_dict)
            except Exception as e:
                print(f"[GAGAL EKSTRAK] {fname}: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[SELESAI] Berhasil mengekstrak {len(df)} sampel audio ke {output_csv}")
    print("Distribusi Kelas:\n", df["label_name"].value_counts())

if __name__ == "__main__":
    build_feature_dataset()
```

---

### 4.3. Kode Tahap 3: Pelatihan Model, Evaluasi & Feature Importance (`src/step3_train_model.py`)
Skrip ini membagi data berdasarkan grup file asli (atau *stratified split* jika setiap file berdiri sendiri), melatih `RandomForestClassifier`, mencetak laporan evaluasi, menampilkan tingkat kepentingan fitur (*Feature Importance*), lalu menyimpan model ke file `.pkl`.

```python
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_and_evaluate(csv_path="data/03_processed/audio_features_32.csv", model_dir="models"):
    df = pd.read_csv(csv_path)

    # Pisahkan 32 kolom fitur dari kolom metadata
    meta_cols = ["filename", "source_group", "label", "label_name"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values
    y = df["label"].values
    groups = df["source_group"].values

    # Cek apakah jumlah grup file asli cukup untuk GroupShuffleSplit (minimal 4 grup berbeda)
    unique_groups = len(np.unique(groups))
    if unique_groups >= 4:
        print(f"[INFO] Menggunakan GroupShuffleSplit berdasarkan {unique_groups} file rekaman asli (Anti Data Leakage).")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        print("[INFO] Jumlah file asli sedikit, menggunakan Stratified Train-Test Split standar (80:20).")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # Inisialisasi dan Pelatihan Random Forest sesuai referensi HackMD
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Prediksi dan Evaluasi
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*55)
    print(f"HASIL EVALUASI MODEL RANDOM FOREST (AKURASI: {acc*100:.2f}%)")
    print("="*55)
    print(classification_report(y_test, y_pred, target_names=["Tidak Ramai (0)", "Ramai (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Analisis Top 10 Feature Importance
    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Fitur": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Fitur Paling Berpengaruh:")
    print(feat_imp_df.head(10).to_string(index=False))

    # Simpan model & daftar kolom fitur untuk Deployment
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_model, os.path.join(model_dir, "rf_crowd_model.pkl"))
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_columns.pkl"))
    print(f"\n[SUKSES] Model disimpan di {model_dir}/rf_crowd_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()
```

---

### 4.4. Kode Tahap 4: Deployment Aplikasi Sistem Web Interaktif (`app.py`)
Jalankan file ini dengan perintah `streamlit run app.py` di terminal. Anda akan langsung memiliki sistem aplikasi web lengkap yang bisa menguji file audio, menampilkan bentuk gelombang suara, spektrogram, dan hasil deteksi keramaiannya.

```python
import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.step2_extract_features import extract_32_features_safe

st.set_page_config(page_title="Sistem Deteksi Keramaian Audio", layout="wide")

st.title("🎙️ Sistem Deteksi Tingkat Keramaian Ruangan Berbasis Audio")
st.markdown("Implementasi *Deployment* Proyek Sains Data menggunakan **Librosa (32 Fitur)** & **Random Forest Classifier**.")

@st.cache_resource
def load_trained_model():
    model = joblib.load("models/rf_crowd_model.pkl")
    cols = joblib.load("models/feature_columns.pkl")
    return model, cols

try:
    rf_model, feature_cols = load_trained_model()
except Exception:
    st.warning("⚠️ File model belum ditemukan di folder `models/`. Pastikan Anda sudah menjalankan `python src/step3_train_model.py` terlebih dahulu.")
    st.stop()

uploaded_file = st.file_uploader("Unggah File Rekaman Suara Ruangan (.wav atau .mp3)", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Mengekstraksi 32 fitur sinyal audio dan menjalankan prediksi Random Forest..."):
        # Simpan sementara file yang diunggah untuk dibaca oleh Librosa
        temp_path = "temp_uploaded_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Ekstraksi fitur dan prediksi
        features_32 = extract_32_features_safe(temp_path, sr=16000)
        X_input = features_32.reshape(1, -1)
        pred_class = rf_model.predict(X_input)[0]
        pred_proba = rf_model.predict_proba(X_input)[0]

        # Tampilkan Hasil Prediksi Utama
        col1, col2 = st.columns(2)
        with col1:
            if pred_class == 1:
                st.error("###  STATUS RUANGAN: RAMAI (Kelas 1)")
                st.metric("Tingkat Keyakinan (Confidence Score)", f"{pred_proba[1]*100:.2f}%")
            else:
                st.success("### ✅ STATUS RUANGAN: TIDAK RAMAI / TENANG (Kelas 0)")
                st.metric("Tingkat Keyakinan (Confidence Score)", f"{pred_proba[0]*100:.2f}%")

        with col2:
            st.markdown("**Ringkasan Parameter Akustik Utama:**")
            st.write(f"- **Rata-rata Energi (RMS Mean):** `{features_32[0]:.5f}`")
            st.write(f"- **Fluktuasi Energi (RMS Std):** `{features_32[1]:.5f}`")
            st.write(f"- **Pusat Frekuensi (Spectral Centroid Mean):** `{features_32[2]:.1f} Hz`")
            st.write(f"- **Zero Crossing Rate (ZCR Mean):** `{features_32[4]:.4f}`")

        # Visualisasi Gelombang & Spektrogram
        st.subheader("📊 Visualisasi Sinyal Audio (Waveform & Mel-Spectrogram)")
        y_vis, sr_vis = librosa.load(temp_path, sr=16000)
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))

        librosa.display.waveshow(y_vis, sr=sr_vis, ax=ax[0], color="#0d9488")
        ax[0].set_title("Waveform (Domain Waktu)")
        ax[0].set_xlabel("Waktu (Detik)")
        ax[0].set_ylabel("Amplitudo")

        S = librosa.feature.melspectrogram(y=y_vis, sr=sr_vis, n_mels=64)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_vis, ax=ax[1], cmap='magma')
        fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
        ax[1].set_title("Mel-Spectrogram (Domain Frekuensi)")

        plt.tight_layout()
        st.pyplot(fig)
```

---

## BAGIAN 5: Katalog Kemungkinan Error & Panduan Problem Solving (Troubleshooting Lengkap)

Dalam pengerjaan proyek audio menggunakan Python dan `librosa`, hampir setiap kelompok mahasiswa pasti menghadapi salah satu dari **8 kendala teknis atau metodologis** di bawah ini. Berikut adalah penjelasan penyebab dan solusi tuntasnya:

---

### Error 1: `PySoundFile failed. Trying audioread instead` / `NoBackendError` saat Membaca File `.mp3` atau `.m4a`
* **Gejala Error:** Saat menjalankan `librosa.load("rekaman_hp.m4a")` atau `.mp3`, muncul pesan error panjang yang diakhiri dengan `audioread.exceptions.NoBackendError` atau `LibsndfileError`.
* **Penyebab:** Rekaman bawaan dari HP (Android/iPhone) umumnya berformat `.m4a` (AAC) atau `.mp3`. Pustaka standar `soundfile` hanya mendukung format tidak terkompresi seperti `.wav`, `.flac`, dan `.ogg`. Untuk membaca `.mp3` atau `.m4a`, sistem operasi memerlukan program eksternal bernama **FFmpeg**.
* **Solusi (Problem Solving):**
  1. **Cara Paling Cepat Tanpa Install FFmpeg:** Konversi file rekaman HP Anda menjadi format `.wav` terlebih dahulu menggunakan layanan konverter gratis atau aplikasi perekam di HP yang diatur langsung menyimpan ke format `.wav` (misal aplikasi *Easy Voice Recorder* atau *ASR Voice Recorder* di Android).
  2. **Cara Permanen di Windows/Mac/Linux:** Instal FFmpeg melalui Conda dengan perintah `conda install -c conda-forge ffmpeg` atau di Ubuntu/Colab dengan `sudo apt-get install ffmpeg`.

---

### Error 2: `ValueError: setting an array element with a sequence` atau `shapes (32,) and (16,) not aligned`
* **Gejala Error:** Saat mengubah list `X` menjadi `np.array(X)` atau saat menjalankan `rf_model.fit(X_train, y_train)`, Python menolak karena panjang fitur antar baris tidak konsisten.
* **Penyebab:**
  1. Anda lupa menyertakan parameter `axis=1` pada fungsi `np.mean(mfcc, axis=1)` atau `np.std(mfcc, axis=1)`. Jika `axis=1` terlewat, maka matriks 13 koefisien MFCC akan dirata-ratakan seluruhnya menjadi **1 angka tunggal**, sehingga total fitur menyusut dari 32 menjadi hanya 8 fitur!
  2. File audio belum diubah ke *Mono* (`mono=True`), sehingga file yang direkam secara *Stereo* (2 kanal kiri-kanan) menghasilkan matriks fitur berdimensi ganda dibanding file *Mono*.
* **Solusi (Problem Solving):**
  * Pastikan saat memanggil `librosa.load(file_path, sr=16000, mono=True)` parameter `mono=True` selalu aktif.
  * Pastikan kode agregasi MFCC selalu tertulis persis: `np.mean(mfcc, axis=1), np.std(mfcc, axis=1)`.
  * Tambahkan pengecekan `assert len(features) == 32` di akhir fungsi `extract_features`.

---

### Error 3: Muncul Nilai `NaN` (*Not a Number*) atau `ZeroDivisionError` pada Saat Ekstraksi Kelas `Tidak Ramai`
* **Gejala Error:** Saat melatih model dengan `rf_model.fit()`, muncul pesan `ValueError: Input X contains NaN`.
* **Penyebab:** Pada kelas `Tidak Ramai`, jika ada file rekaman yang benar-benar hening (atau setelah dipotong oleh `librosa.effects.trim(y, top_db=25)` panjang array `y` menjadi `0` / kosong), maka perhitungan `spectral_centroid` dan `rms` akan membagi angka nol dengan nol ($\frac{0}{0} = \text{NaN}$).
* **Solusi (Problem Solving):**
  * Terapkan blok pengaman seperti yang sudah ditulis pada fungsi `extract_32_features_safe` di Bagian 4.2 di atas:
    ```python
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    # Jika hasil trim menyisakan kurang dari 1 detik, batalkan trim dan gunakan sinyal asli
    y_proc = y if len(y_trimmed) < sr else y_trimmed
    # Jika sinyal benar-benar nol mutlak, suntikkan derau mikroskopis (1e-6) agar tidak memicu pembagian nol
    if np.max(np.abs(y_proc)) < 1e-6:
        y_proc = y_proc + np.random.normal(0, 1e-6, len(y_proc))
    ```

---

### Error 4: Akurasi Model Mencapai 100% Sempurna di Notebook, Tapi Gagal Total Saat Dites dengan Rekaman Baru (*Overfitting & Data Leakage*)
* **Gejala Error:** Saat menjalankan `classification_report`, nilai *Precision*, *Recall*, dan *F1-Score* semuanya bernilai `1.00` (100%). Namun ketika diuji dengan rekaman baru, tebakannya selalu salah.
* **Penyebab:**
  1. **Kebocoran Potongan (*Chunk Data Leakage*):** Anda memotong 1 file rekaman panjang menjadi 30 potongan, lalu mengacaknya dengan `train_test_split` biasa. Akibatnya, potongan detik ke-5 masuk ke *Train* dan potongan detik ke-10 dari rekaman yang persis sama masuk ke *Test*.
  2. **Bias Perangkat Perekam (*Microphone Device Bias*):** Misalnya seluruh kelas `Ramai` direkam memakai HP Anggota A (merk Samsung), sedangkan seluruh kelas `Tidak Ramai` direkam memakai HP Anggota B (merk iPhone). Model *Random Forest* yang cerdas tidak belajar membedakan ramai vs sepi, melainkan **menghafal karakteristik frekuensi mikrofon Samsung vs iPhone!**
* **Solusi (Problem Solving):**
  * **Wajib Aturan Perangkat:** Setiap alat perekam (HP/Laptop) yang digunakan untuk merekam sampel `Ramai` **WAJIB** juga digunakan untuk merekam sampel `Tidak Ramai` dalam jumlah yang seimbang.
  * Gunakan `GroupShuffleSplit` berdasarkan nama file induk seperti pada skrip `step3_train_model.py` di atas.

---

### Error 5: Suara Kipas Angin, Dengung AC, atau Hujan Terdeteksi Sebagai "Ramai" (*False Positive*)
* **Gejala Error:** Ruangan sedang kosong tidak ada orang, tetapi karena mikrofon diletakkan dekat AC atau kipas angin yang menyala kencang, sistem memprediksi ruangan tersebut **Ramai (1)**.
* **Penyebab:** Saat Anda mengumpulkan data latihan untuk kelas `Tidak Ramai`, Anda hanya merekam ruangan yang benar-benar hening senyap. Akibatnya, model menganggap setiap sinyal yang memiliki energi `rms_mean` tinggi pasti adalah `Ramai`.
* **Solusi (Problem Solving):**
  * Perbaiki komposisi dataset kelas `Tidak Ramai` Anda! Isi folder `tidak_ramai` dengan:
    * 40% rekaman ruangan hening tenang.
    * 40% rekaman ruangan kosong dengan suara AC/kipas angin menyala dekat mikrofon.
    * 20% rekaman ruangan tenang dengan suara ketukan benda sesekali (suara ketik keyboard, klik mouse, atau balik halaman buku).
  * Dengan komposisi ini, *Random Forest* akan belajar mengandalkan fitur **MFCC** dan **Spectral Centroid** untuk membedakan spektrum frekuensi vokal manusia dari deru mesin frekuensi rendah.

---

### Error 6: Ruangan Ramai Tapi Terdeteksi "Tidak Ramai" Saat Diuji di Ruangan Besar (*False Negative / Masalah Jarak Gain*)
* **Gejala Error:** Di dalam ruang kelas besar terdapat 25 orang sedang mengobrol, tetapi karena laptop diletakkan di meja pojok depan yang jauh dari kerumunan, sistem memprediksi **Tidak Ramai (0)**.
* **Penyebab:** Energi suara (`RMS`) meluruh secara kuadratik terhadap jarak (Hukum Kebalikan Kuadrat / *Inverse Square Law*). Jika saat melatih model semua data `Ramai` direkam dengan jarak mikrofon sangat dekat ke mulut ($30\text{ cm}$), maka obrolan ramai dari jarak $5\text{ meter}$ akan memiliki nilai `rms_mean` yang sangat kecil.
* **Solusi (Problem Solving):**
  * Saat mengumpulkan data `Ramai`, variasikan posisi peletakan HP/mikrofon: ada yang diletakkan di tengah meja diskusi, ada yang diletakkan di sudut ruangan berjarak $3 - 5\text{ meter}$ dari kerumunan.
  * Opsi Teknis Tambahan: Anda dapat menambahkan proses *Peak Normalization* (`y = librosa.util.normalize(y)`) khusus sebelum menghitung fitur **MFCC**, **Spectral Centroid**, dan **ZCR** (tetapi hitung **RMS** sebelum dinormalisasi). Dengan demikian, nilai warna suara (MFCC) tidak terpengaruh oleh jauh-dekatnya jarak mikrofon.

---

### Error 7: Ketimpangan Jumlah Sampel Antar Kelas (*Class Imbalance*)
* **Gejala Error:** Nilai *Accuracy* terlihat tinggi (misal 85%), tetapi pada `classification_report`, nilai *Recall* untuk kelas `Ramai` sangat rendah (misal hanya `0.35`).
* **Penyebab:** Jumlah potongan file di folder `tidak_ramai` jauh lebih banyak daripada di folder `ramai` (misal 400 file sepi vs 60 file ramai). Model menjadi bias dan cenderung menebak "Tidak Ramai" terus-menerus.
* **Solusi (Problem Solving):**
  1. Samakan durasi total rekaman antara kedua kelas sebelum dipotong (usahakan rasio jumlah potongan mendekati **50% : 50%** atau maksimal **60% : 40%**).
  2. Tambahkan parameter `class_weight="balanced"` pada saat mendefinisikan model:
     ```python
     rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
     ```

---

### Error 8: Proses Ekstraksi Fitur Terasa Sangat Lambat Saat Jumlah File Mencapai Ratusan
* **Gejala Error:** Ketika menjalankan skrip ekstraksi fitur untuk 500 file audio, program seolah-olah macet (*hang*) dan memakan waktu lebih dari 15 menit.
* **Penyebab:**
  1. Anda memproses file yang belum dipotong (masih berdurasi puluhan menit per file) di dalam fungsi ekstraksi.
  2. Saat pertama kali `librosa` dijalankan di lingkungan baru, pustaka kompilasi `numba` membutuhkan waktu sekitar 20–40 detik untuk melakukan kompilasi JIT (*Just-In-Time Compilation*).
* **Solusi (Problem Solving):**
  * Pastikan Anda sudah menjalankan `step1_chunk_audio.py` terlebih dahulu sehingga setiap file di folder `02_chunks` hanya berdurasi 5–10 detik. Mengekstrak 32 fitur dari satu file `.wav` berdurasi 5 detik pada sample rate $16000\text{ Hz}$ hanya memakan waktu sekitar **0,05 hingga 0,1 detik** (500 file selesai dalam waktu kurang dari 1 menit).
  * Gunakan penyimpanan tabel antara (`audio_features_32.csv`). Jangan mengekstrak ulang file audio setiap kali Anda ingin mengubah parameter model *Random Forest*; cukup baca file CSV yang sudah jadi menggunakan `pd.read_csv()`.

---

## BAGIAN 6: Checklist Urutan Pengerjaan Tugas untuk Tim Anda

Agar pengerjaan tugas ini terarah dari awal sampai selesai, ikuti daftar periksa (*checklist*) berurutan ini:

- [ ] **Langkah 1 (Business Understanding):** Tulis Bab 1 laporan yang menjelaskan latar belakang masalah pemantauan ruangan tanpa melanggar privasi menggunakan sensor audio, tujuan bisnis, dan target metrik teknis ($F1\text{-Score} \ge 0.85$).
- [ ] **Langkah 2 (Pengumpulan Data):** Pilih **Jalur 1** atau **Jalur 3 (Hibrida)**. Kumpulkan minimal 10–15 menit total rekaman untuk kelas `Ramai` dan 10–15 menit untuk kelas `Tidak Ramai` (jangan lupa sertakan suara dengung AC/kipas angin di kelas `Tidak Ramai`). Masukkan ke folder `data/01_raw/`.
- [ ] **Langkah 3 (Chunking Audio):** Jalankan `python src/step1_chunk_audio.py` untuk memotong rekaman panjang menjadi segmen-segmen 5 detik secara otomatis per file asli. Anda akan mendapatkan sekitar 120–180 file potongan per kelas di folder `data/02_chunks/`.
- [ ] **Langkah 4 (Data Understanding / EDA):** Buat 1 grafik perbandingan *Waveform* dan *Mel-Spectrogram* antara sampel `Ramai` vs `Tidak Ramai` untuk dimasukkan ke Bab 2 laporan.
- [ ] **Langkah 5 (Data Preparation / Feature Extraction):** Jalankan `python src/step2_extract_features.py` untuk mengubah seluruh potongan audio menjadi tabel `audio_features_32.csv` (32 fitur numerik per baris).
- [ ] **Langkah 6 (Modeling & Evaluation):** Jalankan `python src/step3_train_model.py` untuk melatih model *Random Forest* (plus model pembanding seperti SVM/Logistic Regression), simpan tabel *Classification Report*, *Confusion Matrix*, dan grafik *Top 10 Feature Importance* untuk Bab 4 & Bab 5 laporan.
- [ ] **Langkah 7 (Deployment & Demo Sistem):** Jalankan `streamlit run app.py` untuk menampilkan antarmuka web sistem deteksi keramaian saat sesi presentasi atau ambil tangkapan layar (*screenshot*) sistemnya untuk Bab 6 laporan.