import os
import soundfile as sf
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 300
})

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

CSV_PATH = "data/03_processed/audio_features_32.csv"
df = pd.read_csv(CSV_PATH)

# =========================================================================
# 1. 01_distribusi_kelas.png: Perbandingan Raw Files vs Chunks
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Data
categories = ["Tidak Ramai (0)", "Ramai (1)"]
raw_counts = [89, 97]     # Sesuai data valid di data/01_raw
chunk_counts = [543, 742] # Sesuai data/02_chunks

x = np.arange(len(categories))
width = 0.35

rects1 = ax.bar(x - width/2, raw_counts, width, label="File Mentah Asli", color="#64748b", edgecolor="black", linewidth=0.8)
rects2 = ax.bar(x + width/2, chunk_counts, width, label="Potongan Audio (Chunks @5s)", color=["#10b981", "#ef4444"], edgecolor="black", linewidth=0.8)

ax.set_ylabel("Jumlah File / Sampel")
ax.set_title("Distribusi Dataset: File Rekaman Asli vs Audio Chunks (5 Detik)")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontweight="bold")
ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
ax.set_ylim(0, 850)

# Tambahkan label angka di atas bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, "01_distribusi_kelas.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"[OK] Disimpan: {fig1_path}")

# =========================================================================
# 2. 02_waveform_comparison.png: Perbandingan Waveform Domain Waktu
# =========================================================================
f_ramai = "data/02_chunks/ramai/ramai_001__chunk01.wav"
f_tidak = "data/02_chunks/tidak_ramai/tidak_ramai_001__chunk01.wav"

y_r, sr_r = sf.read(f_ramai)
y_t, sr_t = sf.read(f_tidak)

time_r = np.linspace(0, len(y_r) / sr_r, num=len(y_r))
time_t = np.linspace(0, len(y_t) / sr_t, num=len(y_t))

fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)

# Waveform Ramai
axs[0].plot(time_r, y_r, color="#dc2626", lw=0.7, alpha=0.85)
axs[0].set_title("Waveform Audio Kelas Ramai (Label 1: ramai_001__chunk01.wav)", color="#991b1b")
axs[0].set_ylabel("Amplitudo")
axs[0].axhline(0, color="gray", lw=0.5, ls="--")
axs[0].text(0.1, 0.35, f"RMS Mean: 0.0597 | Peak: {np.max(np.abs(y_r)):.3f}",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#ef4444"))

# Waveform Tidak Ramai
axs[1].plot(time_t, y_t, color="#059669", lw=0.7, alpha=0.85)
axs[1].set_title("Waveform Audio Kelas Tidak Ramai (Label 0: tidak_ramai_001__chunk01.wav)", color="#065f46")
axs[1].set_xlabel("Waktu (Detik)")
axs[1].set_ylabel("Amplitudo")
axs[1].axhline(0, color="gray", lw=0.5, ls="--")
axs[1].text(0.1, 0.35, f"RMS Mean: 0.0199 | Peak: {np.max(np.abs(y_t)):.3f}",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#d1fae5", edgecolor="#10b981"))

plt.suptitle("Perbandingan Karakteristik Sinyal Domain Waktu (Waveform)", y=0.98)
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, "02_waveform_comparison.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"[OK] Disimpan: {fig2_path}")

# =========================================================================
# 3. 03_mel_spectrogram_comparison.png: Perbandingan Mel-Spectrogram (dB)
# =========================================================================
fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# Mel Ramai
S_r = librosa.feature.melspectrogram(y=y_r, sr=sr_r, n_mels=128, fmax=8000)
S_r_db = librosa.power_to_db(S_r, ref=np.max)
img1 = librosa.display.specshow(S_r_db, x_axis="time", y_axis="mel", sr=sr_r, fmax=8000, ax=axs[0], cmap="magma")
axs[0].set_title("Mel-Spectrogram Kelas Ramai (Percakapan Manusia)")
axs[0].set_xlabel("Waktu (Detik)")
axs[0].set_ylabel("Frekuensi (Hz - Skala Mel)")
fig.colorbar(img1, ax=axs[0], format="%+2.0f dB", label="Daya / Intensitas (dB)")

# Mel Tidak Ramai
S_t = librosa.feature.melspectrogram(y=y_t, sr=sr_t, n_mels=128, fmax=8000)
S_t_db = librosa.power_to_db(S_t, ref=np.max)
img2 = librosa.display.specshow(S_t_db, x_axis="time", y_axis="mel", sr=sr_t, fmax=8000, ax=axs[1], cmap="magma")
axs[1].set_title("Mel-Spectrogram Kelas Tidak Ramai (Keheningan / Dengung AC)")
axs[1].set_xlabel("Waktu (Detik)")
axs[1].set_ylabel("")
fig.colorbar(img2, ax=axs[1], format="%+2.0f dB", label="Daya / Intensitas (dB)")

plt.suptitle("Perbandingan Distribusi Frekuensi Mel-Spectrogram: Ramai vs Tidak Ramai", y=1.02)
plt.tight_layout()
fig3_path = os.path.join(FIG_DIR, "03_mel_spectrogram_comparison.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"[OK] Disimpan: {fig3_path}")

# =========================================================================
# 4. 04_boxplot_fitur_utama.png: Boxplot 5 Fitur Utama
# =========================================================================
target_features = [
    ("rms_mean", "RMS Energy (Mean)", "Amplitudo Energi Rata-rata"),
    ("rms_std", "RMS Fluctuation (Std)", "Dinamika Variasi Energi"),
    ("centroid_mean", "Spectral Centroid Mean (Hz)", "Pusat Massa Frekuensi"),
    ("zcr_mean", "Zero Crossing Rate Mean", "Tingkat Transisi Titik Nol"),
    ("mfcc_1_mean", "MFCC-1 Mean (dB)", "Energi Spektrum Pita Dasar")
]

fig, axs = plt.subplots(1, 5, figsize=(18, 4.8))
palette = {"tidak_ramai": "#10b981", "ramai": "#ef4444"}

for i, (col, title, desc) in enumerate(target_features):
    sns.boxplot(data=df, x="class_name", y=col, hue="class_name", ax=axs[i], palette=palette, width=0.45,
                legend=False,
                boxprops=dict(alpha=0.85, edgecolor="black", linewidth=0.8),
                flierprops=dict(marker="o", markersize=2, alpha=0.3))
    axs[i].set_title(title, fontsize=11, fontweight="bold")
    axs[i].set_xlabel("")
    axs[i].set_ylabel(desc, fontsize=10)
    axs[i].set_xticks([0, 1])
    axs[i].set_xticklabels(["Tidak Ramai", "Ramai"], fontweight="bold")

plt.suptitle("Distribusi 5 Fitur Akustik Utama Pembeda Kelas Ramai vs Tidak Ramai", y=1.03)
plt.tight_layout()
fig4_path = os.path.join(FIG_DIR, "04_boxplot_fitur_utama.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"[OK] Disimpan: {fig4_path}")

# =========================================================================
# 5. 05_correlation_heatmap.png: Korelasi Antar Fitur Utama
# =========================================================================
corr_cols = [
    "label", "rms_mean", "rms_std", "centroid_mean", "centroid_std",
    "zcr_mean", "zcr_std", "mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean",
    "mfcc_1_std", "mfcc_2_std", "mfcc_3_std"
]

corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Koefisien Korelasi Pearson"}, ax=ax)
ax.set_title("Matriks Korelasi Pearson Fitur Akustik Utama terhadap Label Target", pad=15)

plt.tight_layout()
fig5_path = os.path.join(FIG_DIR, "05_correlation_heatmap.png")
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"[OK] Disimpan: {fig5_path}")
print("\n[SELESAI] Seluruh 5 figur resolusi tinggi berhasil disimpan ke reports/figures/")
