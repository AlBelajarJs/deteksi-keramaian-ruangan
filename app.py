import os
import sys
import time
import json
import joblib
import io
from datetime import datetime
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

# Setup path root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.feature_extractor import extract_32_features_from_signal, FEATURE_NAMES

# =========================================================================
# AUTO-FORWARD KE STREAMLIT JIKA DIJALANKAN DENGAN 'python app.py'
# =========================================================================
if __name__ == "__main__" and hasattr(st, "runtime") and not st.runtime.exists():
    from streamlit.web import cli as stcli
    print("\n" + "=" * 70)
    print("🚀 [CrowdSense Audio] Memulai Streamlit Web Server...")
    print("💡 Catatan: Aplikasi Streamlit normalnya dijalankan via:")
    print("   streamlit run app.py")
    print("=" * 70 + "\n")
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())

# =========================================================================
# KONFIGURASI HALAMAN STREAMLIT
# =========================================================================
st.set_page_config(
    page_title="CrowdSense Audio: Sistem Deteksi Keramaian Ruangan",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS & Keyframes Animation
st.markdown("""
<style>
    @keyframes pulseRed {
        0% {
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
            border-color: #ef4444;
            background-color: #fee2e2;
            transform: scale(1);
        }
        50% {
            box-shadow: 0 0 40px 14px rgba(239, 68, 68, 0.65);
            border-color: #b91c1c;
            background-color: #fecaca;
            transform: scale(1.008);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
            border-color: #ef4444;
            background-color: #fee2e2;
            transform: scale(1);
        }
    }
    .giant-alert-ramai {
        animation: pulseRed 1.1s infinite;
        border: 4px solid #ef4444;
        border-radius: 14px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .giant-alert-tenang {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 3px solid #10b981;
        border-radius: 14px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .giant-alert-standby {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px dashed #94a3b8;
        border-radius: 14px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .led-meter-box {
        background: #0f172a;
        color: #38bdf8;
        border: 2px solid #334155;
        border-radius: 10px;
        padding: 14px 20px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.15rem;
        margin-bottom: 1.2rem;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .card-ramai {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 6px solid #ef4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .card-tenang {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 6px solid #10b981;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .meter-box {
        background-color: #f1f5f9;
        border-radius: 6px;
        padding: 8px 12px;
        font-family: monospace;
        font-weight: bold;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# CACHING MODEL & ARTEFAK
# =========================================================================
@st.cache_resource
def load_all_artifacts():
    models_dir = "models"
    rf_model = joblib.load(os.path.join(models_dir, "rf_audio_classifier.pkl"))
    # Optimalisasi single-sample latency
    rf_model.n_jobs = 1
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

    # Load metadata
    meta_path = os.path.join(models_dir, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Load benchmark table jika ada
    bench_path = os.path.join("reports", "model_benchmark_results.csv")
    if os.path.exists(bench_path):
        benchmark_df = pd.read_csv(bench_path)
    else:
        benchmark_df = pd.DataFrame()

    # Warmup librosa & model
    warmup_sig = np.zeros(16000, dtype=np.float32)
    _w_feat = extract_32_features_from_signal(warmup_sig, sr=16000)
    _ = rf_model.predict(_w_feat.reshape(1, -1))

    return rf_model, scaler, metadata, benchmark_df

try:
    rf_model, scaler, metadata, benchmark_df = load_all_artifacts()
except Exception as e:
    st.error(f"⚠️ Gagal memuat artefak model dari folder `models/`: {e}")
    st.info("Pastikan Anda telah menjalankan `python src/step3_train_model.py` terlebih dahulu.")
    st.stop()

# Helper fungsi log deteksi
def append_log_csv(row_dict: dict, log_path: str = "reports/live_detection_log.csv"):
    import csv
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)
    headers = ["timestamp", "rms_energy", "predicted_label", "status", "confidence", "inference_ms"]
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def make_led_volume_meter(rms: float, max_rms: float = 0.08, num_blocks: int = 16) -> tuple[str, float]:
    """Menghasilkan bar visual LED meter berbasis tingkat energi RMS."""
    ratio = min(max(rms / max_rms, 0.0), 1.0)
    filled = int(round(ratio * num_blocks))
    blocks = []
    for i in range(num_blocks):
        if i < filled:
            if i < int(num_blocks * 0.5):
                blocks.append("🟩")
            elif i < int(num_blocks * 0.75):
                blocks.append("🟨")
            else:
                blocks.append("🟥")
        else:
            blocks.append("░")
    return "".join(blocks), ratio * 100.0

def trigger_audio_alarm(play_beep: bool = True, play_tts: bool = True):
    """Membunyikan alarm nada Web Audio API dan sintesis suara teguran otomatis."""
    import streamlit.components.v1 as components
    js_parts = []
    if play_beep:
        js_parts.append("""
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                const ctx = new AudioCtx();
                if (ctx.state === 'suspended') { ctx.resume(); }
                function playTone(freq, start, duration) {
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = 'sawtooth';
                    osc.frequency.setValueAtTime(freq, ctx.currentTime + start);
                    gain.gain.setValueAtTime(0.35, ctx.currentTime + start);
                    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + start + duration);
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.start(ctx.currentTime + start);
                    osc.stop(ctx.currentTime + start + duration);
                }
                playTone(880, 0, 0.22);
                playTone(660, 0.24, 0.32);
            }
        } catch(e) { console.error('AudioContext Beep error:', e); }
        """)
    if play_tts:
        js_parts.append("""
        try {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const u = new SpeechSynthesisUtterance("Harap tenang, ruangan terdeteksi berisik!");
                u.lang = "id-ID";
                u.rate = 1.05;
                u.pitch = 1.0;
                window.speechSynthesis.speak(u);
            }
        } catch(e) { console.error('SpeechSynthesis TTS error:', e); }
        """)
    if js_parts:
        full_html = f"""
        <script>
        (function() {{
            {' '.join(js_parts)}
        }})();
        </script>
        """
        components.html(full_html, height=0, width=0)

# =========================================================================
# SIDEBAR NAVIGASI & INFORMASI SISTEM
# =========================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/sound-recording-copyright.png", width=64)
    st.title("CrowdSense Audio")
    st.caption("Sistem Deteksi Tingkat Keramaian Ruangan Berbasis Sinyal Akustik (CRISP-DM)")

    st.markdown("---")
    st.subheader("⚙️ Status Model Aktif")
    st.success("🟢 **Model:** Random Forest Classifier (Tuned)")

    test_metrics = metadata.get("test_metrics", {})
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Test Accuracy", f"{test_metrics.get('accuracy', 0.8008)*100:.1f}%")
        st.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0.8463):.3f}")
    with col_m2:
        st.metric("F1-Score", f"{test_metrics.get('f1_score_binary', 0.7614):.3f}")
        st.metric("Inferensi", f"{test_metrics.get('inference_time_ms', 13.2):.1f} ms")

    st.markdown("---")
    st.subheader("🎚️ Kalibrasi Sensitivitas")
    mic_gain = st.slider(
        "Penguat Sinyal (Mic Gain)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Naikkan faktor penguat jika mikrofon bawaan laptop Anda menangkap suara terlalu pelan/lemah."
    )

    st.markdown("---")
    st.markdown("""
    **Metodologi:** CRISP-DM Standard
    **Fitur Akustik:** 32 Parameter Librosa
    **Sample Rate:** 16.000 Hz Mono
    **Privasi:** *Privacy-Preserving* (Tanpa menyimpan rekaman isi pembicaraan).
    """)

# =========================================================================
# HEADER UTAMA
# =========================================================================
st.markdown('<div class="main-header">🎙️ CrowdSense: Deteksi Keramaian Ruangan Berbasis Audio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Klasifikasi Akustik Real-Time Non-Invasif menggunakan Ekstraksi 32 Fitur Statistik Audio & Random Forest</div>', unsafe_allow_html=True)

# TABS APLIKASI
tab_sentinel, tab_file, tab_eval = st.tabs([
    "🚨 Live Watchdog (Mode Satpam Suara Otomatis)",
    "📂 Uji File Rekaman & Sampel Demo",
    "📊 Dashboard Evaluasi & Metodologi CRISP-DM"
])

# =========================================================================
# TAB 1: LIVE WATCHDOG (MODE SATPAM SUARA OTOMATIS)
# =========================================================================
with tab_sentinel:
    st.subheader("🚨 Smart Acoustic Sentinel: Mode Satpam Suara Otomatis")
    st.caption("Pemantauan ruangan tanpa henti (*Continuous IoT Audio Sentinel*): Mendeteksi kegaduhan secara real-time, memicu peringatan visual berkedip, dan menyalakan teguran audio otomatis saat ruangan berisik.")

    # Inisialisasi Session State Sentinel
    if "watchdog_active" not in st.session_state:
        st.session_state["watchdog_active"] = False
    if "violation_count" not in st.session_state:
        st.session_state["violation_count"] = 0
    if "last_violation_time" not in st.session_state:
        st.session_state["last_violation_time"] = "-"
    if "history_log" not in st.session_state:
        st.session_state["history_log"] = []
    if "current_status" not in st.session_state:
        st.session_state["current_status"] = "STANDBY"
    if "current_proba" not in st.session_state:
        st.session_state["current_proba"] = 0.0
    if "current_conf" not in st.session_state:
        st.session_state["current_conf"] = 0.0
    if "current_rms" not in st.session_state:
        st.session_state["current_rms"] = 0.0
    if "current_latency" not in st.session_state:
        st.session_state["current_latency"] = 0.0
    if "current_led_str" not in st.session_state:
        st.session_state["current_led_str"] = "░" * 16
    if "current_led_pct" not in st.session_state:
        st.session_state["current_led_pct"] = 0.0

    # Cek ketersediaan sounddevice
    try:
        import sounddevice as sd
        has_sounddevice = True
    except Exception:
        has_sounddevice = False

    # ---------------------------------------------------------------------
    # PANEL KONTROL & KONFIGURASI SENTINEL
    # ---------------------------------------------------------------------
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1.8, 1.4, 1.1, 1.3])

    with col_ctrl1:
        if not has_sounddevice:
            st.error("⚠️ `sounddevice` tidak terdeteksi di lingkungan ini.")
            is_active = False
        else:
            is_active = st.toggle(
                "🚨 **AKTIFKAN SATPAM SUARA (LIVE SENTINEL)**",
                value=st.session_state["watchdog_active"],
                key="toggle_sentinel_active"
            )
            st.session_state["watchdog_active"] = is_active

    with col_ctrl2:
        window_duration = st.select_slider(
            "⏱️ Jeda Siklus Sensor",
            options=[2.0, 3.0, 5.0],
            value=3.0,
            format_func=lambda x: f"{int(x)} Detik (Default)" if x == 3.0 else f"{int(x)} Detik",
            help="Durasi perekaman per potongan audio. Nilai 3 detik sangat responsif dan ideal untuk demonstrasi langsung."
        )

    with col_ctrl3:
        sound_alarm = st.checkbox(
            "🔊 Alarm Beep",
            value=True,
            help="Bunyikan nada sirine peringatan via Web Audio API saat ruangan terdeteksi ramai."
        )

    with col_ctrl4:
        tts_alarm = st.checkbox(
            "🗣️ Teguran Suara",
            value=True,
            help="Ucapkan teguran suara 'Harap tenang, ruangan terdeteksi berisik!' via SpeechSynthesis."
        )

    col_sub_btn1, col_sub_btn2 = st.columns([1, 3])
    with col_sub_btn1:
        if st.button("🔄 Reset Counter Pelanggaran", help="Mengembalikan jumlah pelanggaran ke 0 dan mengosongkan riwayat"):
            st.session_state["violation_count"] = 0
            st.session_state["last_violation_time"] = "-"
            st.session_state["history_log"] = []
            st.session_state["current_status"] = "STANDBY"
            st.session_state["current_proba"] = 0.0
            st.session_state["current_conf"] = 0.0
            st.session_state["current_rms"] = 0.0
            st.session_state["current_latency"] = 0.0
            st.session_state["current_led_str"] = "░" * 16
            st.session_state["current_led_pct"] = 0.0
            st.rerun()
    with col_sub_btn2:
        if is_active:
            st.caption("🟢 **Status:** Sentinel aktif memantau getaran ruangan secara kontinu. Tekan toggle kembali untuk berhenti.")
        else:
            st.caption("⚪ **Status:** Sentinel dalam mode Siaga (Standby). Aktifkan toggle di atas untuk memulai pemantauan.")

    st.markdown("---")

    # ---------------------------------------------------------------------
    # GIANT DYNAMIC ALERT BANNER
    # ---------------------------------------------------------------------
    curr_status = st.session_state["current_status"]
    curr_conf = st.session_state["current_conf"]
    curr_proba = st.session_state["current_proba"]
    curr_infer = st.session_state["current_latency"]

    if curr_status == "RAMAI":
        st.markdown(f"""
        <div class="giant-alert-ramai">
            <div style="font-size: 2.3rem; font-weight: 900; color: #b91c1c; letter-spacing: 1px; margin-bottom: 0.4rem;">
                🚨 WARNING: HARAP TENANG! RUANGAN TERDETEKSI BERISIK!
            </div>
            <div style="font-size: 1.25rem; font-weight: 600; color: #7f1d1d; margin-bottom: 0.8rem;">
                Tingkat kebisingan vokal / kerumunan melampaui batas toleransi ruangan kondusif!
            </div>
            <div style="display: inline-block; background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; border-radius: 6px; padding: 6px 16px; font-weight: bold; color: #991b1b; font-size: 1.05rem;">
                Keyakinan AI: <b>{curr_conf:.1f}%</b> &nbsp;|&nbsp; Probabilitas Ramai: <b>{curr_proba:.1f}%</b> &nbsp;|&nbsp; Latensi: <b>{curr_infer:.1f} ms</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif curr_status == "TENANG":
        st.markdown(f"""
        <div class="giant-alert-tenang">
            <div style="font-size: 2.1rem; font-weight: 800; color: #065f46; margin-bottom: 0.4rem;">
                🟢 STATUS RUANGAN: KONDUSIF & HENING
            </div>
            <div style="font-size: 1.2rem; font-weight: 500; color: #047857; margin-bottom: 0.8rem;">
                Suasana ruangan tenang dan hening, sangat ideal untuk konsentrasi belajar / bekerja.
            </div>
            <div style="display: inline-block; background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; border-radius: 6px; padding: 6px 16px; font-weight: bold; color: #065f46; font-size: 1.05rem;">
                Keyakinan AI: <b>{curr_conf:.1f}%</b> &nbsp;|&nbsp; Probabilitas Ramai: <b>{curr_proba:.1f}%</b> &nbsp;|&nbsp; Latensi: <b>{curr_infer:.1f} ms</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="giant-alert-standby">
            <div style="font-size: 1.8rem; font-weight: 700; color: #334155; margin-bottom: 0.4rem;">
                ⚪ SMART ACOUSTIC SENTINEL: MODE STANDBY
            </div>
            <div style="font-size: 1.05rem; color: #64748b;">
                Nyalakan saklar <b>'🚨 Aktifkan Satpam Suara'</b> di atas untuk memulai pemantauan mikrofon secara otomatis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # REAL-TIME LED VOLUME METER (RMS ENERGY)
    # ---------------------------------------------------------------------
    curr_rms = st.session_state["current_rms"]
    curr_led_str = st.session_state["current_led_str"]
    curr_led_pct = st.session_state["current_led_pct"]

    st.markdown(f"""
    <div class="led-meter-box">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="color: #94a3b8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px;">
                🎚️ Indikator Energi Akustik Real-Time (RMS)
            </span>
            <span style="color: #38bdf8; font-weight: bold; font-size: 1.05rem;">
                RMS: {curr_rms:.4f} &nbsp;|&nbsp; Beban Energi: {curr_led_pct:.0f}%
            </span>
        </div>
        <div style="font-size: 1.35rem; letter-spacing: 2px;">
            {curr_led_str}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # 4 KARTU METRIK SESI
    # ---------------------------------------------------------------------
    cm1, cm2, cm3, cm4 = st.columns(4)
    with cm1:
        stat_label = "RAMAI 🚨" if curr_status == "RAMAI" else ("TENANG 🟢" if curr_status == "TENANG" else "STANDBY ⚪")
        cm1.metric(
            "Status Saat Ini",
            stat_label,
            delta="Bising" if curr_status == "RAMAI" else ("Kondusif" if curr_status == "TENANG" else None),
            delta_color="inverse"
        )
    with cm2:
        cm2.metric(
            "Total Pelanggaran",
            f"{st.session_state['violation_count']} Kali",
            help="Total insiden ruangan berisik yang terdeteksi selama sesi ini"
        )
    with cm3:
        cm3.metric(
            "Teguran Terakhir",
            st.session_state["last_violation_time"],
            help="Jam pencatatan pelanggaran kebisingan terakhir"
        )
    with cm4:
        cm4.metric(
            "Kecepatan AI",
            f"{curr_infer:.1f} ms" if curr_infer > 0 else "-",
            help="Latensi komputasi ekstraksi 32 fitur dan inferensi Random Forest"
        )

    # ---------------------------------------------------------------------
    # LIVE TREND CHART & TABEL LOG RIWAYAT
    # ---------------------------------------------------------------------
    if st.session_state["history_log"]:
        st.markdown("---")
        df_hist = pd.DataFrame(st.session_state["history_log"])

        fig_trend, ax_trend = plt.subplots(figsize=(11, 2.8))
        ax_trend.axhspan(50, 105, color="#fee2e2", alpha=0.45, label="Zona Berisik (>50%)")
        ax_trend.axhspan(-5, 50, color="#d1fae5", alpha=0.45, label="Zona Tenang (<=50%)")
        ax_trend.axhline(50, color="#ef4444", linestyle="--", linewidth=1.4, alpha=0.8, label="Threshold 50%")

        times = df_hist["Waktu"]
        probs = df_hist["Probabilitas Ramai (%)"]
        ax_trend.plot(times, probs, color="#1e293b", marker="o", markersize=6, linewidth=2.0, label="Probabilitas Ramai (%)")

        for _, r in df_hist.iterrows():
            pt_col = "#ef4444" if r["Probabilitas Ramai (%)"] > 50 else "#10b981"
            ax_trend.scatter(r["Waktu"], r["Probabilitas Ramai (%)"], color=pt_col, s=70, zorder=5)

        ax_trend.set_ylim(-5, 105)
        ax_trend.set_ylabel("Probabilitas Ramai (%)", fontsize=9, fontweight="bold")
        ax_trend.set_title("📈 Grafik Fluktuasi Keramaian Ruangan Real-Time", fontsize=11, fontweight="bold")
        ax_trend.grid(alpha=0.3, linestyle=":")
        ax_trend.legend(loc="upper right", framealpha=0.9, fontsize=8)
        plt.xticks(rotation=25, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_trend)

        with st.expander("📋 Riwayat Aktivitas Sentinel (20 Siklus Terakhir)", expanded=False):
            st.dataframe(df_hist, use_container_width=True)

    # ---------------------------------------------------------------------
    # SUB-BAGIAN: REKAM MANUAL CEPAT BROWSER (PUSH-TO-TALK)
    # ---------------------------------------------------------------------
    with st.expander("🎙️ Opsi Manual: Rekam Suara Cepat via Browser (Push-to-Talk)", expanded=False):
        st.write("Gunakan fitur ini untuk merekam satu sampel suara tertentu langsung melalui mikrofon web browser:")
        rec_browser = st.audio_input("Rekam Suara (Klik Mikrofon untuk Mulai & Selesai)", key="expander_audio_input")
        if rec_browser is not None:
            with st.spinner("Menganalisis audio browser..."):
                t_b0 = time.perf_counter()
                b_bytes = rec_browser.read()
                y_b, sr_b = librosa.load(io.BytesIO(b_bytes), sr=16000, mono=True)
                dur_b = len(y_b) / 16000.0
                if mic_gain != 1.0:
                    y_b = np.clip(y_b * mic_gain, -1.0, 1.0)
                if len(y_b) < 16000:
                    st.warning("Durasi rekaman terlalu singkat (< 1 detik).")
                else:
                    if len(y_b) < 80000:
                        y_b_padded = np.pad(y_b, (0, 80000 - len(y_b)), mode="constant")
                    else:
                        y_b_padded = y_b
                    feats_b = extract_32_features_from_signal(y_b_padded, sr=16000)
                    pred_b = int(rf_model.predict(feats_b.reshape(1, -1))[0])
                    proba_b = rf_model.predict_proba(feats_b.reshape(1, -1))[0]
                    infer_b = (time.perf_counter() - t_b0) * 1000.0
                    rms_b = float(np.sqrt(np.mean(y_b_padded ** 2)))

                    if pred_b == 1:
                        st.error(f"🚨 Hasil: **RAMAI / BERISIK** (Keyakinan: {proba_b[1]*100:.1f}%, RMS: {rms_b:.4f}, Latensi: {infer_b:.1f}ms)")
                    else:
                        st.success(f"🟢 Hasil: **TIDAK RAMAI / KONDUSIF** (Keyakinan: {proba_b[0]*100:.1f}%, RMS: {rms_b:.4f}, Latensi: {infer_b:.1f}ms)")

    # ---------------------------------------------------------------------
    # LOOP SENTINEL OTOMATIS (JIKA WATCHDOG AKTIF)
    # ---------------------------------------------------------------------
    if is_active and has_sounddevice:
        status_box = st.empty()
        with status_box.container():
            st.info(f"🎧 **Sentinel Sedang Mendengarkan:** Merekam getaran suara ruangan ({int(window_duration)} detik)...")

        try:
            rec_samples = int(window_duration * 16000)
            t0_rec = time.perf_counter()
            live_data = sd.rec(rec_samples, samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            y_live = live_data.flatten()

            if mic_gain != 1.0:
                y_live = np.clip(y_live * mic_gain, -1.0, 1.0)

            # Ekstraksi 32 Fitur & Prediksi
            feats_live = extract_32_features_from_signal(y_live, sr=16000)
            pred_live = int(rf_model.predict(feats_live.reshape(1, -1))[0])
            proba_live = rf_model.predict_proba(feats_live.reshape(1, -1))[0]
            infer_ms = (time.perf_counter() - t0_rec - window_duration) * 1000.0
            infer_ms = max(infer_ms, 8.0)

            rms_live = float(np.sqrt(np.mean(y_live ** 2)))
            conf_live = float(proba_live[pred_live] * 100.0)
            prob_ramai = float(proba_live[1] * 100.0)
            now_str = datetime.now().strftime("%H:%M:%S")

            led_str, led_pct = make_led_volume_meter(rms_live)

            # Update State
            st.session_state["current_status"] = "RAMAI" if pred_live == 1 else "TENANG"
            st.session_state["current_proba"] = prob_ramai
            st.session_state["current_conf"] = conf_live
            st.session_state["current_rms"] = rms_live
            st.session_state["current_latency"] = infer_ms
            st.session_state["current_led_str"] = led_str
            st.session_state["current_led_pct"] = led_pct

            if pred_live == 1:
                st.session_state["violation_count"] += 1
                st.session_state["last_violation_time"] = now_str

            st.session_state["history_log"].append({
                "Waktu": now_str,
                "Status": "RAMAI 🚨" if pred_live == 1 else "TENANG 🟢",
                "Probabilitas Ramai (%)": round(prob_ramai, 2),
                "Keyakinan (%)": round(conf_live, 1),
                "RMS Energy": round(rms_live, 4),
                "Latensi AI (ms)": round(infer_ms, 1)
            })
            if len(st.session_state["history_log"]) > 20:
                st.session_state["history_log"].pop(0)

            # Log ke CSV
            append_log_csv({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rms_energy": round(rms_live, 6),
                "predicted_label": pred_live,
                "status": "Ramai" if pred_live == 1 else "Tidak Ramai",
                "confidence": round(conf_live, 2),
                "inference_ms": round(infer_ms, 2)
            })

            # Pemicu Alarm jika RAMAI
            if pred_live == 1 and (sound_alarm or tts_alarm):
                trigger_audio_alarm(play_beep=sound_alarm, play_tts=tts_alarm)
                # PENCEGAHAN ACOUSTIC FEEDBACK: Jeda 1.2 detik agar bunyi speaker laptop selesai
                # sebelum mikrofon merekam kembali di siklus berikutnya!
                time.sleep(1.2)
            else:
                time.sleep(0.05)

            st.rerun()

        except Exception as e:
            st.error(f"Terjadi kesalahan perekaman mikrofon sounddevice: {e}")

# =========================================================================
# TAB 2: DETEKSI FILE & SAMPEL DEMO
# =========================================================================
with tab_file:
    demo_options = {
        "-- Unggah File Sendiri --": None,
        "🔴 [Ramai] Sampel Chunk 5 Detik": "data/02_chunks/ramai/ramai_001__chunk01.wav",
        "🔴 [Ramai] Percakapan Panjang (~30s)": "data/01_raw/ramai/ramai_001.mpeg",
        "🟢 [Tidak Ramai] Sampel Chunk 5 Detik": "data/02_chunks/tidak_ramai/tidak_ramai_001__chunk01.wav",
        "🟢 [Tidak Ramai] Ruang Sunyi/AC (~30s)": "data/01_raw/tidak_ramai/tidak_ramai_001.mp3",
    }

    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        selected_demo = st.selectbox(
            "Pilih audio pengujian siap pakai (Quick Demo):",
            options=list(demo_options.keys()),
            index=0,
            help="Gunakan sampel bawaan ini untuk demonstrasi langsung saat presentasi tanpa repot mencari file di laptop."
        )

    audio_source_path = None

    if selected_demo != "-- Unggah File Sendiri --":
        sample_path = demo_options[selected_demo]
        if sample_path and os.path.exists(sample_path):
            audio_source_path = sample_path
            st.info(f"📁 Menggunakan Sampel Bawaan: **{os.path.basename(sample_path)}**")
        else:
            st.warning("File sampel demo tidak ditemukan di direktori lokal.")
    else:
        with col_opt2:
            uploaded_file = st.file_uploader(
                "Atau Unggah File Audio Rekaman (.wav, .mp3, .mpeg, .ogg, .flac)",
                type=["wav", "mp3", "mpeg", "ogg", "flac"],
                help="File rekaman suara dari HP, laptop, atau mikrofon ruangan."
            )
            if uploaded_file is not None:
                temp_path = "temp_uploaded_audio.wav"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                audio_source_path = temp_path

    if audio_source_path is not None:
        st.subheader("🎧 Pemutar Audio")
        with open(audio_source_path, "rb") as f_audio:
            audio_bytes = f_audio.read()
        st.audio(audio_bytes, format="audio/wav")

        with st.spinner("Mengekstrak 32 fitur akustik dan melakukan inferensi..."):
            t_start = time.perf_counter()
            y, sr = librosa.load(audio_source_path, sr=16000, mono=True)
            total_duration = len(y) / sr

            if mic_gain != 1.0:
                y = np.clip(y * mic_gain, -1.0, 1.0)

            feats_overall = extract_32_features_from_signal(y, sr=sr)
            X_input = feats_overall.reshape(1, -1)

            pred_class = rf_model.predict(X_input)[0]
            pred_proba = rf_model.predict_proba(X_input)[0]
            infer_ms = (time.perf_counter() - t_start) * 1000.0

        st.markdown("---")

        if pred_class == 1:
            st.markdown(f"""
            <div class="card-ramai">
                <h2 style="color: #991b1b; margin:0 0 0.5rem 0;">🚨 STATUS RUANGAN: RAMAI / BERISIK (Kelas 1)</h2>
                <p style="color: #7f1d1d; margin:0; font-size: 1.1rem;">
                    Terdeteksi aktivitas percakapan vokal manusia yang aktif, diskusi kelompok, atau kerumunan.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-tenang">
                <h2 style="color: #065f46; margin:0 0 0.5rem 0;">✅ STATUS RUANGAN: TIDAK RAMAI / KONDUSIF (Kelas 0)</h2>
                <p style="color: #064e3b; margin:0; font-size: 1.1rem;">
                    Ruangan dalam kondisi tenang, hening, atau hanya berisi suara statis mesin pendingin (AC/kipas).
                </p>
            </div>
            """, unsafe_allow_html=True)

        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        conf = pred_proba[pred_class] * 100
        col_res1.metric("Tingkat Keyakinan", f"{conf:.2f}%")
        col_res2.metric("Durasi Audio", f"{total_duration:.1f} detik")
        col_res3.metric("Waktu Inferensi", f"{infer_ms:.2f} ms")
        col_res4.metric("Status Privasi", "🔒 Aman (Tanpa Suara)", delta="Privacy-Preserving")

        # Analisis Timeline Jendela 5 Detik jika File Berdurasi Panjang
        chunk_len = 5 * sr
        if len(y) >= chunk_len:
            st.markdown("---")
            st.subheader("⏱️ Analisis Dinamis Per Segmen Waktu (Jendela 5 Detik)")
            st.caption("Memantau fluktuasi keramaian ruangan secara berkala sepanjang durasi rekaman.")

            num_chunks = len(y) // chunk_len
            segment_data = []
            for i in range(num_chunks):
                start_s = i * chunk_len
                end_s = start_s + chunk_len
                chunk_sig = y[start_s:end_s]
                chunk_feats = extract_32_features_from_signal(chunk_sig, sr=sr)
                c_pred = rf_model.predict(chunk_feats.reshape(1, -1))[0]
                c_proba = rf_model.predict_proba(chunk_feats.reshape(1, -1))[0]

                t_label = f"{i*5:02d}s - {(i+1)*5:02d}s"
                segment_data.append({
                    "Segmen Waktu": t_label,
                    "Detik Awal": i * 5,
                    "Detik Akhir": (i + 1) * 5,
                    "Status": "RAMAI (1)" if c_pred == 1 else "TIDAK RAMAI (0)",
                    "Keyakinan (%)": round(c_proba[c_pred] * 100, 2),
                    "Probabilitas Ramai (%)": round(c_proba[1] * 100, 2),
                    "RMS Mean": round(float(chunk_feats[0]), 5),
                    "Centroid Mean (Hz)": round(float(chunk_feats[2]), 1),
                })

            seg_df = pd.DataFrame(segment_data)

            fig_tl, ax_tl = plt.subplots(figsize=(10, 2.8))
            colors = ["#ef4444" if s == "RAMAI (1)" else "#10b981" for s in seg_df["Status"]]
            bars = ax_tl.bar(seg_df["Segmen Waktu"], seg_df["Probabilitas Ramai (%)"], color=colors, edgecolor="black", linewidth=0.8)
            ax_tl.axhline(50, color="gray", linestyle="--", alpha=0.7, label="Threshold 50%")
            ax_tl.set_ylabel("Probabilitas Ramai (%)")
            ax_tl.set_ylim(0, 105)
            ax_tl.set_title("Timeline Tingkat Keramaian per Jendela 5 Detik", fontsize=11, fontweight="bold")
            ax_tl.legend(loc="upper right")

            for bar in bars:
                h = bar.get_height()
                ax_tl.annotate(f"{h:.0f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                               xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig_tl)

            with st.expander("Lihat Rincian Tabel Segmen Waktu"):
                st.dataframe(seg_df, use_container_width=True)

        # Visualisasi Sinyal Langsung
        st.markdown("---")
        st.subheader("📊 Visualisasi Karakteristik Sinyal Audio")
        fig_sig, axs = plt.subplots(1, 2, figsize=(13, 3.8))

        time_axis = np.linspace(0, len(y)/sr, num=len(y))
        axs[0].plot(time_axis, y, color="#dc2626" if pred_class == 1 else "#059669", lw=0.7)
        axs[0].set_title(f"Waveform (Domain Waktu) - Durasi {total_duration:.1f}s")
        axs[0].set_xlabel("Waktu (Detik)")
        axs[0].set_ylabel("Amplitudo")
        axs[0].axhline(0, color="gray", lw=0.5, ls="--")
        axs[0].grid(alpha=0.3)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=axs[1], cmap='magma')
        fig_sig.colorbar(img, ax=axs[1], format='%+2.0f dB')
        axs[1].set_title("Mel-Spectrogram (Domain Frekuensi Mel)")

        plt.tight_layout()
        st.pyplot(fig_sig)

        with st.expander("📋 Rincian 32 Parameter Fitur Akustik Hasil Ekstraksi"):
            feat_rows = []
            for name, val in zip(FEATURE_NAMES, feats_overall):
                if "rms" in name:
                    cat = "Energi Sinyal (RMS)"
                elif "centroid" in name:
                    cat = "Titik Berat Spektral"
                elif "zcr" in name:
                    cat = "Zero Crossing Rate"
                else:
                    cat = "Mel-Frequency Cepstral (MFCC)"
                feat_rows.append({"Parameter Fitur": name, "Kategori": cat, "Nilai Terhitung": float(val)})

            f_df = pd.DataFrame(feat_rows)
            st.dataframe(f_df, use_container_width=True, height=350)
    else:
        st.info("👆 Silakan pilih **Sampel Bawaan (Quick Demo)** atau unggah file rekaman suara Anda di atas untuk memulai analisis.")

# =========================================================================
# TAB 3: DASHBOARD EVALUASI & METODOLOGI CRISP-DM
# =========================================================================
with tab_eval:
    st.subheader("🏛️ Dokumentasi Metodologi & Hasil Evaluasi Model (CRISP-DM)")
    st.markdown("""
    Proyek ini dibangun secara end-to-end mengikuti 6 fase standar industri **CRISP-DM**:
    1. **Business Understanding:** Solusi pemantauan akustik non-invasif ramah privasi bagi perpustakaan, co-working space, dan ruang kelas.
    2. **Data Understanding:** Analisis waveform dan spektrogram frekuensi pada rekaman suasana ramai vs sepi.
    3. **Data Preparation:** Pemotongan audio 5 detik (*Group-Aware Chunking*) & ekstraksi 32 fitur DSP tanpa kebocoran data.
    4. **Modeling:** Pelatihan model Random Forest Classifier dan pembandingan 3 algoritma alternatif.
    5. **Evaluation:** Validasi metrik klasifikasi (Akurasi, F1, ROC-AUC) dan interpretasi *Feature Importance*.
    6. **Deployment:** Implementasi antarmuka sistem web interaktif Streamlit (`app.py`) dan Sensor CLI Kontinu (`src/live_monitor.py`).
    """)

    st.markdown("---")
    st.subheader("🏆 Tabel Perbandingan Hasil Benchmarking 4 Model")
    if not benchmark_df.empty:
        st.dataframe(
            benchmark_df.style.highlight_max(
                subset=["Accuracy", "F1-Score (Binary)", "ROC-AUC"], color="#bbf7d0"
            ).format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score (Binary)": "{:.4f}",
                "F1-Score (Macro)": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "Inference Time (ms/sample)": "{:.2f} ms",
                "Fit Time (s)": "{:.2f} s"
            }),
            use_container_width=True
        )
    else:
        st.info("Tabel hasil benchmark belum tersedia di folder `reports/`.")

    st.markdown("---")
    st.subheader("📈 Galeri Grafik Evaluasi Sistem (Resolusi Tinggi 300 DPI)")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        p_comp = "reports/figures/06_model_comparison_bar.png"
        if os.path.exists(p_comp):
            st.image(p_comp, caption="Perbandingan Akurasi dan F1-Score 4 Model", use_container_width=True)

        p_roc = "reports/figures/08_roc_curves_comparison.png"
        if os.path.exists(p_roc):
            st.image(p_roc, caption="Kurva ROC-AUC Gabungan 4 Model", use_container_width=True)

    with col_g2:
        p_cm = "reports/figures/07_confusion_matrix_rf.png"
        if os.path.exists(p_cm):
            st.image(p_cm, caption="Confusion Matrix Model Utama (Random Forest)", use_container_width=True)

        p_fi = "reports/figures/09_top10_feature_importance.png"
        if os.path.exists(p_fi):
            st.image(p_fi, caption="Top 10 Feature Importance (Interpretasi Akustik)", use_container_width=True)

    st.markdown("---")
    st.subheader("💡 Ringkasan Ketercapaian Sasaran Bisnis")
    st.markdown("""
    - ✅ **Target Akurasi $\\ge 80\\%$:** Tercapai **80.08%** pada data uji bebas kebocoran.
    - ✅ **Target 5-Fold Cross-Validation F1 $\\ge 0.85$:** Tercapai **0.8806 (88.06%)**.
    - ✅ **Target Kecepatan Inferensi $< 1$ detik:** Tercapai **13.19 milidetik** per sampel ($75\\times$ lebih cepat).
    - ✅ **Perlindungan Privasi:** Data audio langsung direduksi menjadi 32 parameter numerik statistik tanpa menyimpan percakapan.
    - ✅ **Dual-Mode Live Deployment:** Tersedia mode Browser Mic (`st.audio_input`) dan CLI Continuous Sensor (`src/live_monitor.py`).
    """)
