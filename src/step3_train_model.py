import os
import sys
import time
import json
import warnings
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.feature_extractor import FEATURE_NAMES
except ImportError:
    from feature_extractor import FEATURE_NAMES

def train_and_benchmark(
    csv_path="data/03_processed/audio_features_32.csv",
    models_dir="models",
    reports_dir="reports",
    figures_dir="reports/figures"
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 65)
    print(" FASE 3 CRISP-DM: MODELING, BENCHMARKING & EVALUASI")
    print("=" * 65)

    # 1. Load Data
    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES].values
    y = df["label"].values
    groups = df["parent_file"].values

    print(f"Dataset: {X.shape[0]} sampel baris, {X.shape[1]} kolom fitur.")
    print(f"Total parent files unik: {len(np.unique(groups))}")

    # 2. GroupShuffleSplit (Anti-Leakage Split 80:20)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    # Validasi Ketat Anti-Leakage
    overlap = set(groups_train).intersection(set(groups_test))
    assert len(overlap) == 0, f"DATA LEAKAGE TERDETEKSI! Overlap: {overlap}"
    assert set(groups_train).isdisjoint(set(groups_test)), "Group train dan test harus saling lepas (disjoint)!"

    print("\n[VALIDASI ANTI-LEAKAGE SUKSES - 100% BEBAS KEBOCORAN]")
    print(f" - Train : {len(X_train)} chunks dari {len(np.unique(groups_train))} file induk (Tidak Ramai: {np.sum(y_train==0)}, Ramai: {np.sum(y_train==1)})")
    print(f" - Test  : {len(X_test)} chunks dari {len(np.unique(groups_test))} file induk (Tidak Ramai: {np.sum(y_test==0)}, Ramai: {np.sum(y_test==1)})")

    # 3. Standarisasi Fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    # 4. Inisialisasi Model
    models = {
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
            "use_scaled": False
        },
        "Support Vector Machine": {
            "model": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
            "use_scaled": True
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            "use_scaled": False
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "use_scaled": True
        }
    }

    # 5. Cross-Validation Random Forest pada Data Latih
    print("\n--- Evaluasi 5-Fold GroupKFold Cross Validation (Random Forest) ---")
    gkf = GroupKFold(n_splits=5)
    rf_cv_scores = cross_val_score(models["Random Forest"]["model"], X_train, y_train, groups=groups_train, cv=gkf, scoring="f1")
    print(f"RF CV F1-Scores per Fold : {[round(s, 4) for s in rf_cv_scores]}")
    print(f"RF Mean CV F1-Score      : {np.mean(rf_cv_scores):.4f} (+/- {np.std(rf_cv_scores):.4f})")

    # 6. Training & Benchmarking pada Test Set
    benchmark_records = []
    trained_models = {}
    test_predictions = {}
    test_probabilities = {}

    print("\n" + "=" * 65)
    print(" HASIL EVALUASI 4 MODEL CLASSIFIER PADA TEST SET")
    print("=" * 65)

    for name, cfg in models.items():
        clf = cfg["model"]
        use_scaled = cfg["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        # Fit model
        t_fit_start = time.time()
        clf.fit(X_tr, y_train)
        fit_time = time.time() - t_fit_start

        # Prediksi
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]

        # Waktu inferensi per sampel (ms) - gunakan single thread untuk inferensi 1 sampel
        original_n_jobs = getattr(clf, "n_jobs", None)
        if original_n_jobs is not None:
            clf.n_jobs = 1

        sample_input = X_te[:1]
        for _ in range(50):
            _ = clf.predict(sample_input)  # warmup

        n_iters = 300
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = clf.predict(sample_input)
        avg_infer_ms = (time.perf_counter() - t0) / n_iters * 1000.0

        if original_n_jobs is not None:
            clf.n_jobs = original_n_jobs

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1_bin = f1_score(y_test, y_pred, zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        trained_models[name] = clf
        test_predictions[name] = y_pred
        test_probabilities[name] = y_proba

        record = {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score (Binary)": f1_bin,
            "F1-Score (Macro)": f1_macro,
            "ROC-AUC": roc_auc,
            "Inference Time (ms/sample)": avg_infer_ms,
            "Fit Time (s)": fit_time
        }
        benchmark_records.append(record)

        print(f"[{name}]")
        print(f"  Accuracy : {acc*100:.2f}% | F1 (Binary): {f1_bin:.4f} | F1 (Macro): {f1_macro:.4f}")
        print(f"  Precision: {prec*100:.2f}% | Recall     : {rec*100:.2f}% | ROC-AUC   : {roc_auc:.4f}")
        print(f"  Waktu Inferensi: {avg_infer_ms:.2f} ms/sampel (< 1.000 ms)\n")

    # Simpan hasil benchmark
    benchmark_df = pd.DataFrame(benchmark_records).sort_values(by="F1-Score (Binary)", ascending=False).reset_index(drop=True)
    benchmark_csv_path = os.path.join(reports_dir, "model_benchmark_results.csv")
    benchmark_df.to_csv(benchmark_csv_path, index=False)
    print(f"[SUKSES] Benchmark tersimpan di: {benchmark_csv_path}")

    # Serialisasi Model
    rf_model = trained_models["Random Forest"]
    best_model_name = benchmark_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    joblib.dump(rf_model, os.path.join(models_dir, "rf_audio_classifier.pkl"))
    joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))

    rf_record = next(r for r in benchmark_records if r["Model"] == "Random Forest")
    metadata = {
        "model_name": "Random Forest Classifier",
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "sample_rate": 16000,
        "chunk_duration_sec": 5.0,
        "test_metrics": {
            "accuracy": float(rf_record["Accuracy"]),
            "precision": float(rf_record["Precision"]),
            "recall": float(rf_record["Recall"]),
            "f1_score_binary": float(rf_record["F1-Score (Binary)"]),
            "f1_score_macro": float(rf_record["F1-Score (Macro)"]),
            "roc_auc": float(rf_record["ROC-AUC"]),
            "inference_time_ms": float(rf_record["Inference Time (ms/sample)"])
        },
        "target_classes": {
            0: "tidak_ramai",
            1: "ramai"
        },
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model_overall": best_model_name,
        "anti_leakage_strategy": "GroupShuffleSplit (groups=parent_file)"
    }

    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[SUKSES] Artefak model & metadata tersimpan di '{models_dir}/'")

    # Top 10 Feature Importance
    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": importances,
        "Importance (%)": importances * 100
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    rf_cm = confusion_matrix(y_test, test_predictions["Random Forest"])

    # Visualisasi
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.dpi": 300
    })

    # 1. 06_model_comparison_bar.png
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(benchmark_df))
    width = 0.35

    r1 = ax.bar(x - width/2, benchmark_df["Accuracy"] * 100, width, label="Akurasi (%)", color="#3b82f6", edgecolor="black", linewidth=0.8)
    r2 = ax.bar(x + width/2, benchmark_df["F1-Score (Binary)"] * 100, width, label="F1-Score (%)", color="#10b981", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Nilai Evaluasi (%)", fontweight="bold")
    ax.set_title("Benchmarking 4 Algoritma Klasifikasi Deteksi Keramaian Audio", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmark_df["Model"], fontweight="bold")
    ax.legend(frameon=True, facecolor="white")
    ax.set_ylim(0, 105)
    ax.axhline(85, color="red", ls="--", lw=1.2, label="Target Bisnis (85%)")

    for rect in r1 + r2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p1 = os.path.join(figures_dir, "06_model_comparison_bar.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[OK] Disimpan: {p1}")

    # 2. 07_confusion_matrix_rf.png
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 16, "fontweight": "bold"},
                xticklabels=["Tidak Ramai (0)", "Ramai (1)"],
                yticklabels=["Tidak Ramai (0)", "Ramai (1)"])
    ax.set_xlabel("Prediksi Model", fontweight="bold", labelpad=10)
    ax.set_ylabel("Label Sebenarnya (Ground Truth)", fontweight="bold", labelpad=10)
    ax.set_title(f"Confusion Matrix: Random Forest\n(Akurasi: {rf_record['Accuracy']*100:.2f}% | F1: {rf_record['F1-Score (Binary)']:.4f})", pad=15)

    plt.tight_layout()
    p2 = os.path.join(figures_dir, "07_confusion_matrix_rf.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[OK] Disimpan: {p2}")

    # 3. 08_roc_curves_comparison.png
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "Random Forest": "#dc2626",
        "Support Vector Machine": "#2563eb",
        "Gradient Boosting": "#059669",
        "Logistic Regression": "#d97706"
    }

    for name in models.keys():
        fpr, tpr, _ = roc_curve(y_test, test_probabilities[name])
        auc_val = roc_auc_score(y_test, test_probabilities[name])
        ax.plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {auc_val:.4f})", color=colors[name])

    ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random Chance (AUC = 0.50)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    ax.set_title("Kurva Karakteristik Operasi Penerima (ROC Curves Comparison)", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p3 = os.path.join(figures_dir, "08_roc_curves_comparison.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"[OK] Disimpan: {p3}")

    # 4. 09_top10_feature_importance.png
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    top10 = feat_imp_df.head(10).iloc[::-1]

    bars = ax.barh(top10["Feature"], top10["Importance (%)"], color="#0ea5e9", edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Tingkat Kepentingan Relatif / Feature Importance (%)", fontweight="bold")
    ax.set_title("Top 10 Fitur Akustik Paling Berpengaruh (Random Forest Classifier)", pad=12)
    ax.set_xlim(0, max(top10["Importance (%)"]) * 1.25)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f}%",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, fontweight="bold", color="#0369a1")

    plt.tight_layout()
    p4 = os.path.join(figures_dir, "09_top10_feature_importance.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"[OK] Disimpan: {p4}")

    return benchmark_df, rf_cm, feat_imp_df

if __name__ == "__main__":
    train_and_benchmark()
