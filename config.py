"""
config.py — Konfigurasi Terpusat Pipeline FSCA-MedMNIST
==========================================================
Single source of truth untuk semua path, parameter dataset, dan
hyperparameter yang sebelumnya tersebar/diduplikasi di step1–step11.

CARA PAKAI di setiap step (contoh):
    from config import PATHS, DATASET, TRAIN_HP

    train_ds = DermaMNIST(split="train", download=True, size=DATASET["image_size"])
    optimizer = AdamW(model.parameters(), lr=TRAIN_HP["lr"], weight_decay=TRAIN_HP["weight_decay"])

PENTING — Mengubah image_size BUKAN perubahan ringan:
    Jika kamu ubah DATASET["image_size"] dari 28 ke nilai lain (64/128/224),
    kamu WAJIB menjalankan ulang Step 1 → Step 2 → Step 3 secara berurutan,
    karena:
      1. Step 1 mengunduh ulang dataset dengan resolusi baru dari medmnist
         (medmnist menyediakan size: 28, 64, 128, 224 — bukan resize bebas)
      2. Step 2 menghitung ulang mean/std spesifik resolusi tersebut
      3. Arsitektur FSCA (Step 4) tidak berubah karena pakai AdaptiveAvgPool2d,
         tapi waktu training akan naik signifikan di resolusi lebih besar
    Config ini hanya menyentralkan nilainya — bukan menghilangkan kebutuhan
    re-run pipeline saat resolusi berubah.

PERUBAHAN (finalisasi kode)
----------------------------
- Ditambahkan IMBALANCE, EVAL_CONFIG, EXPERIMENT yang diimpor oleh Step 7
  dan Step 11 tetapi sebelumnya tidak ada di file ini (ImportError).
- Ditambahkan path ablasi (ablation_dir / ablation_report / ablation_chart).
- MODEL_CONFIG["attention"] ditambahkan agar Step 4 bisa membangun
  ResNet-18 dengan modul atensi none / se / eca / cbam / fsca.
- DEVICE_CONFIG["device"] = "auto" (MPS -> CUDA -> CPU), dipakai lewat
  steps/common.py::resolve_device.
- TRAIN_HP["seed"] ditambahkan agar run utama bisa direproduksi.

PERUBAHAN (baseline + tanpa finetune)
--------------------------------------
- Fine-tuning (Fase 2, Step 6 lama) DIHAPUS: FINETUNE_HP, finetune_history,
  dan finetune_curve_chart tidak ada lagi. Model final = checkpoint terbaik
  training FSCA (step5_training.py menyalinnya ke final_model).
- Path baseline ditambahkan (baseline_model, baseline_history,
  baseline_curve_chart, baseline_report, baseline_cm_chart) untuk Step
  "Baseline ResNet-18" (step_baseline.py), yaitu ResNet-18 polos
  (attention="none") dengan resep training yang sama dengan FSCA.

PERUBAHAN (feedback dosen 26 Okt 2026)
---------------------------------------
- TRAIN_HP epochs 100 -> 40 (overfit mulai ~epoch 35).
- IMBALANCE["melanoma_bias"]: bias melanoma post-hoc dituning di validasi.
- IMBALANCE["class_weight_boost"]: opsi pengali bobot loss per kelas.
- MAIN_MODEL + main_model_spec(): pilihan model utama untuk Step 8–10.

PERUBAHAN (finalisasi laporan, Nov 2026)
-----------------------------------------
- MAIN_MODEL = "fsca": FSCA model usulan, baseline pembanding (sesuai laporan).
- EXPERIMENT["attentions"] berisi kelima model (none, se, eca, cbam, fsca)
  untuk Tabel 4.11; run yang sudah selesai dilewati otomatis.
- PATHS["comparison_report"]: tabel baseline vs FSCA dari Step 7.
"""

import os

# MPS (Apple Silicon) belum mendukung semua operator. Variabel ini harus
# diset SEBELUM torch diinisialisasi; config.py selalu diimpor lebih dulu
# oleh setiap step, jadi di sinilah tempat paling aman.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# =========================================================
# PATH DASAR
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")
GRADCAM_DIR  = os.path.join(OUTPUT_DIR, "png", "gradcam")   # Grad-CAM per kelas
ABLATION_DIR = os.path.join(OUTPUT_DIR, "ablation")

PATHS = {
    "output_dir":        OUTPUT_DIR,
    "gradcam_dir":        GRADCAM_DIR,

    # Metadata & config antar-step
    "dataset_meta":       os.path.join(OUTPUT_DIR, "json", "dataset_meta.json"),
    "preprocess_config":  os.path.join(OUTPUT_DIR, "json", "preprocess_config.json"),
    "split_info":         os.path.join(OUTPUT_DIR, "json", "split_info.json"),
    "model_summary":      os.path.join(OUTPUT_DIR, "txt", "model_summary.txt"),

    # Checkpoint & model
    "checkpoint_best":    os.path.join(OUTPUT_DIR, "models", "checkpoint_best.pth"),
    "final_model":        os.path.join(OUTPUT_DIR, "models", "ResNet18_FSCA_DermaMNIST_Best.pth"),

    # Baseline ResNet-18 tanpa modul atensi (step_baseline.py)
    "baseline_model":       os.path.join(OUTPUT_DIR, "models", "ResNet18_Baseline_DermaMNIST_Best.pth"),
    "baseline_history":     os.path.join(OUTPUT_DIR, "json", "baseline_training_history.json"),
    "baseline_curve_chart": os.path.join(OUTPUT_DIR, "png", "baseline_training_curve.png"),
    "baseline_report":      os.path.join(OUTPUT_DIR, "json", "baseline_evaluation_report.json"),
    "baseline_cm_chart":    os.path.join(OUTPUT_DIR, "png", "baseline_confusion_matrix.png"),

    # History training
    "training_history":   os.path.join(OUTPUT_DIR, "json", "training_history.json"),

    # Evaluasi
    "evaluation_report":  os.path.join(OUTPUT_DIR, "json", "evaluation_report.json"),
    "comparison_report":  os.path.join(OUTPUT_DIR, "json", "comparison_report.json"),
    "summary_json":       os.path.join(OUTPUT_DIR, "json", "summary_report.json"),

    # Ablasi multi-seed (Step 11)
    "ablation_dir":       ABLATION_DIR,
    "ablation_report":    os.path.join(OUTPUT_DIR, "json", "ablation_report.json"),
    "ablation_chart":     os.path.join(OUTPUT_DIR, "png", "ablation_comparison.png"),

    # Chart / visualisasi (PNG)
    "split_distribution_chart": os.path.join(OUTPUT_DIR, "png", "split_distribution.png"),
    "training_curve_chart":     os.path.join(OUTPUT_DIR, "png", "training_curve.png"),
    "confusion_matrix_chart":   os.path.join(OUTPUT_DIR, "png", "confusion_matrix.png"),
    "gradcam_grid_chart":       os.path.join(OUTPUT_DIR, "png", "gradcam_grid.png"),
    "summary_report_chart":     os.path.join(OUTPUT_DIR, "png", "summary_report.png"),
    "inference_result_chart":   os.path.join(OUTPUT_DIR, "png", "inference_result.png"),

    # Folder per jenis berkas: models/ json/ png/ txt/ (lihat ensure_dirs)
    "png_dir":            os.path.join(OUTPUT_DIR, "png"),
}


# =========================================================
# KONFIGURASI DATASET
# =========================================================
DATASET = {
    "name":         "dermamnist",
    "image_size":   28,     # WAJIB salah satu dari: 28, 64, 128, 224 (sesuai medmnist)
    "n_channels":   3,      # RGB
    "download":     True,   # auto-download via medmnist (fallback manual ke ~/.medmnist/)

    # Folder cache lokal medmnist (tempat manual download diletakkan
    # jika Zenodo auto-download gagal karena network restriction)
    "medmnist_root": os.path.expanduser("~/.medmnist/"),
}


# =========================================================
# AUGMENTASI (dipakai Step 2, 3, 5, 6 untuk transform train)
# =========================================================
AUGMENTATION = {
    "horizontal_flip_p":  0.5,
    "vertical_flip_p":    0.5,
    "rotation_degrees":   15,
    "color_jitter": {
        "brightness": 0.2,
        "contrast":   0.2,
        "saturation": 0.2,
    },
}


# =========================================================
# HYPERPARAMETER TRAINING
# Dipakai SAMA PERSIS oleh Baseline ResNet-18 (step_baseline.py),
# Training FSCA (step5_training.py), dan ablasi (Step 11).
# =========================================================
TRAIN_HP = {
    # 40 epoch (sebelumnya 100): dengan 100 epoch model mulai overfit sekitar
    # epoch 35 (train acc ~99%, val loss naik 0.95 -> 2.8). Karena CosineAnnealingLR
    # memakai T_max = epochs, LR kini sudah turun ke ~0 di epoch 40.
    "epochs":        40,
    "batch_size":    64,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "patience":      40,     # = epochs -> early stopping praktis nonaktif
    "seed":          42,     # seed run utama (Step 11 memakai EXPERIMENT["seeds"])
    # Metrik validasi untuk memilih checkpoint terbaik (dan early stopping):
    #   "auc"     -> val AUC macro, makin besar makin baik (protokol resmi MedMNIST)
    #   "loss"    -> val loss, makin kecil makin baik
    #   "bal_acc" -> val balanced accuracy
    #   "acc"     -> val accuracy (perilaku lama; berisik & buta overconfidence)
    "select_metric": "auc",
}


# =========================================================
# KONFIGURASI MODEL (Step 4 — ResNet-18 + modul atensi)
# =========================================================
MODEL_CONFIG = {
    "backbone":        "resnet18",
    "pretrained":       True,
    # Modul atensi yang dipasang setelah layer1..layer4.
    # Pilihan: "none" | "se" | "eca" | "cbam" | "fsca"
    "attention":        "fsca",
    "dropout_p":        0.4,
    "fsca_reduction":   16,      # reduction ratio untuk ChannelAttention MLP (juga SE/CBAM)
    "fsca_spatial_kernel": 7,    # kernel size untuk SpatialAttention conv (juga CBAM)
}


# =========================================================
# PENANGANAN CLASS IMBALANCE
# =========================================================
IMBALANCE = {
    # Bobot Weighted Cross-Entropy (dipakai Step 3 & Step 5):
    #   "sqrt"    -> 1/sqrt(n)  (DEFAULT, rasio maks/min ~7.6x)
    #   "inverse" -> 1/n        (WCE penuh, rasio ~58x)
    #   "none"    -> bobot seragam (CE biasa)
    "loss_weight_mode":     "sqrt",

    # Logit adjustment post-hoc saat evaluasi (Step 7):
    #   logit_adj = logit - tau * log(prior_kelas)
    # Jika auto_tune_tau=True, tau dipilih dari tau_grid berdasarkan
    # balanced accuracy di VALIDATION set, lalu dilaporkan di TEST set.
    # Jika auto_tune_tau=False, nilai logit_adjustment_tau dipakai apa adanya
    # (0.0 = tanpa logit adjustment).
    # CATATAN: model dilatih dengan Weighted CE yang SUDAH mengoreksi imbalance,
    # sehingga logit adjustment di atasnya berisiko koreksi ganda. Default: mati.
    "logit_adjustment_tau": 0.0,
    "auto_tune_tau":        False,
    "tau_grid":             [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],

    # --- Fokus kelas melanoma (feedback dosen 26 Okt 2026, poin 1) ---
    # (a) Bias melanoma post-hoc saat evaluasi (Step 4 & Step 7), TANPA training ulang:
    #     logit melanoma + bias. Bias dipilih di VALIDATION: recall melanoma
    #     setinggi mungkin selama balanced accuracy val tidak turun lebih dari
    #     max_bal_acc_drop poin dan accuracy val tidak turun lebih dari
    #     max_acc_drop poin. Lalu dilaporkan di TEST. Angka tanpa bias tetap
    #     disimpan di laporan ("without_tta_or_adjustment" & "melanoma_bias").
    "melanoma_bias": {
        "enabled":          True,
        "class_name":       "melanoma",
        "grid":             [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        "max_bal_acc_drop": 0.0,
        "max_acc_drop":     2.0,
    },
    # (b) Pengali bobot loss per kelas saat TRAINING (opsional, butuh training ulang).
    #     1.0 = mati. Contoh eksperimen: {"melanoma": 1.5}
    "class_weight_boost":   {"melanoma": 1.0},
}


# =========================================================
# MODEL UTAMA (feedback dosen 26 Okt 2026, poin 3)
# =========================================================
# "baseline" -> ResNet-18 tanpa atensi (Step 4) menjadi model utama untuk
#               Grad-CAM (Step 8), Summary (Step 9), dan Inference (Step 10).
#               FSCA dilaporkan sebagai model pembanding (Step 7).
# "fsca"     -> ResNet-18 + FSCA (Step 6) sebagai model utama / model usulan.
# Laporan skripsi (Nov 2026) memposisikan FSCA sebagai model usulan dan
# baseline sebagai pembanding, sehingga default-nya "fsca".
MAIN_MODEL = "fsca"


def main_model_spec():
    """Path dan identitas model utama sesuai MAIN_MODEL."""
    if MAIN_MODEL == "baseline":
        return {"key": "baseline", "attention": "none",
                "label": "ResNet-18 Baseline",
                "model_path": PATHS["baseline_model"],
                "report":     PATHS["baseline_report"],
                "history":    PATHS["baseline_history"],
                "made_by":    "Step 4: Baseline ResNet-18"}
    if MAIN_MODEL == "fsca":
        attn = MODEL_CONFIG["attention"]
        return {"key": "fsca", "attention": attn,
                "label": f"ResNet-18 + {attn.upper()}",
                "model_path": PATHS["final_model"],
                "report":     PATHS["evaluation_report"],
                "history":    PATHS["training_history"],
                "made_by":    "Step 6: Training FSCA"}
    raise ValueError(f"MAIN_MODEL='{MAIN_MODEL}' tidak dikenal. Pilihan: 'baseline' | 'fsca'")


# =========================================================
# EVALUASI (Step 7 & Step 11)
# =========================================================
EVAL_CONFIG = {
    "batch_size": 256,
    # Test-time augmentation 8 arah: rotasi 0/90/180/270 derajat,
    # masing-masing dengan dan tanpa horizontal flip. Logit dirata-rata.
    # Default nonaktif: angka utama tanpa TTA agar sebanding dengan baseline
    # resmi MedMNIST. Hasil run utama: TTA hanya +0.005 AUC, balanced acc -0.7.
    "tta":        False,
}


# =========================================================
# EKSPERIMEN ABLASI MULTI-SEED (Step 11)
# =========================================================
EXPERIMENT = {
    # Kelima model untuk Tabel 4.11 (Rumusan Masalah 1 dan 2). Run yang sudah
    # selesai (none & fsca, ablasi 2 Nov 2026) dilewati dan hanya dievaluasi
    # ulang, jadi yang benar-benar dilatih hanya SE, ECA, dan CBAM.
    # Untuk mengulang perbandingan baseline vs FSCA saja:
    #     python run_ablation.py --attentions none fsca
    "attentions": ["none", "se", "eca", "cbam", "fsca"],
    "seeds":      [0, 1, 2, 3, 4],
    "epochs":     TRAIN_HP["epochs"],
}


# =========================================================
# DEVICE
# =========================================================
DEVICE_CONFIG = {
    # "auto" -> MPS (Apple Silicon) jika ada, lalu CUDA, lalu CPU.
    # Bisa juga dipaksa: "mps" | "cuda" | "cpu"
    "device": "auto",
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def ensure_dirs():
    """Pastikan semua folder output yang dibutuhkan sudah ada."""
    os.makedirs(PATHS["output_dir"], exist_ok=True)
    for sub in ("models", "json", "png", "txt"):
        os.makedirs(os.path.join(PATHS["output_dir"], sub), exist_ok=True)
    os.makedirs(PATHS["gradcam_dir"], exist_ok=True)
    os.makedirs(PATHS["ablation_dir"], exist_ok=True)


def get_png_subdir(step_name):
    """
    Kembalikan path folder PNG khusus untuk step tertentu,
    membuatnya jika belum ada. Konsisten dengan konvensi GUI:
    outputs/png/<step_name>/
    """
    path = os.path.join(PATHS["png_dir"], step_name)
    os.makedirs(path, exist_ok=True)
    return path


def validate_image_size(size):
    """
    Validasi bahwa image_size sesuai pilihan resmi medmnist.
    Panggil ini di awal Step 1 jika kamu mengganti DATASET['image_size'].
    """
    valid_sizes = (28, 64, 128, 224)
    if size not in valid_sizes:
        raise ValueError(
            f"image_size={size} tidak valid. "
            f"medmnist hanya menyediakan resolusi: {valid_sizes}. "
            f"Jika resolusi berubah, jalankan ulang Step 1 -> Step 2 -> Step 3."
        )
    return True


# Folder output dibuat saat config diimpor, supaya step mana pun yang
# dijalankan lebih dulu tidak gagal karena subfolder belum ada.
ensure_dirs()


if __name__ == "__main__":
    # Quick sanity check saat file dijalankan langsung
    ensure_dirs()
    validate_image_size(DATASET["image_size"])
    print("Config OK.")
    print(f"  Output dir   : {PATHS['output_dir']}")
    print(f"  Image size   : {DATASET['image_size']}")
    print(f"  Attention    : {MODEL_CONFIG['attention']}")
    print(f"  Train epochs : {TRAIN_HP['epochs']} (pilih ckpt: {TRAIN_HP['select_metric']})")
    print(f"  TTA          : {EVAL_CONFIG['tta']}")
    print(f"  Auto-tune tau: {IMBALANCE['auto_tune_tau']}")
    print(f"  Bias melanoma: {'aktif' if IMBALANCE['melanoma_bias']['enabled'] else 'nonaktif'}")
    print(f"  Model utama  : {main_model_spec()['label']}")