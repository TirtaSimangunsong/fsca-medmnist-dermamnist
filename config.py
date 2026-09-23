"""
config.py — Konfigurasi Terpusat Pipeline FSCA-MedMNIST
==========================================================
Single source of truth untuk semua path, parameter dataset, dan
hyperparameter yang sebelumnya tersebar/diduplikasi di step1–step11.

CARA PAKAI di setiap step (contoh):
    from config import PATHS, DATASET, TRAIN_HP, FINETUNE_HP

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
GRADCAM_DIR  = os.path.join(OUTPUT_DIR, "gradcam")
ABLATION_DIR = os.path.join(OUTPUT_DIR, "ablation")

PATHS = {
    "output_dir":        OUTPUT_DIR,
    "gradcam_dir":        GRADCAM_DIR,

    # Metadata & config antar-step
    "dataset_meta":       os.path.join(OUTPUT_DIR, "dataset_meta.json"),
    "preprocess_config":  os.path.join(OUTPUT_DIR, "preprocess_config.json"),
    "split_info":         os.path.join(OUTPUT_DIR, "split_info.json"),
    "model_summary":      os.path.join(OUTPUT_DIR, "model_summary.txt"),

    # Checkpoint & model
    "checkpoint_best":    os.path.join(OUTPUT_DIR, "checkpoint_best.pth"),
    "final_model":        os.path.join(OUTPUT_DIR, "ResNet18_FSCA_DermaMNIST_Best.pth"),

    # History training
    "training_history":   os.path.join(OUTPUT_DIR, "training_history.json"),
    "finetune_history":   os.path.join(OUTPUT_DIR, "finetune_history.json"),

    # Evaluasi
    "evaluation_report":  os.path.join(OUTPUT_DIR, "evaluation_report.json"),
    "summary_json":       os.path.join(OUTPUT_DIR, "summary_report.json"),

    # Ablasi multi-seed (Step 11)
    "ablation_dir":       ABLATION_DIR,
    "ablation_report":    os.path.join(OUTPUT_DIR, "ablation_report.json"),
    "ablation_chart":     os.path.join(OUTPUT_DIR, "ablation_comparison.png"),

    # Chart / visualisasi (PNG)
    "split_distribution_chart": os.path.join(OUTPUT_DIR, "split_distribution.png"),
    "training_curve_chart":     os.path.join(OUTPUT_DIR, "training_curve.png"),
    "finetune_curve_chart":     os.path.join(OUTPUT_DIR, "finetune_curve.png"),
    "confusion_matrix_chart":   os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    "gradcam_grid_chart":       os.path.join(OUTPUT_DIR, "gradcam_grid.png"),
    "summary_report_chart":     os.path.join(OUTPUT_DIR, "summary_report.png"),
    "inference_result_chart":   os.path.join(OUTPUT_DIR, "inference_result.png"),

    # Folder PNG terorganisir per step (sesuai konvensi GUI kamu)
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
# HYPERPARAMETER — STEP 5 (TRAINING AWAL / FASE 1)
# =========================================================
TRAIN_HP = {
    "epochs":        100,
    "batch_size":    64,
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "patience":      100,    # = epochs -> early stopping praktis nonaktif
    "seed":          42,     # seed run utama (Step 11 memakai EXPERIMENT["seeds"])
    # Metrik validasi untuk memilih checkpoint terbaik (dan early stopping):
    #   "auc"     -> val AUC macro, makin besar makin baik (protokol resmi MedMNIST)
    #   "loss"    -> val loss, makin kecil makin baik
    #   "bal_acc" -> val balanced accuracy
    #   "acc"     -> val accuracy (perilaku lama; berisik & buta overconfidence)
    "select_metric": "auc",
}


# =========================================================
# HYPERPARAMETER — STEP 6 (FINETUNING / FASE 2)
# =========================================================
FINETUNE_HP = {
    # False -> Step 6 hanya menyalin checkpoint Fase 1 sebagai model final.
    # True  -> fine-tuning dijalankan; model final HANYA diganti bila metrik
    #          validasi (TRAIN_HP["select_metric"]) lebih baik dari Fase 1.
    "enabled":          False,
    "epochs":           10,
    "batch_size":       32,
    "lr":               1e-5,   # 100x lebih kecil dari Fase 1 (Fase 1 sudah konvergen)
    "weight_decay":     1e-4,
    "label_smoothing":  0.1,
    "patience":         5,
    # Modul tingkat-atas yang dilatih saat fine-tuning (dicocokkan persis dengan
    # nama modul paling luar, mis. "layer3.0.conv1" -> "layer3"); sisanya dibekukan
    "unfreeze_keys": ["layer3", "layer4", "fsca3", "fsca4", "fc"],
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
}


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
    # Tahap 1: jawab Rumusan Masalah utama (kontribusi FSCA) dulu.
    # Tahap 2 (opsional, bila waktu cukup): ["none", "se", "eca", "cbam", "fsca"]
    "attentions": ["none", "fsca"],
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
    os.makedirs(PATHS["gradcam_dir"], exist_ok=True)
    os.makedirs(PATHS["png_dir"], exist_ok=True)
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


if __name__ == "__main__":
    # Quick sanity check saat file dijalankan langsung
    ensure_dirs()
    validate_image_size(DATASET["image_size"])
    print("Config OK.")
    print(f"  Output dir   : {PATHS['output_dir']}")
    print(f"  Image size   : {DATASET['image_size']}")
    print(f"  Attention    : {MODEL_CONFIG['attention']}")
    print(f"  Train epochs : {TRAIN_HP['epochs']} (pilih ckpt: {TRAIN_HP['select_metric']})")
    print(f"  Finetune     : {'aktif' if FINETUNE_HP['enabled'] else 'nonaktif'}")
    print(f"  Finetune lr  : {FINETUNE_HP['lr']}")
    print(f"  TTA          : {EVAL_CONFIG['tta']}")
    print(f"  Auto-tune tau: {IMBALANCE['auto_tune_tau']}")