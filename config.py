"""
config.py — Konfigurasi Terpusat Pipeline FSCA-MedMNIST
==========================================================
Single source of truth untuk semua path, parameter dataset, dan
hyperparameter yang sebelumnya tersebar/diduplikasi di step1–step10.

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
"""

import os

# =========================================================
# PATH DASAR
# =========================================================
# Asumsi struktur folder:
#   project_root/
#     ├── config.py          <- file ini
#     ├── steps/              <- step1_extract.py, step2_preprocess.py, dst.
#     └── outputs/            <- semua hasil (json, png, checkpoint)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")
GRADCAM_DIR  = os.path.join(OUTPUT_DIR, "gradcam")

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
# AUGMENTASI (dipakai Step 2 & Step 3 untuk transform train)
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
# HYPERPARAMETER — STEP 5 (TRAINING AWAL)
# =========================================================
TRAIN_HP = {
    "epochs":       30,
    "batch_size":   64,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "patience":     5,      # early stopping
}


# =========================================================
# HYPERPARAMETER — STEP 6 (FINETUNING)
# =========================================================
FINETUNE_HP = {
    "epochs":          15,
    "batch_size":       32,
    "lr":               1e-4,   # 10x lebih kecil dari TRAIN_HP["lr"]
    "weight_decay":     1e-4,
    "label_smoothing":  0.1,
    "patience":         5,
    # Layer yang di-unfreeze saat finetuning (substring match pada nama parameter)
    "unfreeze_keys": ["layer3", "layer4", "fsca3", "fsca4", "fc"],
}


# =========================================================
# KONFIGURASI MODEL (Step 4 — FSCA ResNet-18)
# =========================================================
MODEL_CONFIG = {
    "backbone":        "resnet18",
    "pretrained":       False,   # True jika ingin mulai dari ImageNet weights
    "dropout_p":        0.4,
    "fsca_reduction":   16,      # reduction ratio untuk ChannelAttention MLP
    "fsca_spatial_kernel": 7,    # kernel size untuk SpatialAttention conv
}


# =========================================================
# DEVICE
# =========================================================
DEVICE_CONFIG = {
    "device": "cpu",   # ganti ke "cuda" jika GPU tersedia
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def ensure_dirs():
    """Pastikan semua folder output yang dibutuhkan sudah ada."""
    os.makedirs(PATHS["output_dir"], exist_ok=True)
    os.makedirs(PATHS["gradcam_dir"], exist_ok=True)
    os.makedirs(PATHS["png_dir"], exist_ok=True)


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
    print(f"  Train epochs : {TRAIN_HP['epochs']}")
    print(f"  Finetune lr  : {FINETUNE_HP['lr']}")
