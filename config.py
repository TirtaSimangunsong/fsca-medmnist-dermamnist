"""
config.py — Konfigurasi Terpusat Pipeline FSCA-MedMNIST
==========================================================
Single source of truth untuk semua path, parameter dataset, dan
hyperparameter yang sebelumnya tersebar/diduplikasi di step1-step10.

CARA PAKAI di setiap step (contoh):
    from config import PATHS, DATASET, TRAIN_HP, FINETUNE_HP

    train_ds = DermaMNIST(split="train", download=True, size=DATASET["image_size"])
    optimizer = AdamW(model.parameters(), lr=TRAIN_HP["lr"], weight_decay=TRAIN_HP["weight_decay"])

PENTING - Mengubah image_size BUKAN perubahan ringan:
    Jika kamu ubah DATASET["image_size"] dari 28 ke nilai lain (64/128/224),
    kamu WAJIB menjalankan ulang Step 1 -> Step 2 -> Step 3 secara berurutan,
    karena:
      1. Step 1 mengunduh ulang dataset dengan resolusi baru dari medmnist
         (medmnist menyediakan size: 28, 64, 128, 224 - bukan resize bebas)
      2. Step 2 menghitung ulang mean/std spesifik resolusi tersebut
      3. Arsitektur FSCA (Step 4) tidak berubah karena pakai AdaptiveAvgPool2d,
         tapi waktu training akan naik signifikan di resolusi lebih besar
    Config ini hanya menyentralkan nilainya - bukan menghilangkan kebutuhan
    re-run pipeline saat resolusi berubah.

CHANGELOG (revisi optimasi akurasi):
    - DEVICE_CONFIG["device"] = "auto" -> pakai MPS di Mac Apple Silicon
    - AUGMENTATION diperkuat (RandomResizedCrop, hue jitter, rotasi penuh,
      random erasing) sesuai sifat citra dermoskopi yang tidak punya
      orientasi kanonik
    - IMBALANCE dipisah jadi satu strategi eksplisit supaya tidak lagi
      terjadi koreksi ganda (sampler + weighted loss dipakai bersamaan)
    - MIXUP, EMA, warmup, dan gradient clipping ditambahkan
    - EXPERIMENT untuk multi-seed & ablasi modul atensi
"""

import os

# =========================================================
# PATH DASAR
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")
GRADCAM_DIR  = os.path.join(OUTPUT_DIR, "gradcam")

PATHS = {
    "output_dir":        OUTPUT_DIR,
    "gradcam_dir":       GRADCAM_DIR,

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

    # Eksperimen ablasi / multi-seed (Step 11)
    "ablation_report":    os.path.join(OUTPUT_DIR, "ablation_report.json"),
    "ablation_dir":       os.path.join(OUTPUT_DIR, "ablation"),
    "ablation_chart":     os.path.join(OUTPUT_DIR, "ablation_comparison.png"),

    # Chart / visualisasi (PNG)
    "split_distribution_chart": os.path.join(OUTPUT_DIR, "split_distribution.png"),
    "training_curve_chart":     os.path.join(OUTPUT_DIR, "training_curve.png"),
    "finetune_curve_chart":     os.path.join(OUTPUT_DIR, "finetune_curve.png"),
    "confusion_matrix_chart":   os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    "gradcam_grid_chart":       os.path.join(OUTPUT_DIR, "gradcam_grid.png"),
    "summary_report_chart":     os.path.join(OUTPUT_DIR, "summary_report.png"),
    "inference_result_chart":   os.path.join(OUTPUT_DIR, "inference_result.png"),

    # Folder PNG terorganisir per step (sesuai konvensi GUI)
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
    "num_workers":  0,      # di macOS, >0 kadang bermasalah dgn medmnist; naikkan ke 4 jika stabil

    # Folder cache lokal medmnist
    "medmnist_root": os.path.expanduser("~/.medmnist/"),
}


# =========================================================
# AUGMENTASI
# =========================================================
# Catatan ilmiah untuk BAB III:
# Lesi dermoskopi tidak memiliki orientasi kanonik (tidak ada "atas" atau
# "bawah" yang bermakna secara klinis), sehingga augmentasi rotasi penuh dan
# flip pada kedua sumbu adalah transformasi yang label-preserving. Ini berbeda
# dengan citra natural (ImageNet) di mana vertical flip justru merusak semantik.
AUGMENTATION = {
    # RandomResizedCrop mensimulasikan variasi jarak/magnifikasi dermatoskop
    "random_resized_crop": {
        "enabled": True,
        "scale":   [0.70, 1.00],
        "ratio":   [0.85, 1.18],
    },
    "horizontal_flip_p":  0.5,
    "vertical_flip_p":    0.5,
    "rotation_degrees":   180,      # naik dari 15 -> 180 (lihat catatan di atas)
    "color_jitter": {
        "brightness": 0.25,
        "contrast":   0.25,
        "saturation": 0.25,
        "hue":        0.05,         # warna = petunjuk diagnostik, jitter dibuat kecil
    },
    "random_erasing_p":   0.25,     # oklusi kecil, memaksa model tidak bergantung 1 patch
}


# =========================================================
# PENANGANAN CLASS IMBALANCE
# =========================================================
# PENTING: pilih SATU strategi. Versi lama menjalankan "sampler" dan
# "weighted_loss" bersamaan sehingga kelas minoritas dibobot dua kali
# (over-correction) dan akurasi keseluruhan turun tajam.
#
#   "none"               : CrossEntropy biasa. Akurasi tertinggi, sensitivity
#                          kelas minoritas terendah.
#   "sqrt_weighted_loss" : bobot ~ 1/sqrt(n_c). Kompromi terbaik. DEFAULT.
#   "weighted_loss"       : bobot ~ 1/n_c (WCE penuh, sesuai BAB III versi awal).
#                          Sensitivity naik, akurasi turun paling banyak.
#   "sampler"             : WeightedRandomSampler, loss tanpa bobot.
#
# logit_adjustment_tau: koreksi post-hoc saat inferensi (Menon et al., 2021),
#   logit_c <- logit_c - tau * log(prior_c). tau=0 berarti nonaktif.
#   Step 7 akan mencari tau optimal di validation set secara otomatis.
IMBALANCE = {
    "strategy":             "sqrt_weighted_loss",
    "logit_adjustment_tau": 0.0,
    "auto_tune_tau":        True,   # step7 menyapu tau di val set
}


# =========================================================
# MIXUP / CUTMIX
# =========================================================
MIXUP = {
    "enabled":      True,
    "mixup_alpha":  0.2,
    "cutmix_alpha": 1.0,
    "prob":         0.5,   # peluang sebuah batch dimix sama sekali
    "switch_prob":  0.5,   # jika dimix, peluang pakai CutMix (vs MixUp)
}


# =========================================================
# HYPERPARAMETER - STEP 5 (TRAINING UTAMA / PHASE 1)
# =========================================================
TRAIN_HP = {
    "epochs":            150,     # naik dari 30; cosine butuh horizon panjang
    "batch_size":        128,     # naik dari 64
    "lr":                1e-3,    # LR untuk modul baru (stem, FSCA, fc)
    "backbone_lr_mult":  0.1,     # backbone pretrained pakai lr * 0.1
    "weight_decay":      5e-2,    # AdamW butuh wd jauh lebih besar dari 1e-4
    "warmup_epochs":     5,
    "label_smoothing":   0.1,
    "grad_clip":         1.0,
    "ema_decay":         0.999,   # 0 = matikan EMA
    "patience":          40,      # early stopping longgar; cosine perlu selesai
    "monitor":           "balanced_acc",   # acc | balanced_acc | auc
    "seed":              42,
}


# =========================================================
# HYPERPARAMETER - STEP 6 (FINETUNING / PHASE 2 - POLISH)
# =========================================================
# Phase 2 tidak lagi membekukan layer. Membekukan backbone pada citra 28x28
# dengan stem yang diganti justru merugikan: stem baru diinisialisasi ulang,
# jadi fitur backbone tidak pernah menyesuaikan diri dengan distribusi input.
# Yang dilakukan di sini: LR kecil, MixUp dimatikan, augmentasi diperlembut,
# supaya model "mendarat" bersih di minimum yang sudah ditemukan Phase 1.
FINETUNE_HP = {
    "epochs":          40,
    "batch_size":      128,
    "lr":              1e-4,
    "backbone_lr_mult": 0.5,
    "weight_decay":    1e-2,
    "warmup_epochs":   0,
    "label_smoothing": 0.1,
    "grad_clip":       1.0,
    "ema_decay":       0.999,
    "patience":        15,
    "monitor":         "balanced_acc",
    "mixup":           False,        # matikan mixup di fase polish
    "aug_strength":    0.5,          # skala augmentasi (0.5 = separuh dari Phase 1)
    "freeze":          False,        # set True + unfreeze_keys utk kembali ke perilaku lama
    "unfreeze_keys":   ["layer3", "layer4", "fsca3", "fsca4", "fc"],
}


# =========================================================
# KONFIGURASI MODEL (Step 4)
# =========================================================
MODEL_CONFIG = {
    "backbone":            "resnet18",
    "pretrained":          True,
    "attention":           "fsca",   # fsca | cbam | se | eca | none
    "dropout_p":           0.3,      # turun dari 0.4 (mixup sudah meregularisasi)
    "fsca_reduction":      8,        # turun dari 16; 64 kanal / 16 = terlalu sempit
    "fsca_spatial_kernel": 7,
    "attention_stages":    [1, 2, 3, 4],   # stage mana yang dipasangi modul atensi
    "zero_init_attention": True,     # modul atensi mulai dari identitas -> training stabil
}


# =========================================================
# EVALUASI
# =========================================================
EVAL_CONFIG = {
    "tta":       True,    # test-time augmentation 8-arah (4 rotasi x 2 flip)
    "batch_size": 256,
}


# =========================================================
# EKSPERIMEN (Step 11 - multi-seed & ablasi)
# =========================================================
# Selisih FSCA vs CBAM biasanya < 1%. Tanpa mean +/- std dari beberapa seed,
# klaim keunggulan FSCA tidak bisa dipertahankan di sidang.
EXPERIMENT = {
    "seeds":      [42, 1337, 2024],
    "attentions": ["none", "se", "eca", "cbam", "fsca"],
    "epochs":     100,    # boleh lebih pendek dari TRAIN_HP untuk ablasi
}


# =========================================================
# DEVICE
# =========================================================
DEVICE_CONFIG = {
    # "auto" -> mps (Apple Silicon) > cuda > cpu
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
    """Validasi bahwa image_size sesuai pilihan resmi medmnist."""
    valid_sizes = (28, 64, 128, 224)
    if size not in valid_sizes:
        raise ValueError(
            f"image_size={size} tidak valid. "
            f"medmnist hanya menyediakan resolusi: {valid_sizes}. "
            f"Jika resolusi berubah, jalankan ulang Step 1 -> Step 2 -> Step 3."
        )
    return True


if __name__ == "__main__":
    ensure_dirs()
    validate_image_size(DATASET["image_size"])
    print("Config OK.")
    print(f"  Output dir     : {PATHS['output_dir']}")
    print(f"  Image size     : {DATASET['image_size']}")
    print(f"  Attention      : {MODEL_CONFIG['attention']}")
    print(f"  Pretrained     : {MODEL_CONFIG['pretrained']}")
    print(f"  Imbalance      : {IMBALANCE['strategy']}")
    print(f"  Train epochs   : {TRAIN_HP['epochs']}")
    print(f"  Finetune lr    : {FINETUNE_HP['lr']}")