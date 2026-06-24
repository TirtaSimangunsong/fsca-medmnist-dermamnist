"""
Step 2 - Preprocessing: Normalisasi & Augmentasi
==================================================
Mendefinisikan transform pipeline (augmentasi train, normalisasi val/test)
dan menghitung mean/std dataset untuk normalisasi yang akurat.
Hasilnya disimpan ke outputs/preprocess_config.json.
"""

import os
import json
import numpy as np

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
META_PATH    = os.path.join(OUTPUT_DIR, "dataset_meta.json")
CONFIG_PATH  = os.path.join(OUTPUT_DIR, "preprocess_config.json")


def run(log_fn):
    """
    Entry point dipanggil GUI.
    Menghitung mean/std channel-wise dari training set lalu menyimpan config.
    """
    try:
        import torch
        from torchvision import transforms
        from medmnist import DermaMNIST
    except ImportError as e:
        log_fn(f"ERROR: {e}")
        log_fn("Pastikan torch, torchvision, dan medmnist sudah terinstall.")
        raise

    # Pastikan Step 1 sudah dijalankan
    if not os.path.exists(META_PATH):
        log_fn("ERROR: dataset_meta.json tidak ditemukan. Jalankan Step 1 terlebih dahulu.")
        raise FileNotFoundError(META_PATH)

    with open(META_PATH) as f:
        meta = json.load(f)

    log_fn("Memuat dataset mentah untuk menghitung statistik normalisasi...")

    # Load tanpa augmentasi dulu untuk hitung mean/std
    raw_transform = transforms.Compose([transforms.ToTensor()])
    train_raw = DermaMNIST(split="train", transform=raw_transform, download=True, size=28)

    loader = torch.utils.data.DataLoader(train_raw, batch_size=256, num_workers=0)

    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n    = 0

    log_fn("Menghitung mean per channel (R, G, B)...")
    for imgs, _ in loader:
        # imgs shape: (B, C, H, W)
        mean += imgs.mean(dim=[0, 2, 3]) * imgs.size(0)
        n    += imgs.size(0)
    mean /= n

    log_fn("Menghitung std per channel (R, G, B)...")
    for imgs, _ in loader:
        std += ((imgs - mean[None, :, None, None]) ** 2).mean(dim=[0, 2, 3]) * imgs.size(0)
    std = (std / n).sqrt()

    mean_list = mean.tolist()
    std_list  = std.tolist()

    log_fn(f"Mean (R,G,B) : {[round(v,4) for v in mean_list]}")
    log_fn(f"Std  (R,G,B) : {[round(v,4) for v in std_list]}")

    # Definisi augmentasi training
    train_aug = [
        "RandomHorizontalFlip(p=0.5)",
        "RandomVerticalFlip(p=0.5)",
        "RandomRotation(degrees=15)",
        "ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)",
        "ToTensor()",
        f"Normalize(mean={[round(v,4) for v in mean_list]}, std={[round(v,4) for v in std_list]})",
    ]
    val_aug = [
        "ToTensor()",
        f"Normalize(mean={[round(v,4) for v in mean_list]}, std={[round(v,4) for v in std_list]})",
    ]

    log_fn("")
    log_fn("Pipeline Augmentasi Training:")
    for t in train_aug:
        log_fn(f"  → {t}")
    log_fn("Pipeline Augmentasi Val/Test:")
    for t in val_aug:
        log_fn(f"  → {t}")

    config = {
        "mean":      mean_list,
        "std":       std_list,
        "train_aug": train_aug,
        "val_aug":   val_aug,
        "image_size": 28,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    log_fn(f"Config preprocessing tersimpan → {CONFIG_PATH}")
    return config
