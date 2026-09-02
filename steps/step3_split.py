"""
Step 3 - Split Data
=====================
Menggunakan split resmi DermaMNIST (train/val/test) dengan transform
yang sudah dikonfigurasi di Step 2. Menyimpan info split + class weights
untuk menangani class imbalance saat training.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, DATASET, AUGMENTATION, ensure_dirs

OUTPUT_DIR   = PATHS["output_dir"]
META_PATH    = PATHS["dataset_meta"]
CONFIG_PATH  = PATHS["preprocess_config"]
SPLIT_PATH   = PATHS["split_info"]

# Path chart yang akan ditampilkan di GUI
CHART_PATH   = PATHS["split_distribution_chart"]


def run(log_fn):
    """
    Membangun DataLoader train/val/test dan menghitung class weights
    untuk WeightedRandomSampler (menangani imbalance DermaMNIST).
    """
    try:
        import torch
        from torchvision import transforms
        from medmnist import DermaMNIST
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError as e:
        log_fn(f"ERROR: {e}")
        raise

    for path, name in [(META_PATH, "dataset_meta.json"), (CONFIG_PATH, "preprocess_config.json")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {name} tidak ditemukan. Jalankan Step sebelumnya.")
            raise FileNotFoundError(path)

    with open(META_PATH)   as f: meta   = json.load(f)
    with open(CONFIG_PATH) as f: config = json.load(f)

    mean = config["mean"]
    std  = config["std"]
    img_size = DATASET["image_size"]
    aug = AUGMENTATION

    # ----- Build transforms -----
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
        transforms.RandomVerticalFlip(p=aug["vertical_flip_p"]),
        transforms.RandomRotation(degrees=aug["rotation_degrees"]),
        transforms.ColorJitter(
            brightness=aug["color_jitter"]["brightness"],
            contrast=aug["color_jitter"]["contrast"],
            saturation=aug["color_jitter"]["saturation"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    log_fn("Memuat split resmi DermaMNIST (train / val / test)...")
    train_ds = DermaMNIST(split="train", transform=train_transform, download=DATASET["download"], size=img_size)
    val_ds   = DermaMNIST(split="val",   transform=eval_transform,  download=DATASET["download"], size=img_size)
    test_ds  = DermaMNIST(split="test",  transform=eval_transform,  download=DATASET["download"], size=img_size)

    n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)
    n_total = n_train + n_val + n_test
    log_fn(f"Train : {n_train:,}  ({100*n_train/n_total:.1f}%)")
    log_fn(f"Val   : {n_val:,}  ({100*n_val/n_total:.1f}%)")
    log_fn(f"Test  : {n_test:,}  ({100*n_test/n_total:.1f}%)")

    # ----- Hitung class weights -----
    labels = np.array([train_ds[i][1].item() for i in range(n_train)])
    class_map = meta["class_map"]
    n_classes = meta["n_classes"]

    class_counts = np.bincount(labels, minlength=n_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights /= class_weights.sum()   # normalize
    sample_weights = class_weights[labels]

    # TAMBAHAN: hitung loss_class_weights (skala berbeda, untuk CrossEntropyLoss)
    loss_class_weights = 1.0 / (class_counts + 1e-6)
    loss_class_weights = loss_class_weights / loss_class_weights.mean()  # rata-rata = 1

    log_fn("")
    log_fn("Distribusi kelas di Training Set:")
    for idx in range(n_classes):
        name = class_map[str(idx)]
        log_fn(f"  [{idx}] {name:<40}: {class_counts[idx]:>5} sampel  (weight: {class_weights[idx]:.4f})")

    # ----- Buat chart distribusi -----
    log_fn("Membuat chart distribusi kelas...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a1a")

    colors = plt.cm.Set2.colors
    names  = [class_map[str(i)] for i in range(n_classes)]

    # Pie chart
    ax1 = axes[0]
    ax1.set_facecolor("#1a1a1a")
    wedges, texts, autotexts = ax1.pie(
        class_counts, labels=names, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": "white", "fontsize": 8},
    )
    ax1.set_title("Distribusi Kelas (Train)", color="white", fontsize=11, fontweight="bold")

    # Bar chart
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a1a")
    bars = ax2.bar(names, class_counts, color=colors[:n_classes], edgecolor="#333")
    ax2.set_title("Jumlah Sampel per Kelas (Train)", color="white", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Kelas", color="white")
    ax2.set_ylabel("Jumlah Sampel", color="white")
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("#555")
    ax2.spines["left"].set_color("#555")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for spine in ["bottom", "left"]: ax2.spines[spine].set_color("#555")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", color="white", fontsize=7)
    for bar, cnt in zip(bars, class_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(cnt), ha="center", va="bottom", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()

    # ----- Simpan info split -----
    split_info = {
    "n_train":             n_train,
    "n_val":                n_val,
    "n_test":               n_test,
    "class_counts":         class_counts.tolist(),
    "class_weights":        class_weights.tolist(),
    "sample_weights":       sample_weights.tolist(),   # boleh dibiarkan tersimpan meski tak dipakai lagi
    "loss_class_weights":   loss_class_weights.tolist(),  # <- key yang dicari Step 5
    "chart_path":           CHART_PATH,
    }
    with open(SPLIT_PATH, "w") as f:
        json.dump(split_info, f, indent=2)

    log_fn(f"Split info tersimpan → {SPLIT_PATH}")
    log_fn(f"Chart tersimpan      → {CHART_PATH}")
    return split_info