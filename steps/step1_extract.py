"""
Step 1 - Ekstraksi Dataset DermaMNIST
======================================
Mengunduh DermaMNIST dari library medmnist secara otomatis
dan menyimpan metadata ke file JSON untuk step selanjutnya.
"""

import os
import json
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
META_PATH  = os.path.join(OUTPUT_DIR, "dataset_meta.json")


def run(log_fn):
    """
    Entry point dipanggil oleh GUI.
    log_fn : callable(str) → dikirim ke terminal GUI
    return : dict berisi info dataset
    """
    try:
        import medmnist
        from medmnist import DermaMNIST
    except ImportError:
        log_fn("ERROR: library 'medmnist' belum terinstall.")
        log_fn("Jalankan: pip install medmnist")
        raise

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_fn("Menghubungi server MedMNIST untuk mengunduh DermaMNIST...")
    log_fn(f"Library medmnist versi: {medmnist.__version__}")

    # Download otomatis — medmnist menyimpan ke ~/.medmnist/
    train_ds = DermaMNIST(split="train", download=True, size=28)
    val_ds   = DermaMNIST(split="val",   download=True, size=28)
    test_ds  = DermaMNIST(split="test",  download=True, size=28)

    n_train = len(train_ds)
    n_val   = len(val_ds)
    n_test  = len(test_ds)
    n_total = n_train + n_val + n_test

    # Ambil label info dari medmnist
    info       = medmnist.INFO["dermamnist"]
    class_map  = info["label"]          # {0: "acne", 1: "bcc", ...}
    n_channels = info["n_channels"]
    task       = info["task"]

    # Hitung distribusi kelas di training set
    labels_train = np.array([train_ds[i][1].item() for i in range(n_train)])
    class_counts = {}
    for idx, name in class_map.items():
        count = int(np.sum(labels_train == int(idx)))
        class_counts[name] = count
        log_fn(f"  Kelas [{idx}] {name:<40}: {count:>5} sampel (train)")

    log_fn(f"")
    log_fn(f"Total sampel  : {n_total:,}")
    log_fn(f"  Train       : {n_train:,}")
    log_fn(f"  Val         : {n_val:,}")
    log_fn(f"  Test        : {n_test:,}")
    log_fn(f"Resolusi      : 28x28 piksel, {n_channels} channel")
    log_fn(f"Task          : {task}")
    log_fn(f"Jumlah kelas  : {len(class_map)}")

    # Simpan metadata agar step lain bisa membacanya
    meta = {
        "n_train":      n_train,
        "n_val":        n_val,
        "n_test":       n_test,
        "n_total":      n_total,
        "n_classes":    len(class_map),
        "class_map":    class_map,
        "class_counts": class_counts,
        "n_channels":   n_channels,
        "task":         task,
        "image_size":   28,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    log_fn(f"Metadata tersimpan → {META_PATH}")
    return meta
