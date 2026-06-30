"""
Step 5 - Training
==================
Training loop lengkap dengan:
- WeightedRandomSampler untuk menangani class imbalance
- AdamW optimizer + CosineAnnealingLR scheduler
- Early stopping (patience=5)
- Checkpoint model terbaik (val accuracy)
- Plot loss & accuracy curve

Output: outputs/checkpoint_best.pth, outputs/training_curve.png
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, DATASET, AUGMENTATION, TRAIN_HP

OUTPUT_DIR    = PATHS["output_dir"]
META_PATH     = PATHS["dataset_meta"]
CONFIG_PATH   = PATHS["preprocess_config"]
SPLIT_PATH    = PATHS["split_info"]
CKPT_PATH     = PATHS["checkpoint_best"]
HISTORY_PATH  = PATHS["training_history"]
CURVE_PATH    = PATHS["training_curve_chart"]

# Hyperparameter default kini berasal dari config.py (TRAIN_HP).
# Tetap diekspos sebagai DEFAULT_CONFIG agar kode lama yang mereferensikan
# nama ini (mis. step lain atau notebook) tidak rusak.
DEFAULT_CONFIG = TRAIN_HP


def _build_loaders(meta, config, split_info):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import transforms
    from medmnist import DermaMNIST

    mean = config["mean"]
    std  = config["std"]
    bs   = DEFAULT_CONFIG["batch_size"]
    img_size = DATASET["image_size"]
    aug = AUGMENTATION

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

    train_ds = DermaMNIST(split="train", transform=train_transform, download=DATASET["download"], size=img_size)
    val_ds   = DermaMNIST(split="val",   transform=eval_transform,  download=DATASET["download"], size=img_size)

    # WeightedRandomSampler — atasi class imbalance
    sample_weights = torch.tensor(split_info["sample_weights"], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,  num_workers=0)
    return train_loader, val_loader


def _plot_curves(history, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a1a")

    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_color("#555")

    # Loss
    axes[0].plot(epochs, history["train_loss"], color="#4fc3f7", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   color="#ef5350", label="Val Loss",   linewidth=2)
    axes[0].set_title("Training & Validation Loss", color="white", fontweight="bold")
    axes[0].set_xlabel("Epoch", color="white")
    axes[0].set_ylabel("Loss", color="white")
    axes[0].legend(facecolor="#333", labelcolor="white")

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], color="#4fc3f7", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"],   color="#ef5350", label="Val Acc",   linewidth=2)
    axes[1].set_title("Training & Validation Accuracy", color="white", fontweight="bold")
    axes[1].set_xlabel("Epoch", color="white")
    axes[1].set_ylabel("Accuracy (%)", color="white")
    axes[1].legend(facecolor="#333", labelcolor="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()


def run(log_fn, hp_override=None):
    """
    hp_override: dict optional dari Step 6 untuk hyperparameter tuning.
    """
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # Pastikan step sebelumnya sudah dijalankan
    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2"), (SPLIT_PATH, "Step 3")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} terlebih dahulu.")
            raise FileNotFoundError(path)

    with open(META_PATH)   as f: meta       = json.load(f)
    with open(CONFIG_PATH) as f: config     = json.load(f)
    with open(SPLIT_PATH)  as f: split_info = json.load(f)

    # Merge hyperparameter
    hp = {**DEFAULT_CONFIG, **(hp_override or {})}
    n_classes = meta["n_classes"]

    log_fn(f"Hyperparameter Training:")
    for k, v in hp.items(): log_fn(f"  {k:<15}: {v}")
    log_fn("")

    # Build model
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model

    device = torch.device("cpu")
    log_fn(f"Device: {device}")

    model = build_model(n_classes=n_classes, pretrained=False).to(device)
    log_fn(f"Model berhasil diinisialisasi.")

    # DataLoaders
    log_fn("Memuat DataLoader...")
    train_loader, val_loader = _build_loaders(meta, config, split_info)
    log_fn(f"  Train batches : {len(train_loader)}")
    log_fn(f"  Val batches   : {len(val_loader)}")

    # Loss (CrossEntropy standard — sampler sudah handle imbalance)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=hp["epochs"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc  = 0.0
    patience_cnt  = 0
    best_epoch    = 0

    log_fn("\nMemulai Training Loop...")
    log_fn("-" * 55)

    for epoch in range(1, hp["epochs"] + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.squeeze(1).long().to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            preds          = outputs.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.squeeze(1).long().to(device)
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
                val_loss    += loss.item() * imgs.size(0)
                preds        = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        # Metrics
        t_loss = train_loss / train_total
        v_loss = val_loss   / val_total
        t_acc  = 100 * train_correct / train_total
        v_acc  = 100 * val_correct   / val_total
        elapsed = time.time() - t0

        history["train_loss"].append(round(t_loss, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["train_acc"].append(round(t_acc, 2))
        history["val_acc"].append(round(v_acc, 2))

        marker = ""
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), CKPT_PATH)
            marker = " ✓ SAVED"
        else:
            patience_cnt += 1

        log_fn(
            f"Epoch [{epoch:>3}/{hp['epochs']}] "
            f"Loss: {t_loss:.4f}/{v_loss:.4f}  "
            f"Acc: {t_acc:.1f}%/{v_acc:.1f}%  "
            f"({elapsed:.0f}s){marker}"
        )

        # Early stopping
        if patience_cnt >= hp["patience"]:
            log_fn(f"\nEarly stopping pada epoch {epoch} (patience={hp['patience']} tercapai).")
            break

    log_fn(f"\nBest Val Accuracy : {best_val_acc:.2f}%  (epoch {best_epoch})")
    log_fn(f"Checkpoint disimpan → {CKPT_PATH}")

    # Simpan history
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    # Plot curves
    log_fn("Membuat plot training curve...")
    _plot_curves(history, CURVE_PATH)
    log_fn(f"Chart tersimpan → {CURVE_PATH}")

    return {
        "best_val_acc": best_val_acc,
        "best_epoch":   best_epoch,
        "history":      history,
        "chart_path":   CURVE_PATH,
    }