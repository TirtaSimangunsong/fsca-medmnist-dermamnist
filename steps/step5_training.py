"""
Step 5 - Training (Fase 1)
============================
Melatih SELURUH model (tidak ada layer yang dibekukan) dari bobot awal
ImageNet, dengan:
- Penanganan class imbalance: Weighted Cross-Entropy. Bobot per kelas
  = 1/sqrt(n_kelas), dinormalisasi agar rata-ratanya 1
  (IMBALANCE["loss_weight_mode"] di config.py).
  TIDAK memakai WeightedRandomSampler; DataLoader train memakai shuffle biasa.
- Augmentasi train: HFlip, VFlip, rotasi, ColorJitter (AUGMENTATION di config.py)
- AdamW optimizer + CosineAnnealingLR scheduler
- Early stopping (TRAIN_HP["patience"])
- Checkpoint model terbaik berdasarkan val accuracy
- Plot loss & accuracy curve

Fungsi train_one_run() juga dipakai Step 11 (ablasi multi-seed).

Output: outputs/checkpoint_best.pth, outputs/training_curve.png,
        outputs/training_history.json
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, TRAIN_HP, MODEL_CONFIG, IMBALANCE

OUTPUT_DIR    = PATHS["output_dir"]
META_PATH     = PATHS["dataset_meta"]
CONFIG_PATH   = PATHS["preprocess_config"]
SPLIT_PATH    = PATHS["split_info"]
CKPT_PATH     = PATHS["checkpoint_best"]
HISTORY_PATH  = PATHS["training_history"]
CURVE_PATH    = PATHS["training_curve_chart"]

# Hyperparameter default berasal dari config.py (TRAIN_HP).
# Tetap diekspos sebagai DEFAULT_CONFIG agar kode lama yang mereferensikan
# nama ini (mis. step lain atau notebook) tidak rusak.
DEFAULT_CONFIG = TRAIN_HP


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


def train_one_run(hp, log_fn, ckpt_path=CKPT_PATH, attention=None, seed=None, tag=None):
    """
    Satu run training Fase 1.

    hp        : dict berisi epochs, batch_size, lr, weight_decay, patience (+ seed opsional)
    ckpt_path : lokasi simpan checkpoint terbaik (val accuracy)
    attention : "none" | "se" | "eca" | "cbam" | "fsca" | None (=MODEL_CONFIG)
    seed      : jika diberikan, mengunci inisialisasi dan urutan data
    tag       : prefix log (dipakai Step 11 agar log tiap run mudah dibedakan)

    Return: dict {best_val_acc, best_epoch, history}
    """
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model
    import common

    prefix = f"[{tag}] " if tag else ""
    seed = seed if seed is not None else hp.get("seed")

    with open(META_PATH)   as f: meta   = json.load(f)
    with open(CONFIG_PATH) as f: config = json.load(f)
    n_classes = meta["n_classes"]

    if seed is not None:
        common.set_seed(seed)
        log_fn(f"{prefix}Seed: {seed}")

    device = common.resolve_device(log_fn=log_fn)

    # DataLoader (split resmi, shuffle biasa, augmentasi di train)
    log_fn(f"{prefix}Memuat DataLoader...")
    loaders, dinfo = common.build_dataloaders(
        mean=config["mean"], std=config["std"],
        batch_size=hp["batch_size"], n_classes=n_classes,
        splits=("train", "val"), train_augment=True, seed=seed, log_fn=log_fn,
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    # Model (bobot awal ImageNet sesuai MODEL_CONFIG["pretrained"])
    model = build_model(n_classes=n_classes,
                        pretrained=MODEL_CONFIG["pretrained"],
                        attention=attention).to(device)
    log_fn(f"{prefix}Model: ResNet-18 + {model.attention_name.upper()} "
           f"(pretrained={MODEL_CONFIG['pretrained']})")

    # Loss: Weighted Cross-Entropy (tanpa WeightedRandomSampler)
    mode = IMBALANCE["loss_weight_mode"]
    weights = common.compute_loss_class_weights(dinfo["class_counts"], mode)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    log_fn(f"{prefix}Loss: Weighted CE (mode={mode}, rasio maks/min "
           f"{weights.max() / weights.min():.1f}x)")

    # Seluruh model dilatih. Skema "Phase 1 head-only" tidak dipakai karena
    # stem sudah diganti (conv 3x3 stride 1), sehingga membekukan backbone
    # justru menghalangi fitur menyesuaikan diri.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_fn(f"{prefix}Melatih seluruh model. Trainable params: {trainable:,}")

    optimizer = AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=hp["epochs"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_cnt = 0
    best_epoch   = 0

    log_fn(f"\n{prefix}Memulai Training Loop...")
    log_fn("-" * 55)

    for epoch in range(1, hp["epochs"] + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.view(-1).long().to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.view(-1).long().to(device)
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        t_loss = train_loss / train_total
        v_loss = val_loss   / val_total
        t_acc  = 100 * train_correct / train_total
        v_acc  = 100 * val_correct   / val_total

        history["train_loss"].append(round(t_loss, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["train_acc"].append(round(t_acc, 2))
        history["val_acc"].append(round(v_acc, 2))

        marker = ""
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
            marker = " ✓ SAVED"
        else:
            patience_cnt += 1

        log_fn(
            f"{prefix}Epoch [{epoch:>3}/{hp['epochs']}] "
            f"Loss: {t_loss:.4f}/{v_loss:.4f}  "
            f"Acc: {t_acc:.1f}%/{v_acc:.1f}%  "
            f"({time.time() - t0:.0f}s){marker}"
        )

        if patience_cnt >= hp["patience"]:
            log_fn(f"\n{prefix}Early stopping pada epoch {epoch} "
                   f"(patience={hp['patience']} tercapai).")
            break

    log_fn(f"\n{prefix}Best Val Accuracy : {best_val_acc:.2f}%  (epoch {best_epoch})")
    log_fn(f"{prefix}Checkpoint disimpan → {ckpt_path}")

    return {"best_val_acc": best_val_acc, "best_epoch": best_epoch, "history": history}


def run(log_fn, hp_override=None):
    """
    Entry point GUI (Step 5).
    hp_override: dict opsional untuk menimpa TRAIN_HP.
    """
    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2"), (SPLIT_PATH, "Step 3")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} terlebih dahulu.")
            raise FileNotFoundError(path)

    hp = {**DEFAULT_CONFIG, **(hp_override or {})}
    log_fn("Hyperparameter Training:")
    for k, v in hp.items():
        log_fn(f"  {k:<15}: {v}")
    log_fn("")

    result = train_one_run(hp, log_fn, ckpt_path=CKPT_PATH)
    history = result["history"]

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    log_fn("Membuat plot training curve...")
    _plot_curves(history, CURVE_PATH)
    log_fn(f"Chart tersimpan → {CURVE_PATH}")

    return {
        "best_val_acc": result["best_val_acc"],
        "best_epoch":   result["best_epoch"],
        "history":      history,
        "chart_path":   CURVE_PATH,
    }