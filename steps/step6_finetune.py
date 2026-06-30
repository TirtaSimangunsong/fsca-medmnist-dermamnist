"""
Step 6 - Finetuning & Hyperparameter Tuning
=============================================
Melakukan fine-tuning dari checkpoint Step 5 dengan:
- Learning rate lebih kecil (1/10 dari training awal)
- Unfreeze selektif: hanya layer3, layer4, FSCA, fc
- Label smoothing pada CrossEntropyLoss
- Simpan model final sebagai 'ResNet18_FSCA_DermaMNIST_Best.pth'

Output: outputs/ResNet18_FSCA_DermaMNIST_Best.pth
        outputs/finetune_curve.png
        outputs/finetune_history.json
"""

import os
import json
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, DATASET, AUGMENTATION, FINETUNE_HP

OUTPUT_DIR      = PATHS["output_dir"]
META_PATH       = PATHS["dataset_meta"]
CONFIG_PATH     = PATHS["preprocess_config"]
SPLIT_PATH      = PATHS["split_info"]
CKPT_PATH       = PATHS["checkpoint_best"]
FINAL_MODEL_PATH = PATHS["final_model"]
HISTORY_PATH    = PATHS["finetune_history"]
CURVE_PATH      = PATHS["finetune_curve_chart"]

# FINETUNE_HP kini berasal dari config.py — termasuk unfreeze_keys


def _freeze_except(model, unfreeze_keys):
    """Freeze semua parameter kecuali yang namanya mengandung unfreeze_keys."""
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in unfreeze_keys)


def run(log_fn):
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import transforms
    from medmnist import DermaMNIST

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model

    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2"),
                       (SPLIT_PATH, "Step 3"), (CKPT_PATH, "Step 5")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} terlebih dahulu.")
            raise FileNotFoundError(path)

    with open(META_PATH)   as f: meta       = json.load(f)
    with open(CONFIG_PATH) as f: config     = json.load(f)
    with open(SPLIT_PATH)  as f: split_info = json.load(f)

    n_classes = meta["n_classes"]
    device    = torch.device("cpu")

    # Load checkpoint
    log_fn("Memuat checkpoint dari Step 5...")
    model = build_model(n_classes=n_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    log_fn("Checkpoint berhasil dimuat.")

    # Freeze layer awal, unfreeze layer3/4, FSCA, fc (key dari config.py)
    unfreeze_keys = FINETUNE_HP["unfreeze_keys"]
    _freeze_except(model, unfreeze_keys)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_fn(f"Parameter trainable setelah freeze: {trainable:,}")
    log_fn(f"Layer yang di-unfreeze: {unfreeze_keys}")

    # DataLoaders
    mean, std = config["mean"], config["std"]
    bs = FINETUNE_HP["batch_size"]
    img_size = DATASET["image_size"]
    aug = AUGMENTATION

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(aug["horizontal_flip_p"]),
        transforms.RandomVerticalFlip(aug["vertical_flip_p"]),
        transforms.RandomRotation(aug["rotation_degrees"]),
        transforms.ColorJitter(
            aug["color_jitter"]["brightness"],
            aug["color_jitter"]["contrast"],
            aug["color_jitter"]["saturation"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = DermaMNIST(split="train", transform=train_transform, download=DATASET["download"], size=img_size)
    val_ds   = DermaMNIST(split="val",   transform=eval_transform,  download=DATASET["download"], size=img_size)

    sample_weights = torch.tensor(split_info["sample_weights"], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,   num_workers=0)

    criterion = nn.CrossEntropyLoss(label_smoothing=FINETUNE_HP["label_smoothing"])
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_HP["lr"],
        weight_decay=FINETUNE_HP["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_HP["epochs"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, patience_cnt, best_epoch = 0.0, 0, 0

    log_fn(f"\nHyperparameter Finetuning:")
    for k, v in FINETUNE_HP.items():
        if k == "unfreeze_keys":
            continue
        log_fn(f"  {k:<20}: {v}")
    log_fn("\nMemulai Finetuning...")
    log_fn("-" * 55)

    for epoch in range(1, FINETUNE_HP["epochs"] + 1):
        t0 = time.time()

        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.squeeze(1).long().to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)
        scheduler.step()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.squeeze(1).long().to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)

        tl = t_loss / t_total; vl = v_loss / v_total
        ta = 100 * t_correct / t_total; va = 100 * v_correct / v_total
        history["train_loss"].append(round(tl, 4))
        history["val_loss"].append(round(vl, 4))
        history["train_acc"].append(round(ta, 2))
        history["val_acc"].append(round(va, 2))

        marker = ""
        if va > best_val_acc:
            best_val_acc, best_epoch = va, epoch
            patience_cnt = 0
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            marker = " ✓ SAVED"
        else:
            patience_cnt += 1

        log_fn(
            f"Finetune [{epoch:>2}/{FINETUNE_HP['epochs']}] "
            f"Loss: {tl:.4f}/{vl:.4f}  "
            f"Acc: {ta:.1f}%/{va:.1f}%  "
            f"({time.time()-t0:.0f}s){marker}"
        )

        if patience_cnt >= FINETUNE_HP["patience"]:
            log_fn(f"\nEarly stopping pada epoch {epoch}.")
            break

    log_fn(f"\nBest Val Accuracy (Finetune): {best_val_acc:.2f}%  (epoch {best_epoch})")
    log_fn(f"Model final tersimpan → {FINAL_MODEL_PATH}")

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    # Plot
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

    axes[0].plot(epochs, history["train_loss"], "#4fc3f7", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   "#ef5350", label="Val",   linewidth=2)
    axes[0].set_title("Finetune Loss Curve", color="white", fontweight="bold")
    axes[0].set_xlabel("Epoch", color="white"); axes[0].set_ylabel("Loss", color="white")
    axes[0].legend(facecolor="#333", labelcolor="white")

    axes[1].plot(epochs, history["train_acc"], "#4fc3f7", label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"],   "#ef5350", label="Val",   linewidth=2)
    axes[1].set_title("Finetune Accuracy Curve", color="white", fontweight="bold")
    axes[1].set_xlabel("Epoch", color="white"); axes[1].set_ylabel("Accuracy (%)", color="white")
    axes[1].legend(facecolor="#333", labelcolor="white")

    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()

    log_fn(f"Chart tersimpan → {CURVE_PATH}")
    return {
        "best_val_acc":   best_val_acc,
        "best_epoch":     best_epoch,
        "final_model":    FINAL_MODEL_PATH,
        "chart_path":     CURVE_PATH,
    }