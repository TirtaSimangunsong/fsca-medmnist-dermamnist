"""
common.py - Utilitas Bersama Pipeline
=======================================
Fungsi yang dipakai lebih dari satu step:

- resolve_device()            : pilih MPS / CUDA / CPU sesuai DEVICE_CONFIG
- set_seed()                  : reproduktibilitas (random, numpy, torch)
- build_transforms()          : transform train (augmentasi) / eval
- build_dataloaders()         : DataLoader split resmi DermaMNIST + jumlah sampel per kelas
- compute_loss_class_weights(): bobot Weighted Cross-Entropy (1/sqrt(n), 1/n, atau seragam)
- evaluate()                  : metrik test (opsional TTA 8 arah + logit adjustment)
- make_logit_adjustment()     : vektor tau * log(prior)
- tune_logit_adjustment()     : sapuan tau di VALIDATION set
- validate()                  : loss + acc + balanced acc + AUC di val set (per epoch)
- selection_key()/is_better() : aturan memilih checkpoint terbaik (TRAIN_HP["select_metric"])
- plot_history()              : kurva loss / accuracy / val AUC

Dipakai oleh: step5_training (train_one_run), step6_finetune, step7_evaluation,
step11_ablation.
"""

import os
import sys
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATASET, AUGMENTATION, DEVICE_CONFIG, IMBALANCE


def _noop(*_args, **_kwargs):
    pass


# =========================================================
# DEVICE & SEED
# =========================================================
def resolve_device(log_fn=None):
    import torch

    log_fn = log_fn or _noop
    pref = DEVICE_CONFIG.get("device", "auto")

    if pref == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    else:
        dev = pref

    log_fn(f"Device: {dev}")
    return torch.device(dev)


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# DATA
# =========================================================
def build_transforms(mean, std, train):
    from torchvision import transforms

    if not train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    aug = AUGMENTATION
    return transforms.Compose([
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


def build_dataloaders(mean, std, batch_size, n_classes,
                      splits=("train", "val", "test"),
                      train_augment=True, seed=None, log_fn=None):
    """
    Membangun DataLoader untuk split resmi DermaMNIST.

    - Split "train" diacak (shuffle biasa, TANPA WeightedRandomSampler) dan
      diberi augmentasi jika train_augment=True.
    - Split lain memakai transform eval (tanpa augmentasi).
    - Jika seed diberikan, urutan shuffle train ikut terkunci ke seed.

    Return: (loaders: dict split -> DataLoader,
             info:    dict {"class_counts": list[int] | None, "sizes": dict})
    """
    import torch
    from torch.utils.data import DataLoader
    from medmnist import DermaMNIST

    log_fn = log_fn or _noop
    loaders, sizes = {}, {}
    class_counts = None

    for split in splits:
        is_train = split == "train"
        ds = DermaMNIST(
            split=split,
            transform=build_transforms(mean, std, train=is_train and train_augment),
            download=DATASET["download"],
            size=DATASET["image_size"],
        )

        generator = None
        if is_train and seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=is_train,
            num_workers=0, generator=generator,
        )
        sizes[split] = len(ds)

        if is_train:
            labels = np.asarray(ds.labels).reshape(-1).astype(int)
            class_counts = np.bincount(labels, minlength=n_classes).tolist()

        log_fn(f"  {split:<5}: {len(ds):>5} sampel, {len(loaders[split])} batch")

    return loaders, {"class_counts": class_counts, "sizes": sizes}


def compute_loss_class_weights(class_counts, mode=None):
    """
    Bobot per kelas untuk nn.CrossEntropyLoss(weight=...).
    Dinormalisasi agar RATA-RATA bobot = 1 (skala loss tetap sebanding
    dengan CE biasa). Rumus identik dengan Step 3.
    """
    mode = mode or IMBALANCE["loss_weight_mode"]
    counts = np.asarray(class_counts, dtype=float)

    if mode == "inverse":
        w = 1.0 / (counts + 1e-6)
    elif mode == "sqrt":
        w = 1.0 / np.sqrt(counts + 1e-6)
    elif mode == "none":
        w = np.ones_like(counts)
    else:
        raise ValueError(f"loss_weight_mode tidak dikenal: {mode}")

    return w / w.mean()


# =========================================================
# EVALUASI
# =========================================================
def _forward(model, imgs, use_tta):
    """Logit untuk satu batch. TTA 8 arah = 4 rotasi x (asli, flip horizontal)."""
    import torch

    if not use_tta:
        return model(imgs)

    outs = []
    for k in range(4):
        rot = torch.rot90(imgs, k, dims=(2, 3))
        outs.append(model(rot))
        outs.append(model(torch.flip(rot, dims=(3,))))
    return torch.stack(outs, dim=0).mean(dim=0)


def collect_logits(model, loader, device, use_tta=False):
    """Jalankan model di seluruh loader. Return (logits [N,C], y_true [N]) numpy."""
    import torch

    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = _forward(model, imgs, use_tta)
            all_logits.append(logits.float().cpu().numpy())
            all_y.append(labels.view(-1).long().numpy())
    return np.concatenate(all_logits), np.concatenate(all_y)


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_metrics(logits, y_true, n_classes):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, roc_auc_score)

    probs = _softmax(logits)
    y_pred = probs.argmax(axis=1)

    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                            labels=list(range(n_classes)))
    except ValueError:
        auc = float("nan")   # mis. ada kelas yang tidak muncul di y_true

    return {
        "acc":          100.0 * accuracy_score(y_true, y_pred),
        "balanced_acc": 100.0 * balanced_accuracy_score(y_true, y_pred),
        "f1_weighted":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":     f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_macro":    float(auc),
        "y_true":       y_true.tolist(),
        "y_pred":       y_pred.tolist(),
    }


def make_logit_adjustment(class_counts, tau):
    """
    Vektor penyesuaian logit (Menon et al., 2021, post-hoc):
        logit_adj = logit - tau * log(prior)
    Return numpy [C] yang DIKURANGKAN dari logit.
    """
    counts = np.asarray(class_counts, dtype=float)
    prior = counts / counts.sum()
    return (float(tau) * np.log(prior + 1e-12)).astype(np.float32)


def evaluate(model, loader, device, n_classes, use_tta=False, logit_adjust=None):
    logits, y_true = collect_logits(model, loader, device, use_tta=use_tta)
    if logit_adjust is not None:
        logits = logits - np.asarray(logit_adjust, dtype=np.float32)[None, :]
    return compute_metrics(logits, y_true, n_classes)


def tune_logit_adjustment(model, val_loader, device, n_classes, class_counts,
                          metric="balanced_acc", use_tta=False, log_fn=None,
                          grid=None):
    """
    Pilih tau terbaik di VALIDATION set. Logit val dihitung sekali, lalu
    setiap tau di grid hanya menggeser logit (murah).
    Jika ada nilai metrik yang sama, tau terkecil yang dipilih.

    Return: (best_tau: float, table: list[dict])
    """
    log_fn = log_fn or _noop
    grid = grid if grid is not None else IMBALANCE.get("tau_grid", [0.0, 0.5, 1.0])

    logits, y_true = collect_logits(model, val_loader, device, use_tta=use_tta)
    return _sweep_tau(logits, y_true, n_classes, class_counts, grid, metric, log_fn)


def _sweep_tau(logits, y_true, n_classes, class_counts, grid, metric, log_fn):
    log_fn(f"Sapuan logit adjustment di VALIDATION (metrik: {metric})")
    log_fn(f"  {'tau':>6} {'acc':>8} {'bal_acc':>8} {'f1_macro':>9}")

    table, best_tau, best_score = [], 0.0, -np.inf
    for tau in grid:
        adj = make_logit_adjustment(class_counts, tau)
        m = compute_metrics(logits - adj[None, :], y_true, n_classes)
        row = {
            "tau":          float(tau),
            "acc":          round(float(m["acc"]), 4),
            "balanced_acc": round(float(m["balanced_acc"]), 4),
            "f1_macro":     round(float(m["f1_macro"]), 4),
        }
        table.append(row)
        log_fn(f"  {tau:>6.2f} {row['acc']:>8.2f} {row['balanced_acc']:>8.2f} "
               f"{row['f1_macro']:>9.4f}")
        if m[metric] > best_score:
            best_score, best_tau = m[metric], float(tau)

    log_fn(f"  -> tau terpilih: {best_tau}")
    return best_tau, table


# =========================================================
# VALIDASI PER EPOCH & PEMILIHAN CHECKPOINT
# =========================================================
# nama di config -> (kunci di dict metrik, True jika makin besar makin baik)
_SELECT_METRICS = {
    "auc":     ("auc_macro",    True),
    "loss":    ("loss",         False),
    "bal_acc": ("balanced_acc", True),
    "acc":     ("acc",          True),
}


def selection_key(name):
    """Return (kunci_metrik, higher_is_better) untuk TRAIN_HP["select_metric"]."""
    if name not in _SELECT_METRICS:
        raise ValueError(f"select_metric '{name}' tidak dikenal. "
                         f"Pilihan: {list(_SELECT_METRICS)}")
    return _SELECT_METRICS[name]


def is_better(score, best, higher):
    if best is None:
        return True
    if score != score:          # NaN tidak pernah dianggap lebih baik
        return False
    return score > best if higher else score < best


def validate(model, loader, device, n_classes, criterion=None):
    """
    Evaluasi satu kali di loader (tanpa TTA, tanpa logit adjustment).
    Return dict: loss, acc, balanced_acc, f1_macro, f1_weighted, auc_macro.
    """
    import torch

    model.eval()
    total_loss, n = 0.0, 0
    logits_all, y_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            y = labels.view(-1).long().to(device)
            out = model(imgs)
            if criterion is not None:
                total_loss += criterion(out, y).item() * imgs.size(0)
            n += imgs.size(0)
            logits_all.append(out.float().cpu().numpy())
            y_all.append(y.cpu().numpy())

    m = compute_metrics(np.concatenate(logits_all), np.concatenate(y_all), n_classes)
    m.pop("y_true"); m.pop("y_pred")
    m["loss"] = total_loss / n if criterion is not None else float("nan")
    return m


def plot_history(history, save_path, title_prefix="Training"):
    """Kurva loss, accuracy, dan val AUC (jika tercatat) dengan tema gelap GUI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_auc = bool(history.get("val_auc"))
    n_panels = 3 if has_auc else 2
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    fig.patch.set_facecolor("#1a1a1a")
    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#555")

    axes[0].plot(epochs, history["train_loss"], color="#4fc3f7", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   color="#ef5350", label="Val Loss",   linewidth=2)
    axes[0].set_title(f"{title_prefix} - Loss", color="white", fontweight="bold")
    axes[0].set_xlabel("Epoch", color="white"); axes[0].set_ylabel("Loss", color="white")

    axes[1].plot(epochs, history["train_acc"], color="#4fc3f7", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"],   color="#ef5350", label="Val Acc",   linewidth=2)
    if history.get("val_bal_acc"):
        axes[1].plot(epochs, history["val_bal_acc"], color="#ba68c8",
                     label="Val Balanced Acc", linewidth=1.5)
    axes[1].set_title(f"{title_prefix} - Accuracy", color="white", fontweight="bold")
    axes[1].set_xlabel("Epoch", color="white"); axes[1].set_ylabel("Accuracy (%)", color="white")

    if has_auc:
        axes[2].plot(epochs, history["val_auc"], color="#81c784", label="Val AUC", linewidth=2)
        axes[2].set_title(f"{title_prefix} - Val AUC", color="white", fontweight="bold")
        axes[2].set_xlabel("Epoch", color="white"); axes[2].set_ylabel("AUC macro", color="white")

    best_epoch = history.get("best_epoch")
    for ax in axes:
        if best_epoch:
            ax.axvline(best_epoch, color="#ffd54f", linestyle="--", linewidth=1,
                       label=f"ckpt terpilih (ep {best_epoch})")
        ax.legend(facecolor="#333", labelcolor="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()