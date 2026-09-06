"""
common.py - Utilitas Bersama Pipeline FSCA-MedMNIST
=====================================================
Modul baru yang menampung logika yang sebelumnya diduplikasi (dan
tidak konsisten) di step3/step5/step6/step7:

  - resolve_device()        : auto-detect MPS / CUDA / CPU
  - set_seed()              : reproducibility penuh
  - build_transforms()      : pipeline augmentasi tunggal
  - build_dataloaders()     : loader train/val/test + strategi imbalance
  - MixupCutmix             : MixUp & CutMix + loss-nya
  - ModelEMA                : exponential moving average bobot
  - build_optimizer()       : param group (backbone LR lebih kecil)
  - WarmupCosine            : cosine annealing + linear warmup
  - evaluate()              : inferensi + metrik (acc, balanced acc, F1, AUC)
  - tta_logits()            : test-time augmentation 8-arah
  - tune_logit_adjustment() : sapu tau di validation set

Semua step lain sebaiknya mengimpor dari sini agar tidak ada lagi
divergensi konfigurasi antar-step.
"""

import os
import sys
import math
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATASET, AUGMENTATION, IMBALANCE, EVAL_CONFIG, DEVICE_CONFIG


# =========================================================
# DEVICE & SEED
# =========================================================
def resolve_device(preference=None, log_fn=print):
    """
    Kembalikan torch.device terbaik yang tersedia.

    Versi lama meng-hardcode "cpu" di step5/6/7. Di MacBook Apple Silicon,
    backend MPS memberi percepatan besar untuk training ResNet-18.
    """
    import torch

    pref = preference or DEVICE_CONFIG.get("device", "auto")

    if pref != "auto":
        dev = torch.device(pref)
        log_fn(f"Device (manual): {dev}")
        return dev

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
        log_fn("Device: mps (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
        log_fn(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        dev = torch.device("cpu")
        log_fn("Device: cpu (tidak ada akselerator terdeteksi)")
    return dev


def set_seed(seed=42, deterministic=False):
    """Kunci seluruh sumber keacakan agar hasil bisa direproduksi."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    return seed


# =========================================================
# TRANSFORMS
# =========================================================
def build_transforms(mean, std, train=True, strength=1.0):
    """
    Bangun pipeline transform.

    strength : skala intensitas augmentasi (1.0 = penuh, 0.5 = separuh).
               Dipakai Step 6 untuk memperlembut augmentasi di fase polish.
    """
    from torchvision import transforms

    if not train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    aug = AUGMENTATION
    size = DATASET["image_size"]
    s = float(strength)
    ops = []

    rrc = aug.get("random_resized_crop", {})
    if rrc.get("enabled", False) and s > 0:
        lo = 1.0 - (1.0 - rrc["scale"][0]) * s      # strength kecil -> crop lebih lembut
        ops.append(transforms.RandomResizedCrop(
            size,
            scale=(max(0.05, lo), rrc["scale"][1]),
            ratio=tuple(rrc["ratio"]),
            antialias=True,
        ))

    if aug["horizontal_flip_p"] > 0:
        ops.append(transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]))
    if aug["vertical_flip_p"] > 0:
        ops.append(transforms.RandomVerticalFlip(p=aug["vertical_flip_p"]))
    if aug["rotation_degrees"] > 0:
        ops.append(transforms.RandomRotation(degrees=aug["rotation_degrees"] * s))

    cj = aug["color_jitter"]
    ops.append(transforms.ColorJitter(
        brightness=cj["brightness"] * s,
        contrast=cj["contrast"] * s,
        saturation=cj["saturation"] * s,
        hue=cj.get("hue", 0.0) * s,
    ))

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=mean, std=std))

    p_erase = aug.get("random_erasing_p", 0.0) * s
    if p_erase > 0:
        ops.append(transforms.RandomErasing(p=p_erase, scale=(0.02, 0.15), value="random"))

    return transforms.Compose(ops)


def describe_transforms(tf):
    """String ringkas isi Compose - untuk logging & dokumentasi BAB III."""
    try:
        return [t.__class__.__name__ for t in tf.transforms]
    except AttributeError:
        return [tf.__class__.__name__]


# =========================================================
# CLASS WEIGHTS & DATALOADERS
# =========================================================
def get_class_counts(dataset, n_classes):
    """
    Hitung jumlah sampel per kelas TANPA mengiterasi transform.

    Versi lama di step3 memakai `[train_ds[i][1] for i in range(n)]`, yang
    memicu decode + augmentasi seluruh 7.007 citra hanya untuk membaca label.
    medmnist menyimpan label mentah di atribut `.labels`.
    """
    labels = np.asarray(dataset.labels).reshape(-1).astype(int)
    return np.bincount(labels, minlength=n_classes), labels


def compute_loss_weights(class_counts, strategy):
    """
    Kembalikan (weights_tensor_or_None, sample_weights_or_None).

    Bobot loss dinormalisasi agar rata-ratanya 1.0 supaya skala loss - dan
    karenanya learning rate efektif - konsisten antar strategi dan antar step.
    Ini memperbaiki bug lama di mana step5 dan step6 memakai normalisasi
    berbeda (mean=1 vs sum=1), membuat loss step6 ~7x lebih kecil.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    counts = np.clip(counts, 1.0, None)

    if strategy == "none":
        return None, None

    if strategy == "sampler":
        inv = 1.0 / counts
        return None, inv

    if strategy == "weighted_loss":
        w = 1.0 / counts
    elif strategy == "sqrt_weighted_loss":
        w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"strategy tidak dikenal: {strategy}")

    w = w / w.mean()          # rata-rata = 1.0
    return w, None


def build_dataloaders(mean, std, batch_size, n_classes, splits=("train", "val"),
                      train_strength=1.0, log_fn=print, seed=42):
    """
    Bangun DataLoader untuk split yang diminta, sekaligus mengembalikan
    class_counts dan bobot loss sesuai IMBALANCE["strategy"].

    Returns: dict {split: DataLoader}, info dict
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from medmnist import DermaMNIST

    img_size = DATASET["image_size"]
    nw       = DATASET.get("num_workers", 0)
    strategy = IMBALANCE["strategy"]

    train_tf = build_transforms(mean, std, train=True,  strength=train_strength)
    eval_tf  = build_transforms(mean, std, train=False)

    loaders = {}
    counts = None
    sample_weights = None
    loss_weights = None

    for sp in splits:
        tf = train_tf if sp == "train" else eval_tf
        ds = DermaMNIST(split=sp, transform=tf,
                        download=DATASET["download"], size=img_size)

        if sp == "train":
            counts, labels = get_class_counts(ds, n_classes)
            loss_weights, inv = compute_loss_weights(counts, strategy)

            if inv is not None:
                sample_weights = inv[labels]
                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(ds),
                    replacement=True,
                )
                loaders[sp] = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                                         num_workers=nw, drop_last=True)
            else:
                g = torch.Generator()
                g.manual_seed(seed)
                loaders[sp] = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         num_workers=nw, drop_last=True, generator=g)
        else:
            loaders[sp] = DataLoader(ds, batch_size=max(batch_size, EVAL_CONFIG["batch_size"]),
                                     shuffle=False, num_workers=nw)

    info = {
        "strategy":     strategy,
        "class_counts": counts.tolist() if counts is not None else None,
        "loss_weights": loss_weights.tolist() if loss_weights is not None else None,
        "uses_sampler": sample_weights is not None,
        "train_transform": describe_transforms(train_tf),
        "eval_transform":  describe_transforms(eval_tf),
    }

    log_fn(f"Strategi imbalance : {strategy}")
    if loss_weights is not None:
        log_fn(f"  Bobot loss (mean=1): {[round(float(v), 3) for v in loss_weights]}")
    if sample_weights is not None:
        log_fn("  WeightedRandomSampler aktif, loss TANPA bobot (hindari koreksi ganda)")

    return loaders, info

def ensure_ssl_certificates():
    """
    macOS: Python tidak memakai keychain sistem, sehingga unduhan HTTPS
    (bobot ImageNet torchvision, dataset medmnist dari Zenodo) gagal dengan
    CERTIFICATE_VERIFY_FAILED. Arahkan ke CA bundle certifi.

    Ini tetap MEMVERIFIKASI sertifikat dengan benar - bukan mematikan
    verifikasi seperti trik ssl._create_unverified_context yang banyak
    beredar di forum.
    """
    try:
        import certifi
    except ImportError:
        return False

    path = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", path)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", path)
    try:
        import ssl
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=path)
    except Exception:
        pass
    return True


ensure_ssl_certificates()

# =========================================================
# MIXUP / CUTMIX
# =========================================================
class MixupCutmix:
    """
    MixUp (Zhang et al., 2018) dan CutMix (Yun et al., 2019).

    Regularisasi ini efektif pada dataset kecil dan tidak seimbang seperti
    DermaMNIST karena memaksa model belajar batas keputusan yang linier
    di antara pasangan sampel, bukan menghafal citra individual.
    """

    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5):
        self.mixup_alpha  = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob         = prob
        self.switch_prob  = switch_prob

    def __call__(self, x, y):
        """Returns (x_mixed, y_a, y_b, lam)."""
        import torch

        if np.random.rand() > self.prob:
            return x, y, y, 1.0

        idx = torch.randperm(x.size(0), device=x.device)
        use_cutmix = np.random.rand() < self.switch_prob

        if use_cutmix and self.cutmix_alpha > 0:
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            H, W = x.size(2), x.size(3)
            r = math.sqrt(1.0 - lam)
            cut_h, cut_w = int(H * r), int(W * r)
            cy, cx = np.random.randint(H), np.random.randint(W)
            y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, H)
            x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, W)
            x = x.clone()
            x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
        else:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            x = lam * x + (1.0 - lam) * x[idx]

        return x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss untuk batch yang sudah dimix. Kompatibel dengan class weight + label smoothing."""
    if lam >= 1.0:
        return criterion(pred, y_a)
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# =========================================================
# EMA
# =========================================================
class ModelEMA:
    """
    Exponential Moving Average bobot model.

    Bobot rata-rata biasanya menghasilkan validasi lebih tinggi dan jauh
    lebih stabil dari epoch ke epoch dibanding bobot mentah, terutama pada
    dataset kecil di mana kurva validasi sangat berisik.
    """

    def __init__(self, model, decay=0.999):
        import copy
        import torch

        self.decay  = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _is_float(t):
        return t.dtype.is_floating_point

    def update(self, model):
        import torch

        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                mv = msd[k].detach()
                if self._is_float(v):
                    v.mul_(self.decay).add_(mv, alpha=1.0 - self.decay)
                else:
                    v.copy_(mv)   # buffer integer (mis. num_batches_tracked)


# =========================================================
# OPTIMIZER & SCHEDULER
# =========================================================
def build_optimizer(model, lr, weight_decay, backbone_lr_mult=0.1, log_fn=print):
    """
    Dua penyempurnaan dibanding versi lama:

    1. Param group terpisah. Backbone berbobot ImageNet hanya perlu LR kecil,
       sedangkan stem baru, modul atensi, dan classifier diinisialisasi acak
       sehingga butuh LR penuh. Ini menggantikan skema freeze/unfreeze yang
       sebelumnya justru membekukan backbone yang belum terlatih.
    2. Norm & bias dikeluarkan dari weight decay - praktik standar yang
       konsisten memberi sedikit kenaikan akurasi.
    """
    from torch.optim import AdamW

    backbone_keys = ("layer1", "layer2", "layer3", "layer4")

    groups = {
        "backbone_decay":   {"params": [], "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
        "backbone_nodecay": {"params": [], "lr": lr * backbone_lr_mult, "weight_decay": 0.0},
        "new_decay":        {"params": [], "lr": lr,                    "weight_decay": weight_decay},
        "new_nodecay":      {"params": [], "lr": lr,                    "weight_decay": 0.0},
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = any(k in name for k in backbone_keys)
        no_decay    = p.ndim <= 1 or name.endswith(".bias")
        key = ("backbone_" if is_backbone else "new_") + ("nodecay" if no_decay else "decay")
        groups[key]["params"].append(p)

    param_groups = [g for g in groups.values() if g["params"]]
    n_bb  = sum(p.numel() for g in ("backbone_decay", "backbone_nodecay") for p in groups[g]["params"])
    n_new = sum(p.numel() for g in ("new_decay", "new_nodecay") for p in groups[g]["params"])
    log_fn(f"Optimizer AdamW  | backbone: {n_bb:,} params @ lr={lr * backbone_lr_mult:.1e}"
           f"  |  modul baru: {n_new:,} params @ lr={lr:.1e}")

    return AdamW(param_groups)


class WarmupCosine:
    """Linear warmup lalu cosine annealing ke 0. Dipanggil per-epoch."""

    def __init__(self, optimizer, total_epochs, warmup_epochs=0, min_lr_ratio=0.0):
        self.opt = optimizer
        self.total = max(1, total_epochs)
        self.warmup = max(0, warmup_epochs)
        self.min_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.epoch = 0
        self.step(0)

    def _factor(self, epoch):
        if self.warmup and epoch < self.warmup:
            return (epoch + 1) / float(self.warmup)
        prog = (epoch - self.warmup) / max(1, self.total - self.warmup)
        prog = min(max(prog, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return self.min_ratio + (1.0 - self.min_ratio) * cos

    def step(self, epoch=None):
        self.epoch = self.epoch + 1 if epoch is None else epoch
        f = self._factor(self.epoch)
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = base * f
        return f

    def current_lr(self):
        return self.opt.param_groups[-1]["lr"]


# =========================================================
# TTA & EVALUASI
# =========================================================
def tta_logits(model, x):
    """
    Test-time augmentation 8-arah: 4 rotasi 90 derajat x 2 (identitas + flip).

    Valid untuk dermoskopi karena rotasi/flip adalah transformasi
    label-preserving. Biasanya menaikkan akurasi 0.5-1 poin secara gratis.
    Logit dirata-ratakan (bukan probabilitas) agar konsisten dengan
    perhitungan AUC.
    """
    import torch

    out = 0.0
    n = 0
    for k in range(4):
        xr = torch.rot90(x, k, dims=(2, 3))
        out = out + model(xr)
        out = out + model(torch.flip(xr, dims=(3,)))
        n += 2
    return out / n


def evaluate(model, loader, device, n_classes, criterion=None, use_tta=False,
             logit_adjust=None):
    """
    Jalankan inferensi dan hitung metrik lengkap.

    Returns dict: loss, acc, balanced_acc, f1_macro, f1_weighted, auc_macro,
                  y_true, y_pred, y_prob
    """
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    )

    model.eval()
    all_logits, all_labels = [], []
    total_loss, total_n = 0.0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze(1).long().to(device)

            logits = tta_logits(model, imgs) if use_tta else model(imgs)

            if criterion is not None:
                total_loss += criterion(logits, labels).item() * imgs.size(0)
            total_n += imgs.size(0)

            if logit_adjust is not None:
                logits = logits - logit_adjust.to(logits.device)

            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    y_true = torch.cat(all_labels).numpy()
    y_prob = F.softmax(logits, dim=1).numpy()
    y_pred = y_prob.argmax(1)

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    return {
        "loss":         total_loss / max(1, total_n) if criterion is not None else None,
        "acc":          100.0 * accuracy_score(y_true, y_pred),
        "balanced_acc": 100.0 * balanced_accuracy_score(y_true, y_pred),
        "f1_macro":     f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "auc_macro":    float(auc),
        "y_true":       y_true,
        "y_pred":       y_pred,
        "y_prob":       y_prob,
        "logits":       logits.numpy(),
    }


# =========================================================
# LOGIT ADJUSTMENT
# =========================================================
def make_logit_adjustment(class_counts, tau):
    """
    Vektor koreksi post-hoc (Menon et al., ICLR 2021):
        logit_c <- logit_c - tau * log(prior_c)

    Keunggulannya untuk skripsi: tidak mengubah proses training sama sekali,
    sehingga kamu bisa melaporkan SATU model terlatih pada beberapa nilai tau
    dan menampilkan kurva trade-off akurasi vs sensitivity kelas minoritas.
    """
    import torch

    counts = np.asarray(class_counts, dtype=np.float64)
    prior = counts / counts.sum()
    return torch.tensor(tau * np.log(prior + 1e-12), dtype=torch.float32)


def tune_logit_adjustment(model, val_loader, device, n_classes, class_counts,
                          metric="balanced_acc", taus=None, use_tta=False, log_fn=print):
    """
    Sapu tau di validation set, kembalikan (tau_terbaik, tabel_hasil).

    PENTING: tuning HARUS di validation, tidak pernah di test set.
    """
    taus = taus if taus is not None else [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    rows, best_tau, best_score = [], 0.0, -1.0

    log_fn("Menyapu tau untuk logit adjustment (di validation set)...")
    log_fn(f"  {'tau':>5} {'acc':>8} {'bal_acc':>9} {'f1_macro':>9}")

    for t in taus:
        adj = make_logit_adjustment(class_counts, t)
        m = evaluate(model, val_loader, device, n_classes, use_tta=use_tta, logit_adjust=adj)
        rows.append({"tau": t, "acc": m["acc"], "balanced_acc": m["balanced_acc"],
                     "f1_macro": m["f1_macro"]})
        log_fn(f"  {t:>5.2f} {m['acc']:>7.2f}% {m['balanced_acc']:>8.2f}% {m['f1_macro']:>9.4f}")
        if m[metric] > best_score:
            best_score, best_tau = m[metric], t

    log_fn(f"  -> tau terbaik = {best_tau} (kriteria: {metric})")
    return best_tau, rows