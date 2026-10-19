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
- AdamW optimizer + CosineAnnealingLR scheduler (T_max = jumlah epoch)
- Checkpoint terbaik dipilih berdasarkan TRAIN_HP["select_metric"]
  (default: val AUC macro, sesuai protokol resmi MedMNIST). Val accuracy
  tidak dipakai karena berisik pada val set kecil dan tidak menghukum
  model yang overconfident (val loss naik sementara val acc stagnan).
- Early stopping memakai metrik yang sama (TRAIN_HP["patience"];
  patience = epochs berarti praktis nonaktif)
- Setiap epoch mencatat: loss, acc, balanced acc, dan AUC validasi

Resep bisa ditimpa per run lewat `hp` (dipakai Step 12, ablasi resep):
  pretrained       (default MODEL_CONFIG["pretrained"])
  augment          (default TRAIN_HP["augment"])
  loss_weight_mode (default IMBALANCE["loss_weight_mode"]; "none" = CE biasa)
  optimizer        (default TRAIN_HP["optimizer"]: "adamw" | "adam")
  scheduler        (default TRAIN_HP["scheduler"]: "cosine" | "multistep")
  dropout_p        (default MODEL_CONFIG["dropout_p"])
  batch_size, weight_decay, lr, epochs
Tanpa override, perilakunya identik dengan resep utama skripsi.

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


def resolve_recipe(hp):
    """
    Resep efektif sebuah run: default resep utama, ditimpa oleh kunci di `hp`.
    Dipakai train_one_run() dan oleh Step 11/12/13 untuk memastikan checkpoint
    lama yang dipakai ulang memang berasal dari resep yang sama.
    """
    return {
        "pretrained":       hp.get("pretrained", MODEL_CONFIG["pretrained"]),
        "augment":          hp.get("augment", TRAIN_HP.get("augment", True)),
        "loss_weight_mode": hp.get("loss_weight_mode", IMBALANCE["loss_weight_mode"]),
        "optimizer":        hp.get("optimizer", TRAIN_HP.get("optimizer", "adamw")),
        "scheduler":        hp.get("scheduler", TRAIN_HP.get("scheduler", "cosine")),
        "dropout_p":        hp.get("dropout_p", MODEL_CONFIG["dropout_p"]),
        "batch_size":       hp["batch_size"],
        "lr":               hp["lr"],
        "weight_decay":     hp["weight_decay"],
        "epochs":           hp["epochs"],
    }


def train_one_run(hp, log_fn, ckpt_path=CKPT_PATH, attention=None, seed=None, tag=None):
    """
    Satu run training Fase 1.

    hp        : dict berisi epochs, batch_size, lr, weight_decay, patience,
                select_metric (opsional, default "auc"), seed (opsional),
                dan override resep opsional (lihat docstring modul)
    ckpt_path : lokasi simpan checkpoint terbaik
    attention : "none" | "se" | "eca" | "cbam" | "fsca" | None (=MODEL_CONFIG)
    seed      : jika diberikan, mengunci inisialisasi dan urutan data
    tag       : prefix log (dipakai Step 11 agar log tiap run mudah dibedakan)

    Return: dict {best_epoch, best_score, select_metric, best_val_acc,
                  best_val_auc, history}
    """
    import torch
    import torch.nn as nn

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model, model_label
    import common

    prefix = f"[{tag}] " if tag else ""
    seed = seed if seed is not None else hp.get("seed")
    select_metric = hp.get("select_metric", "auc")
    sel_key, higher = common.selection_key(select_metric)

    recipe = resolve_recipe(hp)

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
        batch_size=recipe["batch_size"], n_classes=n_classes,
        splits=("train", "val"), train_augment=recipe["augment"], seed=seed, log_fn=log_fn,
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    # Model
    model = build_model(n_classes=n_classes,
                        pretrained=recipe["pretrained"],
                        attention=attention,
                        dropout_p=recipe["dropout_p"]).to(device)
    log_fn(f"{prefix}Model: {model_label(model.attention_name)} "
           f"(pretrained={recipe['pretrained']}, dropout={recipe['dropout_p']})")
    log_fn(f"{prefix}Augmentasi: {'aktif' if recipe['augment'] else 'nonaktif'}")

    # Loss: Weighted Cross-Entropy (tanpa WeightedRandomSampler); "none" = CE biasa
    mode = recipe["loss_weight_mode"]
    weights = common.compute_loss_class_weights(dinfo["class_counts"], mode)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    if mode == "none":
        log_fn(f"{prefix}Loss: Cross-Entropy biasa (tanpa bobot kelas)")
    else:
        log_fn(f"{prefix}Loss: Weighted CE (mode={mode}, rasio maks/min "
               f"{weights.max() / weights.min():.1f}x)")

    # Seluruh model dilatih. Skema "Phase 1 head-only" tidak dipakai karena
    # stem sudah diganti (conv 3x3 stride 1), sehingga membekukan backbone
    # justru menghalangi fitur menyesuaikan diri.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_fn(f"{prefix}Melatih seluruh model. Trainable params: {trainable:,}")
    log_fn(f"{prefix}Checkpoint dipilih berdasarkan: val {select_metric} "
           f"({'maksimum' if higher else 'minimum'})")

    optimizer = common.make_optimizer(model.parameters(), recipe["optimizer"],
                                      recipe["lr"], recipe["weight_decay"])
    scheduler = common.make_scheduler(
        optimizer, recipe["scheduler"], hp["epochs"],
        milestones=TRAIN_HP.get("multistep_milestones", (0.5, 0.75)),
        gamma=TRAIN_HP.get("multistep_gamma", 0.1),
    )
    log_fn(f"{prefix}Optimizer: {recipe['optimizer']} (lr={recipe['lr']}, "
           f"weight_decay={recipe['weight_decay']}) | scheduler: {recipe['scheduler']} "
           f"| batch: {recipe['batch_size']}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "val_bal_acc": [], "val_auc": [], "lr": []}
    best_score, best_epoch, best_vm = None, 0, None
    patience_cnt = 0

    log_fn(f"\n{prefix}Memulai Training Loop...")
    log_fn("-" * 72)

    for epoch in range(1, hp["epochs"] + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

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
        vm = common.validate(model, val_loader, device, n_classes, criterion)

        t_loss = train_loss / train_total
        t_acc  = 100 * train_correct / train_total

        history["train_loss"].append(round(t_loss, 4))
        history["val_loss"].append(round(vm["loss"], 4))
        history["train_acc"].append(round(t_acc, 2))
        history["val_acc"].append(round(vm["acc"], 2))
        history["val_bal_acc"].append(round(vm["balanced_acc"], 2))
        history["val_auc"].append(round(vm["auc_macro"], 4))
        history["lr"].append(lr_now)

        score = vm[sel_key]
        marker = ""
        if common.is_better(score, best_score, higher):
            best_score, best_epoch, best_vm = score, epoch, vm
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
            marker = " ✓ SAVED"
        else:
            patience_cnt += 1

        log_fn(
            f"{prefix}Epoch [{epoch:>3}/{hp['epochs']}] "
            f"Loss: {t_loss:.4f}/{vm['loss']:.4f}  "
            f"Acc: {t_acc:.1f}%/{vm['acc']:.1f}%  "
            f"BalAcc: {vm['balanced_acc']:.1f}%  AUC: {vm['auc_macro']:.4f}  "
            f"({time.time() - t0:.0f}s){marker}"
        )

        if patience_cnt >= hp["patience"]:
            log_fn(f"\n{prefix}Early stopping pada epoch {epoch} "
                   f"(patience={hp['patience']} tercapai).")
            break

    history["best_epoch"]    = best_epoch
    history["select_metric"] = select_metric
    history["recipe"]        = recipe

    log_fn(f"\n{prefix}Checkpoint terpilih: epoch {best_epoch} | "
           f"val AUC {best_vm['auc_macro']:.4f} | val acc {best_vm['acc']:.2f}% | "
           f"val bal acc {best_vm['balanced_acc']:.2f}% | val loss {best_vm['loss']:.4f}")
    log_fn(f"{prefix}Checkpoint disimpan → {ckpt_path}")

    return {
        "best_epoch":    best_epoch,
        "best_score":    best_score,
        "select_metric": select_metric,
        "best_val_acc":  best_vm["acc"],
        "best_val_auc":  best_vm["auc_macro"],
        "history":       history,
    }


def run(log_fn, hp_override=None):
    """
    Entry point GUI (Step 5).
    hp_override: dict opsional untuk menimpa TRAIN_HP.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    import common

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
    common.plot_history(history, CURVE_PATH, title_prefix="Fase 1")
    log_fn(f"Chart tersimpan → {CURVE_PATH}")

    return {
        "best_val_acc": result["best_val_acc"],
        "best_val_auc": result["best_val_auc"],
        "best_epoch":   result["best_epoch"],
        "history":      history,
        "chart_path":   CURVE_PATH,
    }