"""
Step 6 - Finetuning (Fase 2, OPSIONAL)
========================================
Dikendalikan oleh FINETUNE_HP["enabled"] di config.py.

enabled = False (DEFAULT)
    Checkpoint Fase 1 (Step 5) disalin apa adanya sebagai model final.
    Alasan: dengan 100 epoch cosine, LR Fase 1 sudah turun ke ~0 dan model
    sudah konvergen; Step 11 (ablasi) juga hanya melatih Fase 1, jadi model
    utama harus memakai resep yang sama agar perbandingan adil.

enabled = True
    Fine-tuning dari checkpoint Fase 1 dengan:
    - Hanya modul FINETUNE_HP["unfreeze_keys"] yang dilatih (default:
      layer3, layer4, fsca3, fsca4, fc); modul lain dibekukan, TERMASUK
      statistik BatchNorm-nya (dipaksa mode eval), sehingga benar-benar diam.
    - LR kecil (FINETUNE_HP["lr"]) + CosineAnnealingLR
    - Weighted CE (bobot sama dengan Fase 1) + label smoothing
    - Titik awal pembanding = skor validasi checkpoint Fase 1. Model final
      HANYA diganti bila metrik TRAIN_HP["select_metric"] (default val AUC)
      lebih baik dari Fase 1. Jika tidak pernah lebih baik, checkpoint Fase 1
      tetap menjadi model final. Fine-tuning tidak bisa memperburuk hasil
      (diukur di validation set).

Output: outputs/ResNet18_FSCA_DermaMNIST_Best.pth
        outputs/finetune_curve.png, outputs/finetune_history.json (jika enabled)
"""

import os
import json
import time
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, FINETUNE_HP, TRAIN_HP, IMBALANCE

OUTPUT_DIR       = PATHS["output_dir"]
META_PATH        = PATHS["dataset_meta"]
CONFIG_PATH      = PATHS["preprocess_config"]
CKPT_PATH        = PATHS["checkpoint_best"]
FINAL_MODEL_PATH = PATHS["final_model"]
HISTORY_PATH     = PATHS["finetune_history"]
CURVE_PATH       = PATHS["finetune_curve_chart"]
TRAIN_CURVE_PATH = PATHS["training_curve_chart"]


def _top_module(param_or_module_name):
    """'layer3.0.conv1.weight' -> 'layer3'."""
    return param_or_module_name.split(".")[0]


def _freeze_except(model, unfreeze_keys):
    """Bekukan semua parameter kecuali milik modul tingkat-atas di unfreeze_keys."""
    for name, param in model.named_parameters():
        param.requires_grad = _top_module(name) in unfreeze_keys


def _freeze_bn_stats(model, unfreeze_keys):
    """
    Paksa BatchNorm di modul yang dibekukan ke mode eval, agar running
    mean/var-nya tidak ikut bergeser selama model.train(). Panggil SETIAP
    kali setelah model.train().
    """
    import torch.nn as nn
    for name, m in model.named_modules():
        if name and isinstance(m, nn.modules.batchnorm._BatchNorm) \
                and _top_module(name) not in unfreeze_keys:
            m.eval()


def _copy_phase1(log_fn, reason):
    shutil.copyfile(CKPT_PATH, FINAL_MODEL_PATH)
    log_fn(reason)
    log_fn(f"Checkpoint Fase 1 disalin sebagai model final -> {FINAL_MODEL_PATH}")


def run(log_fn):
    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2"),
                       (CKPT_PATH, "Step 5")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} terlebih dahulu.")
            raise FileNotFoundError(path)

    # ---------------- Mode nonaktif ----------------
    if not FINETUNE_HP.get("enabled", False):
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)   # buang history finetune lama agar Step 9 tidak memakai data basi
        _copy_phase1(log_fn, "Fine-tuning dinonaktifkan (FINETUNE_HP['enabled'] = False).")
        return {"best_val_acc": None, "best_epoch": None, "finetuned": False,
                "final_model": FINAL_MODEL_PATH, "chart_path": TRAIN_CURVE_PATH}

    # ---------------- Mode aktif ----------------
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model
    import common

    with open(META_PATH)   as f: meta   = json.load(f)
    with open(CONFIG_PATH) as f: config = json.load(f)
    n_classes = meta["n_classes"]

    seed = TRAIN_HP.get("seed")
    if seed is not None:
        common.set_seed(seed)
        log_fn(f"Seed: {seed}")

    device = common.resolve_device(log_fn=log_fn)
    select_metric = TRAIN_HP.get("select_metric", "auc")
    sel_key, higher = common.selection_key(select_metric)

    # Load checkpoint Fase 1
    log_fn("Memuat checkpoint dari Step 5...")
    model = build_model(n_classes=n_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    log_fn(f"Checkpoint berhasil dimuat (atensi: {model.attention_name.upper()}).")

    # DataLoader (identik dengan Step 5, batch size dari FINETUNE_HP)
    log_fn("Memuat DataLoader...")
    loaders, dinfo = common.build_dataloaders(
        mean=config["mean"], std=config["std"],
        batch_size=FINETUNE_HP["batch_size"], n_classes=n_classes,
        splits=("train", "val"), train_augment=True, seed=seed, log_fn=log_fn,
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    # Loss: bobot kelas SAMA dengan Fase 1, plus label smoothing
    mode = IMBALANCE["loss_weight_mode"]
    weights = common.compute_loss_class_weights(dinfo["class_counts"], mode)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=FINETUNE_HP["label_smoothing"],
    )
    log_fn(f"Loss: Weighted CE (mode={mode}) + label smoothing "
           f"{FINETUNE_HP['label_smoothing']}")

    # Titik awal: skor checkpoint Fase 1 (bukan 0), agar model final tidak
    # pernah digantikan oleh hasil fine-tuning yang lebih buruk.
    vm0 = common.validate(model, val_loader, device, n_classes, criterion)
    best_score = vm0[sel_key]
    log_fn(f"Titik awal (Fase 1): val AUC {vm0['auc_macro']:.4f} | "
           f"acc {vm0['acc']:.2f}% | bal acc {vm0['balanced_acc']:.2f}% | "
           f"loss {vm0['loss']:.4f}")
    log_fn(f"Model final hanya diganti bila val {select_metric} "
           f"{'>' if higher else '<'} {best_score:.4f}")

    # Bekukan modul selain unfreeze_keys (bobot + statistik BatchNorm)
    unfreeze_keys = FINETUNE_HP["unfreeze_keys"]
    _freeze_except(model, unfreeze_keys)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_fn(f"Parameter trainable setelah freeze: {trainable:,}")
    log_fn(f"Modul yang dilatih: {unfreeze_keys}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_HP["lr"],
        weight_decay=FINETUNE_HP["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_HP["epochs"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "val_bal_acc": [], "val_auc": [],
               "phase1_val": {k: round(float(v), 4) for k, v in vm0.items()}}
    best_epoch, patience_cnt = 0, 0

    log_fn("\nHyperparameter Finetuning:")
    for k, v in FINETUNE_HP.items():
        if k == "unfreeze_keys":
            continue
        log_fn(f"  {k:<20}: {v}")
    log_fn("\nMemulai Finetuning...")
    log_fn("-" * 72)

    for epoch in range(1, FINETUNE_HP["epochs"] + 1):
        t0 = time.time()

        model.train()
        _freeze_bn_stats(model, unfreeze_keys)
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.view(-1).long().to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)
        scheduler.step()

        vm = common.validate(model, val_loader, device, n_classes, criterion)
        tl, ta = t_loss / t_total, 100 * t_correct / t_total

        history["train_loss"].append(round(tl, 4))
        history["val_loss"].append(round(vm["loss"], 4))
        history["train_acc"].append(round(ta, 2))
        history["val_acc"].append(round(vm["acc"], 2))
        history["val_bal_acc"].append(round(vm["balanced_acc"], 2))
        history["val_auc"].append(round(vm["auc_macro"], 4))

        marker = ""
        if common.is_better(vm[sel_key], best_score, higher):
            best_score, best_epoch = vm[sel_key], epoch
            patience_cnt = 0
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            marker = " ✓ SAVED"
        else:
            patience_cnt += 1

        log_fn(
            f"Finetune [{epoch:>2}/{FINETUNE_HP['epochs']}] "
            f"Loss: {tl:.4f}/{vm['loss']:.4f}  "
            f"Acc: {ta:.1f}%/{vm['acc']:.1f}%  "
            f"BalAcc: {vm['balanced_acc']:.1f}%  AUC: {vm['auc_macro']:.4f}  "
            f"({time.time()-t0:.0f}s){marker}"
        )

        if patience_cnt >= FINETUNE_HP["patience"]:
            log_fn(f"\nEarly stopping pada epoch {epoch}.")
            break

    history["best_epoch"]    = best_epoch or None
    history["select_metric"] = select_metric

    if best_epoch == 0:
        log_fn("")
        _copy_phase1(log_fn, f"Fine-tuning TIDAK memperbaiki val {select_metric} "
                             f"dibanding Fase 1 ({best_score:.4f}).")
    else:
        log_fn(f"\nFine-tuning memperbaiki val {select_metric}: "
               f"{vm0[sel_key]:.4f} -> {best_score:.4f} (epoch {best_epoch})")
        log_fn(f"Model final tersimpan -> {FINAL_MODEL_PATH}")

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    common.plot_history(history, CURVE_PATH, title_prefix="Fine-tuning")
    log_fn(f"Chart tersimpan -> {CURVE_PATH}")

    return {
        "best_epoch":  best_epoch or None,
        "best_score":  best_score,
        "finetuned":   best_epoch > 0,
        "final_model": FINAL_MODEL_PATH,
        "chart_path":  CURVE_PATH,
    }