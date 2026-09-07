"""
Step 5 - Training (Phase 1)
=============================
Training loop utama dengan:
- Backbone ImageNet pretrained yang BENAR-BENAR dipakai dan ikut dilatih
- Discriminative learning rate (backbone lr x 0.1, modul baru lr penuh)
- SATU strategi class imbalance (tidak lagi sampler + weighted loss sekaligus)
- MixUp / CutMix
- Label smoothing
- Linear warmup + cosine annealing
- EMA bobot
- Gradient clipping
- Seleksi checkpoint berdasarkan balanced accuracy / AUC (bukan val loss)

Output: outputs/checkpoint_best.pth, outputs/training_curve.png,
        outputs/training_history.json

RINGKASAN PERBAIKAN dari versi sebelumnya
------------------------------------------
1. BUG UTAMA. Versi lama memanggil `build_model(..., pretrained=False)` lalu
   membekukan semua parameter kecuali `fc` dan `fsca` ketika
   MODEL_CONFIG["pretrained"] bernilai True. Akibatnya backbone berbobot ACAK
   dibekukan dan tidak pernah dilatih - model hanya melatih classifier di atas
   ekstraktor fitur acak. Ini penyebab utama akurasi tidak naik.
2. Koreksi imbalance ganda. Versi lama memakai WeightedRandomSampler DAN
   CrossEntropyLoss(weight=...) bersamaan, sehingga kelas minoritas dibobot
   dua kali. Sekarang dipilih satu lewat IMBALANCE["strategy"].
3. Device tidak lagi di-hardcode "cpu" - otomatis memakai MPS di Mac.
4. Early stopping tidak lagi memantau akurasi mentah. Pada data tidak seimbang,
   akurasi bisa terlihat naik hanya karena model makin bias ke kelas mayoritas.
"""

import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (PATHS, DATASET, TRAIN_HP, MODEL_CONFIG, IMBALANCE,
                    MIXUP, ensure_dirs)

OUTPUT_DIR   = PATHS["output_dir"]
META_PATH    = PATHS["dataset_meta"]
CONFIG_PATH  = PATHS["preprocess_config"]
SPLIT_PATH   = PATHS["split_info"]
CKPT_PATH    = PATHS["checkpoint_best"]
HISTORY_PATH = PATHS["training_history"]
CURVE_PATH   = PATHS["training_curve_chart"]

# Dipertahankan untuk kompatibilitas dengan kode/notebook lama
DEFAULT_CONFIG = TRAIN_HP


def train_one_run(hp, log_fn, ckpt_path=CKPT_PATH, attention=None,
                  seed=None, train_strength=1.0, use_mixup=None,
                  init_state=None, tag="Train"):
    """
    Satu proses training penuh. Dipakai oleh Step 5, Step 6, dan Step 11
    (ablasi multi-seed) supaya resep trainingnya identik dan perbandingannya adil.

    Returns: (history dict, best_metrics dict, info dict)
    """
    import torch
    import torch.nn as nn

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model
    import common

    with open(META_PATH) as f:
        meta = json.load(f)
    with open(CONFIG_PATH) as f:
        pre = json.load(f)

    n_classes = meta["n_classes"]
    seed = hp.get("seed", 42) if seed is None else seed
    common.set_seed(seed)
    device = common.resolve_device(log_fn=log_fn)

    log_fn(f"Seed: {seed}")
    log_fn("Hyperparameter:")
    for k, v in hp.items():
        log_fn(f"  {k:<18}: {v}")
    log_fn("")

    # ---------- Data ----------
    loaders, dinfo = common.build_dataloaders(
        mean=pre["mean"], std=pre["std"],
        batch_size=hp["batch_size"], n_classes=n_classes,
        splits=("train", "val"), train_strength=train_strength,
        log_fn=log_fn, seed=seed,
    )
    train_loader, val_loader = loaders["train"], loaders["val"]
    class_counts = dinfo["class_counts"]
    log_fn(f"  Train batches : {len(train_loader)}")
    log_fn(f"  Val batches   : {len(val_loader)}")

    # ---------- Model ----------
    # pretrained diambil dari MODEL_CONFIG, tidak lagi dipaksa False.
    model = build_model(n_classes=n_classes, attention=attention).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
        log_fn("Bobot awal dimuat dari checkpoint sebelumnya.")

    total_p = sum(p.numel() for p in model.parameters())
    log_fn(f"Model: ResNet-18 + {model.attention_name.upper()}  ({total_p:,} params)")

    # ---------- Freeze opsional ----------
    if hp.get("freeze", False):
        keys = hp.get("unfreeze_keys", [])
        for name, p in model.named_parameters():
            p.requires_grad = any(k in name for k in keys)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_fn(f"Freeze aktif. Trainable: {n_tr:,} params (unfreeze: {keys})")

    # ---------- Loss ----------
    w = dinfo["loss_weights"]
    weight_tensor = torch.tensor(w, dtype=torch.float, device=device) if w else None
    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=hp.get("label_smoothing", 0.0),
    )
    log_fn(f"Loss: CrossEntropy(label_smoothing={hp.get('label_smoothing', 0.0)}, "
           f"weighted={'ya' if weight_tensor is not None else 'tidak'})")

    # ---------- Optimizer & scheduler ----------
    optimizer = common.build_optimizer(
        model, lr=hp["lr"], weight_decay=hp["weight_decay"],
        backbone_lr_mult=hp.get("backbone_lr_mult", 0.1),
        eps=hp.get("adam_eps", 1e-6), log_fn=log_fn,
    )
    scheduler = common.WarmupCosine(
        optimizer, total_epochs=hp["epochs"],
        warmup_epochs=hp.get("warmup_epochs", 0),
    )

    # ---------- MixUp ----------
    mix_on = MIXUP["enabled"] if use_mixup is None else use_mixup
    mixer = common.MixupCutmix(
        mixup_alpha=MIXUP["mixup_alpha"], cutmix_alpha=MIXUP["cutmix_alpha"],
        prob=MIXUP["prob"], switch_prob=MIXUP["switch_prob"],
    ) if mix_on else None
    log_fn(f"MixUp/CutMix: {'aktif' if mix_on else 'nonaktif'}")

    # ---------- EMA ----------
    ema_decay = hp.get("ema_decay", 0.0)
    ema_warmup = hp.get("ema_warmup_steps", 3 * max(1, len(train_loader)))
    ema = common.ModelEMA(model, decay=ema_decay, warmup_steps=ema_warmup) if ema_decay else None
    if ema:
        log_fn(f"EMA aktif (decay={ema_decay}, warmup {ema_warmup} langkah)")

    monitor = hp.get("monitor", "balanced_acc")
    clip = hp.get("grad_clip", 0.0)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "val_balanced_acc": [], "val_auc": [], "val_f1_macro": [], "lr": [],
               "grad_norm": []}
    best = {"score": -1.0, "epoch": 0, "source": "raw"}
    patience_cnt = 0

    skipped_batches = 0
    max_skipped = hp.get("max_skipped_batches", 5)   # cukup; forensik sudah dicetak di kejadian pertama
    clip_warned = False
    first_epoch_norms = []

    if clip:
        log_fn(f"Gradient clipping: {clip}")
    else:
        log_fn("Gradient clipping: nonaktif")

    log_fn(f"\nMemulai {tag} Loop (monitor: {monitor})...")
    log_fn("-" * 78)

    for epoch in range(1, hp["epochs"] + 1):
        t0 = time.time()
        lr_now = scheduler.current_lr()

        # ----- Train -----
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        epoch_grad_norm = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.squeeze(1).long().to(device, non_blocking=True)

            if mixer is not None:
                imgs_m, y_a, y_b, lam = mixer(imgs, labels)
                outputs = model(imgs_m)
                loss = common.mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            # --- PENJAGA NaN ---
            # Tanpa ini, run sebelumnya membuang 40 epoch memproses NaN dan
            # menulis checkpoint rusak. Sekarang berhenti pada kejadian pertama.
            if not torch.isfinite(loss):
                log_fn("")
                log_fn("!" * 78)
                log_fn(f"DIVERGENSI pada epoch {epoch}: loss = {loss.item()}")
                log_fn("Training dihentikan. Yang biasanya menyebabkan ini:")
                log_fn("  - learning rate terlalu besar (turunkan TRAIN_HP['lr'])")
                log_fn("  - weight_decay terlalu besar (coba 1e-2 alih-alih 5e-2)")
                log_fn("  - aktivasi meledak dari inisialisasi stem")
                log_fn("  - bug numerik backend MPS")
                log_fn("Jalankan `python diagnose.py` untuk mengisolasi penyebabnya.")
                log_fn("!" * 78)
                raise RuntimeError(
                    f"Loss menjadi non-finite pada epoch {epoch}. "
                    f"Jalankan diagnose.py."
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Norm gradien dihitung dengan implementasi aman-MPS.
            grad_norm, bad_params = common.safe_clip_grad_norm(
                model.named_parameters(), clip
            )

            if bad_params or not np.isfinite(grad_norm):
                # Satu batch buruk BUKAN alasan membatalkan seluruh run.
                # Batch itu dilewati, optimizer tidak melangkah, training lanjut.
                # Run dibatalkan hanya kalau ini terjadi berulang kali.
                skipped_batches += 1
                if skipped_batches == 1:
                    # Forensik lengkap pada kejadian PERTAMA, di kondisi nyata.
                    try:
                        common.forensics_report(
                            model, imgs, labels, outputs, loss, log_fn,
                            dump_path=os.path.join(OUTPUT_DIR, "batch_gagal.pt"),
                        )
                    except Exception as e:
                        log_fn(f"  (forensik gagal dijalankan: {e})")
                if skipped_batches <= 3:
                    names = ", ".join(bad_params[:3]) if bad_params else "(norm total)"
                    log_fn(f"  [epoch {epoch}] batch dilewati - gradien non-finite "
                           f"pada: {names}")
                if skipped_batches > max_skipped:
                    log_fn("")
                    log_fn("!" * 78)
                    log_fn(f"DIVERGENSI: {skipped_batches} batch dilewati, melewati "
                           f"batas {max_skipped}.")
                    log_fn("Training dihentikan. Jalankan `python diagnose_step.py`")
                    log_fn("untuk mengetahui parameter mana yang meledak.")
                    log_fn("!" * 78)
                    raise RuntimeError(
                        f"Terlalu banyak batch dengan gradien non-finite "
                        f"({skipped_batches}). Jalankan diagnose_step.py."
                    )
                optimizer.zero_grad(set_to_none=True)
                continue

            epoch_grad_norm = max(epoch_grad_norm, float(grad_norm))
            if epoch == 1:
                first_epoch_norms.append(float(grad_norm))

            optimizer.step()
            if ema:
                ema.update(model)

            run_loss += loss.item() * imgs.size(0)
            # akurasi train hanya indikatif saat mixup aktif
            run_correct += (outputs.argmax(1) == labels).sum().item()
            run_total += imgs.size(0)

        # Peringatkan kalau clipping terlalu agresif dibanding norm sebenarnya.
        # Clipping seharusnya menangkap LONJAKAN, bukan memotong tiap langkah.
        if epoch == 1 and clip and first_epoch_norms and not clip_warned:
            med = float(np.median(first_epoch_norms))
            if clip < 0.5 * med:
                log_fn("")
                log_fn(f"  PERINGATAN: grad_clip={clip} jauh di bawah norm gradien "
                       f"khas ({med:.1f}).")
                log_fn(f"  Setiap langkah dipotong ~{med/max(clip,1e-9):.0f}x, bukan "
                       f"sekadar menangkap lonjakan.")
                log_fn(f"  Ini bisa membuat sqrt(v_hat) di AdamW jatuh ke wilayah eps "
                       f"dan meledakkan update.")
                log_fn(f"  Saran: set grad_clip=0.0 (nonaktif) atau "
                       f"{2*med:.0f} di config.py.")
                log_fn("")
            clip_warned = True

        scheduler.step()

        # ----- Validation -----
        m_raw = common.evaluate(model, val_loader, device, n_classes, criterion=criterion)
        m_use = m_raw
        source = "raw"
        # EMA hanya boleh ikut seleksi checkpoint setelah cukup langkah.
        # Di run sebelumnya, EMA "menang" di epoch 1 - saat isinya masih
        # hampir seluruhnya bobot inisialisasi - dan checkpoint rusak itulah
        # yang tersimpan sebagai model terbaik.
        if ema and ema.ready():
            m_ema = common.evaluate(ema.module, val_loader, device, n_classes, criterion=criterion)
            if m_ema[monitor] > m_raw[monitor]:
                m_use, source = m_ema, "ema"

        t_loss = run_loss / max(1, run_total)
        t_acc = 100.0 * run_correct / max(1, run_total)

        history["train_loss"].append(round(t_loss, 4))
        history["val_loss"].append(round(m_use["loss"], 4))
        history["train_acc"].append(round(t_acc, 2))
        history["val_acc"].append(round(m_use["acc"], 2))
        history["val_balanced_acc"].append(round(m_use["balanced_acc"], 2))
        history["val_auc"].append(round(m_use["auc_macro"], 4))
        history["val_f1_macro"].append(round(m_use["f1_macro"], 4))
        history["lr"].append(lr_now)
        history["grad_norm"].append(round(epoch_grad_norm, 3))

        score = m_use[monitor]
        marker = ""
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "source": source,
                    "acc": m_use["acc"], "balanced_acc": m_use["balanced_acc"],
                    "auc": m_use["auc_macro"], "f1_macro": m_use["f1_macro"]}
            patience_cnt = 0
            state = (ema.module if source == "ema" else model).state_dict()
            torch.save(state, ckpt_path)
            marker = f"  <- SAVED ({source})"
        else:
            patience_cnt += 1

        log_fn(
            f"{tag} [{epoch:>3}/{hp['epochs']}] "
            f"lr {lr_now:.2e} | loss {t_loss:.3f}/{m_use['loss']:.3f} | "
            f"acc {m_use['acc']:.2f}% | bal {m_use['balanced_acc']:.2f}% | "
            f"auc {m_use['auc_macro']:.4f} | gn {epoch_grad_norm:.1f} | "
            f"{time.time()-t0:.0f}s{marker}"
        )

        if skipped_batches:
            log_fn(f"    (kumulatif {skipped_batches} batch dilewati karena gradien non-finite)")

        if patience_cnt >= hp.get("patience", 10**9):
            log_fn(f"\nEarly stopping pada epoch {epoch} "
                   f"(tidak ada perbaikan {monitor} selama {patience_cnt} epoch).")
            break

    log_fn("-" * 78)
    log_fn(f"Terbaik pada epoch {best['epoch']} (bobot {best['source']}): "
           f"acc {best['acc']:.2f}% | balanced {best['balanced_acc']:.2f}% | "
           f"auc {best['auc']:.4f}")
    log_fn(f"Checkpoint -> {ckpt_path}")

    info = {**dinfo, "seed": seed, "device": str(device),
            "attention": model.attention_name, "n_classes": n_classes,
            "class_counts": class_counts}
    return history, best, info


def _plot_curve(history, path, title="Training Curve"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor("#1a1a1a")

    panels = [
        (axes[0], "Loss", [("train_loss", "Train", "#4FC3F7"), ("val_loss", "Val", "#FF8A65")]),
        (axes[1], "Accuracy (%)", [("train_acc", "Train", "#4FC3F7"),
                                   ("val_acc", "Val", "#FF8A65"),
                                   ("val_balanced_acc", "Val (balanced)", "#81C784")]),
        (axes[2], "Val AUC / F1-macro", [("val_auc", "AUC", "#BA68C8"),
                                         ("val_f1_macro", "F1-macro", "#FFD54F")]),
    ]
    for ax, label, series in panels:
        ax.set_facecolor("#1a1a1a")
        for key, name, color in series:
            if history.get(key):
                ax.plot(ep, history[key], label=name, color=color, linewidth=1.6)
        ax.set_title(label, color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", color="white")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("#555")
        ax.grid(alpha=0.15)
        ax.legend(facecolor="#2a2a2a", edgecolor="#555", labelcolor="white", fontsize=8)

    fig.suptitle(title, color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()


def run(log_fn, hp_override=None):
    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} dulu.")
            raise FileNotFoundError(path)

    ensure_dirs()
    hp = {**TRAIN_HP, **(hp_override or {})}

    log_fn("=" * 78)
    log_fn("  STEP 5 - TRAINING PHASE 1")
    log_fn("=" * 78)

    history, best, info = train_one_run(hp, log_fn, ckpt_path=CKPT_PATH, tag="Train")

    log_fn("\nMembuat kurva training...")
    _plot_curve(history, CURVE_PATH,
                title=f"Phase 1 - ResNet-18 + {info['attention'].upper()}")
    log_fn(f"Kurva tersimpan -> {CURVE_PATH}")

    payload = {"history": history, "best": best, "hyperparameters": hp,
               "data_info": info, "chart_path": CURVE_PATH}
    with open(HISTORY_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    log_fn(f"History tersimpan -> {HISTORY_PATH}")

    # split_info tetap ditulis agar step lain yang masih membacanya tidak rusak
    with open(SPLIT_PATH, "w") as f:
        json.dump({
            "class_counts":       info["class_counts"],
            "loss_class_weights": info["loss_weights"] or [1.0] * info["n_classes"],
            "class_weights":      info["loss_weights"] or [1.0] * info["n_classes"],
            "strategy":           info["strategy"],
        }, f, indent=2)

    return payload