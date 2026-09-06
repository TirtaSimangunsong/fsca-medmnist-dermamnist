"""
Step 6 - Finetuning (Phase 2 / Polish)
========================================
Melanjutkan dari checkpoint Step 5 dengan:
- Learning rate lebih kecil
- MixUp dimatikan dan augmentasi diperlembut (aug_strength)
- Label smoothing yang KINI benar-benar dipasang
- Bobot loss dengan skala yang sama seperti Step 5
- Cosine annealing tanpa warmup

Output: outputs/ResNet18_FSCA_DermaMNIST_Best.pth
        outputs/finetune_curve.png
        outputs/finetune_history.json

RINGKASAN PERBAIKAN dari versi sebelumnya
------------------------------------------
1. Versi lama memakai `split_info["class_weights"]` (dinormalisasi agar
   jumlahnya 1, jadi rata-ratanya ~0.14) sementara Step 5 memakai
   `loss_class_weights` (rata-rata 1). Efeknya loss Phase 2 terskala ~7x lebih
   kecil dari Phase 1; digabung LR yang sudah 10x lebih kecil, fine-tuning
   praktis tidak menggerakkan bobot sama sekali. Kini keduanya memakai
   normalisasi yang sama lewat common.compute_loss_weights().
2. `label_smoothing` ada di FINETUNE_HP dan disebut di docstring lama, tetapi
   tidak pernah diteruskan ke nn.CrossEntropyLoss. Sekarang dipasang.
3. Freezing default dimatikan. Pada citra 28x28 dengan stem yang diganti,
   membekukan backbone menghalangi fitur menyesuaikan diri dengan distribusi
   input yang baru. Discriminative LR (backbone_lr_mult) memberi efek
   perlindungan yang sama tanpa kerugian tersebut. Set FINETUNE_HP["freeze"]
   = True jika ingin kembali ke perilaku lama.
4. Device tidak lagi di-hardcode "cpu".
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, FINETUNE_HP, ensure_dirs

OUTPUT_DIR       = PATHS["output_dir"]
META_PATH        = PATHS["dataset_meta"]
CONFIG_PATH      = PATHS["preprocess_config"]
SPLIT_PATH       = PATHS["split_info"]
CKPT_PATH        = PATHS["checkpoint_best"]
FINAL_MODEL_PATH = PATHS["final_model"]
HISTORY_PATH     = PATHS["finetune_history"]
CURVE_PATH       = PATHS["finetune_curve_chart"]


def run(log_fn, hp_override=None):
    import torch

    sys.path.insert(0, os.path.dirname(__file__))
    from step5_training import train_one_run, _plot_curve

    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2"),
                       (CKPT_PATH, "Step 5")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} dulu.")
            raise FileNotFoundError(path)

    ensure_dirs()
    hp = {**FINETUNE_HP, **(hp_override or {})}

    log_fn("=" * 78)
    log_fn("  STEP 6 - FINETUNING PHASE 2 (POLISH)")
    log_fn("=" * 78)
    log_fn("Memuat checkpoint terbaik dari Step 5...")
    init_state = torch.load(CKPT_PATH, map_location="cpu")
    log_fn("Checkpoint berhasil dimuat.")

    history, best, info = train_one_run(
        hp, log_fn,
        ckpt_path=FINAL_MODEL_PATH,
        train_strength=hp.get("aug_strength", 0.5),
        use_mixup=hp.get("mixup", False),
        init_state=init_state,
        tag="Finetune",
    )

    log_fn("\nMembuat kurva finetuning...")
    _plot_curve(history, CURVE_PATH,
                title=f"Phase 2 - ResNet-18 + {info['attention'].upper()}")
    log_fn(f"Kurva tersimpan -> {CURVE_PATH}")

    payload = {"history": history, "best": best, "hyperparameters": hp,
               "data_info": info, "chart_path": CURVE_PATH,
               "final_model": FINAL_MODEL_PATH}
    with open(HISTORY_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    log_fn(f"History tersimpan -> {HISTORY_PATH}")
    log_fn(f"Model final -> {FINAL_MODEL_PATH}")

    # Peringatan bila Phase 2 tidak memperbaiki apa pun
    try:
        with open(PATHS["training_history"]) as f:
            p1 = json.load(f)["best"]
        if best["score"] <= p1["score"]:
            log_fn("")
            log_fn("CATATAN: Phase 2 tidak melampaui Phase 1. Yang biasanya membantu:")
            log_fn("  - turunkan FINETUNE_HP['lr'] satu tingkat lagi (mis. 5e-5)")
            log_fn("  - naikkan FINETUNE_HP['epochs'] agar cosine sempat menurun penuh")
            log_fn("  - atau lewati Step 6 dan pakai checkpoint Phase 1 langsung")
    except Exception:
        pass

    return payload