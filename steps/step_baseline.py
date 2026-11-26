"""
Step 4 (GUI) - Baseline ResNet-18 (tanpa modul atensi) — MODEL PEMBANDING
==========================================================================
Melatih ResNet-18 POLOS (attention="none"), lalu langsung mengevaluasinya di
test set. Laporan skripsi memposisikan model ini sebagai PEMBANDING, sedangkan
ResNet-18 + FSCA (Step 5–6) adalah model usulan (config.MAIN_MODEL = "fsca").
Bila dosen memilih baseline sebagai model utama, cukup set MAIN_MODEL = "baseline".

Agar perbandingan dengan FSCA adil, baseline memakai komponen yang SAMA:
- Arsitektur  : step4_model.build_model(attention="none") -> stem 3x3 stride 1,
                tanpa maxpool, pretrained ImageNet, dropout, FC yang sama.
                Satu-satunya beda dengan model FSCA: slot fsca1..fsca4 = Identity.
- Training    : step5_training.train_one_run() dengan TRAIN_HP yang sama
                (epoch, batch size, lr, weight decay, seed, select_metric,
                Weighted CE, augmentasi).
- Evaluasi    : step7_evaluation.evaluate_checkpoint() dengan EVAL_CONFIG dan
                IMBALANCE yang sama dengan Step 7.

Output: outputs/models/ResNet18_Baseline_DermaMNIST_Best.pth
        outputs/json/baseline_training_history.json
        outputs/png/baseline_training_curve.png
        outputs/json/baseline_evaluation_report.json
        outputs/png/baseline_confusion_matrix.png
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, TRAIN_HP

META_PATH     = PATHS["dataset_meta"]
CONFIG_PATH   = PATHS["preprocess_config"]
SPLIT_PATH    = PATHS["split_info"]
MODEL_PATH    = PATHS["baseline_model"]
HISTORY_PATH  = PATHS["baseline_history"]
CURVE_PATH    = PATHS["baseline_curve_chart"]
REPORT_PATH   = PATHS["baseline_report"]
CM_PATH       = PATHS["baseline_cm_chart"]

ATTENTION = "none"


def run(log_fn, hp_override=None):
    sys.path.insert(0, os.path.dirname(__file__))
    import common
    from step5_training import train_one_run
    from step7_evaluation import evaluate_checkpoint
    from step4_model import build_model

    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2"), (SPLIT_PATH, "Step 3")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} terlebih dahulu.")
            raise FileNotFoundError(path)

    with open(META_PATH) as f:
        n_classes = json.load(f)["n_classes"]

    # Ringkasan arsitektur baseline
    model = build_model(n_classes=n_classes, pretrained=False, attention=ATTENTION)
    total_params = sum(p.numel() for p in model.parameters())
    del model
    log_fn("Model: ResNet-18 baseline (tanpa modul atensi)")
    log_fn(f"Total parameter: {total_params:,}")
    log_fn("Resep training identik dengan ResNet-18 + FSCA (TRAIN_HP di config.py).")
    log_fn("")

    hp = {**TRAIN_HP, **(hp_override or {})}
    log_fn("Hyperparameter Training:")
    for k, v in hp.items():
        log_fn(f"  {k:<15}: {v}")
    log_fn("")

    # ---------------- Training ----------------
    result = train_one_run(hp, log_fn, ckpt_path=MODEL_PATH,
                           attention=ATTENTION, tag="Baseline")
    history = result["history"]
    history["attention"] = ATTENTION

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    common.plot_history(history, CURVE_PATH, title_prefix="ResNet-18 Baseline")
    log_fn(f"Training curve tersimpan -> {CURVE_PATH}")

    # ---------------- Evaluasi test set ----------------
    log_fn("\n" + "=" * 58)
    log_fn("  EVALUASI TEST SET — ResNet-18 Baseline")
    log_fn("=" * 58)
    report = evaluate_checkpoint(MODEL_PATH, log_fn, attention=ATTENTION,
                                 report_path=REPORT_PATH, cm_path=CM_PATH,
                                 history_path=HISTORY_PATH, label="ResNet-18 Baseline")

    return {
        "best_epoch":   result["best_epoch"],
        "best_val_auc": result["best_val_auc"],
        "test_auc":     report["auc_macro"],
        "test_acc":     report["accuracy"],
        "test_bal_acc": report["balanced_accuracy"],
        "test_melanoma_recall": report.get("melanoma_recall"),
        "total_params": total_params,
        "model_path":   MODEL_PATH,
        "curve_path":   CURVE_PATH,
        "cm_path":      CM_PATH,
    }
