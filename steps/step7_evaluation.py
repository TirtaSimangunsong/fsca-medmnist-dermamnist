"""
Step 7 - Evaluation pada Test Set
====================================
Evaluasi model final dari Step 6 pada test set DermaMNIST.

Menghasilkan:
- Accuracy, Balanced Accuracy, Weighted/Macro F1, Macro AUC
- Confusion matrix (count + row-normalized)
- Metrik per kelas
- Sapuan logit adjustment (tau dituning di VALIDATION, dilaporkan di TEST)
- Test-time augmentation 8-arah

Output: outputs/confusion_matrix.png, outputs/evaluation_report.json

RINGKASAN PERBAIKAN dari versi sebelumnya
------------------------------------------
1. Device tidak lagi di-hardcode "cpu".
2. `build_model(pretrained=False)` diganti - saat memuat state_dict, bobot
   ImageNet tidak relevan, tapi arsitektur (jumlah/penempatan modul atensi)
   harus persis sama dengan saat training. Kini diambil dari MODEL_CONFIG.
3. Ditambahkan Balanced Accuracy. Dengan kelas `nv` menguasai ~67% data,
   akurasi mentah bisa terlihat tinggi walau kelas minoritas nyaris tak pernah
   terdeteksi. Penguji hampir pasti menanyakan hal ini.
4. Ditambahkan TTA dan logit adjustment - keduanya menaikkan angka tanpa
   melatih ulang apa pun.
5. Confusion matrix ternormalisasi baris ditambahkan agar pola kesalahan
   antar-kelas terbaca meski jumlah sampel per kelas timpang.
"""

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, EVAL_CONFIG, IMBALANCE, ensure_dirs

OUTPUT_DIR  = PATHS["output_dir"]
META_PATH   = PATHS["dataset_meta"]
CONFIG_PATH = PATHS["preprocess_config"]
FINAL_MODEL = PATHS["final_model"]
CKPT_PATH   = PATHS["checkpoint_best"]
REPORT_PATH = PATHS["evaluation_report"]
CM_PATH     = PATHS["confusion_matrix_chart"]


def _plot_confusion(cm, class_names, acc, bal, auc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    fig.patch.set_facecolor("#1a1a1a")

    for ax, data, fmt, cmap, sub in [
        (axes[0], cm, "d", "Blues", "Jumlah"),
        (axes[1], cm_norm, ".2f", "Greens", "Normalisasi per baris (recall)"),
    ]:
        ax.set_facecolor("#1a1a1a")
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    linewidths=0.5, linecolor="#333", cbar=False,
                    annot_kws={"size": 8, "color": "white"})
        ax.set_title(sub, color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", color="white", fontsize=9)
        ax.set_ylabel("Actual", color="white", fontsize=9)
        ax.tick_params(colors="white")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="white", fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, color="white", fontsize=7)

    fig.suptitle(f"Confusion Matrix  |  Acc {acc:.2f}%  |  Balanced Acc {bal:.2f}%  |  AUC {auc:.4f}",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()


def run(log_fn):
    import torch
    from sklearn.metrics import classification_report, confusion_matrix

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model
    import common

    model_path = FINAL_MODEL if os.path.exists(FINAL_MODEL) else CKPT_PATH
    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2"),
                       (model_path, "Step 5/6")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} dulu.")
            raise FileNotFoundError(path)

    ensure_dirs()
    with open(META_PATH) as f:
        meta = json.load(f)
    with open(CONFIG_PATH) as f:
        pre = json.load(f)

    n_classes = meta["n_classes"]
    class_map = meta["class_map"]
    class_names = [class_map[str(i)] for i in range(n_classes)]

    device = common.resolve_device(log_fn=log_fn)

    log_fn(f"Memuat model final: {os.path.basename(model_path)}")
    model = build_model(n_classes=n_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    log_fn(f"Model berhasil dimuat (atensi: {model.attention_name.upper()}).")

    loaders, dinfo = common.build_dataloaders(
        mean=pre["mean"], std=pre["std"],
        batch_size=EVAL_CONFIG["batch_size"], n_classes=n_classes,
        splits=("train", "val", "test"), log_fn=log_fn,
    )
    val_loader, test_loader = loaders["val"], loaders["test"]
    class_counts = dinfo["class_counts"]

    use_tta = EVAL_CONFIG.get("tta", False)
    log_fn(f"TTA 8-arah: {'aktif' if use_tta else 'nonaktif'}")

    # ---------- Baseline: tanpa TTA, tanpa logit adjustment ----------
    log_fn("\nEvaluasi dasar (tanpa TTA, tanpa logit adjustment)...")
    m_plain = common.evaluate(model, test_loader, device, n_classes, use_tta=False)
    log_fn(f"  acc {m_plain['acc']:.2f}% | bal {m_plain['balanced_acc']:.2f}% | "
           f"auc {m_plain['auc_macro']:.4f}")

    # ---------- Logit adjustment: tau dituning di VALIDATION ----------
    tau = float(IMBALANCE.get("logit_adjustment_tau", 0.0))
    tau_table = []
    if IMBALANCE.get("auto_tune_tau", False):
        log_fn("")
        tau, tau_table = common.tune_logit_adjustment(
            model, val_loader, device, n_classes, class_counts,
            metric="balanced_acc", use_tta=use_tta, log_fn=log_fn,
        )

    adj = common.make_logit_adjustment(class_counts, tau) if tau else None

    # ---------- Evaluasi final ----------
    log_fn(f"\nEvaluasi final (TTA={use_tta}, tau={tau})...")
    m = common.evaluate(model, test_loader, device, n_classes,
                        use_tta=use_tta, logit_adjust=adj)

    acc, bal = m["acc"], m["balanced_acc"]
    f1_w, f1_macro, auc = m["f1_weighted"], m["f1_macro"], m["auc_macro"]
    y_true, y_pred = m["y_true"], m["y_pred"]

    log_fn("")
    log_fn("=" * 58)
    log_fn(f"  Accuracy (Test)      : {acc:.2f}%")
    log_fn(f"  Balanced Accuracy    : {bal:.2f}%")
    log_fn(f"  F1 Weighted          : {f1_w:.4f}")
    log_fn(f"  F1 Macro             : {f1_macro:.4f}")
    log_fn(f"  AUC Macro (OvR)      : {auc:.4f}")
    log_fn("=" * 58)
    log_fn(f"  Kontribusi TTA + logit adj : {acc - m_plain['acc']:+.2f} poin akurasi, "
           f"{bal - m_plain['balanced_acc']:+.2f} poin balanced")
    log_fn("=" * 58)

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    log_fn("\nPer-Class Report:")
    log_fn(f"  {'Kelas':<42} {'P':>7} {'R':>7} {'F1':>7} {'N':>6}")
    for cls in class_names:
        r = report[cls]
        log_fn(f"  {cls:<42} {r['precision']:>7.3f} {r['recall']:>7.3f} "
               f"{r['f1-score']:>7.3f} {int(r['support']):>6}")

    log_fn("\nMembuat confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion(cm, class_names, acc, bal, auc, CM_PATH)
    log_fn(f"Confusion matrix tersimpan -> {CM_PATH}")

    eval_result = {
        "model_path":     model_path,
        "attention":      model.attention_name,
        "device":         str(device),
        "tta":            use_tta,
        "logit_adjustment_tau": tau,
        "tau_sweep_val":  tau_table,
        "accuracy":       round(acc, 4),
        "balanced_accuracy": round(bal, 4),
        "f1_weighted":    round(f1_w, 4),
        "f1_macro":       round(f1_macro, 4),
        "auc_macro":      round(auc, 4),
        "without_tta_or_adjustment": {
            "accuracy":          round(m_plain["acc"], 4),
            "balanced_accuracy": round(m_plain["balanced_acc"], 4),
            "f1_macro":          round(m_plain["f1_macro"], 4),
            "auc_macro":         round(m_plain["auc_macro"], 4),
        },
        "confusion_matrix": cm.tolist(),
        "class_names":      class_names,
        "class_counts_train": class_counts,
        "per_class": {
            name: {k: round(v, 4) for k, v in data.items() if isinstance(v, (int, float))}
            for name, data in report.items() if isinstance(data, dict)
        },
        "chart_path": CM_PATH,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(eval_result, f, indent=2)
    log_fn(f"Evaluation report tersimpan -> {REPORT_PATH}")

    return eval_result