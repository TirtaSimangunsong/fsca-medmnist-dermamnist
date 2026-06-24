"""
Step 7 - Evaluation pada Test Set
====================================
Evaluasi model final dari Step 6 menggunakan test set DermaMNIST.
Menghasilkan:
- Accuracy, Weighted F1, Macro AUC
- Confusion Matrix (heatmap)
- Per-class precision/recall/F1

Output: outputs/confusion_matrix.png
        outputs/evaluation_report.json
"""

import os
import json
import sys
import numpy as np

OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "outputs")
META_PATH     = os.path.join(OUTPUT_DIR, "dataset_meta.json")
CONFIG_PATH   = os.path.join(OUTPUT_DIR, "preprocess_config.json")
FINAL_MODEL   = os.path.join(OUTPUT_DIR, "ResNet18_FSCA_DermaMNIST_Best.pth")
REPORT_PATH   = os.path.join(OUTPUT_DIR, "evaluation_report.json")
CM_PATH       = os.path.join(OUTPUT_DIR, "confusion_matrix.png")


def run(log_fn):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from medmnist import DermaMNIST
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report,
        confusion_matrix, roc_auc_score
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model

    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2"), (FINAL_MODEL, "Step 6")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} terlebih dahulu.")
            raise FileNotFoundError(path)

    with open(META_PATH)   as f: meta   = json.load(f)
    with open(CONFIG_PATH) as f: config = json.load(f)

    n_classes = meta["n_classes"]
    class_map = meta["class_map"]
    class_names = [class_map[str(i)] for i in range(n_classes)]

    device = torch.device("cpu")

    log_fn("Memuat model final...")
    model = build_model(n_classes=n_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(FINAL_MODEL, map_location=device))
    model.eval()
    log_fn("Model berhasil dimuat.")

    # Test DataLoader
    mean, std = config["mean"], config["std"]
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_ds = DermaMNIST(split="test", transform=test_transform, download=True, size=28)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    log_fn(f"Mengevaluasi {len(test_ds):,} sampel test...")

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device)
            labels = labels.squeeze(1).long()
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # Metrics
    acc      = 100 * accuracy_score(all_labels, all_preds)
    f1_w     = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    log_fn("")
    log_fn("=" * 50)
    log_fn(f"  Accuracy (Test)     : {acc:.2f}%")
    log_fn(f"  F1 Weighted         : {f1_w:.4f}")
    log_fn(f"  F1 Macro            : {f1_macro:.4f}")
    log_fn(f"  AUC Macro (OvR)     : {auc:.4f}")
    log_fn("=" * 50)

    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    log_fn("\nPer-Class Report:")
    for cls in class_names:
        r = report[cls]
        log_fn(f"  {cls:<40}: P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}")

    # Confusion Matrix
    log_fn("\nMembuat confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="#333",
        annot_kws={"size": 9, "color": "white"}
    )
    ax.set_title(f"Confusion Matrix\nAccuracy: {acc:.2f}% | F1-W: {f1_w:.3f} | AUC: {auc:.3f}",
                 color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", color="white", fontsize=10)
    ax.set_ylabel("Actual",    color="white", fontsize=10)
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="white", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0,  color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()

    log_fn(f"Confusion matrix tersimpan → {CM_PATH}")

    # Simpan report
    eval_result = {
        "accuracy":   round(acc, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro":   round(f1_macro, 4),
        "auc_macro":  round(auc, 4),
        "per_class":  {
            cls_name: {metric: round(val, 4) for metric, val in cls_data.items() if isinstance(val, float)}
            for cls_name, cls_data in report.items()
            if isinstance(cls_data, dict)
        },
        "chart_path": CM_PATH,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(eval_result, f, indent=2)

    log_fn(f"Evaluation report tersimpan → {REPORT_PATH}")
    return eval_result