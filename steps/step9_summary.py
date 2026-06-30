"""
Step 9 - Summary Report
=========================
Mengumpulkan semua hasil dari Step 1–8 dan membuat:
- Summary card: accuracy, F1, AUC, model size, params
- Comparison table: FSCA vs baseline ResNet-18 (tanpa attention)
- PDF-ready summary chart

Output: outputs/summary_report.png
        outputs/summary_report.json
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS

OUTPUT_DIR   = PATHS["output_dir"]
META_PATH    = PATHS["dataset_meta"]
REPORT_PATH  = PATHS["evaluation_report"]
HISTORY_PATH = PATHS["finetune_history"]
FINAL_MODEL  = PATHS["final_model"]
SUMMARY_PNG  = PATHS["summary_report_chart"]
SUMMARY_JSON = PATHS["summary_json"]


def _get_model_size_mb(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception:
        return 0.0


def run(log_fn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Cek dependensi file
    missing = []
    for path, name in [(META_PATH, "Step 1"), (REPORT_PATH, "Step 7"), (FINAL_MODEL, "Step 6")]:
        if not os.path.exists(path):
            missing.append(f"{name} ({os.path.basename(path)})")
    if missing:
        for m in missing: log_fn(f"ERROR: Belum ada output dari {m}.")
        raise FileNotFoundError("Output step sebelumnya tidak ditemukan.")

    with open(META_PATH)   as f: meta    = json.load(f)
    with open(REPORT_PATH) as f: report  = json.load(f)

    history = {}
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f: history = json.load(f)

    # Metrics utama
    acc      = report["accuracy"]
    f1_w     = report["f1_weighted"]
    f1_mac   = report["f1_macro"]
    auc      = report["auc_macro"]
    model_mb = _get_model_size_mb(FINAL_MODEL)

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model
    import torch
    model = build_model(n_classes=meta["n_classes"])
    total_params = sum(p.numel() for p in model.parameters())

    log_fn("=" * 60)
    log_fn("       RINGKASAN PERFORMA MODEL FSCA ResNet-18")
    log_fn("=" * 60)
    log_fn(f"  Dataset        : DermaMNIST (28×28 piksel)")
    log_fn(f"  Jumlah Kelas   : {meta['n_classes']}")
    log_fn(f"  Total Sampel   : {meta['n_total']:,}")
    log_fn(f"")
    log_fn(f"  Accuracy Test  : {acc:.2f}%")
    log_fn(f"  F1 Weighted    : {f1_w:.4f}")
    log_fn(f"  F1 Macro       : {f1_mac:.4f}")
    log_fn(f"  AUC Macro      : {auc:.4f}")
    log_fn(f"")
    log_fn(f"  Total Params   : {total_params:,}")
    log_fn(f"  Model Size     : {model_mb:.1f} MB")
    log_fn("=" * 60)

    # Perbandingan dengan literatur (dari paper)
    log_fn("\nPerbandingan dengan State-of-the-Art:")
    comparison = [
        ("GLCM + Multi-SVM [Hasanah 2021]",         "N/A",   "—",     "—"),
        ("CNN + ViT Hybrid [Arshed 2023]",            "high",  "—",     "—"),
        ("Transfer + Deep Attention [Alotaibi 2025]", "high",  "—",     "—"),
        ("EDA-ResNet50 [Hosny 2025]",                 "high",  "—",     "—"),
        (f"ResNet-18 + FSCA (Ours)",                  "28×28", f"{acc:.2f}%", f"{f1_w:.4f}"),
    ]
    log_fn(f"  {'Model':<48} {'Res':>6} {'Acc':>8} {'F1-W':>8}")
    log_fn("  " + "-" * 75)
    for row in comparison:
        log_fn(f"  {row[0]:<48} {row[1]:>6} {row[2]:>8} {row[3]:>8}")

    # ---- BUAT SUMMARY CHART ----
    log_fn("\nMembuat summary chart...")
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a1a")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # 1. Metric cards (bar horizontal)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#222222")
    metrics_names = ["Accuracy (%)", "F1 Weighted", "F1 Macro", "AUC Macro"]
    metrics_vals  = [acc, f1_w * 100, f1_mac * 100, auc * 100]
    colors_m = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292"]
    bars = ax1.barh(metrics_names, metrics_vals, color=colors_m, edgecolor="#333", height=0.5)
    ax1.set_xlim(0, 105)
    ax1.set_title("Metrics Utama Model FSCA ResNet-18", color="white", fontweight="bold")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values(): spine.set_color("#555")
    for bar, val in zip(bars, metrics_vals):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", color="white", fontsize=10, fontweight="bold")

    # 2. Model info card
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#222222")
    ax2.axis("off")
    info_text = (
        f"Model Info\n"
        f"{'─'*22}\n"
        f"Arch   : ResNet-18\n"
        f"Modul  : FSCA ×4\n"
        f"Params : {total_params/1e6:.2f}M\n"
        f"Size   : {model_mb:.1f} MB\n"
        f"Input  : 28×28 px\n"
        f"Kelas  : {meta['n_classes']}\n"
        f"Device : CPU"
    )
    ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes,
             color="white", fontsize=9.5, va="top", fontfamily="monospace",
             bbox=dict(facecolor="#333", alpha=0.6, boxstyle="round,pad=0.5"))
    ax2.set_title("Info Model", color="white", fontweight="bold")

    # 3. Finetune loss curve (bawah kiri)
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor("#1a1a1a")
    if history:
        ep = range(1, len(history["val_acc"]) + 1)
        ax3.plot(ep, history["val_loss"], color="#ef5350", label="Val Loss",    linewidth=2)
        ax3.plot(ep, [v/100 for v in history["val_acc"]], color="#4fc3f7", label="Val Acc (scaled)", linewidth=2)
        ax3.axhline(y=f1_w, color="#81c784", linestyle="--", label=f"F1-W={f1_w:.3f}", linewidth=1.5)
        ax3.legend(facecolor="#333", labelcolor="white", fontsize=8)
    ax3.set_title("Finetune Curve & Final F1", color="white", fontweight="bold")
    ax3.set_xlabel("Epoch", color="white"); ax3.set_ylabel("Value", color="white")
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values(): spine.set_color("#555")

    # 4. Per-class F1 (bawah kanan)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor("#1a1a1a")
    class_names = [meta["class_map"][str(i)] for i in range(meta["n_classes"])]
    per_class_f1 = [report["per_class"].get(cn, {}).get("f1-score", 0) for cn in class_names]
    colors_c = plt.cm.Set2.colors
    ax4.barh(class_names, per_class_f1, color=colors_c[:len(class_names)], edgecolor="#333")
    ax4.set_xlim(0, 1.1)
    ax4.set_title("F1 Per Kelas", color="white", fontweight="bold")
    ax4.set_xlabel("F1-Score", color="white")
    ax4.tick_params(colors="white")
    plt.setp(ax4.get_yticklabels(), color="white", fontsize=7)
    for spine in ax4.spines.values(): spine.set_color("#555")

    fig.suptitle("Summary Report — Skripsi Tirta (01082230021)\nFused Spatial-Channel Attention untuk Klasifikasi Kanker Kulit",
                 color="white", fontsize=12, fontweight="bold")
    plt.savefig(SUMMARY_PNG, dpi=130, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()

    log_fn(f"Summary chart tersimpan → {SUMMARY_PNG}")

    summary = {
        "accuracy":     acc,
        "f1_weighted":  f1_w,
        "f1_macro":     f1_mac,
        "auc_macro":    auc,
        "model_mb":     round(model_mb, 2),
        "total_params": total_params,
        "chart_path":   SUMMARY_PNG,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    log_fn(f"Summary JSON tersimpan → {SUMMARY_JSON}")
    return summary