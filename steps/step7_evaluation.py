"""
Step 7 - Evaluation pada Test Set
====================================
Evaluasi model final ResNet-18 + FSCA (dari Step 6: Training FSCA) pada test
set DermaMNIST. Fungsi evaluate_checkpoint() juga dipakai Step 4: Baseline
ResNet-18 agar kedua model dievaluasi dengan protokol yang identik.

Menghasilkan:
- Accuracy, Balanced Accuracy, Weighted/Macro F1, Macro AUC
- Confusion matrix (count + row-normalized)
- Metrik per kelas
- Sapuan logit adjustment (tau dituning di VALIDATION, dilaporkan di TEST)
- Test-time augmentation 8-arah

Output: outputs/confusion_matrix.png, outputs/evaluation_report.json

RINGKASAN PERBAIKAN dari versi sebelumnya
------------------------------------------
1. Device tidak lagi di-hardcode "cpu" (lewat common.resolve_device).
2. Arsitektur (modul atensi) diambil dari MODEL_CONFIG lewat build_model(),
   sehingga persis sama dengan saat training.
3. Ditambahkan Balanced Accuracy. Dengan kelas `nv` menguasai ~67% data,
   akurasi mentah bisa terlihat tinggi walau kelas minoritas nyaris tak pernah
   terdeteksi.
4. Ditambahkan TTA dan logit adjustment - keduanya tanpa melatih ulang.
   Hasil tanpa keduanya tetap disimpan di "without_tta_or_adjustment".
5. Confusion matrix ternormalisasi baris ditambahkan agar pola kesalahan
   antar-kelas terbaca meski jumlah sampel per kelas timpang.
"""

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, EVAL_CONFIG, IMBALANCE, MODEL_CONFIG, ensure_dirs

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
    """
    Entry point GUI (Step 7: Evaluasi & Perbandingan).
    Mengevaluasi ulang KEDUA model dengan protokol yang sama (hanya evaluasi,
    tanpa training):
      - ResNet-18 Baseline (Step 4) -> baseline_evaluation_report.json
      - ResNet-18 + FSCA   (Step 6) -> evaluation_report.json
    lalu membuat tabel perbandingan -> comparison_report.json.
    Model yang belum dilatih dilewati dengan peringatan.
    """
    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} dulu.")
            raise FileNotFoundError(path)

    fsca_path = FINAL_MODEL if os.path.exists(FINAL_MODEL) else CKPT_PATH
    attn = MODEL_CONFIG["attention"]
    candidates = [
        ("baseline", "ResNet-18 Baseline", PATHS["baseline_model"], "none",
         PATHS["baseline_report"], PATHS["baseline_cm_chart"], PATHS["baseline_history"],
         "Step 4: Baseline ResNet-18"),
        ("fsca", f"ResNet-18 + {attn.upper()}", fsca_path, attn,
         REPORT_PATH, CM_PATH, PATHS["training_history"], "Step 6: Training FSCA"),
    ]

    reports = {}
    for key, label, path, attention, rep_path, cm_path, hist_path, made_by in candidates:
        if not os.path.exists(path):
            log_fn(f"PERINGATAN: model {label} belum ada ({os.path.basename(path)}). "
                   f"Jalankan {made_by} dulu. Dilewati.")
            continue
        log_fn("\n" + "#" * 58)
        log_fn(f"  EVALUASI {label.upper()}")
        log_fn("#" * 58)
        reports[key] = evaluate_checkpoint(path, log_fn, attention=attention,
                                           report_path=rep_path, cm_path=cm_path,
                                           history_path=hist_path, label=label)

    if not reports:
        raise FileNotFoundError("Belum ada model. Jalankan Step 4 dan/atau Step 6 dulu.")
    if len(reports) == 2:
        compare_reports(reports["baseline"], reports["fsca"], log_fn)

    return {"evaluated": list(reports)}   # tanpa chart_path -> GUI menampilkan log


def _share(y_true, y_pred, true_idx, pred_idx):
    """Porsi sampel kelas true_idx yang diprediksi sebagai pred_idx (0..1)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true == true_idx
    return float((y_pred[mask] == pred_idx).mean()) if mask.any() else float("nan")


def compare_reports(base, other, log_fn):
    """
    Tabel perbandingan baseline vs model lain (FSCA) dari dua laporan
    evaluate_checkpoint(). Disimpan ke PATHS["comparison_report"].
    CATATAN: ini 1 seed (TRAIN_HP["seed"]); kesimpulan akhir tetap dari Step 11.
    """
    rows = [("Accuracy (%)", "accuracy", 2), ("Balanced Acc (%)", "balanced_accuracy", 2),
            ("F1 Macro", "f1_macro", 4), ("AUC Macro", "auc_macro", 4),
            ("Recall melanoma", "melanoma_recall", 3)]
    b_name = base.get("label", "Baseline")
    o_name = other.get("label", "FSCA")

    log_fn("\n" + "=" * 72)
    log_fn("  PERBANDINGAN (test set, protokol evaluasi identik, 1 seed)")
    log_fn("=" * 72)
    log_fn(f"  {'Metrik':<22} {b_name:>20} {o_name:>20} {'Selisih':>8}")
    table = []
    for title, key, nd in rows:
        bv, ov = base.get(key), other.get(key)
        if bv is None or ov is None:
            continue
        table.append({"metric": title, "baseline": bv, "other": ov, "diff": round(ov - bv, 4)})
        log_fn(f"  {title:<22} {bv:>20.{nd}f} {ov:>20.{nd}f} {ov - bv:>+8.{nd}f}")
    for title, key in [("Parameter", "total_params"), ("Epoch checkpoint", "best_epoch")]:
        bv, ov = base.get(key), other.get(key)
        if bv is not None and ov is not None:
            table.append({"metric": title, "baseline": bv, "other": ov, "diff": ov - bv})
            log_fn(f"  {title:<22} {bv:>20,} {ov:>20,} {ov - bv:>+8,}")

    log_fn("\n  Recall per kelas:")
    per_class = []
    for cls in base["class_names"]:
        bv = base["per_class"].get(cls, {}).get("recall")
        ov = other["per_class"].get(cls, {}).get("recall")
        if bv is None or ov is None:
            continue
        per_class.append({"class": cls, "baseline": bv, "other": ov, "diff": round(ov - bv, 4)})
        log_fn(f"  {cls:<42} {bv:>7.3f} {ov:>7.3f} {ov - bv:>+8.3f}")
    log_fn("=" * 72)
    log_fn("  Catatan: 1 seed. Untuk kesimpulan (mean +/- std 5 seed) jalankan Step 11.")

    comparison = {"baseline": b_name, "other": o_name, "metrics": table,
                  "recall_per_class": per_class,
                  "note": "1 seed (TRAIN_HP['seed']); kesimpulan akhir dari Step 11"}
    with open(PATHS["comparison_report"], "w") as f:
        json.dump(comparison, f, indent=2)
    log_fn(f"Tabel perbandingan tersimpan -> {PATHS['comparison_report']}")
    return comparison


def evaluate_checkpoint(model_path, log_fn, attention=None,
                        report_path=REPORT_PATH, cm_path=CM_PATH,
                        history_path=None, label=None):
    """
    Evaluasi satu checkpoint di test set dengan protokol yang sama
    (EVAL_CONFIG + IMBALANCE). Dipakai Step 7 (kedua model) dan
    step_baseline.py agar angka baseline dan FSCA bisa dibandingkan langsung.

    attention    : None -> MODEL_CONFIG["attention"]; "none" untuk baseline.
    history_path : history training model ini (untuk mencatat epoch checkpoint).
    Return       : dict laporan evaluasi (juga disimpan ke report_path).
    """
    import torch
    from sklearn.metrics import classification_report, confusion_matrix

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model
    import common

    ensure_dirs()
    with open(META_PATH) as f:
        meta = json.load(f)
    with open(CONFIG_PATH) as f:
        pre = json.load(f)

    n_classes = meta["n_classes"]
    class_map = meta["class_map"]
    class_names = [class_map[str(i)] for i in range(n_classes)]

    device = common.resolve_device(log_fn=log_fn)

    log_fn(f"Memuat model: {os.path.basename(model_path)}")
    model = build_model(n_classes=n_classes, pretrained=False, attention=attention).to(device)
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

    # ---------- Bias melanoma: dituning di VALIDATION (feedback dosen poin 1) ----------
    mel_cfg  = IMBALANCE.get("melanoma_bias", {})
    mel_name = mel_cfg.get("class_name", "melanoma")
    mel_idx  = class_names.index(mel_name) if mel_name in class_names else None
    nevi_idx = class_names.index("melanocytic nevi") if "melanocytic nevi" in class_names else None
    mel_info = {"enabled": False}
    if mel_cfg.get("enabled", False) and mel_idx is not None:
        log_fn("")
        val_logits, val_y = common.collect_logits(model, val_loader, device, use_tta=use_tta)
        if adj is not None:
            val_logits = val_logits - adj[None, :]
        bias, bias_table = common.tune_class_bias(
            val_logits, val_y, n_classes, mel_idx, mel_cfg.get("grid", [0.0]),
            max_bal_acc_drop=mel_cfg.get("max_bal_acc_drop", 0.0),
            max_acc_drop=mel_cfg.get("max_acc_drop", 2.0),
            log_fn=log_fn, class_label=mel_name,
        )
        m0 = common.evaluate(model, test_loader, device, n_classes,
                             use_tta=use_tta, logit_adjust=adj)
        mel_info = {
            "enabled": True, "class_name": mel_name, "bias": bias,
            "rule": {k: mel_cfg.get(k) for k in ("max_bal_acc_drop", "max_acc_drop")},
            "val_sweep": bias_table,
            "test_without_bias": {
                "accuracy":          round(m0["acc"], 4),
                "balanced_accuracy": round(m0["balanced_acc"], 4),
                "f1_macro":          round(m0["f1_macro"], 4),
                "auc_macro":         round(m0["auc_macro"], 4),
                "recall":    round(common.class_recall(m0["y_true"], m0["y_pred"], mel_idx), 4),
                "precision": round(common.class_precision(m0["y_true"], m0["y_pred"], mel_idx), 4),
                "as_nevi":   round(_share(m0["y_true"], m0["y_pred"], mel_idx, nevi_idx), 4)
                             if nevi_idx is not None else None,
            },
        }
        if bias:
            bvec = common.class_bias_vector(n_classes, mel_idx, bias)
            adj = bvec if adj is None else adj + bvec

    # ---------- Evaluasi final ----------
    log_fn(f"\nEvaluasi final (TTA={use_tta}, tau={tau}, "
           f"bias {mel_name}={mel_info.get('bias', 0.0)})...")
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
    log_fn(f"  Kontribusi TTA + adj + bias: {acc - m_plain['acc']:+.2f} poin akurasi, "
           f"{bal - m_plain['balanced_acc']:+.2f} poin balanced")
    log_fn("=" * 58)

    mel_recall = (common.class_recall(y_true, y_pred, mel_idx)
                  if mel_idx is not None else None)
    if mel_info["enabled"]:
        before = mel_info["test_without_bias"]
        mel_info["test_with_bias"] = {
            "recall":    round(mel_recall, 4),
            "precision": round(common.class_precision(y_true, y_pred, mel_idx), 4),
            "as_nevi":   round(_share(y_true, y_pred, mel_idx, nevi_idx), 4)
                         if nevi_idx is not None else None,
        }
        after = mel_info["test_with_bias"]
        log_fn(f"  Recall {mel_name} (test)     : {before['recall']:.3f} -> {after['recall']:.3f} "
               f"(bias {mel_info['bias']})")
        log_fn(f"  Presisi {mel_name} (test)    : {before['precision']:.3f} -> {after['precision']:.3f}")
        if before.get("as_nevi") is not None:
            log_fn(f"  {mel_name} ditebak nevi      : {100 * before['as_nevi']:.1f}% -> "
                   f"{100 * after['as_nevi']:.1f}%")
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
    _plot_confusion(cm, class_names, acc, bal, auc, cm_path)
    log_fn(f"Confusion matrix tersimpan -> {cm_path}")

    best_epoch = None
    if history_path and os.path.exists(history_path):
        with open(history_path) as f:
            best_epoch = json.load(f).get("best_epoch")

    eval_result = {
        "label":          label or f"ResNet-18 + {model.attention_name.upper()}",
        "model_path":     model_path,
        "attention":      model.attention_name,
        "total_params":   sum(p.numel() for p in model.parameters()),
        "best_epoch":     best_epoch,
        "device":         str(device),
        "tta":            use_tta,
        "logit_adjustment_tau": tau,
        "tau_sweep_val":  tau_table,
        "melanoma_bias":  mel_info,
        "melanoma_recall": round(mel_recall, 4) if mel_recall is not None else None,
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
        "chart_path": cm_path,
    }
    with open(report_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    log_fn(f"Evaluation report tersimpan -> {report_path}")
    return eval_result