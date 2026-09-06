"""
Step 11 - Ablasi Multi-Seed (BARU)
====================================
Menjalankan matriks eksperimen {modul atensi} x {seed} pada backbone,
resep training, dan pembagian data yang identik, lalu melaporkan
mean +/- standard deviation untuk setiap metrik.

Kenapa step ini wajib ada
--------------------------
Selisih performa antara FSCA, CBAM, SE, dan ECA pada DermaMNIST biasanya
di bawah 1 poin. Sementara itu, variasi antar-seed pada dataset sebesar ini
saja sudah bisa mencapai 0.7-1.0 poin. Artinya, satu kali run per modul tidak
bisa membedakan "FSCA lebih baik" dari "FSCA kebetulan dapat seed bagus".
Penguji yang teliti akan menanyakan ini, dan tanpa standard deviation
klaim novelty di BAB I tidak bisa dipertahankan.

Output: outputs/ablation_report.json
        outputs/ablation_comparison.png
        outputs/ablation/<attention>_seed<seed>.pth
"""

import os
import sys
import json
import time
import itertools

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (PATHS, TRAIN_HP, EVAL_CONFIG, EXPERIMENT, ensure_dirs)

META_PATH      = PATHS["dataset_meta"]
CONFIG_PATH    = PATHS["preprocess_config"]
ABLATION_DIR   = PATHS["ablation_dir"]
REPORT_PATH    = PATHS["ablation_report"]
CHART_PATH     = PATHS["ablation_chart"]

METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "auc_macro"]


def _evaluate_checkpoint(ckpt, attention, log_fn):
    """Muat checkpoint dan ukur di test set dengan pengaturan evaluasi standar."""
    import torch

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model
    import common

    with open(META_PATH) as f:
        meta = json.load(f)
    with open(CONFIG_PATH) as f:
        pre = json.load(f)

    n_classes = meta["n_classes"]
    device = common.resolve_device(log_fn=lambda *_: None)

    model = build_model(n_classes=n_classes, pretrained=False, attention=attention).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    loaders, _ = common.build_dataloaders(
        mean=pre["mean"], std=pre["std"],
        batch_size=EVAL_CONFIG["batch_size"], n_classes=n_classes,
        splits=("test",), log_fn=lambda *_: None,
    )
    m = common.evaluate(model, loaders["test"], device, n_classes,
                        use_tta=EVAL_CONFIG.get("tta", False))

    total_params = sum(p.numel() for p in model.parameters())
    return {
        "accuracy":          m["acc"],
        "balanced_accuracy": m["balanced_acc"],
        "f1_macro":          m["f1_macro"],
        "auc_macro":         m["auc_macro"],
        "params":            total_params,
    }


def _plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary.keys())
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.5))
    fig.patch.set_facecolor("#1a1a1a")
    colors = ["#607D8B", "#4FC3F7", "#81C784", "#FFD54F", "#FF8A65"]

    for ax, metric in zip(np.atleast_1d(axes), METRICS):
        ax.set_facecolor("#1a1a1a")
        means = [summary[n][metric]["mean"] for n in names]
        stds = [summary[n][metric]["std"] for n in names]
        ax.bar(names, means, yerr=stds, capsize=5,
               color=[colors[i % len(colors)] for i in range(len(names))])
        lo = min(m - s for m, s in zip(means, stds))
        hi = max(m + s for m, s in zip(means, stds))
        pad = (hi - lo) * 0.4 + 1e-6
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(metric, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("#555")
        ax.grid(alpha=0.15, axis="y")

    fig.suptitle("Ablasi Modul Atensi - mean +/- std antar seed",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()


def run(log_fn, attentions=None, seeds=None, epochs=None):
    sys.path.insert(0, os.path.dirname(__file__))
    from step5_training import train_one_run

    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} dulu.")
            raise FileNotFoundError(path)

    ensure_dirs()
    os.makedirs(ABLATION_DIR, exist_ok=True)

    attentions = attentions or EXPERIMENT["attentions"]
    seeds      = seeds      or EXPERIMENT["seeds"]
    epochs     = epochs     or EXPERIMENT["epochs"]

    combos = list(itertools.product(attentions, seeds))
    log_fn("=" * 78)
    log_fn("  STEP 11 - ABLASI MULTI-SEED")
    log_fn("=" * 78)
    log_fn(f"Modul atensi : {attentions}")
    log_fn(f"Seed         : {seeds}")
    log_fn(f"Epoch/run    : {epochs}")
    log_fn(f"Total run    : {len(combos)}")
    log_fn("PERINGATAN: ini akan berjalan lama. Jalankan semalaman.")
    log_fn("")

    raw = {a: [] for a in attentions}
    t_start = time.time()

    for i, (attn, seed) in enumerate(combos, 1):
        ckpt = os.path.join(ABLATION_DIR, f"{attn}_seed{seed}.pth")
        log_fn("-" * 78)
        log_fn(f"[{i}/{len(combos)}]  attention={attn}  seed={seed}")
        log_fn("-" * 78)

        if os.path.exists(ckpt):
            log_fn(f"Checkpoint sudah ada, training dilewati: {os.path.basename(ckpt)}")
        else:
            hp = {**TRAIN_HP, "epochs": epochs, "seed": seed}
            train_one_run(hp, log_fn, ckpt_path=ckpt, attention=attn,
                          seed=seed, tag=f"{attn}/s{seed}")

        res = _evaluate_checkpoint(ckpt, attn, log_fn)
        raw[attn].append({"seed": seed, **res})
        log_fn(f"  TEST -> acc {res['accuracy']:.2f}% | bal {res['balanced_accuracy']:.2f}% | "
               f"f1m {res['f1_macro']:.4f} | auc {res['auc_macro']:.4f}")

    # ---------- Agregasi ----------
    summary = {}
    for attn, runs in raw.items():
        if not runs:
            continue
        entry = {"n_runs": len(runs), "params": runs[0]["params"], "seeds": [r["seed"] for r in runs]}
        for metric in METRICS:
            vals = np.array([r[metric] for r in runs], dtype=float)
            entry[metric] = {
                "mean": float(vals.mean()),
                "std":  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "runs": [float(v) for v in vals],
            }
        summary[attn] = entry

    log_fn("")
    log_fn("=" * 92)
    log_fn("  HASIL AKHIR (mean +/- std)")
    log_fn("=" * 92)
    log_fn(f"  {'Modul':<8} {'Params':>12} {'Accuracy':>18} {'Balanced Acc':>18} "
           f"{'F1-macro':>16} {'AUC':>16}")
    for attn in attentions:
        if attn not in summary:
            continue
        s = summary[attn]
        log_fn(
            f"  {attn:<8} {s['params']:>12,} "
            f"{s['accuracy']['mean']:>11.2f} +/-{s['accuracy']['std']:>4.2f} "
            f"{s['balanced_accuracy']['mean']:>11.2f} +/-{s['balanced_accuracy']['std']:>4.2f} "
            f"{s['f1_macro']['mean']:>9.4f} +/-{s['f1_macro']['std']:>5.4f} "
            f"{s['auc_macro']['mean']:>9.4f} +/-{s['auc_macro']['std']:>5.4f}"
        )
    log_fn("=" * 92)

    # ---------- Uji signifikansi FSCA vs pembanding terbaik ----------
    if "fsca" in summary and len(summary) > 1:
        try:
            from scipy import stats
            fs = summary["fsca"]["accuracy"]["runs"]
            rivals = {k: v for k, v in summary.items() if k != "fsca"}
            best = max(rivals, key=lambda k: rivals[k]["accuracy"]["mean"])
            rv = rivals[best]["accuracy"]["runs"]
            if len(fs) > 1 and len(rv) > 1:
                t, p = stats.ttest_ind(fs, rv, equal_var=False)
                log_fn(f"\nUji Welch t-test, FSCA vs {best} (accuracy): "
                       f"t={t:.3f}, p={p:.4f}")
                if p < 0.05:
                    log_fn("  -> selisih signifikan secara statistik (p < 0.05).")
                else:
                    log_fn("  -> selisih BELUM signifikan. Tambah seed, atau bingkai ulang "
                           "klaim sebagai 'setara dengan biaya komputasi lebih rendah'.")
        except ImportError:
            log_fn("\n(scipy tidak tersedia - uji t dilewati. pip install scipy)")

    log_fn(f"\nTotal waktu: {(time.time() - t_start) / 60:.1f} menit")

    _plot(summary, CHART_PATH)
    log_fn(f"Chart perbandingan -> {CHART_PATH}")

    payload = {"summary": summary, "raw": raw, "attentions": attentions,
               "seeds": seeds, "epochs": epochs, "chart_path": CHART_PATH}
    with open(REPORT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    log_fn(f"Report tersimpan -> {REPORT_PATH}")

    return payload