"""
Step 12 - Ablasi Resep Training (ResNet-18 tanpa modul atensi)
================================================================
Menjawab pertanyaan: komponen resep mana yang membuat ResNet-18 lebih baik
dari baseline resmi MedMNIST pada DermaMNIST 28x28?

Desain: leave-one-out. Setiap varian = resep lengkap dengan SATU komponen
dimatikan, masing-masing 5 seed (RECIPE_ABLATION di config.py):

  full         resep lengkap (pretrained, augmentasi, WCE 1/sqrt(n), AdamW + cosine)
  no_pretrain  - bobot ImageNet (dilatih dari nol)
  no_aug       - augmentasi
  no_wce       - Weighted CE (pakai CE biasa)
  adam_step    Adam + MultiStep (resmi) menggantikan AdamW + cosine
  official     semua dimatikan + batch 128, tanpa dropout, 100 epoch = resep resmi MedMNIST v2

Selisih (full - varian) per seed = kontribusi komponen tersebut.
Selisih (full - official) = total kenaikan dari resep.
Perbedaan yang tersisa dengan kode resmi MedMNIST: normalisasi memakai
mean/std per kanal (resmi: 0.5/0.5) dan inisialisasi conv1 3x3; keduanya
sama di semua varian sehingga tidak memengaruhi perbandingan antar varian.

Semua varian memakai train_one_run() dari Step 5 (aturan pemilihan
checkpoint, seed, dan evaluasi identik). Hasil "full" memakai ulang
checkpoint Step 11 (attention="none") bila ada DAN resepnya sama; checkpoint
dari resep lain tidak pernah dipakai diam-diam (common.reusable).

Output: outputs/recipe_ablation_report.json
        outputs/recipe_ablation_comparison.png
        outputs/recipe_ablation/<varian>_seed<s>_ep<E>_<metric>.pth (+ .history.json)
"""

import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, TRAIN_HP, EVAL_CONFIG, RECIPE_ABLATION, ensure_dirs

META_PATH   = PATHS["dataset_meta"]
CONFIG_PATH = PATHS["preprocess_config"]
OUT_DIR     = PATHS["recipe_ablation_dir"]
REPORT_PATH = PATHS["recipe_ablation_report"]
CHART_PATH  = PATHS["recipe_ablation_chart"]

METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "auc_macro"]
CLASS_SHORT = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def _run_name(variant, seed, epochs, select_metric):
    return f"{variant}_seed{seed}_ep{epochs}_{select_metric}"


def _evaluate(ckpt, attention, log_fn):
    """Metrik test + recall per kelas (tanpa TTA bila EVAL_CONFIG['tta']=False)."""
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
    return {
        "accuracy":          m["acc"],
        "balanced_accuracy": m["balanced_acc"],
        "f1_macro":          m["f1_macro"],
        "auc_macro":         m["auc_macro"],
        "recall_per_class":  common.per_class_recall(m["y_true"], m["y_pred"], n_classes),
    }


def _variant_hp(variant, seed, epochs):
    """hp lengkap sebuah varian. Varian boleh menimpa epochs (mis. "official" = 100)."""
    override = RECIPE_ABLATION["variants"][variant]
    return {**TRAIN_HP, "epochs": epochs, "seed": seed, **override}


def _find_checkpoint(variant, seed, epochs, select_metric, attention, log_fn=None, retire=True):
    """
    Lokasi checkpoint varian + apakah sudah bisa dipakai (resep cocok).
    Untuk "full" dengan attention="none", checkpoint Step 11 dipakai ulang
    bila ada dan resepnya sama.
    Return (ckpt, history_path, reused_from_step11: bool, ready: bool).
    """
    from step5_training import resolve_recipe
    import common

    hp = _variant_hp(variant, seed, epochs)
    expected = resolve_recipe(hp)
    name = _run_name(variant, seed, hp["epochs"], select_metric)
    own_ckpt = os.path.join(OUT_DIR, f"{name}.pth")
    own_hist = os.path.join(OUT_DIR, f"{name}.history.json")
    if common.reusable(own_ckpt, own_hist, expected, log_fn, retire=retire):
        return own_ckpt, own_hist, False, True

    if variant == "full" and attention == "none":
        s11 = f"none_seed{seed}_ep{hp['epochs']}_{select_metric}"
        s11_ckpt = os.path.join(PATHS["ablation_dir"], f"{s11}.pth")
        s11_hist = os.path.join(PATHS["ablation_dir"], f"{s11}.history.json")
        # Checkpoint Step 11 tidak di-.stale-kan dari sini (milik Step 11).
        if common.reusable(s11_ckpt, s11_hist, expected, log_fn, retire=False):
            return s11_ckpt, s11_hist, True, True

    return own_ckpt, own_hist, False, False


def _per_class_table(summary, variants, labels, log_fn):
    log_fn("\nRecall per kelas (%) - rata-rata antar seed:")
    head = f"  {'Varian':<40}" + "".join(f"{c:>7}" for c in CLASS_SHORT)
    log_fn(head)
    for v in variants:
        if v not in summary:
            continue
        rec = summary[v]["recall_per_class"]["mean"]
        log_fn(f"  {labels.get(v, v):<40}" + "".join(f"{r:>7.1f}" for r in rec))


def run(log_fn, variants=None, seeds=None, epochs=None):
    sys.path.insert(0, os.path.dirname(__file__))
    from step5_training import train_one_run
    from step11_ablation import _paired_tests, _plot

    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} dulu.")
            raise FileNotFoundError(path)

    ensure_dirs()
    os.makedirs(OUT_DIR, exist_ok=True)

    all_variants = RECIPE_ABLATION["variants"]
    labels       = RECIPE_ABLATION.get("labels", {})
    attention    = RECIPE_ABLATION.get("attention", "none")
    variants = variants or list(all_variants)
    unknown = [v for v in variants if v not in all_variants]
    if unknown:
        raise ValueError(f"Varian tidak dikenal: {unknown}. Pilihan: {list(all_variants)}")
    if "full" not in variants:
        variants = ["full"] + variants      # referensi wajib ada
    seeds  = seeds  or RECIPE_ABLATION["seeds"]
    epochs = epochs or RECIPE_ABLATION["epochs"]
    select_metric = TRAIN_HP.get("select_metric", "auc")

    combos = [(v, s) for v in variants for s in seeds]
    log_fn("=" * 78)
    log_fn("  STEP 12 - ABLASI RESEP TRAINING (leave-one-out)")
    log_fn("=" * 78)
    log_fn(f"Model           : ResNet-18 (attention={attention})")
    for v in variants:
        log_fn(f"  {v:<12} {labels.get(v, ''):<42} override={all_variants[v] or '-'}")
    log_fn(f"Seed            : {seeds}")
    log_fn(f"Epoch/run       : {epochs}")
    log_fn(f"Pilih checkpoint: val {select_metric}")
    log_fn(f"TTA evaluasi    : {'aktif' if EVAL_CONFIG.get('tta') else 'nonaktif'}")
    log_fn(f"Total run       : {len(combos)}")
    log_fn("")

    raw = {v: [] for v in variants}
    t_start = time.time()
    durations = []

    for i, (variant, seed) in enumerate(combos, 1):
        ckpt, hist_p, reused, ready = _find_checkpoint(variant, seed, epochs, select_metric,
                                                       attention, log_fn)
        partial = ckpt.replace(".pth", ".partial.pth")

        log_fn("-" * 78)
        log_fn(f"[{i}/{len(combos)}]  varian={variant}  seed={seed}")
        log_fn("-" * 78)

        best_epoch = None
        if ready:
            src = "Step 11" if reused else "run sebelumnya"
            log_fn(f"Checkpoint sudah ada ({src}), training dilewati: {os.path.basename(ckpt)}")
            if os.path.exists(hist_p):
                with open(hist_p) as f:
                    best_epoch = json.load(f).get("best_epoch")
        else:
            if os.path.exists(partial):
                log_fn("Ditemukan run terputus (.partial) - diulang dari awal.")
                os.remove(partial)
            t_run = time.time()
            hp = _variant_hp(variant, seed, epochs)
            result = train_one_run(hp, log_fn, ckpt_path=partial, attention=attention,
                                   seed=seed, tag=f"{variant}/s{seed}")
            os.replace(partial, ckpt)
            best_epoch = result["best_epoch"]
            with open(hist_p, "w") as f:
                json.dump(result["history"], f, indent=2)
            durations.append(time.time() - t_run)

        res = _evaluate(ckpt, attention, log_fn)
        raw[variant].append({"seed": seed, "best_epoch": best_epoch, **res})
        log_fn(f"  TEST -> acc {res['accuracy']:.2f}% | bal {res['balanced_accuracy']:.2f}% | "
               f"f1m {res['f1_macro']:.4f} | auc {res['auc_macro']:.4f} | ckpt epoch {best_epoch}")

        remaining = sum(
            1 for v, s in combos[i:]
            if not _find_checkpoint(v, s, epochs, select_metric, attention, retire=False)[3]
        )
        if durations and remaining:
            eta = np.mean(durations) * remaining / 60
            log_fn(f"  Estimasi sisa waktu: ~{eta:.0f} menit ({remaining} run perlu dilatih, "
                   f"rata-rata {np.mean(durations)/60:.1f} menit/run)")

    # ---------- Agregasi ----------
    summary = {}
    for v, runs in raw.items():
        if not runs:
            continue
        entry = {"label": labels.get(v, v), "override": all_variants[v], "n_runs": len(runs),
                 "seeds": [r["seed"] for r in runs],
                 "best_epochs": [r["best_epoch"] for r in runs]}
        for metric in METRICS:
            vals = np.array([r[metric] for r in runs], dtype=float)
            entry[metric] = {"mean": float(vals.mean()),
                             "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                             "runs": [float(x) for x in vals]}
        rec = np.array([r["recall_per_class"] for r in runs], dtype=float)
        entry["recall_per_class"] = {"classes": CLASS_SHORT,
                                     "mean": [float(x) for x in rec.mean(axis=0)],
                                     "std": [float(x) for x in (rec.std(axis=0, ddof=1)
                                                                if len(rec) > 1 else rec[0] * 0)]}
        summary[v] = entry

    log_fn("")
    log_fn("=" * 104)
    log_fn("  HASIL AKHIR (mean +/- std antar seed)")
    log_fn("=" * 104)
    log_fn(f"  {'Varian':<40} {'Accuracy':>15} {'Balanced Acc':>15} {'F1-macro':>17} {'AUC':>17}")
    for v in variants:
        if v not in summary:
            continue
        s = summary[v]
        log_fn(f"  {s['label']:<40} "
               f"{s['accuracy']['mean']:>8.2f} +/-{s['accuracy']['std']:>4.2f} "
               f"{s['balanced_accuracy']['mean']:>8.2f} +/-{s['balanced_accuracy']['std']:>4.2f} "
               f"{s['f1_macro']['mean']:>9.4f} +/-{s['f1_macro']['std']:>6.4f} "
               f"{s['auc_macro']['mean']:>9.4f} +/-{s['auc_macro']['std']:>6.4f}")
    log_fn("=" * 104)

    _per_class_table(summary, variants, labels, log_fn)

    # ---------- Uji berpasangan: full vs tiap varian ----------
    tests = {}
    if len(summary) > 1:
        log_fn("\nUji berpasangan per seed. 'mean selisih' = FULL - varian:")
        log_fn("  positif -> komponen yang dimatikan MEMBANTU (resep lengkap lebih baik)")
        log_fn("  negatif -> komponen yang dimatikan justru MERUGIKAN")
        log_fn("  baris 'official' -> total kenaikan resep dibanding resep resmi MedMNIST")
        tests = _paired_tests(raw, "full", log_fn)

    log_fn(f"\nTotal waktu: {(time.time() - t_start) / 60:.1f} menit")

    _plot(summary, CHART_PATH, title="Ablasi Resep Training ResNet-18",
          labels={v: labels.get(v, v) for v in summary})
    log_fn(f"Chart perbandingan -> {CHART_PATH}")

    payload = {"summary": summary, "raw": raw, "paired_tests_vs_full": tests,
               "variants": variants, "seeds": seeds, "epochs": epochs,
               "attention": attention, "select_metric": select_metric,
               "tta": EVAL_CONFIG.get("tta", False), "chart_path": CHART_PATH}
    with open(REPORT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    log_fn(f"Report tersimpan -> {REPORT_PATH}")
    return payload


if __name__ == "__main__":
    run(print)