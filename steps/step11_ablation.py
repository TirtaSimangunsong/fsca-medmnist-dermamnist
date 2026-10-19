"""
Step 11 - Ablasi Multi-Seed
=============================
Menjalankan matriks eksperimen {modul atensi} x {seed} dengan backbone,
resep training, pembagian data, dan aturan pemilihan checkpoint yang
IDENTIK (train_one_run dari Step 5), lalu melaporkan mean +/- std setiap
metrik dan uji signifikansi berpasangan per seed.

Kenapa step ini wajib ada
--------------------------
Selisih performa antar modul atensi pada DermaMNIST biasanya di bawah 1 poin,
sementara variasi antar-seed saja bisa 0.7-1.0 poin. Satu run per modul tidak
bisa membedakan "FSCA lebih baik" dari "FSCA kebetulan dapat seed bagus".
Perbandingan "none" vs "fsca" di sini adalah jawaban langsung Rumusan
Masalah poin 1: kontribusi FSCA dengan resep training yang sama.

Perbaikan dari versi sebelumnya
--------------------------------
0. Checkpoint lama hanya dipakai ulang bila resepnya sama dengan config saat
   ini (common.reusable); bila berbeda, ditandai .stale dan dilatih ulang.
1. Run yang terputus TIDAK lagi dianggap selesai. Training menulis ke file
   sementara (*.partial.pth) dan baru di-rename setelah run tuntas.
2. Nama checkpoint memuat jumlah epoch dan metrik seleksi, sehingga
   checkpoint dari resep lama tidak terpakai ulang secara diam-diam.
3. Uji signifikansi berpasangan per seed (paired t-test) pada AUC dan
   F1-macro, bukan Welch t-test pada akurasi. Seed yang sama berarti urutan
   data yang sama, jadi pasangan per seed adalah desain yang benar.
4. History validasi tiap run disimpan (kurva per run untuk lampiran).
5. Estimasi waktu selesai (ETA) dicatat setelah setiap run.

Output: outputs/ablation_report.json
        outputs/ablation_comparison.png
        outputs/ablation/<attention>_seed<seed>_ep<E>_<metric>.pth
        outputs/ablation/<attention>_seed<seed>_ep<E>_<metric>.history.json
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

METRICS      = ["accuracy", "balanced_accuracy", "f1_macro", "auc_macro"]
TEST_METRICS = ["auc_macro", "f1_macro", "balanced_accuracy", "accuracy"]


def _run_name(attn, seed, epochs, select_metric):
    return f"{attn}_seed{seed}_ep{epochs}_{select_metric}"


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


def _paired_tests(raw, reference, log_fn):
    """
    Bandingkan `reference` (mis. "fsca") dengan setiap modul lain, berpasangan
    per seed. Return dict {rival: {metric: {...}}}.
    """
    try:
        from scipy import stats
    except ImportError:
        log_fn("\n(scipy tidak tersedia - uji signifikansi dilewati. pip install scipy)")
        return {}

    ref_by_seed = {r["seed"]: r for r in raw.get(reference, [])}
    results = {}
    for rival, runs in raw.items():
        if rival == reference:
            continue
        riv_by_seed = {r["seed"]: r for r in runs}
        seeds = sorted(set(ref_by_seed) & set(riv_by_seed))
        if len(seeds) < 2:
            continue

        results[rival] = {"seeds": seeds}
        log_fn(f"\n{reference.upper()} vs {rival.upper()} "
               f"(berpasangan per seed, n={len(seeds)}):")
        log_fn(f"  {'Metrik':<18} {'mean selisih':>13} {'menang':>8} {'t':>8} {'p':>8}")
        for metric in TEST_METRICS:
            a = np.array([ref_by_seed[s][metric] for s in seeds], dtype=float)
            b = np.array([riv_by_seed[s][metric] for s in seeds], dtype=float)
            diff = a - b
            wins = int((diff > 0).sum())
            if np.allclose(diff, diff[0]):
                t, p = float("nan"), float("nan")
            else:
                t, p = stats.ttest_rel(a, b)
            results[rival][metric] = {
                "mean_diff": float(diff.mean()),
                "std_diff":  float(diff.std(ddof=1)),
                "per_seed_diff": [float(d) for d in diff],
                "wins": wins, "n": len(seeds),
                "t": float(t), "p": float(p),
            }
            log_fn(f"  {metric:<18} {diff.mean():>+13.4f} "
                   f"{wins:>4}/{len(seeds):<3} {t:>8.3f} {p:>8.4f}")
    return results


def _plot(summary, path, title="Ablasi Modul Atensi", labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary.keys())
    ticks = [(labels or {}).get(n, n) for n in names]
    rotate = max(len(t) for t in ticks) > 8
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5.5 if rotate else 4.5))
    fig.patch.set_facecolor("#1a1a1a")
    colors = ["#607D8B", "#4FC3F7", "#81C784", "#FFD54F", "#FF8A65", "#BA68C8"]

    for ax, metric in zip(np.atleast_1d(axes), METRICS):
        ax.set_facecolor("#1a1a1a")
        means = [summary[n][metric]["mean"] for n in names]
        stds = [summary[n][metric]["std"] for n in names]
        ax.bar(range(len(names)), means, yerr=stds, capsize=5,
               color=[colors[i % len(colors)] for i in range(len(names))])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(ticks, rotation=35 if rotate else 0,
                           ha="right" if rotate else "center", fontsize=8 if rotate else 10)
        # titik per seed agar sebaran terlihat
        for i, n in enumerate(names):
            runs = summary[n][metric]["runs"]
            ax.scatter([i] * len(runs), runs, color="white", s=12, zorder=3)
        lo = min(min(summary[n][metric]["runs"]) for n in names)
        hi = max(max(summary[n][metric]["runs"]) for n in names)
        pad = (hi - lo) * 0.4 + 1e-6
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(metric, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("#555")
        ax.grid(alpha=0.15, axis="y")

    fig.suptitle(f"{title} - mean +/- std antar seed (titik = tiap seed)",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()


def run(log_fn, attentions=None, seeds=None, epochs=None):
    sys.path.insert(0, os.path.dirname(__file__))
    from step5_training import train_one_run, resolve_recipe
    import common

    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} dulu.")
            raise FileNotFoundError(path)

    ensure_dirs()
    os.makedirs(ABLATION_DIR, exist_ok=True)

    attentions = attentions or EXPERIMENT["attentions"]
    seeds      = seeds      or EXPERIMENT["seeds"]
    epochs     = epochs     or EXPERIMENT["epochs"]
    select_metric = TRAIN_HP.get("select_metric", "auc")

    combos = list(itertools.product(attentions, seeds))
    log_fn("=" * 78)
    log_fn("  STEP 11 - ABLASI MULTI-SEED")
    log_fn("=" * 78)
    log_fn(f"Modul atensi    : {attentions}")
    log_fn(f"Seed            : {seeds}")
    log_fn(f"Epoch/run       : {epochs}")
    log_fn(f"Pilih checkpoint: val {select_metric}")
    log_fn(f"TTA evaluasi    : {'aktif' if EVAL_CONFIG.get('tta') else 'nonaktif'}")
    log_fn(f"Total run       : {len(combos)}")
    log_fn("PERINGATAN: ini akan berjalan lama. Jalankan semalaman.")
    log_fn("")

    raw = {a: [] for a in attentions}
    t_start = time.time()
    trained_durations = []

    for i, (attn, seed) in enumerate(combos, 1):
        name    = _run_name(attn, seed, epochs, select_metric)
        ckpt    = os.path.join(ABLATION_DIR, f"{name}.pth")
        partial = os.path.join(ABLATION_DIR, f"{name}.partial.pth")
        hist_p  = os.path.join(ABLATION_DIR, f"{name}.history.json")

        log_fn("-" * 78)
        log_fn(f"[{i}/{len(combos)}]  attention={attn}  seed={seed}")
        log_fn("-" * 78)

        best_epoch = None
        hp = {**TRAIN_HP, "epochs": epochs, "seed": seed}
        if common.reusable(ckpt, hist_p, resolve_recipe(hp), log_fn):
            log_fn(f"Run sudah selesai sebelumnya, training dilewati: {os.path.basename(ckpt)}")
            if os.path.exists(hist_p):
                with open(hist_p) as f:
                    best_epoch = json.load(f).get("best_epoch")
        else:
            if os.path.exists(partial):
                log_fn("Ditemukan run terputus (.partial) - diulang dari awal.")
                os.remove(partial)
            t_run = time.time()
            result = train_one_run(hp, log_fn, ckpt_path=partial, attention=attn,
                                   seed=seed, tag=f"{attn}/s{seed}")
            os.replace(partial, ckpt)          # baru dianggap selesai di sini
            best_epoch = result["best_epoch"]
            with open(hist_p, "w") as f:
                json.dump(result["history"], f, indent=2)
            trained_durations.append(time.time() - t_run)

        res = _evaluate_checkpoint(ckpt, attn, log_fn)
        raw[attn].append({"seed": seed, "best_epoch": best_epoch, **res})
        log_fn(f"  TEST -> acc {res['accuracy']:.2f}% | bal {res['balanced_accuracy']:.2f}% | "
               f"f1m {res['f1_macro']:.4f} | auc {res['auc_macro']:.4f} "
               f"| ckpt epoch {best_epoch}")

        remaining = len(combos) - i
        if trained_durations and remaining:
            eta_min = np.mean(trained_durations) * remaining / 60
            log_fn(f"  Estimasi sisa waktu: ~{eta_min:.0f} menit "
                   f"({remaining} run lagi, rata-rata {np.mean(trained_durations)/60:.1f} menit/run)")

    # ---------- Agregasi ----------
    summary = {}
    for attn, runs in raw.items():
        if not runs:
            continue
        entry = {"n_runs": len(runs), "params": runs[0]["params"],
                 "seeds": [r["seed"] for r in runs],
                 "best_epochs": [r["best_epoch"] for r in runs]}
        for metric in METRICS:
            vals = np.array([r[metric] for r in runs], dtype=float)
            entry[metric] = {
                "mean": float(vals.mean()),
                "std":  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "runs": [float(v) for v in vals],
            }
        summary[attn] = entry

    log_fn("")
    log_fn("=" * 96)
    log_fn("  HASIL AKHIR (mean +/- std antar seed)")
    log_fn("=" * 96)
    log_fn(f"  {'Modul':<8} {'Params':>12} {'Accuracy':>16} {'Balanced Acc':>16} "
           f"{'F1-macro':>18} {'AUC':>18}")
    for attn in attentions:
        if attn not in summary:
            continue
        s = summary[attn]
        log_fn(
            f"  {attn:<8} {s['params']:>12,} "
            f"{s['accuracy']['mean']:>9.2f} +/-{s['accuracy']['std']:>4.2f} "
            f"{s['balanced_accuracy']['mean']:>9.2f} +/-{s['balanced_accuracy']['std']:>4.2f} "
            f"{s['f1_macro']['mean']:>10.4f} +/-{s['f1_macro']['std']:>6.4f} "
            f"{s['auc_macro']['mean']:>10.4f} +/-{s['auc_macro']['std']:>6.4f}"
        )
    log_fn("=" * 96)

    # ---------- Uji signifikansi berpasangan ----------
    tests = {}
    reference = "fsca" if "fsca" in summary else None
    if reference and len(summary) > 1:
        tests = _paired_tests(raw, reference, log_fn)
        log_fn("\nCatatan: p < 0.05 berarti selisih signifikan. Dengan n=5 seed, "
               "selisih kecil yang konsisten (mis. menang 5/5) tetap layak dilaporkan "
               "walau p belum < 0.05.")

    log_fn(f"\nTotal waktu: {(time.time() - t_start) / 60:.1f} menit")

    _plot(summary, CHART_PATH)
    log_fn(f"Chart perbandingan -> {CHART_PATH}")

    payload = {"summary": summary, "raw": raw, "paired_tests": tests,
               "attentions": attentions, "seeds": seeds, "epochs": epochs,
               "select_metric": select_metric, "tta": EVAL_CONFIG.get("tta", False),
               "chart_path": CHART_PATH}
    with open(REPORT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    log_fn(f"Report tersimpan -> {REPORT_PATH}")

    return payload


if __name__ == "__main__":
    run(print)