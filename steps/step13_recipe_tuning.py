"""
Step 13 - Uji Resep (dijalankan SEBELUM resep final dikunci)
==============================================================
Membandingkan kandidat resep (RECIPE_TUNING di config.py) untuk ResNet-18
tanpa modul atensi, masing-masing beberapa seed.

Kenapa step ini ada
--------------------
Pada resep 100 epoch, model mencapai puncak val AUC sekitar epoch 20-30,
saat learning rate cosine masih tinggi; fase LR kecil (epoch 60-100) terjadi
ketika model sudah overfit. Step ini menguji apakah jadwal lebih pendek
dan/atau weight decay lebih besar memberi puncak yang lebih baik.

Aturan pemilihan (ditetapkan sebelum run, lihat config.py)
-----------------------------------------------------------
- Yang dibandingkan HANYA metrik validasi (val AUC di epoch checkpoint
  terpilih, rata-rata antar seed). Data test TIDAK dihitung di step ini,
  sehingga pemilihan resep tidak "mencontek" test.
- Pemenang = rata-rata val AUC tertinggi. Bila selisih dengan kandidat lain
  <= tie_tolerance, kandidat dengan epoch lebih sedikit yang dipilih.

Pemakaian ulang: kandidat resep saat ini (100 epoch, wd 1e-4) memakai history
Step 11 (attention="none") untuk seed yang sama, bila resepnya cocok.

Output: outputs/recipe_tuning_report.json
        outputs/recipe_tuning_val_auc.png
        outputs/recipe_tuning/<kandidat>_seed<s>_<metric>.pth (+ .history.json)
"""

import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, TRAIN_HP, RECIPE_TUNING, ensure_dirs

META_PATH   = PATHS["dataset_meta"]
CONFIG_PATH = PATHS["preprocess_config"]
OUT_DIR     = PATHS["recipe_tuning_dir"]
REPORT_PATH = PATHS["recipe_tuning_report"]
CHART_PATH  = PATHS["recipe_tuning_chart"]


def _candidate_hp(cand, seed):
    return {**TRAIN_HP, "seed": seed, **RECIPE_TUNING["candidates"][cand]}


def _locate(cand, seed, select_metric, attention, log_fn=None, retire=True):
    """
    Return (ckpt, hist, source, ready). source: "recipe_tuning" | "step11".
    ready=True bila history tersedia dan resepnya cocok dengan kandidat.
    """
    from step5_training import resolve_recipe
    import common

    hp = _candidate_hp(cand, seed)
    expected = resolve_recipe(hp)
    name = f"{cand}_seed{seed}_{select_metric}"
    ckpt = os.path.join(OUT_DIR, f"{name}.pth")
    hist = os.path.join(OUT_DIR, f"{name}.history.json")
    if os.path.exists(hist) and common.reusable(ckpt, hist, expected, log_fn, retire=retire):
        return ckpt, hist, "recipe_tuning", True

    if attention == "none":
        s11 = f"none_seed{seed}_ep{hp['epochs']}_{select_metric}"
        s11_ckpt = os.path.join(PATHS["ablation_dir"], f"{s11}.pth")
        s11_hist = os.path.join(PATHS["ablation_dir"], f"{s11}.history.json")
        if os.path.exists(s11_hist) and common.reusable(s11_ckpt, s11_hist, expected,
                                                        log_fn, retire=False):
            return s11_ckpt, s11_hist, "step11", True

    return ckpt, hist, "recipe_tuning", False


def _val_at_best(history):
    """Metrik validasi pada epoch checkpoint terpilih."""
    be = history.get("best_epoch")
    if not be or not history.get("val_auc"):
        raise ValueError("History tidak memuat best_epoch / val_auc.")
    i = be - 1
    return {
        "best_epoch":   be,
        "val_auc":      float(history["val_auc"][i]),
        "val_loss":     float(history["val_loss"][i]),
        "val_acc":      float(history["val_acc"][i]),
        "val_bal_acc":  float(history["val_bal_acc"][i]) if history.get("val_bal_acc") else float("nan"),
        "epochs_run":   len(history["val_auc"]),
    }


def _plot(curves, summary, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#90A4AE", "#4FC3F7", "#FFB74D", "#81C784", "#BA68C8"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a1a")
    for ax, key, ylab in [(axes[0], "val_auc", "Val AUC macro"),
                          (axes[1], "val_loss", "Val loss")]:
        ax.set_facecolor("#1a1a1a")
        for i, cand in enumerate(curves):
            runs = curves[cand][key]
            n = min(len(r) for r in runs)
            arr = np.array([r[:n] for r in runs], dtype=float)
            ep = np.arange(1, n + 1)
            c = colors[i % len(colors)]
            ax.plot(ep, arr.mean(axis=0), color=c, linewidth=2,
                    label=f"{labels.get(cand, cand)} (n={len(runs)})")
            ax.fill_between(ep, arr.min(axis=0), arr.max(axis=0), color=c, alpha=0.12)
            if key == "val_auc":
                be = np.mean(summary[cand]["best_epochs"])
                ax.axvline(be, color=c, linestyle="--", linewidth=1)
        ax.set_title(f"{ylab} per epoch (garis = rata-rata seed, bayangan = min-maks)",
                     color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", color="white"); ax.set_ylabel(ylab, color="white")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("#555")
        ax.grid(alpha=0.15)
        ax.legend(facecolor="#333", labelcolor="white", fontsize=8)
    axes[0].text(0.01, 0.02, "garis putus-putus = rata-rata epoch checkpoint terpilih",
                 transform=axes[0].transAxes, color="#aaaaaa", fontsize=8)
    fig.suptitle("Uji Resep ResNet-18 - hanya data validasi", color="white",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()


def run(log_fn, candidates=None, seeds=None):
    sys.path.insert(0, os.path.dirname(__file__))
    from step5_training import train_one_run

    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} dulu.")
            raise FileNotFoundError(path)

    ensure_dirs()
    os.makedirs(OUT_DIR, exist_ok=True)

    all_cands = RECIPE_TUNING["candidates"]
    labels    = RECIPE_TUNING.get("labels", {})
    attention = RECIPE_TUNING.get("attention", "none")
    tol       = RECIPE_TUNING.get("tie_tolerance", 0.001)
    candidates = candidates or list(all_cands)
    unknown = [c for c in candidates if c not in all_cands]
    if unknown:
        raise ValueError(f"Kandidat tidak dikenal: {unknown}. Pilihan: {list(all_cands)}")
    seeds = seeds or RECIPE_TUNING["seeds"]
    select_metric = TRAIN_HP.get("select_metric", "auc")
    current = next(iter(all_cands))          # kandidat pertama = resep saat ini

    combos = [(c, s) for c in candidates for s in seeds]
    log_fn("=" * 78)
    log_fn("  STEP 13 - UJI RESEP (hanya validasi; data test tidak dipakai)")
    log_fn("=" * 78)
    log_fn(f"Model           : ResNet-18 (attention={attention})")
    for c in candidates:
        log_fn(f"  {c:<14} {labels.get(c, ''):<44} {all_cands[c]}")
    log_fn(f"Seed            : {seeds}")
    log_fn(f"Pilih checkpoint: val {select_metric}")
    log_fn(f"Aturan pemenang : rata-rata val AUC tertinggi; seri (<= {tol}) -> epoch lebih sedikit")
    log_fn(f"Total run       : {len(combos)}")
    log_fn("")

    raw = {c: [] for c in candidates}
    curves = {c: {"val_auc": [], "val_loss": []} for c in candidates}
    durations = []

    for i, (cand, seed) in enumerate(combos, 1):
        log_fn("-" * 78)
        log_fn(f"[{i}/{len(combos)}]  kandidat={cand}  seed={seed}")
        log_fn("-" * 78)

        ckpt, hist_p, source, ready = _locate(cand, seed, select_metric, attention, log_fn)
        if ready:
            src = "Step 11" if source == "step11" else "run sebelumnya"
            log_fn(f"History sudah ada ({src}), training dilewati: {os.path.basename(hist_p)}")
            with open(hist_p) as f:
                history = json.load(f)
        else:
            partial = ckpt[:-4] + ".partial.pth"
            if os.path.exists(partial):
                log_fn("Ditemukan run terputus (.partial) - diulang dari awal.")
                os.remove(partial)
            t_run = time.time()
            result = train_one_run(_candidate_hp(cand, seed), log_fn, ckpt_path=partial,
                                   attention=attention, seed=seed, tag=f"{cand}/s{seed}")
            os.replace(partial, ckpt)
            history = result["history"]
            with open(hist_p, "w") as f:
                json.dump(history, f, indent=2)
            durations.append(time.time() - t_run)

        v = _val_at_best(history)
        raw[cand].append({"seed": seed, "source": source, **v})
        curves[cand]["val_auc"].append(history["val_auc"])
        curves[cand]["val_loss"].append(history["val_loss"])
        log_fn(f"  VAL (epoch {v['best_epoch']}/{v['epochs_run']}) -> AUC {v['val_auc']:.4f} | "
               f"acc {v['val_acc']:.2f}% | bal {v['val_bal_acc']:.2f}% | loss {v['val_loss']:.4f}")

        remaining = sum(1 for c, s in combos[i:]
                        if not _locate(c, s, select_metric, attention, retire=False)[3])
        if durations and remaining:
            log_fn(f"  Estimasi sisa waktu: ~{np.mean(durations) * remaining / 60:.0f} menit "
                   f"({remaining} run lagi; durasi per run bergantung jumlah epoch)")

    # ---------- Ringkasan ----------
    summary = {}
    for c, runs in raw.items():
        aucs = np.array([r["val_auc"] for r in runs], dtype=float)
        summary[c] = {
            "label": labels.get(c, c), "override": all_cands[c],
            "epochs": all_cands[c].get("epochs", TRAIN_HP["epochs"]),
            "seeds": [r["seed"] for r in runs],
            "val_auc_mean": float(aucs.mean()),
            "val_auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
            "val_auc_runs": [float(a) for a in aucs],
            "val_loss_mean": float(np.mean([r["val_loss"] for r in runs])),
            "val_bal_acc_mean": float(np.mean([r["val_bal_acc"] for r in runs])),
            "best_epochs": [r["best_epoch"] for r in runs],
        }

    log_fn("")
    log_fn("=" * 96)
    log_fn("  HASIL (validasi, pada epoch checkpoint terpilih; rata-rata antar seed)")
    log_fn("=" * 96)
    log_fn(f"  {'Kandidat':<44} {'Val AUC':>17} {'Val loss':>9} {'Val bal acc':>12} {'Epoch ckpt':>14}")
    for c in candidates:
        s = summary[c]
        log_fn(f"  {s['label']:<44} {s['val_auc_mean']:>9.4f} +/-{s['val_auc_std']:>6.4f} "
               f"{s['val_loss_mean']:>9.4f} {s['val_bal_acc_mean']:>11.2f}% "
               f"{str(s['best_epochs']):>14}")

    # Selisih berpasangan per seed terhadap resep saat ini
    comparisons = {}
    if current in raw:
        cur = {r["seed"]: r["val_auc"] for r in raw[current]}
        log_fn(f"\nSelisih val AUC terhadap resep saat ini ({labels.get(current, current)}), per seed:")
        for c in candidates:
            if c == current:
                continue
            other = {r["seed"]: r["val_auc"] for r in raw[c]}
            common_seeds = sorted(set(cur) & set(other))
            diffs = [other[s] - cur[s] for s in common_seeds]
            if not diffs:
                continue
            wins = sum(d > 0 for d in diffs)
            comparisons[c] = {"seeds": common_seeds, "diffs": diffs,
                              "mean_diff": float(np.mean(diffs)), "wins": wins}
            log_fn(f"  {labels.get(c, c):<44} rata-rata {np.mean(diffs):+.4f} | lebih baik di "
                   f"{wins}/{len(diffs)} seed | per seed: "
                   + ", ".join(f"{d:+.4f}" for d in diffs))

    # ---------- Pemenang (aturan yang sudah ditetapkan) ----------
    best_mean = max(summary[c]["val_auc_mean"] for c in candidates)
    tied = [c for c in candidates if best_mean - summary[c]["val_auc_mean"] <= tol]
    winner = min(tied, key=lambda c: (summary[c]["epochs"], -summary[c]["val_auc_mean"]))
    w = all_cands[winner]

    log_fn("")
    log_fn("=" * 78)
    log_fn(f"  PEMENANG: {labels.get(winner, winner)}")
    if len(tied) > 1:
        log_fn(f"  (seri dalam toleransi {tol} dengan: "
               + ", ".join(labels.get(c, c) for c in tied if c != winner)
               + " -> dipilih epoch paling sedikit)")
    log_fn("=" * 78)
    log_fn("Ubah TRAIN_HP di config.py menjadi:")
    log_fn(f'    "epochs":       {w.get("epochs", TRAIN_HP["epochs"])},')
    log_fn(f'    "weight_decay": {w.get("weight_decay", TRAIN_HP["weight_decay"])},')
    log_fn(f'    "patience":     {w.get("epochs", TRAIN_HP["epochs"])},   # = epochs (early stopping nonaktif)')
    log_fn("Lalu jalankan ulang: Step 5-9 (model utama), run_ablation.py (Step 11),")
    log_fn("dan run_recipe_ablation.py (Step 12), agar semua hasil memakai resep final.")

    _plot(curves, summary, labels, CHART_PATH)
    log_fn(f"\nGrafik kurva validasi -> {CHART_PATH}")

    payload = {"summary": summary, "raw": raw, "comparisons_vs_current": comparisons,
               "winner": winner, "winner_override": w, "tied": tied,
               "tie_tolerance": tol, "seeds": seeds, "select_metric": select_metric,
               "note": "Pemilihan hanya berdasarkan validasi; data test tidak dievaluasi.",
               "chart_path": CHART_PATH}
    with open(REPORT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    log_fn(f"Report tersimpan -> {REPORT_PATH}")
    return payload


if __name__ == "__main__":
    run(print)