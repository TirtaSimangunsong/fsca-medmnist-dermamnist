"""
make_report_figures.py - Membuat ulang gambar hasil pada Bab IV laporan
=========================================================================
Semua gambar dibuat dari berkas JSON di outputs/, jadi angka pada gambar
selalu sama dengan angka pada tabel. Gaya gambar: latar putih, font serif,
koma desimal (format laporan berbahasa Indonesia).

  Gambar 4.2  gambar_4_2_multi_seed.png        <- ablation_report.json (Step 11)
  Gambar 4.3  gambar_4_3_kurva_pelatihan.png   <- *training_history.json
  Gambar 4.4  gambar_4_4_bias_melanoma.png     <- *evaluation_report.json (val_sweep)
  Gambar 4.5  gambar_4_5_confusion_matrix.png  <- *evaluation_report.json
  Gambar 4.6  gambar_4_6_gradcam.png           <- salinan outputs/gradcam_grid.png (Step 8)
  Gambar 4.7  gambar_4_7_gradcam_fsca.png      <- salinan outputs/gradcam_fsca_effect.png (Step 8)

Urutan menjalankan: Step 4 (baseline), Step 6 (FSCA), Step 7 (evaluasi),
Step 8 (Grad-CAM), run_ablation.py (Step 11), lalu:
    python make_report_figures.py

Output: outputs/report_figures/
Gambar yang datanya belum ada dilewati dengan pesan, tidak membuat error.
"""

import os
import sys
import json
import shutil

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.ticker import FuncFormatter          # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from config import PATHS, TRAIN_HP                   # noqa: E402

OUT_DIR = os.path.join(PATHS["output_dir"], "report_figures")

GREY, BLUE, INK, MUTED, GRID = "#8a8f98", "#1f5fae", "#1a1a1a", "#555555", "#e6e6e6"
ABBR = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11,
    "axes.edgecolor": "#9a9a9a", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.facecolor": "white",
})


def _c(text):
    """Koma desimal untuk teks gambar."""
    return str(text).replace(".", ",")


COMMA = FuncFormatter(lambda v, _pos: _c(f"{v:.10g}"))


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _need(name, *items):
    missing = [os.path.relpath(p, BASE_DIR) for p, d in items if d is None]
    if missing:
        print(f"[lewati] {name}: belum ada {', '.join(missing)}")
        return False
    return True


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[ok] {path}")


# ---------------------------------------------------------------------------
# Gambar 4.3 - kurva loss dan AUC validasi (run tunggal)
# ---------------------------------------------------------------------------
def fig_training_curve(hb, hf):
    n_ep = len(hf["val_loss"])
    ep = np.arange(1, n_ep + 1)
    ticks = [1] + list(range(10, n_ep + 1, 10))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

    ax = axes[0]
    for h, c, lab in [(hb, GREY, "Baseline"), (hf, BLUE, "FSCA")]:
        ax.plot(ep, h["train_loss"], color=c, lw=1.4, ls="--")
        ax.plot(ep, h["val_loss"], color=c, lw=1.8)
        ax.text(n_ep + 0.6, h["val_loss"][-1], f"{lab} (val)", color=INK, va="center", fontsize=9)
        ax.text(n_ep + 0.6, h["train_loss"][-1], f"{lab} (train)", color=MUTED, va="center", fontsize=9)
    ax.set_title("Loss pelatihan (putus-putus) dan validasi (utuh)", color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted cross-entropy loss")
    ax.set_xlim(1, n_ep * 1.25)
    ax.set_xticks(ticks)
    ax.yaxis.set_major_formatter(COMMA)
    ax.grid(axis="y", color=GRID)

    ax = axes[1]
    lo = min(min(hb["val_auc"]), min(hf["val_auc"]))
    for i, (h, c, lab, mk) in enumerate([(hb, GREY, "Baseline", "o"), (hf, BLUE, "FSCA", "s")]):
        ax.plot(ep, h["val_auc"], color=c, lw=1.8)
        b = h["best_epoch"]
        v = h["val_auc"][b - 1]
        ax.plot([b], [v], marker=mk, color=c, ms=7, mec="white", mew=1.5, zorder=3)
        ax.annotate(_c(f"{lab}: epoch {b}, AUC {v:.4f}"), (b, v),
                    xytext=(n_ep * 0.42, lo + (0.035 if i == 0 else 0.015)),
                    fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8))
    ax.set_title("AUC validasi; penanda = checkpoint terbaik", color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC makro (validasi)")
    ax.set_xticks(ticks)
    ax.yaxis.set_major_formatter(COMMA)
    ax.grid(axis="y", color=GRID)
    _save(fig, "gambar_4_3_kurva_pelatihan.png")


# ---------------------------------------------------------------------------
# Gambar 4.4 - sapuan bias logit melanoma pada data validasi
# ---------------------------------------------------------------------------
def fig_melanoma_bias(rb, rf):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    chosen = None
    for r, c, lab, mk in [(rb, GREY, "Baseline", "o"), (rf, BLUE, "FSCA", "s")]:
        mb = r.get("melanoma_bias") or {}
        sw = mb.get("val_sweep") or []
        if not sw:
            continue
        chosen = mb.get("bias", chosen)
        b = [x["bias"] for x in sw]
        axes[0].plot(b, [x["recall"] for x in sw], color=c, lw=1.8, marker=mk, ms=4)
        axes[1].plot(b, [x["balanced_acc"] for x in sw], color=c, lw=1.8, marker=mk, ms=4)
        axes[0].text(b[-1] + 0.05, sw[-1]["recall"], lab, va="center", fontsize=9, color=INK)
        axes[1].text(b[-1] + 0.05, sw[-1]["balanced_acc"], lab, va="center", fontsize=9, color=INK)
    for ax in axes:
        if chosen is not None:
            ax.axvline(chosen, color="#b0b0b0", lw=1, ls=":")
        ax.grid(axis="y", color=GRID)
        ax.set_xlabel("Bias logit melanoma")
        ax.xaxis.set_major_formatter(COMMA)
        ax.yaxis.set_major_formatter(COMMA)
    if chosen is not None:
        axes[0].text(chosen + 0.05, 0.93, _c(f"bias terpilih = {chosen:g}"), fontsize=9,
                     color=MUTED, transform=axes[0].get_xaxis_transform())
    axes[0].set_ylabel("Recall melanoma (validasi)")
    axes[0].set_title("Recall melanoma naik seiring bias", color=INK)
    axes[1].set_ylabel("Balanced accuracy (%) (validasi)")
    axes[1].set_title("Balanced accuracy turun pada bias besar", color=INK)
    _save(fig, "gambar_4_4_bias_melanoma.png")


# ---------------------------------------------------------------------------
# Gambar 4.5 - confusion matrix test set (persentase per baris)
# ---------------------------------------------------------------------------
def fig_confusion(rb, rf):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    mel, nv = ABBR.index("MEL"), ABBR.index("NV")
    for ax, r, ttl in [(axes[0], rb, "ResNet-18 Baseline"), (axes[1], rf, "ResNet-18 + FSCA")]:
        cm = np.array(r["confusion_matrix"], dtype=float)
        pct = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(pct.shape[0]):
            for j in range(pct.shape[1]):
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if pct[i, j] > 55 else INK)
        ax.set_xticks(range(len(ABBR)))
        ax.set_yticks(range(len(ABBR)))
        ax.set_xticklabels(ABBR, fontsize=8.5)
        ax.set_yticklabels(ABBR, fontsize=8.5)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label sebenarnya")
        biased = (r.get("melanoma_bias") or {}).get("bias", 0)
        note = "setelah bias melanoma" if biased else "tanpa bias"
        ax.set_title(_c(f"{ttl} ({note})\nrecall MEL {r['melanoma_recall']:.3f}, "
                        f"MEL→NV {pct[mel, nv]:.0f}%"), color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    _save(fig, "gambar_4_5_confusion_matrix.png")


# ---------------------------------------------------------------------------
# Gambar 4.2 - hasil lima seed, berpasangan per seed
# ---------------------------------------------------------------------------
def fig_multi_seed(ab):
    raw = ab.get("raw", {})
    if "none" not in raw or "fsca" not in raw:
        print("[lewati] Gambar 4.2: ablation_report.json belum memuat 'none' dan 'fsca'")
        return
    by_n = {r["seed"]: r for r in raw["none"]}
    by_f = {r["seed"]: r for r in raw["fsca"]}
    seeds = sorted(set(by_n) & set(by_f))

    def mel(r):
        return r.get("recall_melanoma", r["recall_per_class"]["melanoma"])

    metrics = [(lambda r: r["auc_macro"], "AUC makro", "{:.4f}"),
               (lambda r: r["balanced_accuracy"], "Balanced accuracy (%)", "{:.2f}"),
               (mel, "Recall melanoma", "{:.3f}"),
               (lambda r: r["accuracy"], "Accuracy (%)", "{:.2f}")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(9.6, 3.4))
    for ax, (get, ttl, fmt) in zip(axes, metrics):
        a = [get(by_n[s]) for s in seeds]
        b = [get(by_f[s]) for s in seeds]
        for x0, x1 in zip(a, b):
            ax.plot([0, 1], [x0, x1], color="#c8c8c8", lw=1, zorder=1)
        ax.scatter([0] * len(a), a, color=GREY, s=26, zorder=2, edgecolor="white", linewidth=0.8)
        ax.scatter([1] * len(b), b, color=BLUE, s=26, zorder=2, marker="s",
                   edgecolor="white", linewidth=0.8)
        for x, v, c in [(0, np.mean(a), GREY), (1, np.mean(b), BLUE)]:
            ax.plot([x - 0.18, x + 0.18], [v, v], color=c, lw=2.2, zorder=3)
            ax.text(x + (0.22 if x else -0.22), v, _c(fmt.format(v)), va="center",
                    ha="left" if x else "right", fontsize=8.5, color=INK)
        wins = sum(y > x for x, y in zip(a, b))
        ax.set_title(f"{ttl}\nFSCA lebih tinggi di {wins}/{len(seeds)} seed", color=INK, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "FSCA"])
        ax.set_xlim(-0.7, 1.7)
        ax.yaxis.set_major_formatter(COMMA)
        ax.grid(axis="y", color=GRID)
    _save(fig, "gambar_4_2_multi_seed.png")


def copy_step8(src, name):
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(OUT_DIR, name))
        print(f"[ok] {os.path.join(OUT_DIR, name)} (salinan dari Step 8)")
    else:
        print(f"[lewati] {name}: jalankan Step 8 (Grad-CAM) dulu")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hb, hf = _load(PATHS["baseline_history"]), _load(PATHS["training_history"])
    rb, rf = _load(PATHS["baseline_report"]), _load(PATHS["evaluation_report"])
    ab = _load(PATHS["ablation_report"])

    hist_items = [(PATHS["baseline_history"], hb), (PATHS["training_history"], hf)]
    rep_items = [(PATHS["baseline_report"], rb), (PATHS["evaluation_report"], rf)]
    if _need("Gambar 4.3", *hist_items):
        fig_training_curve(hb, hf)
    if _need("Gambar 4.4", *rep_items):
        fig_melanoma_bias(rb, rf)
    if _need("Gambar 4.5", *rep_items):
        fig_confusion(rb, rf)
    if _need("Gambar 4.2", (PATHS["ablation_report"], ab)):
        if ab.get("epochs") != TRAIN_HP["epochs"]:
            print(f"[info] ablation_report.json memakai {ab.get('epochs')} epoch, "
                  f"resep final {TRAIN_HP['epochs']} epoch")
        fig_multi_seed(ab)
    copy_step8(PATHS["gradcam_grid_chart"], "gambar_4_6_gradcam.png")
    copy_step8(os.path.join(PATHS["output_dir"], "gradcam_fsca_effect.png"),
               "gambar_4_7_gradcam_fsca.png")
    print(f"\nSelesai. Gambar ada di {OUT_DIR}")


if __name__ == "__main__":
    main()
