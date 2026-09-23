"""
Step 8 - Grad-CAM (Explainable AI) — VERSI PERBAIKAN
======================================================
Menghasilkan heatmap Grad-CAM untuk setiap kelas DermaMNIST
menggunakan model final dari Step 6/7.

RINGKASAN PERBAIKAN dari versi sebelumnya
------------------------------------------
1. BUG KRITIS — label "pred:" palsu.
   Versi lama memanggil `generate(target_class=cls_idx)` lalu memakai nilai
   kembalian itu sebagai "prediksi". Karena target_class diisi manual,
   cabang `argmax` tidak pernah dieksekusi, sehingga judul gambar SELALU
   menampilkan label asli seolah model selalu benar. Kini prediksi model
   dihitung terpisah dan ditampilkan apa adanya, lengkap dengan penanda
   BENAR / SALAH.

2. Hook dipindah dari `layer4` ke `fsca4`.
   Pada forward, urutannya layer4 -> fsca4 -> avgpool -> fc. Hook di layer4
   membaca fitur SEBELUM modul FSCA bekerja, sehingga heatmap sama sekali
   tidak memuat kontribusi FSCA — padahal itu justru yang ingin divalidasi
   (Rumusan Masalah poin 3).

3. Multi-layer. CAM dihasilkan dari fsca3 (7x7) DAN fsca4 (4x4).
   Pada input 28x28, peta fitur layer4 hanya 4x4, jadi CAM-nya cuma 16 nilai
   yang di-upsample — mustahil terlihat presisi dan selalu bias ke tengah.
   fsca3 memberi resolusi spasial ~3x lebih baik.

4. Deteksi CAM kosong.
   Versi lama selalu meregangkan CAM ke [0,1], sehingga CAM datar tetap
   tampil warna-warni. Kini nilai mentah sebelum normalisasi diperiksa;
   bila mendekati nol, panel ditandai "CAM KOSONG" alih-alih menampilkan
   gradasi palsu.

5. Pemilihan sampel berbasis kualitas.
   Versi lama mengambil sampel PERTAMA yang ditemukan per kelas tanpa
   memeriksa apakah model memprediksinya benar. Kini dipilih sampel yang
   diprediksi BENAR dengan confidence tertinggi; bila tidak ada satu pun
   yang benar untuk kelas tersebut, hal itu dilaporkan eksplisit.

6. Penangkapan gradien lewat tensor hook, bukan `register_full_backward_hook`
   pada nn.Sequential. BasicBlock torchvision memakai operasi in-place
   (`out += identity`), yang membuat full backward hook pada modul kontainer
   tidak selalu andal.

7. Figure pembanding layer4 vs fsca4 — bukti visual langsung bahwa modul
   FSCA menggeser fokus model. Ini yang menjawab Rumusan Masalah poin 3.

8. Device dipilih lewat common.resolve_device() (DEVICE_CONFIG "auto":
   MPS -> CUDA -> CPU), sama seperti Step 5, 6, 7, 10, dan 11.

Output: outputs/gradcam_grid.png            (grid utama, multi-layer)
        outputs/gradcam_fsca_effect.png     (layer4 vs fsca4)
        outputs/gradcam/class_{i}_{name}.png
        outputs/gradcam/gradcam_diagnostics.json
"""

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, DATASET, ensure_dirs

OUTPUT_DIR   = PATHS["output_dir"]
GRADCAM_DIR  = PATHS["gradcam_dir"]
META_PATH    = PATHS["dataset_meta"]
CONFIG_PATH  = PATHS["preprocess_config"]
FINAL_MODEL  = PATHS["final_model"]
GRID_PATH    = PATHS["gradcam_grid_chart"]

EFFECT_PATH      = os.path.join(OUTPUT_DIR, "gradcam_fsca_effect.png")
DIAGNOSTICS_PATH = os.path.join(GRADCAM_DIR, "gradcam_diagnostics.json")

# Ambang di bawah mana CAM dianggap kosong (tidak ada aktivasi positif)
DEAD_CAM_EPS = 1e-6

# Lapisan yang divisualisasikan pada grid utama, berurutan dari resolusi
# tinggi ke rendah. Nama harus cocok dengan atribut pada ResNet18_FSCA.
GRID_LAYERS = ["fsca3", "fsca4"]


# =========================================================
# GRAD-CAM
# =========================================================
class GradCAM:
    """
    Grad-CAM tanpa pustaka eksternal.

    Catatan kompatibilitas: `generate()` tetap mengembalikan
    (cam, target_class) seperti versi lama agar Step 10 tidak rusak.
    Informasi tambahan disimpan sebagai atribut setelah pemanggilan:
        .last_pred        - kelas hasil argmax model (prediksi sebenarnya)
        .last_probs       - vektor probabilitas softmax
        .last_raw_max     - nilai maksimum CAM SEBELUM normalisasi
        .last_is_dead     - True bila CAM praktis kosong
        .last_cam_shape   - dimensi CAM asli sebelum di-upsample
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._handles     = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output
            if output.requires_grad:
                # Tensor hook: lebih andal daripada full_backward_hook pada
                # modul kontainer yang memuat operasi in-place.
                output.register_hook(self._save_gradient)

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def remove(self):
        """Lepas hook. Panggil ini bila membuat banyak objek GradCAM."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def generate(self, input_tensor, target_class=None, target_size=None):
        import torch
        import torch.nn.functional as F

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)
        probs  = F.softmax(output, dim=1).detach().squeeze(0)

        # Prediksi sebenarnya selalu dihitung, terlepas dari target_class.
        self.last_pred  = int(output.argmax(dim=1).item())
        self.last_probs = probs.cpu().numpy()

        if target_class is None:
            target_class = self.last_pred

        score = output[0, target_class]
        score.backward()

        if self.gradients is None:
            raise RuntimeError(
                "Gradien tidak tertangkap. Pastikan target_layer benar-benar "
                "dilewati pada forward pass dan model tidak dalam no_grad()."
            )

        acts  = self.activations.detach()
        grads = self.gradients

        weights = grads.mean(dim=[2, 3], keepdim=True)          # (1, C, 1, 1)
        cam     = (weights * acts).sum(dim=1, keepdim=True)      # (1, 1, H, W)
        cam     = F.relu(cam)

        self.last_cam_shape = tuple(cam.shape[-2:])

        # Periksa nilai MENTAH sebelum normalisasi. Versi lama melewatkan ini,
        # sehingga CAM datar tetap tampil seolah ada aktivasi kuat.
        raw_max = float(cam.max().item())
        self.last_raw_max = raw_max
        self.last_is_dead = raw_max <= DEAD_CAM_EPS

        if self.last_is_dead:
            cam = torch.zeros_like(cam)
        else:
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

        if target_size is None:
            target_size = (DATASET["image_size"], DATASET["image_size"])
        cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)

        return cam.squeeze().cpu().numpy(), target_class


def _denormalize(tensor, mean, std):
    """Kembalikan tensor ternormalisasi ke citra [0,1] untuk visualisasi."""
    import torch
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (tensor * s + m).clamp(0, 1)


# =========================================================
# PEMILIHAN SAMPEL
# =========================================================
def _select_best_samples(model, test_ds, n_classes, device, log_fn):
    """
    Untuk tiap kelas, pilih sampel yang diprediksi BENAR dengan confidence
    tertinggi. Bila tidak ada yang benar, ambil confidence tertinggi terhadap
    kelas asli dan tandai sebagai tidak-terprediksi-benar.

    Grad-CAM pada sampel yang salah prediksi menghasilkan gradien terhadap
    kelas yang model sendiri tidak yakini — hasilnya nyaris selalu berantakan
    dan tidak layak dijadikan bukti interpretabilitas.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            all_probs.append(F.softmax(out, dim=1).cpu().numpy())
            all_labels.append(y.numpy().reshape(-1))
    probs  = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds  = probs.argmax(axis=1)

    log_fn(f"\nAkurasi test set (tanpa TTA): {100*(preds == labels).mean():.2f}%")

    selected = {}
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        if len(idx_c) == 0:
            log_fn(f"  kelas {c}: tidak ada sampel di test set.")
            continue

        correct = idx_c[preds[idx_c] == c]
        if len(correct) > 0:
            best = correct[probs[correct, c].argmax()]
            ok   = True
        else:
            # Tidak satu pun sampel kelas ini diprediksi benar.
            best = idx_c[probs[idx_c, c].argmax()]
            ok   = False
            log_fn(f"  kelas {c}: TIDAK ADA prediksi benar "
                   f"({len(idx_c)} sampel). Model tidak mengenali kelas ini.")

        selected[c] = {
            "index":      int(best),
            "confidence": float(probs[best, c]),
            "pred":       int(preds[best]),
            "correct":    bool(ok),
            "n_total":    int(len(idx_c)),
            "n_correct":  int(len(correct)),
        }
    return selected, probs, labels, preds


# =========================================================
# ENTRY POINT
# =========================================================
def run(log_fn):
    import torch
    from torchvision import transforms
    from medmnist import DermaMNIST
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model
    import common

    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2"),
                       (FINAL_MODEL, "Step 6")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. "
                   f"Jalankan {name} terlebih dahulu.")
            raise FileNotFoundError(path)

    with open(META_PATH)   as f: meta   = json.load(f)
    with open(CONFIG_PATH) as f: config = json.load(f)

    n_classes = meta["n_classes"]
    class_map = meta["class_map"]
    mean, std = config["mean"], config["std"]
    img_size  = DATASET["image_size"]

    device = common.resolve_device(log_fn=log_fn)

    log_fn("Memuat model untuk Grad-CAM...")
    model = build_model(n_classes=n_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(FINAL_MODEL, map_location=device))
    model.eval()
    log_fn(f"Model: ResNet-18 + {model.attention_name.upper()}")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_ds = DermaMNIST(split="test", transform=test_transform,
                         download=DATASET["download"], size=img_size)

    selected, probs, labels, preds = _select_best_samples(
        model, test_ds, n_classes, device, log_fn)

    os.makedirs(GRADCAM_DIR, exist_ok=True)
    ensure_dirs()

    # Siapkan objek GradCAM untuk tiap lapisan yang diminta
    cams = {}
    for layer_name in GRID_LAYERS:
        if not hasattr(model, layer_name):
            log_fn(f"PERINGATAN: model tidak punya atribut '{layer_name}', dilewati.")
            continue
        cams[layer_name] = GradCAM(model, target_layer=getattr(model, layer_name))
    if not cams:
        raise RuntimeError("Tidak ada target layer valid untuk Grad-CAM.")
    log_fn(f"Grad-CAM dipasang pada: {', '.join(cams.keys())}")

    colormap = plt.get_cmap("jet")
    n_cols   = 1 + 2 * len(cams)          # Original + (CAM, Overlay) per layer
    diagnostics = {}

    fig, axes = plt.subplots(n_classes, n_cols,
                             figsize=(3.0 * n_cols, n_classes * 2.9))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle("Grad-CAM — ResNet-18 + FSCA (DermaMNIST 28x28)\n"
                 "Sampel terpilih: prediksi benar dengan confidence tertinggi",
                 color="white", fontsize=13, fontweight="bold", y=1.005)

    for cls_idx in range(n_classes):
        row      = axes[cls_idx] if n_classes > 1 else axes
        cls_name = class_map[str(cls_idx)]

        if cls_idx not in selected:
            for ax in row:
                ax.axis("off"); ax.set_facecolor("#1a1a1a")
            continue

        info       = selected[cls_idx]
        img_tensor, _ = test_ds[info["index"]]
        inp        = img_tensor.unsqueeze(0).to(device)

        img_np = _denormalize(img_tensor, mean, std).permute(1, 2, 0).numpy()

        status = "BENAR" if info["correct"] else "SALAH"
        log_fn(f"  [{cls_idx}] {cls_name:<42} "
               f"conf {info['confidence']*100:5.1f}% | {status} | "
               f"{info['n_correct']}/{info['n_total']} benar di test set")

        row[0].imshow(img_np, interpolation="nearest")
        row[0].set_title(f"Original\n{cls_name}", color="white", fontsize=7.5)

        diagnostics[cls_name] = {
            "sample_index":   info["index"],
            "confidence":     round(info["confidence"], 4),
            "predicted":      class_map[str(info["pred"])],
            "correct":        info["correct"],
            "n_correct":      info["n_correct"],
            "n_total":        info["n_total"],
            "layers":         {},
        }

        col = 1
        for layer_name, gc in cams.items():
            cam, _ = gc.generate(inp, target_class=cls_idx)

            diagnostics[cls_name]["layers"][layer_name] = {
                "cam_shape":  list(gc.last_cam_shape),
                "raw_max":    round(gc.last_raw_max, 8),
                "is_dead":    gc.last_is_dead,
                "model_pred": class_map[str(gc.last_pred)],
            }

            h, w = gc.last_cam_shape
            if gc.last_is_dead:
                # Jangan tampilkan gradasi palsu untuk CAM yang kosong.
                row[col].imshow(np.zeros((img_size, img_size, 3)) + 0.15)
                row[col].text(0.5, 0.5, "CAM KOSONG\n(tidak ada\naktivasi positif)",
                              transform=row[col].transAxes, ha="center", va="center",
                              color="#ff8a80", fontsize=7.5, fontweight="bold")
                overlay = img_np
            else:
                heatmap = colormap(cam)[:, :, :3]
                row[col].imshow(heatmap, interpolation="nearest")
                overlay = (0.55 * img_np + 0.45 * heatmap).clip(0, 1)

            row[col].set_title(f"Grad-CAM {layer_name}\n(peta asli {h}x{w})",
                               color="white", fontsize=7.5)

            row[col + 1].imshow(overlay, interpolation="nearest")
            pred_name  = class_map[str(gc.last_pred)]
            pred_color = "#81c784" if gc.last_pred == cls_idx else "#ff8a80"
            row[col + 1].set_title(f"Overlay {layer_name}\npred: {pred_name}",
                                   color=pred_color, fontsize=7.5)
            col += 2

        for ax in row:
            ax.axis("off")
            ax.set_facecolor("#1a1a1a")

        # Simpan figure per kelas
        fig_s, ax_s = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.2))
        fig_s.patch.set_facecolor("#1a1a1a")
        fig_s.suptitle(f"[{cls_idx}] {cls_name}", color="white", fontweight="bold")
        for j in range(n_cols):
            src = row[j].images
            if src:
                ax_s[j].imshow(src[0].get_array(), interpolation="nearest")
            ax_s[j].set_title(row[j].get_title(), color="white", fontsize=8)
            ax_s[j].axis("off")
        single_path = os.path.join(GRADCAM_DIR, f"class_{cls_idx}_{cls_name}.png")
        plt.savefig(single_path, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
        plt.close(fig_s)

    plt.tight_layout()
    plt.savefig(GRID_PATH, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    log_fn(f"\nGrid Grad-CAM tersimpan -> {GRID_PATH}")

    for gc in cams.values():
        gc.remove()

    # ---- Figure pembanding: sebelum vs sesudah FSCA ----
    _render_fsca_effect(model, test_ds, selected, class_map, n_classes,
                        mean, std, img_size, device, colormap, log_fn)

    with open(DIAGNOSTICS_PATH, "w") as f:
        json.dump(diagnostics, f, indent=2)
    log_fn(f"Diagnostik tersimpan -> {DIAGNOSTICS_PATH}")

    return {"chart_path": GRID_PATH, "effect_path": EFFECT_PATH,
            "gradcam_dir": GRADCAM_DIR, "diagnostics": diagnostics}


def _render_fsca_effect(model, test_ds, selected, class_map, n_classes,
                        mean, std, img_size, device, colormap, log_fn):
    """
    Bandingkan CAM dari layer4 (SEBELUM modul FSCA) dengan fsca4 (SESUDAH).
    Pergeseran fokus antara dua kolom ini adalah bukti visual langsung bahwa
    FSCA mengubah perhatian model — inilah yang menjawab Rumusan Masalah
    poin 3, bukan sekadar heatmap tunggal.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not (hasattr(model, "layer4") and hasattr(model, "fsca4")):
        log_fn("Lewati figure pembanding: model tidak punya layer4/fsca4.")
        return

    gc_before = GradCAM(model, target_layer=model.layer4)
    gc_after  = GradCAM(model, target_layer=model.fsca4)

    fig, axes = plt.subplots(n_classes, 3, figsize=(9.5, n_classes * 2.9))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle("Efek Modul FSCA terhadap Fokus Model\n"
                 "layer4 (sebelum FSCA) vs fsca4 (sesudah FSCA)",
                 color="white", fontsize=12, fontweight="bold", y=1.005)

    for cls_idx in range(n_classes):
        row = axes[cls_idx] if n_classes > 1 else axes
        if cls_idx not in selected:
            for ax in row:
                ax.axis("off")
            continue

        img_tensor, _ = test_ds[selected[cls_idx]["index"]]
        inp    = img_tensor.unsqueeze(0).to(device)
        img_np = _denormalize(img_tensor, mean, std).permute(1, 2, 0).numpy()

        row[0].imshow(img_np, interpolation="nearest")
        row[0].set_title(f"Original\n{class_map[str(cls_idx)]}",
                         color="white", fontsize=7.5)

        for ax, gc, tag in [(row[1], gc_before, "layer4 (sebelum)"),
                            (row[2], gc_after,  "fsca4 (sesudah)")]:
            cam, _ = gc.generate(inp, target_class=cls_idx)
            if gc.last_is_dead:
                ax.imshow(np.zeros((img_size, img_size, 3)) + 0.15)
                ax.text(0.5, 0.5, "CAM KOSONG", transform=ax.transAxes,
                        ha="center", va="center", color="#ff8a80",
                        fontsize=8, fontweight="bold")
            else:
                heat    = colormap(cam)[:, :, :3]
                overlay = (0.55 * img_np + 0.45 * heat).clip(0, 1)
                ax.imshow(overlay, interpolation="nearest")
            ax.set_title(tag, color="white", fontsize=7.5)

        for ax in row:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(EFFECT_PATH, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)

    gc_before.remove()
    gc_after.remove()
    log_fn(f"Figure efek FSCA tersimpan -> {EFFECT_PATH}")


if __name__ == "__main__":
    run(print)