"""
Step 10 - Inference pada Citra Baru
======================================
Memungkinkan pengguna mengunggah citra dermatoskopi baru
dan mendapatkan prediksi kelas + confidence score + Grad-CAM overlay.

Output: outputs/inference_result.png
"""

import os
import json
import sys

OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "outputs")
META_PATH     = os.path.join(OUTPUT_DIR, "dataset_meta.json")
CONFIG_PATH   = os.path.join(OUTPUT_DIR, "preprocess_config.json")
FINAL_MODEL   = os.path.join(OUTPUT_DIR, "ResNet18_FSCA_DermaMNIST_Best.pth")
RESULT_PATH   = os.path.join(OUTPUT_DIR, "inference_result.png")


def predict_image(image_path, log_fn):
    """
    Melakukan prediksi pada satu file gambar.
    Dipanggil oleh GUI setelah user memilih file.
    """
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm_mod
    import numpy as np

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model
    from step8_gradcam import GradCAM

    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2"), (FINAL_MODEL, "Step 6")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name}.")
            return None

    with open(META_PATH)   as f: meta   = json.load(f)
    with open(CONFIG_PATH) as f: config = json.load(f)

    n_classes = meta["n_classes"]
    class_map = meta["class_map"]
    mean, std = config["mean"], config["std"]

    device = torch.device("cpu")
    model  = build_model(n_classes=n_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(FINAL_MODEL, map_location=device))
    model.eval()

    log_fn(f"Input citra: {os.path.basename(image_path)}")

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        log_fn(f"ERROR membuka file: {e}")
        return None

    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        probs  = F.softmax(output, dim=1).squeeze().cpu().numpy()
        pred   = probs.argmax()

    pred_class = class_map[str(pred)]
    confidence = probs[pred] * 100

    log_fn("")
    log_fn(f"Prediksi    : [{pred}] {pred_class}")
    log_fn(f"Confidence  : {confidence:.2f}%")
    log_fn("")
    log_fn("Top-3 Prediksi:")
    top3 = np.argsort(probs)[::-1][:3]
    for rank, idx in enumerate(top3, 1):
        log_fn(f"  #{rank}  [{idx}] {class_map[str(idx)]:<40} : {probs[idx]*100:.2f}%")

    # Grad-CAM
    grad_cam = GradCAM(model, target_layer=model.layer4)
    cam, _ = grad_cam.generate(img_tensor, target_class=int(pred))

    # Visualisasi
    img_disp = np.array(pil_img.resize((28, 28), Image.NEAREST)) / 255.0
    _cmap    = cm_mod.colormaps["jet"] if hasattr(cm_mod, "colormaps") else plt.get_cmap("jet")
    heatmap  = _cmap(cam)[:, :, :3]
    overlay  = (0.55 * img_disp + 0.45 * heatmap).clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.patch.set_facecolor("#1a1a1a")

    titles = ["Input (resized 28×28)", "Grad-CAM Heatmap", f"Overlay\n{pred_class}\n({confidence:.1f}%)"]
    imgs   = [img_disp, heatmap, overlay]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=9)
        ax.axis("off")
        ax.set_facecolor("#1a1a1a")

    # Bar chart probabilitas
    fig2, ax = plt.subplots(figsize=(7, 3.5))
    fig2.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    cls_names = [class_map[str(i)] for i in range(n_classes)]
    bar_colors = ["#4fc3f7" if i == pred else "#555555" for i in range(n_classes)]
    ax.barh(cls_names, probs * 100, color=bar_colors, edgecolor="#333")
    ax.set_xlabel("Probabilitas (%)", color="white")
    ax.set_title(f"Distribusi Probabilitas\nPrediksi: {pred_class} ({confidence:.1f}%)",
                 color="white", fontweight="bold")
    ax.tick_params(colors="white")
    plt.setp(ax.get_yticklabels(), color="white", fontsize=8)
    for spine in ax.spines.values(): spine.set_color("#555")

    # Gabungkan dua figure
    fig_combined, axes_c = plt.subplots(2, 3, figsize=(13, 7))
    fig_combined.patch.set_facecolor("#1a1a1a")
    fig_combined.suptitle(f"Inference Result — {os.path.basename(image_path)}",
                          color="white", fontsize=11, fontweight="bold")

    for ax in axes_c.flat: ax.axis("off"); ax.set_facecolor("#1a1a1a")

    axes_c[0, 0].imshow(img_disp); axes_c[0, 0].set_title("Input 28×28", color="white", fontsize=9); axes_c[0, 0].axis("off")
    axes_c[0, 1].imshow(heatmap);  axes_c[0, 1].set_title("Grad-CAM",    color="white", fontsize=9); axes_c[0, 1].axis("off")
    axes_c[0, 2].imshow(overlay);  axes_c[0, 2].set_title(f"Overlay\n{pred_class}", color="white", fontsize=9); axes_c[0, 2].axis("off")

    # Prob bar chart di baris bawah (span 3 kolom)
    ax_bar = fig_combined.add_subplot(2, 1, 2)
    ax_bar.set_facecolor("#1a1a1a")
    ax_bar.barh(cls_names, probs * 100, color=bar_colors, edgecolor="#333", height=0.6)
    ax_bar.set_xlabel("Probabilitas (%)", color="white")
    ax_bar.set_title("Distribusi Probabilitas per Kelas", color="white", fontweight="bold")
    ax_bar.tick_params(colors="white")
    plt.setp(ax_bar.get_yticklabels(), color="white", fontsize=8)
    for spine in ax_bar.spines.values(): spine.set_color("#555")
    for i, (name, prob) in enumerate(zip(cls_names, probs)):
        ax_bar.text(prob*100 + 0.3, i, f"{prob*100:.1f}%", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULT_PATH, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close("all")

    log_fn(f"\nHasil inference tersimpan → {RESULT_PATH}")
    return {
        "pred_class": pred_class,
        "pred_idx":   int(pred),
        "confidence": float(confidence),
        "probs":      probs.tolist(),
        "chart_path": RESULT_PATH,
    }


def run(log_fn, image_path=None):
    """
    Entry point dari GUI.
    Jika image_path=None → GUI akan membuka file dialog.
    """
    if image_path is None:
        log_fn("Membuka dialog pilih file citra...")
        import tkinter as tk
        from tkinter import filedialog

        root_tmp = tk.Tk()
        root_tmp.withdraw()
        image_path = filedialog.askopenfilename(
            title="Pilih Citra Dermatoskopi",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        root_tmp.destroy()

        if not image_path:
            log_fn("Tidak ada file dipilih. Inference dibatalkan.")
            return None

    return predict_image(image_path, log_fn)