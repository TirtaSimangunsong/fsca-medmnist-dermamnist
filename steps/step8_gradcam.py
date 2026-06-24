"""
Step 8 - Grad-CAM (Explainable AI)
=====================================
Menghasilkan heatmap Grad-CAM untuk setiap kelas DermaMNIST
menggunakan model final dari Step 6/7.

Proses:
1. Ambil 1 sampel representatif per kelas dari test set
2. Hitung Grad-CAM menggunakan layer4 (feature map terakhir)
3. Overlay heatmap ke citra asli
4. Simpan grid visualisasi semua kelas

Output: outputs/gradcam_grid.png
        outputs/gradcam/class_{name}.png  (per kelas)
"""

import os
import json
import sys
import numpy as np

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
GRADCAM_DIR  = os.path.join(OUTPUT_DIR, "gradcam")
META_PATH    = os.path.join(OUTPUT_DIR, "dataset_meta.json")
CONFIG_PATH  = os.path.join(OUTPUT_DIR, "preprocess_config.json")
FINAL_MODEL  = os.path.join(OUTPUT_DIR, "ResNet18_FSCA_DermaMNIST_Best.pth")
GRID_PATH    = os.path.join(OUTPUT_DIR, "gradcam_grid.png")


# =========================================================
# GRAD-CAM IMPLEMENTATION (no external library)
# =========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        import torch
        import torch.nn.functional as F

        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Grad-CAM: global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam     = F.relu(cam)

        # Normalize ke [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize ke input size (28x28)
        cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy(), target_class


def _denormalize(tensor, mean, std):
    """Kembalikan tensor yang sudah dinormalisasi ke citra asli [0,1]."""
    import torch
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (tensor * s + m).clamp(0, 1)


def run(log_fn):
    import torch
    from torchvision import transforms
    from medmnist import DermaMNIST
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    sys.path.insert(0, os.path.dirname(__file__))
    from step4_model import build_model

    for path, name in [(META_PATH, "Step 1"), (CONFIG_PATH, "Step 2"), (FINAL_MODEL, "Step 6")]:
        if not os.path.exists(path):
            log_fn(f"ERROR: {os.path.basename(path)} tidak ditemukan. Jalankan {name} terlebih dahulu.")
            raise FileNotFoundError(path)

    with open(META_PATH)   as f: meta   = json.load(f)
    with open(CONFIG_PATH) as f: config = json.load(f)

    n_classes   = meta["n_classes"]
    class_map   = meta["class_map"]
    mean, std   = config["mean"], config["std"]

    device = torch.device("cpu")

    log_fn("Memuat model untuk Grad-CAM...")
    model = build_model(n_classes=n_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(FINAL_MODEL, map_location=device))
    model.eval()

    # Pasang Grad-CAM pada layer4 (layer terakhir sebelum avgpool)
    grad_cam = GradCAM(model, target_layer=model.layer4)
    log_fn("Grad-CAM dipasang pada layer4 (feature map 512 channel).")

    # Load test set
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_ds = DermaMNIST(split="test", transform=test_transform, download=True, size=28)

    # Ambil 1 sampel per kelas
    log_fn(f"\nMencari sampel representatif per kelas ({n_classes} kelas)...")
    class_samples = {}
    for i in range(len(test_ds)):
        img_tensor, label = test_ds[i]
        label = label.item()
        if label not in class_samples:
            class_samples[label] = img_tensor
        if len(class_samples) == n_classes:
            break

    os.makedirs(GRADCAM_DIR, exist_ok=True)

    # Generate Grad-CAM per kelas
    fig, axes = plt.subplots(n_classes, 3, figsize=(9, n_classes * 2.8))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle("Grad-CAM Visualization — ResNet-18 + FSCA\n(DermaMNIST 28×28)",
                 color="white", fontsize=13, fontweight="bold", y=1.01)

    colormap = plt.get_cmap("jet")

    for cls_idx in range(n_classes):
        cls_name  = class_map[str(cls_idx)]
        log_fn(f"  Memproses kelas [{cls_idx}] {cls_name}...")

        if cls_idx not in class_samples:
            log_fn(f"    SKIP: tidak ada sampel kelas {cls_idx} di test set.")
            continue

        img_tensor = class_samples[cls_idx].unsqueeze(0).to(device)
        cam, pred_class = grad_cam.generate(img_tensor, target_class=cls_idx)

        # Denormalize untuk visualisasi
        img_vis = _denormalize(class_samples[cls_idx], mean, std)
        img_np  = img_vis.permute(1, 2, 0).numpy()

        # Heatmap
        heatmap_rgba = colormap(cam)[:, :, :3]

        # Overlay: blend citra asli + heatmap
        alpha   = 0.45
        overlay = (1 - alpha) * img_np + alpha * heatmap_rgba
        overlay = overlay.clip(0, 1)

        # Plot 3 kolom: Original | Heatmap | Overlay
        row = axes[cls_idx] if n_classes > 1 else axes
        row[0].imshow(img_np, interpolation="nearest")
        row[0].set_title(f"Original\n{cls_name}", color="white", fontsize=7.5)
        row[1].imshow(heatmap_rgba, interpolation="nearest")
        row[1].set_title("Grad-CAM", color="white", fontsize=7.5)
        row[2].imshow(overlay, interpolation="nearest")
        row[2].set_title(f"Overlay\n(pred: {class_map[str(pred_class)]})", color="white", fontsize=7.5)

        for ax in row:
            ax.axis("off")
            ax.set_facecolor("#1a1a1a")

        # Simpan per kelas
        fig_single, ax_s = plt.subplots(1, 3, figsize=(9, 3))
        fig_single.patch.set_facecolor("#1a1a1a")
        fig_single.suptitle(f"[{cls_idx}] {cls_name}", color="white", fontweight="bold")
        ax_s[0].imshow(img_np); ax_s[0].set_title("Original", color="white", fontsize=9); ax_s[0].axis("off")
        ax_s[1].imshow(heatmap_rgba); ax_s[1].set_title("Grad-CAM", color="white", fontsize=9); ax_s[1].axis("off")
        ax_s[2].imshow(overlay); ax_s[2].set_title("Overlay", color="white", fontsize=9); ax_s[2].axis("off")
        single_path = os.path.join(GRADCAM_DIR, f"class_{cls_idx}_{cls_name}.png")
        plt.savefig(single_path, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
        plt.close(fig_single)

    plt.tight_layout()
    plt.savefig(GRID_PATH, dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)

    log_fn(f"\nGrid Grad-CAM tersimpan → {GRID_PATH}")
    log_fn(f"Per-kelas tersimpan di → {GRADCAM_DIR}/")
    return {"chart_path": GRID_PATH, "gradcam_dir": GRADCAM_DIR}