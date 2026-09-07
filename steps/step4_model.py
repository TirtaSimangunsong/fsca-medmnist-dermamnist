"""
Step 4 - Arsitektur Model FSCA (Fused Spatial-Channel Attention)
=================================================================
Mendefinisikan modul FSCA dan mengintegrasikannya ke ResNet-18.
FSCA menggabungkan Spatial Attention + Channel Attention (CBAM-style)
secara SIMULTAN (bukan sekuensial) — inilah novelty utama paper ini.

Struktur modul:
  Input feature map X
       │
  ┌────┴─────────────────────────┐
  │ Channel Attention (CA)       │  Global Avg + Max Pool → MLP → sigmoid
  │ Spatial Attention (SA)       │  Channel Avg + Max → Conv → sigmoid
  └────┬─────────────────────────┘
       │  Fused: X * CA * SA  (bukan X*CA lalu X*SA)
       ▼
  Output: refined feature map

Disimpan ke outputs/model_summary.txt untuk dokumentasi.
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, MODEL_CONFIG

OUTPUT_DIR   = PATHS["output_dir"]
META_PATH    = PATHS["dataset_meta"]
SUMMARY_PATH = PATHS["model_summary"]


# =========================================================
# MODUL FSCA
# =========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        avg = x.mean(dim=[2, 3])                  # (B, C)
        mx  = x.amax(dim=[2, 3])                  # (B, C)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
        return att.unsqueeze(-1).unsqueeze(-1)     # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention menggunakan channel avg dan max pooling."""
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)         # (B, 1, H, W)
        mx  = x.amax(dim=1, keepdim=True)         # (B, 1, H, W)
        cat = torch.cat([avg, mx], dim=1)          # (B, 2, H, W)
        return torch.sigmoid(self.conv(cat))       # (B, 1, H, W)


class FSCAModule(nn.Module):
    """
    True fused version: CA dan SA dihitung paralel, di-broadcast ke shape sama,
    di-concatenate di dimensi channel, lalu disaring conv 1x1 (feature selector).
    """
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
        self.fuse = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        ca_weight = self.ca(x).expand(-1, -1, H, W)   # (B, C, H, W)
        sa_weight = self.sa(x).expand(-1, C, -1, -1)   # (B, C, H, W)

        ca_feat = x * ca_weight
        sa_feat = x * sa_weight

        fused = torch.cat([ca_feat, sa_feat], dim=1)   # (B, 2C, H, W)
        out = self.fuse(fused)                          # (B, C, H, W) via conv 1x1
        return F.relu(self.bn(out) + x)                 # residual connection


# =========================================================
# MODEL UTAMA: ResNet-18 + FSCA
# =========================================================
class ResNet18_FSCA(nn.Module):
    def __init__(self, n_classes=7, pretrained=MODEL_CONFIG["pretrained"]):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # STEM MODIFICATION untuk citra kecil (28x28)
        # Ganti conv1 7x7/stride2 -> 3x3/stride1, dan buang maxpool
        new_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if pretrained:
            # Inisialisasi dari pretrained conv1 (rata-ratakan kernel 7x7 ke tengah 3x3)
            with torch.no_grad():
                new_conv1.weight.copy_(base.conv1.weight[:, :, 2:5, 2:5])
        self.layer0 = nn.Sequential(new_conv1, base.bn1, base.relu)  # maxpool DIBUANG

        self.layer1 = base.layer1   # 64  channels
        self.layer2 = base.layer2   # 128 channels
        self.layer3 = base.layer3   # 256 channels
        self.layer4 = base.layer4   # 512 channels

        # Pasang FSCA setelah tiap residual block
        self.fsca1 = FSCAModule(64)
        self.fsca2 = FSCAModule(128)
        self.fsca3 = FSCAModule(256)
        self.fsca4 = FSCAModule(512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=MODEL_CONFIG["dropout_p"])
        self.fc      = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.layer0(x)

        x = self.layer1(x)
        x = self.fsca1(x)

        x = self.layer2(x)
        x = self.fsca2(x)

        x = self.layer3(x)
        x = self.fsca3(x)

        x = self.layer4(x)
        x = self.fsca4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# =========================================================
# ENTRY POINT
# =========================================================
def build_model(n_classes=7, pretrained=MODEL_CONFIG["pretrained"]):
    """Factory function — dipanggil oleh step lain."""
    return ResNet18_FSCA(n_classes=n_classes, pretrained=pretrained)


def run(log_fn):
    if not os.path.exists(META_PATH):
        log_fn("ERROR: dataset_meta.json tidak ditemukan. Jalankan Step 1.")
        raise FileNotFoundError(META_PATH)

    with open(META_PATH) as f:
        meta = json.load(f)

    n_classes = meta["n_classes"]
    log_fn(f"Membangun ResNet-18 + FSCA untuk {n_classes} kelas...")

    model = build_model(n_classes=n_classes, pretrained=False)

    # Hitung parameter
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fsca_params     = sum(
        p.numel() for name, p in model.named_parameters()
        if "fsca" in name
    )

    log_fn(f"")
    log_fn(f"Arsitektur: ResNet-18 + FSCA (4 modul)")
    log_fn(f"Total parameter       : {total_params:,}")
    log_fn(f"Parameter trainable   : {trainable_params:,}")
    log_fn(f"Parameter FSCA saja   : {fsca_params:,}  ({100*fsca_params/total_params:.2f}% dari total)")
    log_fn(f"")

    # Detail per modul FSCA
    log_fn("Detail Modul FSCA:")
    for i, tag in enumerate(["fsca1", "fsca2", "fsca3", "fsca4"], 1):
        ch  = [64, 128, 256, 512][i-1]
        p   = sum(p.numel() for name, p in model.named_parameters() if tag in name)
        log_fn(f"  FSCA{i} (layer{i}, {ch:>3} ch): {p:>6,} params")

    # Simpan summary
    import sys, io
    buf = io.StringIO()
    buf.write(f"Model: ResNet-18 + FSCA\n")
    buf.write(f"Total params   : {total_params:,}\n")
    buf.write(f"Trainable      : {trainable_params:,}\n")
    buf.write(f"FSCA params    : {fsca_params:,}\n")
    buf.write(f"n_classes      : {n_classes}\n")
    buf.write(f"\nModules:\n{model}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        f.write(buf.getvalue())

    log_fn(f"Model summary tersimpan → {SUMMARY_PATH}")

    info = {
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "fsca_params":      fsca_params,
        "n_classes":        n_classes,
    }
    return info