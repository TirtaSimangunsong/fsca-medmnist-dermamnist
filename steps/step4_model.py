"""
Step 4 - Arsitektur Model: ResNet-18 + Modul Atensi (FSCA / pembanding)
========================================================================
Mendefinisikan modul FSCA (Fused Spatial-Channel Attention) beserta modul
pembanding (SE, ECA, CBAM) dan mengintegrasikannya ke ResNet-18.

Struktur modul FSCA:
  Input feature map X  (B, C, H, W)
       │
       ├── Channel Attention (CA): GAP + GMP → MLP bersama → sigmoid  → (B,C,1,1)
       └── Spatial Attention (SA): mean + max antar-kanal → conv 7x7 → sigmoid → (B,1,H,W)
                (kedua cabang dihitung PARALEL dari X yang sama)
       │
  ca_feat = X * CA        sa_feat = X * SA
       │
  concat([ca_feat, sa_feat], dim=channel)  → (B, 2C, H, W)
       │
  conv 1x1 (feature selector, 2C → C) → BatchNorm
       │
  Output = ReLU(BN(conv1x1(...)) + X)      ← residual connection

Pembanding (dipilih lewat MODEL_CONFIG["attention"] atau argumen attention):
  - "none" : tanpa modul atensi (ResNet-18 polos dengan stem yang sama)
  - "se"   : Squeeze-and-Excitation (Hu et al., 2018)
  - "eca"  : Efficient Channel Attention (Wang et al., 2020)
  - "cbam" : CBAM, CA lalu SA secara SEKUENSIAL (Woo et al., 2018)
  - "fsca" : modul yang diusulkan (default)

Modul atensi dipasang setelah layer1, layer2, layer3, dan layer4.
Nama atribut slotnya tetap fsca1..fsca4 untuk SEMUA pilihan atensi agar:
  (a) checkpoint lama tetap bisa dimuat,
  (b) FINETUNE_HP["unfreeze_keys"] ("fsca3", "fsca4") tetap berlaku,
  (c) hook Grad-CAM ke model.fsca4 tetap valid.

Disimpan ke outputs/model_summary.txt untuk dokumentasi.
"""

import os
import sys
import json
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS, MODEL_CONFIG

OUTPUT_DIR   = PATHS["output_dir"]
META_PATH    = PATHS["dataset_meta"]
SUMMARY_PATH = PATHS["model_summary"]

ATTENTIONS = ("none", "se", "eca", "cbam", "fsca")


# =========================================================
# BLOK PENYUSUN ATENSI
# =========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention gaya CBAM: GAP + GMP → MLP bersama → sigmoid."""
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


# =========================================================
# MODUL FSCA (DIUSULKAN)
# =========================================================
class FSCAModule(nn.Module):
    """
    CA dan SA dihitung paralel, di-broadcast ke shape sama,
    di-concatenate di dimensi channel, lalu disaring conv 1x1 (feature selector),
    dinormalisasi BatchNorm, dan dijumlahkan dengan input (residual).
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
# MODUL PEMBANDING
# =========================================================
class SEModule(nn.Module):
    """Squeeze-and-Excitation: GAP → FC → ReLU → FC → sigmoid → skala kanal."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x):
        w = torch.sigmoid(self.fc(x.mean(dim=[2, 3])))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class ECAModule(nn.Module):
    """Efficient Channel Attention: GAP → conv1d (kernel adaptif) → sigmoid."""
    def __init__(self, in_channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(in_channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        y = x.mean(dim=[2, 3]).unsqueeze(1)          # (B, 1, C)
        w = torch.sigmoid(self.conv(y)).squeeze(1)   # (B, C)
        return x * w.unsqueeze(-1).unsqueeze(-1)


class CBAMModule(nn.Module):
    """CBAM: channel attention lalu spatial attention secara SEKUENSIAL."""
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        return x * self.sa(x)


def make_attention(name, channels):
    r = MODEL_CONFIG["fsca_reduction"]
    k = MODEL_CONFIG["fsca_spatial_kernel"]
    if name == "none":
        return nn.Identity()
    if name == "se":
        return SEModule(channels, r)
    if name == "eca":
        return ECAModule(channels)
    if name == "cbam":
        return CBAMModule(channels, r, k)
    if name == "fsca":
        return FSCAModule(channels, r, k)
    raise ValueError(f"attention='{name}' tidak dikenal. Pilihan: {ATTENTIONS}")


# =========================================================
# MODEL UTAMA: ResNet-18 + modul atensi
# =========================================================
class ResNet18_FSCA(nn.Module):
    def __init__(self, n_classes=7, pretrained=MODEL_CONFIG["pretrained"], attention=None):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        attention = (attention or MODEL_CONFIG.get("attention", "fsca")).lower()
        if attention not in ATTENTIONS:
            raise ValueError(f"attention='{attention}' tidak dikenal. Pilihan: {ATTENTIONS}")
        self.attention_name = attention

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # STEM untuk citra kecil (28x28), disamakan dengan baseline resmi MedMNIST:
        # conv1 7x7/stride2 -> 3x3/stride1, dan maxpool dibuang.
        new_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if pretrained:
            # Inisialisasi dari pretrained conv1: ambil potongan TENGAH 3x3
            # dari kernel 7x7 ImageNet (indeks 2..4).
            with torch.no_grad():
                new_conv1.weight.copy_(base.conv1.weight[:, :, 2:5, 2:5])
        self.layer0 = nn.Sequential(new_conv1, base.bn1, base.relu)  # maxpool DIBUANG

        self.layer1 = base.layer1   # 64  channels
        self.layer2 = base.layer2   # 128 channels
        self.layer3 = base.layer3   # 256 channels
        self.layer4 = base.layer4   # 512 channels

        # Modul atensi setelah tiap stage residual (nama slot tetap fsca1..fsca4)
        self.fsca1 = make_attention(attention, 64)
        self.fsca2 = make_attention(attention, 128)
        self.fsca3 = make_attention(attention, 256)
        self.fsca4 = make_attention(attention, 512)

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
def build_model(n_classes=7, pretrained=MODEL_CONFIG["pretrained"], attention=None):
    """
    Factory function — dipanggil oleh step lain.
    attention=None -> pakai MODEL_CONFIG["attention"] (default "fsca").
    """
    return ResNet18_FSCA(n_classes=n_classes, pretrained=pretrained, attention=attention)


def run(log_fn):
    if not os.path.exists(META_PATH):
        log_fn("ERROR: dataset_meta.json tidak ditemukan. Jalankan Step 1.")
        raise FileNotFoundError(META_PATH)

    with open(META_PATH) as f:
        meta = json.load(f)

    n_classes = meta["n_classes"]
    model = build_model(n_classes=n_classes, pretrained=False)
    attn = model.attention_name.upper()
    log_fn(f"Membangun ResNet-18 + {attn} untuk {n_classes} kelas...")

    # Hitung parameter
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attn_params      = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith("fsca")
    )

    log_fn("")
    log_fn(f"Arsitektur: ResNet-18 + {attn} (4 modul)")
    log_fn(f"Total parameter        : {total_params:,}")
    log_fn(f"Parameter trainable    : {trainable_params:,}")
    log_fn(f"Parameter modul atensi : {attn_params:,}  ({100*attn_params/total_params:.2f}% dari total)")
    log_fn("")

    # Detail per modul atensi
    log_fn(f"Detail Modul {attn}:")
    for i, tag in enumerate(["fsca1", "fsca2", "fsca3", "fsca4"], 1):
        ch = [64, 128, 256, 512][i - 1]
        p  = sum(p.numel() for name, p in model.named_parameters() if name.startswith(tag))
        log_fn(f"  {attn}{i} (layer{i}, {ch:>3} ch): {p:>7,} params")

    # Simpan summary
    import io
    buf = io.StringIO()
    buf.write(f"Model: ResNet-18 + {attn}\n")
    buf.write(f"Total params   : {total_params:,}\n")
    buf.write(f"Trainable      : {trainable_params:,}\n")
    buf.write(f"Attention params: {attn_params:,}\n")
    buf.write(f"n_classes      : {n_classes}\n")
    buf.write(f"\nModules:\n{model}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        f.write(buf.getvalue())

    log_fn(f"Model summary tersimpan → {SUMMARY_PATH}")

    return {
        "attention":        model.attention_name,
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "fsca_params":      attn_params,
        "n_classes":        n_classes,
    }