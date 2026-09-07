"""
Step 4 - Arsitektur Model Atensi (FSCA + Pembanding)
======================================================
Mendefinisikan modul FSCA dan mengintegrasikannya ke ResNet-18, lengkap
dengan modul pembanding (SE, ECA, CBAM) yang dibutuhkan untuk analisis
komparatif di BAB IV.

Struktur modul FSCA:
  Input feature map X
       |
  +----+-------------------------------+
  | Channel Attention (CA)             |  Global Avg + Max Pool -> MLP -> sigmoid
  | Spatial Attention (SA)             |  Channel Avg + Max -> Conv -> sigmoid
  +----+-------------------------------+
       |  concat[X*CA, X*SA] -> Conv1x1 (Feature Selector) -> BN -> + X
       v
  Output: refined feature map

Bedanya dengan CBAM: CBAM menerapkan CA lalu SA secara SEKUENSIAL
(X -> X*CA -> (X*CA)*SA). FSCA menghitung keduanya PARALEL dari X yang sama,
lalu membiarkan konvolusi 1x1 belajar sendiri bagaimana menimbang kedua
cabang tersebut. Inilah novelty yang diklaim di BAB I.

PERUBAHAN dari versi sebelumnya:
  1. build_model() menerima argumen `attention` -> satu file untuk semua
     varian (none/se/eca/cbam/fsca), sehingga ablasi berjalan adil pada
     backbone yang identik.
  2. Inisialisasi conv1 3x3 dari bobot pretrained 7x7 diperbaiki. Versi lama
     memotong bagian tengah (`w[:, :, 2:5, 2:5]`) sehingga membuang sebagian
     besar energi kernel dan menghasilkan aktivasi yang jauh terlalu kecil.
     Sekarang memakai adaptive average pooling dengan koreksi skala agar
     jumlah bobot (dan karenanya skala respons) terjaga.
  3. Zero-init pada BN terakhir modul atensi -> modul mulai sebagai identitas.
     Model karenanya tidak pernah lebih buruk dari baseline di awal training,
     dan gradien tidak terganggu modul acak.
  4. Penempatan modul atensi bisa dipilih lewat MODEL_CONFIG["attention_stages"].

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


import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# BLOK ATENSI DASAR
# =========================================================
class ChannelAttention(nn.Module):
    """Channel attention gaya Squeeze-and-Excitation dengan avg + max pooling."""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x):
        avg = x.mean(dim=[2, 3])
        mx = x.amax(dim=[2, 3])
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return att.unsqueeze(-1).unsqueeze(-1)          # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention dari statistik avg & max lintas kanal."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))   # (B, 1, H, W)


# =========================================================
# VARIAN ATENSI
# =========================================================
class FSCAModule(nn.Module):
    """
    Fused Spatial-Channel Attention (usulan penelitian ini).

    CA dan SA dihitung PARALEL dari input yang sama, di-broadcast ke bentuk
    yang sama, di-concat pada dimensi kanal, lalu disaring konvolusi 1x1
    yang berperan sebagai Feature Selector.
    """

    def __init__(self, in_channels, reduction=8, spatial_kernel=7, zero_init=True):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
        self.fuse = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        if zero_init:
            nn.init.zeros_(self.bn.weight)   # modul mulai sebagai identitas

    def forward(self, x):
        B, C, H, W = x.shape
        ca_feat = x * self.ca(x).expand(-1, -1, H, W)
        sa_feat = x * self.sa(x).expand(-1, C, -1, -1)
        fused = torch.cat([ca_feat, sa_feat], dim=1)       # (B, 2C, H, W)
        out = self.bn(self.fuse(fused))
        return F.relu(out + x)                              # residual


class CBAMModule(nn.Module):
    """CBAM (Woo et al., 2018) - CA lalu SA secara sekuensial."""

    def __init__(self, in_channels, reduction=8, spatial_kernel=7, zero_init=False):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class SEModule(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018) - hanya channel attention."""

    def __init__(self, in_channels, reduction=8, spatial_kernel=7, zero_init=False):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x.mean(dim=[2, 3]))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class ECAModule(nn.Module):
    """Efficient Channel Attention (Wang et al., 2020) - konv 1D tanpa reduksi dimensi."""

    def __init__(self, in_channels, reduction=8, spatial_kernel=7, zero_init=False, gamma=2, b=1):
        super().__init__()
        import math
        t = int(abs((math.log2(in_channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        y = x.mean(dim=[2, 3]).unsqueeze(1)            # (B, 1, C)
        y = torch.sigmoid(self.conv(y)).transpose(1, 2)  # (B, C, 1)
        return x * y.unsqueeze(-1)


ATTENTION_REGISTRY = {
    "fsca": FSCAModule,
    "cbam": CBAMModule,
    "se":   SEModule,
    "eca":  ECAModule,
    "none": None,
}


# =========================================================
# UTIL: INISIALISASI STEM DARI BOBOT PRETRAINED
# =========================================================
def _init_conv1_from_pretrained(new_conv, pretrained_weight, mode="kaiming"):
    """
    Inisialisasi stem 3x3.

    PERINGATAN HASIL EKSPERIMEN
    ----------------------------
    Versi sebelumnya memakai `adaptive_avg_pool2d(w7, 3) * (49/9)`. Faktor 5.44
    itu dimaksudkan mempertahankan jumlah bobot, tetapi hasilnya aktivasi
    conv1 menjadi jauh lebih besar dari yang diharapkan `bn1` pretrained.
    Kombinasi itu berkontribusi pada divergensi NaN. JANGAN dipakai lagi.

    mode:
      "kaiming" (DEFAULT, paling aman) - init acak Kaiming. Stem 3x3 stride 1
          secara struktural memang berbeda dari stem 7x7 stride 2, sehingga
          bobot ImageNet-nya tidak benar-benar transferable. Yang penting
          untuk transfer learning adalah layer1-layer4, dan itu tetap dimuat.
      "pool" - rata-rata 7x7 -> 3x3 TANPA penskalaan.
      "center" - potong bagian tengah 3x3 (perilaku kode Anda yang asli).
    """
    with torch.no_grad():
        if mode == "kaiming":
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        elif mode == "pool":
            new_conv.weight.copy_(F.adaptive_avg_pool2d(pretrained_weight, 3))
        elif mode == "center":
            new_conv.weight.copy_(pretrained_weight[:, :, 2:5, 2:5])
        else:
            raise ValueError(f"mode stem tidak dikenal: {mode}")


# =========================================================
# MODEL UTAMA
# =========================================================
class ResNet18_Attn(nn.Module):
    """ResNet-18 termodifikasi untuk citra 28x28 + modul atensi yang dapat dipilih."""

    def __init__(self, n_classes=7, pretrained=None, attention=None,
                 reduction=None, spatial_kernel=None, stages=None,
                 dropout_p=None, zero_init=None):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        pretrained     = MODEL_CONFIG["pretrained"]          if pretrained     is None else pretrained
        attention      = MODEL_CONFIG["attention"]           if attention      is None else attention
        reduction      = MODEL_CONFIG["fsca_reduction"]      if reduction      is None else reduction
        spatial_kernel = MODEL_CONFIG["fsca_spatial_kernel"] if spatial_kernel is None else spatial_kernel
        stages         = MODEL_CONFIG["attention_stages"]    if stages         is None else stages
        dropout_p      = MODEL_CONFIG["dropout_p"]           if dropout_p      is None else dropout_p
        zero_init      = MODEL_CONFIG["zero_init_attention"] if zero_init      is None else zero_init

        attention = str(attention).lower()
        if attention not in ATTENTION_REGISTRY:
            raise ValueError(f"attention='{attention}' tidak dikenal. "
                             f"Pilihan: {list(ATTENTION_REGISTRY)}")

        self.attention_name = attention
        self.stages = list(stages)

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # --- STEM untuk citra 28x28 ---
        # conv1 bawaan (7x7, stride 2) + maxpool akan menyusutkan 28 -> 7 sebelum
        # blok residual pertama sempat bekerja. Diganti 3x3 stride 1 dan maxpool
        # dibuang, sehingga peta fitur menjadi 28 -> 28 -> 14 -> 7 -> 4.
        new_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        stem_init = MODEL_CONFIG.get("stem_init", "center")
        _init_conv1_from_pretrained(new_conv1, base.conv1.weight, mode=stem_init)

        bn1 = base.bn1

        # CATATAN KEGAGALAN
        # -----------------
        # Versi sebelumnya memanggil bn1.reset_running_stats(). Itu TIDAK
        # memengaruhi gradien di train-mode, jadi bukan penyebab langsung -
        # tetapi seluruh divergensi yang teramati selalu muncul persis di
        # layer0.0.weight dan layer0.1.weight, sehingga stem dikembalikan ke
        # konfigurasi paling konservatif.
        #
        # Backward BatchNorm mengandung 1/sqrt(var + eps). Kalau sebuah kanal
        # keluaran conv1 punya variansi batch mendekati nol, suku itu meledak.
        # eps default 1e-5 memberi batas atas 316; eps 1e-3 menurunkannya ke
        # 31.6, cukup untuk mencegah ledakan tanpa mengubah perilaku normal.
        if MODEL_CONFIG.get("stem_bn_eps"):
            bn1.eps = float(MODEL_CONFIG["stem_bn_eps"])

        # conv1 baru tidak berpasangan dengan gamma/beta ImageNet, yang
        # dikalibrasi untuk stem 7x7 stride 2. Direset ke identitas.
        if stem_init == "kaiming":
            nn.init.ones_(bn1.weight)
            nn.init.zeros_(bn1.bias)

        self.layer0 = nn.Sequential(new_conv1, bn1, base.relu)

        self.layer1 = base.layer1   # 64  ch, 28x28
        self.layer2 = base.layer2   # 128 ch, 14x14
        self.layer3 = base.layer3   # 256 ch, 7x7
        self.layer4 = base.layer4   # 512 ch, 4x4

        # --- Modul atensi per stage ---
        Block = ATTENTION_REGISTRY[attention]
        channels = {1: 64, 2: 128, 3: 256, 4: 512}
        for i in (1, 2, 3, 4):
            if Block is not None and i in self.stages:
                mod = Block(channels[i], reduction=reduction,
                            spatial_kernel=spatial_kernel, zero_init=zero_init)
            else:
                mod = nn.Identity()
            setattr(self, f"fsca{i}", mod)   # nama dipertahankan agar step8 Grad-CAM tetap jalan

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.fsca1(self.layer1(x))
        x = self.fsca2(self.layer2(x))
        x = self.fsca3(self.layer3(x))
        x = self.fsca4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(self.dropout(x))


# Alias agar kode lama yang mengimpor ResNet18_FSCA tetap berjalan
ResNet18_FSCA = ResNet18_Attn


# =========================================================
# ENTRY POINT
# =========================================================
def build_model(n_classes=7, pretrained=None, attention=None, **kwargs):
    """
    Factory function - dipanggil step lain.

    CATATAN PENTING: default `pretrained` sekarang mengikuti MODEL_CONFIG,
    bukan lagi di-hardcode False di pemanggil. Bug lama di step5 membangun
    model dengan bobot acak lalu membekukan backbone-nya, sehingga ekstraktor
    fitur tidak pernah dilatih sama sekali.
    """
    return ResNet18_Attn(n_classes=n_classes, pretrained=pretrained,
                         attention=attention, **kwargs)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    attn = sum(p.numel() for n, p in model.named_parameters() if "fsca" in n)
    return total, attn


def run(log_fn):
    if not os.path.exists(META_PATH):
        log_fn("ERROR: dataset_meta.json tidak ditemukan. Jalankan Step 1.")
        raise FileNotFoundError(META_PATH)

    with open(META_PATH) as f:
        meta = json.load(f)

    n_classes = meta["n_classes"]
    attn_name = MODEL_CONFIG["attention"]

    log_fn(f"Membangun ResNet-18 + {attn_name.upper()} untuk {n_classes} kelas...")
    log_fn(f"  pretrained  : {MODEL_CONFIG['pretrained']}")
    log_fn(f"  stages      : {MODEL_CONFIG['attention_stages']}")
    log_fn(f"  reduction   : {MODEL_CONFIG['fsca_reduction']}")
    log_fn(f"  zero-init   : {MODEL_CONFIG['zero_init_attention']}")

    model = build_model(n_classes=n_classes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attn_params = sum(p.numel() for n, p in model.named_parameters() if "fsca" in n)

    log_fn("")
    log_fn(f"Arsitektur : ResNet-18 + {attn_name.upper()}")
    log_fn(f"Total parameter     : {total_params:,}")
    log_fn(f"Parameter trainable : {trainable_params:,}")
    pct = 100 * attn_params / total_params if total_params else 0.0
    log_fn(f"Parameter atensi    : {attn_params:,}  ({pct:.2f}% dari total)")
    log_fn("")

    log_fn("Detail per stage:")
    for i, ch in zip((1, 2, 3, 4), (64, 128, 256, 512)):
        p = sum(pp.numel() for n, pp in model.named_parameters() if f"fsca{i}" in n)
        status = "aktif" if p else "Identity"
        log_fn(f"  stage{i} ({ch:>3} ch): {p:>7,} params  [{status}]")

    # Verifikasi bentuk peta fitur - bukti kuantitatif untuk BAB III
    log_fn("")
    log_fn("Jejak dimensi peta fitur (input 1x3x28x28):")
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 28, 28)
        x = model.layer0(x);  log_fn(f"  setelah layer0 : {tuple(x.shape)}")
        for i in (1, 2, 3, 4):
            x = getattr(model, f"fsca{i}")(getattr(model, f"layer{i}")(x))
            log_fn(f"  setelah layer{i} : {tuple(x.shape)}")

    # Perbandingan biaya antar-varian atensi (untuk tabel efisiensi BAB IV)
    log_fn("")
    log_fn("Perbandingan parameter antar modul atensi:")
    log_fn(f"  {'Modul':<8} {'Total':>12} {'Atensi':>10} {'Overhead':>10}")
    base_total = None
    variant_table = {}
    for name in ("none", "se", "eca", "cbam", "fsca"):
        m = build_model(n_classes=n_classes, pretrained=False, attention=name)
        t, a = count_params(m)
        if name == "none":
            base_total = t
        over = 100 * (t - base_total) / base_total if base_total else 0.0
        variant_table[name] = {"total": t, "attention": a, "overhead_pct": round(over, 3)}
        log_fn(f"  {name:<8} {t:>12,} {a:>10,} {over:>9.2f}%")

    buf = [
        f"Model          : ResNet-18 + {attn_name.upper()}",
        f"Pretrained     : {MODEL_CONFIG['pretrained']}",
        f"Total params   : {total_params:,}",
        f"Trainable      : {trainable_params:,}",
        f"Attention params: {attn_params:,}",
        f"n_classes      : {n_classes}",
        "",
        "Modules:",
        str(model),
    ]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(buf))
    log_fn(f"\nModel summary tersimpan -> {SUMMARY_PATH}")

    return {
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "fsca_params":      attn_params,
        "attention":        attn_name,
        "n_classes":        n_classes,
        "variant_table":    variant_table,
    }