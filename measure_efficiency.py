"""
Pengukuran Efisiensi dan Kompleksitas Komputasi (Laporan Bab IV, Subbab 4.5)
=============================================================================
Mengukur, untuk setiap model yang checkpoint-nya tersedia:
  - Jumlah parameter total dan parameter modul atensi
  - FLOPs per citra (dihitung dari MAC lapisan Conv1d/Conv2d/Linear lewat
    forward hook; FLOPs = 2 x MAC). Operasi elemen-per-elemen (ReLU, sigmoid,
    perkalian atensi, BatchNorm) tidak dihitung, sama seperti konvensi umum.
  - Latensi inferensi per citra (batch 1) dan throughput (batch 256)
    di CPU dan, bila ada, di MPS (Apple Silicon).
  - Ukuran berkas checkpoint (.pth).

Model yang diukur (dilewati bila checkpoint belum ada):
  - ResNet-18 Baseline  : outputs/ResNet18_Baseline_DermaMNIST_Best.pth
  - ResNet-18 + FSCA    : outputs/ResNet18_FSCA_DermaMNIST_Best.pth
  - ResNet-18 + SE/ECA/CBAM : outputs/ablation/<attn>_seed0_ep<E>_auc.pth
    (hasil Step 11; hanya bila ablasi SE/ECA/CBAM sudah dijalankan)

Cara pakai (di folder repo):
    PYTORCH_ENABLE_MPS_FALLBACK=1 python measure_efficiency.py

Output: outputs/efficiency_report.json (+ tabel di terminal)
"""

import os
import sys
import json
import time
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "steps"))

from config import PATHS, DATASET, EXPERIMENT  # noqa: E402

REPORT_PATH = os.path.join(PATHS["output_dir"], "efficiency_report.json")
WARMUP, RUNS = 20, 200          # batch 1
THROUGHPUT_BATCH, THROUGHPUT_RUNS = 256, 20


def _candidates():
    ep = EXPERIMENT["epochs"]
    abl = PATHS["ablation_dir"]
    return [
        ("ResNet-18 Baseline", "none", PATHS["baseline_model"]),
        ("ResNet-18 + SE",     "se",   os.path.join(abl, f"se_seed0_ep{ep}_auc.pth")),
        ("ResNet-18 + ECA",    "eca",  os.path.join(abl, f"eca_seed0_ep{ep}_auc.pth")),
        ("ResNet-18 + CBAM",   "cbam", os.path.join(abl, f"cbam_seed0_ep{ep}_auc.pth")),
        ("ResNet-18 + FSCA",   "fsca", PATHS["final_model"]),
    ]


def count_flops(model, x):
    """MAC dari Conv1d/Conv2d/Linear lewat forward hook, untuk SATU citra."""
    import torch.nn as nn
    macs = {"total": 0, "attention": 0}
    hooks = []

    def make_hook(in_attention):
        def hook(mod, inp, out):
            n = 0
            if isinstance(mod, nn.Conv2d):
                k = mod.kernel_size[0] * mod.kernel_size[1]
                n = out.numel() // out.shape[0] * (mod.in_channels // mod.groups) * k
            elif isinstance(mod, nn.Conv1d):
                n = out.numel() // out.shape[0] * (mod.in_channels // mod.groups) * mod.kernel_size[0]
            elif isinstance(mod, nn.Linear):
                n = out.numel() // out.shape[0] * mod.in_features
            macs["total"] += n
            if in_attention:
                macs["attention"] += n
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            in_att = name.startswith(("fsca1", "fsca2", "fsca3", "fsca4"))
            hooks.append(mod.register_forward_hook(make_hook(in_att)))
    import torch
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return macs


def _sync(device):
    import torch
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def measure_latency(model, device, size):
    import torch
    model = model.to(device).eval()
    x1 = torch.randn(1, 3, size, size, device=device)
    xb = torch.randn(THROUGHPUT_BATCH, 3, size, size, device=device)
    times = []
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x1)
        _sync(device)
        for _ in range(RUNS):
            t0 = time.perf_counter()
            model(x1)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)
        for _ in range(3):
            model(xb)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(THROUGHPUT_RUNS):
            model(xb)
        _sync(device)
        elapsed = time.perf_counter() - t0
    return {
        "latency_ms_mean":   round(statistics.mean(times), 3),
        "latency_ms_median": round(statistics.median(times), 3),
        "latency_ms_std":    round(statistics.stdev(times), 3),
        "throughput_img_per_s": round(THROUGHPUT_BATCH * THROUGHPUT_RUNS / elapsed, 1),
    }


def main():
    import torch
    from step4_model import build_model

    size = DATASET["image_size"]
    with open(PATHS["dataset_meta"]) as f:
        n_classes = json.load(f)["n_classes"]

    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    elif torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    results = []
    for label, attn, ckpt in _candidates():
        if not os.path.exists(ckpt):
            print(f"[lewati] {label}: checkpoint tidak ada ({os.path.relpath(ckpt, BASE_DIR)})")
            continue
        model = build_model(n_classes=n_classes, pretrained=False, attention=attn)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()

        total = sum(p.numel() for p in model.parameters())
        att = sum(p.numel() for n, p in model.named_parameters()
                  if n.startswith(("fsca1", "fsca2", "fsca3", "fsca4")))
        macs = count_flops(model, torch.randn(1, 3, size, size))
        entry = {
            "model": label, "attention": attn,
            "checkpoint": os.path.relpath(ckpt, BASE_DIR),
            "params_total": total, "params_attention": att,
            "macs_total": macs["total"], "macs_attention": macs["attention"],
            "gflops_total": round(2 * macs["total"] / 1e9, 4),
            "file_size_mb": round(os.path.getsize(ckpt) / 1e6, 2),
            "latency": {},
        }
        for dev in devices:
            print(f"Mengukur {label} di {dev.type} ...")
            entry["latency"][dev.type] = measure_latency(model, dev, size)
        results.append(entry)

    if not results:
        print("Tidak ada checkpoint yang bisa diukur. Jalankan Step 4 dan Step 6 dulu.")
        return

    report = {
        "image_size": size, "torch": torch.__version__,
        "devices": [d.type for d in devices],
        "protocol": {"warmup": WARMUP, "runs_batch1": RUNS,
                     "throughput_batch": THROUGHPUT_BATCH, "throughput_runs": THROUGHPUT_RUNS},
        "models": results,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 100)
    print(f"{'Model':<22}{'Parameter':>13}{'Param atensi':>14}{'GFLOPs':>9}{'File (MB)':>11}"
          + "".join(f"{('Latensi ' + d.type + ' (ms)'):>20}" for d in devices))
    for r in results:
        lat = "".join(f"{r['latency'][d.type]['latency_ms_mean']:>11.2f} ± {r['latency'][d.type]['latency_ms_std']:<6.2f}"
                      for d in devices)
        print(f"{r['model']:<22}{r['params_total']:>13,}{r['params_attention']:>14,}"
              f"{r['gflops_total']:>9.4f}{r['file_size_mb']:>11.2f}{lat}")
    print("=" * 100)
    print(f"Laporan tersimpan -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
