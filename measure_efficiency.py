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
  - Rincian parameter dan MAC per slot atensi (fsca1..fsca4) beserta dimensi
    peta fiturnya, untuk Tabel 4.4.

Model yang diukur (kelimanya SELALU diukur):
  - ResNet-18 Baseline  : outputs/models/ResNet18_Baseline_DermaMNIST_Best.pth
  - ResNet-18 + FSCA    : outputs/models/ResNet18_FSCA_DermaMNIST_Best.pth
  - ResNet-18 + SE/ECA/CBAM : outputs/ablation/<attn>_seed0_ep<E>_auc.pth
    (hasil Step 11)
Parameter, FLOPs, dan latensi hanya bergantung pada ARSITEKTUR, bukan pada
nilai bobot. Jadi bila checkpoint suatu model belum ada (mis. ablasi SE/ECA/
CBAM belum dijalankan), model dibangun dengan bobot acak dan tetap diukur.
Ukuran berkas dihitung dari torch.save(state_dict) ke memori, sehingga sama
persis dengan ukuran .pth yang akan dihasilkan training.

Cara pakai (di folder repo):
    PYTORCH_ENABLE_MPS_FALLBACK=1 python measure_efficiency.py

Output: outputs/json/efficiency_report.json (+ tabel di terminal dan baris
        Tabel 4.4 dan Tabel 4.13 dalam format laporan, koma desimal)
"""

import io
import os
import sys
import json
import time
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "steps"))

from config import PATHS, DATASET, EXPERIMENT  # noqa: E402

REPORT_PATH = os.path.join(PATHS["output_dir"], "json", "efficiency_report.json")
WARMUP, RUNS = 20, 200          # batch 1
SLOTS = ("fsca1", "fsca2", "fsca3", "fsca4")
SLOT_AFTER = {"fsca1": "layer1", "fsca2": "layer2", "fsca3": "layer3", "fsca4": "layer4"}
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
    """
    MAC dari Conv1d/Conv2d/Linear lewat forward hook, untuk SATU citra.
    Return {"total", "attention", "slots": {fsca1: MAC, ...}, "shapes": {fsca1: (C, H, W)}}.
    """
    import torch
    import torch.nn as nn
    macs = {"total": 0, "attention": 0, "slots": {s: 0 for s in SLOTS}, "shapes": {}}
    hooks = []

    def make_hook(slot):
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
            if slot:
                macs["attention"] += n
                macs["slots"][slot] += n
        return hook

    def shape_hook(slot):
        def hook(mod, inp, out):
            macs["shapes"][slot] = tuple(int(d) for d in inp[0].shape[1:])
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            slot = name.split(".")[0] if name.split(".")[0] in SLOTS else None
            hooks.append(mod.register_forward_hook(make_hook(slot)))
    for slot in SLOTS:
        if hasattr(model, slot):
            hooks.append(getattr(model, slot).register_forward_hook(shape_hook(slot)))
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


def _fmt(x, nd):
    """Angka format laporan: koma desimal, titik ribuan."""
    return f"{x:,.{nd}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def state_dict_mb(model):
    """Ukuran torch.save(state_dict) dalam MB (sama dengan berkas .pth)."""
    import torch
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


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
        model = build_model(n_classes=n_classes, pretrained=False, attention=attn)
        has_ckpt = os.path.exists(ckpt)
        if has_ckpt:
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        else:
            print(f"[info] {label}: checkpoint belum ada ({os.path.relpath(ckpt, BASE_DIR)}), "
                  f"diukur dari arsitektur dengan bobot acak.")
        model.eval()

        total = sum(p.numel() for p in model.parameters())
        att = sum(p.numel() for n, p in model.named_parameters()
                  if n.split(".")[0] in SLOTS)
        macs = count_flops(model, torch.randn(1, 3, size, size))
        slots = {}
        for slot in SLOTS:
            slots[slot] = {
                "after": SLOT_AFTER[slot],
                "shape_chw": list(macs["shapes"].get(slot, ())),
                "params": sum(p.numel() for n, p in model.named_parameters()
                              if n.split(".")[0] == slot),
                "macs": macs["slots"][slot],
            }
        entry = {
            "model": label, "attention": attn,
            "checkpoint": os.path.relpath(ckpt, BASE_DIR) if has_ckpt else None,
            "weights": "checkpoint" if has_ckpt else "acak (arsitektur saja)",
            "params_total": total, "params_attention": att,
            "macs_total": macs["total"], "macs_attention": macs["attention"],
            "gflops_total": round(2 * macs["total"] / 1e9, 4),
            "slots": slots,
            "file_size_mb": round(os.path.getsize(ckpt) / 1e6, 2) if has_ckpt
                            else round(state_dict_mb(model), 2),
            "latency": {},
        }
        for dev in devices:
            print(f"Mengukur {label} di {dev.type} ...")
            entry["latency"][dev.type] = measure_latency(model, dev, size)
        results.append(entry)

    report = {
        "image_size": size, "torch": torch.__version__,
        "devices": [d.type for d in devices],
        "protocol": {"warmup": WARMUP, "runs_batch1": RUNS,
                     "throughput_batch": THROUGHPUT_BATCH, "throughput_runs": THROUGHPUT_RUNS},
        "models": results,
    }

    print("\n" + "=" * 100)
    print(f"{'Model':<22}{'Parameter':>13}{'Param atensi':>14}{'GFLOPs':>9}{'File (MB)':>11}"
          + "".join(f"{('Latensi ' + d.type + ' (ms)'):>20}" for d in devices))
    for r in results:
        lat = "".join(f"{r['latency'][d.type]['latency_ms_mean']:>11.2f} ± {r['latency'][d.type]['latency_ms_std']:<6.2f}"
                      for d in devices)
        print(f"{r['model']:<22}{r['params_total']:>13,}{r['params_attention']:>14,}"
              f"{r['gflops_total']:>9.4f}{r['file_size_mb']:>11.2f}{lat}")
    print("=" * 100)

    # Baris Tabel 4.13 (format laporan)
    lat_dev = [d for d in ("cpu", "mps", "cuda") if d in report["devices"]]
    print("\nBARIS TABEL 4.13 (salin ke laporan; kolom dipisah ' | ')")
    print("Model | Parameter | GFLOPs per citra | Ukuran berkas (MB) | "
          + " | ".join(f"Latensi {d.upper()} (ms)" for d in lat_dev))
    rows = []
    for r in results:
        cells = [r["model"], _fmt(r["params_total"], 0), _fmt(r["gflops_total"], 4),
                 _fmt(r["file_size_mb"], 1)]
        for d in lat_dev:
            L = r["latency"][d]
            cells.append(f"{_fmt(L['latency_ms_mean'], 2)} ± {_fmt(L['latency_ms_std'], 2)}")
        rows.append(cells)
        print(" | ".join(cells))
    base = next((r for r in results if r["attention"] == "none"), None)
    if base:
        print("\nSelisih terhadap baseline:")
        for r in results:
            if r is base:
                continue
            line = (f"  {r['model']:<18} +{r['params_attention']:,} parameter "
                    f"(+{100 * r['params_attention'] / base['params_total']:.2f}%), "
                    f"+{r['gflops_total'] - base['gflops_total']:.4f} GFLOPs")
            for d in lat_dev:
                b, m = base["latency"][d]["latency_ms_mean"], r["latency"][d]["latency_ms_mean"]
                line += f", latensi {d} {100 * (m - b) / b:+.1f}%"
            print(line)
    report["table_4_13_rows"] = rows

    # Baris Tabel 4.4 (rincian per slot) untuk setiap model beratensi
    report["table_4_4_rows"] = {}
    for r in results:
        if r["attention"] == "none":
            continue
        print(f"\nBARIS TABEL 4.4 ({r['model']}, per slot)")
        print("Posisi | Dimensi peta fitur (C × H × W) | Parameter | MAC per citra")
        rows4 = []
        for slot, d in r["slots"].items():
            dims = " × ".join(str(v) for v in d["shape_chw"])
            cells = [f"{slot} (setelah {d['after']})", dims, _fmt(d["params"], 0), _fmt(d["macs"], 0)]
            rows4.append(cells)
            print(" | ".join(cells))
        tot = ["Total", "", _fmt(r["params_attention"], 0), _fmt(r["macs_attention"], 0)]
        rows4.append(tot)
        print(" | ".join(tot))
        report["table_4_4_rows"][r["attention"]] = rows4
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nLaporan tersimpan -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
