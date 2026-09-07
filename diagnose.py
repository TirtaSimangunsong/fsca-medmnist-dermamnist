"""
diagnose.py - Pelacak Penyebab Divergensi NaN
================================================
Jalankan SEBELUM training panjang berikutnya:

    python diagnose.py

Skrip ini menjalankan enam pemeriksaan berurutan, masing-masing beberapa
detik sampai beberapa menit. Setiap pemeriksaan mengisolasi satu tersangka.
Pemeriksaan pertama yang GAGAL adalah penyebabnya.

Letakkan file ini di root proyek (sejajar dengan config.py).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps"))

import numpy as np
import torch
import torch.nn as nn

import common
from config import DATASET, MODEL_CONFIG

OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"
results = {}


def hr(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# =========================================================
# 1. Sanity MPS: apakah backend memberi hasil yang sama dengan CPU?
# =========================================================
def check_mps_correctness():
    hr("1. Kebenaran numerik backend MPS")

    if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        print(f"{OK} MPS tidak tersedia, pemeriksaan dilewati.")
        return True

    torch.manual_seed(0)
    x = torch.randn(8, 64, 14, 14)
    problems = []

    ops = {
        "amax(dim=[2,3])":        lambda t: t.amax(dim=[2, 3]),
        "amax(dim=1,keepdim)":    lambda t: t.amax(dim=1, keepdim=True),
        "mean(dim=[2,3])":        lambda t: t.mean(dim=[2, 3]),
        "rot90(dims=(2,3))":      lambda t: torch.rot90(t, 1, dims=(2, 3)),
        "flip(dims=(3,))":        lambda t: torch.flip(t, dims=(3,)),
        "softmax(dim=1)":         lambda t: torch.softmax(t.flatten(1), dim=1),
    }

    for name, fn in ops.items():
        c = fn(x)
        m = fn(x.to("mps")).cpu()
        diff = (c - m).abs().max().item()
        status = OK if diff < 1e-4 else FAIL
        print(f"{status} {name:<24} selisih maks vs CPU: {diff:.3e}")
        if diff >= 1e-4:
            problems.append(name)

    # BatchNorm sering jadi sumber masalah di MPS
    bn = nn.BatchNorm2d(64)
    bn.train()
    c = bn(x)
    bn_m = nn.BatchNorm2d(64).to("mps")
    bn_m.load_state_dict(bn.state_dict())
    bn_m.train()
    m = bn_m(x.to("mps")).cpu()
    diff = (c - m).abs().max().item()
    status = OK if diff < 1e-3 else FAIL
    print(f"{status} {'BatchNorm2d (train)':<24} selisih maks vs CPU: {diff:.3e}")
    if diff >= 1e-3:
        problems.append("BatchNorm2d")

    if problems:
        print(f"\n{FAIL} MPS memberi hasil berbeda pada: {problems}")
        print("      -> Set DEVICE_CONFIG['device'] = 'cpu' di config.py,")
        print("         atau upgrade PyTorch: pip install --upgrade torch torchvision")
        return False

    print(f"\n{OK} MPS konsisten dengan CPU.")
    return True


# =========================================================
# 2. Magnitudo forward pass: apakah stem meledak?
# =========================================================
def check_forward_magnitudes():
    hr("2. Magnitudo aktivasi forward pass")

    from step4_model import build_model

    model = build_model(n_classes=7).eval()
    x = torch.randn(16, 3, 28, 28)

    acts = {}
    with torch.no_grad():
        h = model.layer0(x); acts["layer0"] = h
        for i in (1, 2, 3, 4):
            h = getattr(model, f"layer{i}")(h)
            acts[f"layer{i}"] = h
            h = getattr(model, f"fsca{i}")(h)
            acts[f"fsca{i}"] = h
        pooled = model.avgpool(h).flatten(1)
        logits = model.fc(pooled)

    bad = []
    for name, t in acts.items():
        std, mx = t.std().item(), t.abs().max().item()
        status = OK
        if not np.isfinite(std) or std > 20 or mx > 200:
            status, _ = FAIL, bad.append(name)
        print(f"{status} {name:<8} std={std:>8.3f}  |maks|={mx:>9.3f}  bentuk={tuple(t.shape)}")

    lstd, lmax = logits.std().item(), logits.abs().max().item()
    status = OK if (np.isfinite(lstd) and lmax < 50) else FAIL
    print(f"{status} {'logits':<8} std={lstd:>8.3f}  |maks|={lmax:>9.3f}")

    if bad or status == FAIL:
        print(f"\n{FAIL} Aktivasi meledak. Tersangka utama: skala init conv1 (49/9).")
        return False

    print(f"\n{OK} Magnitudo aktivasi wajar.")
    return True


# =========================================================
# 3. Uji overfit: bisakah model menghafal 2 batch?
# =========================================================
def check_overfit(device_str=None, use_mixup=False, weight_decay=1e-2, steps=300):
    hr(f"3. Uji overfit 2 batch (mixup={use_mixup}, wd={weight_decay})")
    print("Model SEHAT harus mencapai ~100% akurasi dan loss mendekati nol.")
    print("Kalau gagal di sini, tidak ada gunanya melatih 150 epoch.\n")

    from step4_model import build_model
    from medmnist import DermaMNIST
    from torch.utils.data import DataLoader

    device = torch.device(device_str) if device_str else common.resolve_device(log_fn=lambda *_: None)
    print(f"Device: {device}")

    mean = [0.7631, 0.5381, 0.5614]
    std = [0.1366, 0.1543, 0.1692]
    tf = common.build_transforms(mean, std, train=False)   # tanpa augmentasi
    ds = DermaMNIST(split="train", transform=tf, download=True, size=DATASET["image_size"])

    loader = DataLoader(ds, batch_size=64, shuffle=False)
    batches = []
    for i, b in enumerate(loader):
        batches.append((b[0].to(device), b[1].squeeze(1).long().to(device)))
        if i >= 1:
            break

    common.set_seed(0)
    model = build_model(n_classes=7).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    mixer = common.MixupCutmix() if use_mixup else None

    first_nan = None
    for step in range(steps):
        model.train()
        for imgs, labels in batches:
            if mixer:
                im, ya, yb, lam = mixer(imgs, labels)
                out = model(im)
                loss = common.mixup_criterion(crit, out, ya, yb, lam)
            else:
                out = model(imgs)
                loss = crit(out, labels)

            if not torch.isfinite(loss) and first_nan is None:
                first_nan = step
                print(f"{FAIL} Loss menjadi non-finite pada step {step}.")
                return False

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
            if not np.isfinite(gn):
                print(f"{FAIL} Norm gradien non-finite pada step {step}.")
                return False
            opt.step()

        if step % 50 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                correct = total = 0
                for imgs, labels in batches:
                    correct += (model(imgs).argmax(1) == labels).sum().item()
                    total += labels.numel()
            print(f"  step {step:>3}  loss {loss.item():.4f}  "
                  f"grad_norm {gn:>8.2f}  acc(eval) {100*correct/total:.1f}%")

    ok = correct / total > 0.95
    print(f"\n{OK if ok else FAIL} Akurasi overfit akhir: {100*correct/total:.1f}%")
    if not ok:
        print("      Model tidak bisa menghafal 128 citra -> ada yang rusak secara")
        print("      struktural, bukan sekadar hyperparameter.")
    return ok


# =========================================================
# 4. Divergensi train-mode vs eval-mode (statistik BatchNorm)
# =========================================================
def check_train_eval_gap():
    hr("4. Selisih train-mode vs eval-mode (statistik BatchNorm)")
    print("Log Anda menunjukkan train_loss 0.40 vs val_loss 2.25 sejak epoch 1.")
    print("Kalau selisihnya besar di sini, penyebabnya running stats BatchNorm.\n")

    from step4_model import build_model
    from medmnist import DermaMNIST
    from torch.utils.data import DataLoader

    device = common.resolve_device(log_fn=lambda *_: None)
    mean = [0.7631, 0.5381, 0.5614]
    std = [0.1366, 0.1543, 0.1692]
    tf = common.build_transforms(mean, std, train=False)
    ds = DermaMNIST(split="train", transform=tf, download=True, size=DATASET["image_size"])
    loader = DataLoader(ds, batch_size=128, shuffle=True)

    common.set_seed(0)
    model = build_model(n_classes=7).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()

    for i, (imgs, labels) in enumerate(loader):
        if i >= 30:
            break
        imgs, labels = imgs.to(device), labels.squeeze(1).long().to(device)
        loss = crit(model(imgs), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(device), labels.squeeze(1).long().to(device)

    model.train()
    with torch.no_grad():
        l_train = crit(model(imgs), labels).item()
    model.eval()
    with torch.no_grad():
        l_eval = crit(model(imgs), labels).item()

    gap = abs(l_eval - l_train)
    status = OK if gap < 0.5 else FAIL
    print(f"{status} loss train-mode: {l_train:.4f}   eval-mode: {l_eval:.4f}   selisih: {gap:.4f}")

    print("\nStatistik running BatchNorm:")
    worst = 0.0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            rv = mod.running_var.max().item()
            rm = mod.running_mean.abs().max().item()
            worst = max(worst, rv)
            if rv > 100 or rm > 50 or not np.isfinite(rv):
                print(f"{FAIL} {name:<28} running_var maks={rv:>12.2f}  |mean| maks={rm:>10.2f}")
    print(f"     running_var terbesar di seluruh model: {worst:.2f}")
    if worst > 100:
        print(f"{FAIL} Statistik BatchNorm meledak -> hampir pasti init conv1.")
        return False

    return gap < 0.5


# =========================================================
# 5. Isolasi komponen: mana yang memicu NaN?
# =========================================================
def check_components():
    hr("5. Isolasi komponen satu per satu")
    print("Menjalankan uji overfit pendek dengan komponen dinyalakan bertahap.\n")

    configs = [
        ("baseline (tanpa mixup, wd=1e-2)", dict(use_mixup=False, weight_decay=1e-2)),
        ("+ weight_decay 5e-2",             dict(use_mixup=False, weight_decay=5e-2)),
        ("+ mixup",                         dict(use_mixup=True,  weight_decay=1e-2)),
        ("+ mixup + wd 5e-2",               dict(use_mixup=True,  weight_decay=5e-2)),
    ]
    out = {}
    for label, kw in configs:
        print(f"\n--- {label} ---")
        try:
            out[label] = check_overfit(steps=120, **kw)
        except Exception as e:
            print(f"{FAIL} exception: {e}")
            out[label] = False

    print("\nRingkasan:")
    for label, passed in out.items():
        print(f"  {OK if passed else FAIL} {label}")
    return all(out.values())


# =========================================================
# 6. CPU sebagai kontrol
# =========================================================
def check_cpu_control():
    hr("6. Kontrol: uji yang sama di CPU")
    print("Kalau CPU lolos tapi MPS gagal, penyebabnya backend, bukan kode Anda.\n")
    return check_overfit(device_str="cpu", steps=120, weight_decay=1e-2)


if __name__ == "__main__":
    print("DIAGNOSTIK PIPELINE FSCA-DERMAMNIST")
    print(f"PyTorch {torch.__version__}")

    checks = [
        ("MPS correctness",  check_mps_correctness),
        ("Forward magnitude", check_forward_magnitudes),
        ("Overfit test",     lambda: check_overfit(steps=300)),
        ("Train/eval gap",   check_train_eval_gap),
        ("Component isolation", check_components),
        ("CPU control",      check_cpu_control),
    ]

    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            print(f"\n{FAIL} {name} melempar exception:")
            traceback.print_exc()
            results[name] = False

    hr("RINGKASAN")
    for name, passed in results.items():
        print(f"  {OK if passed else FAIL}  {name}")

    failed = [n for n, p in results.items() if not p]
    if failed:
        print(f"\nPemeriksaan pertama yang gagal: {failed[0]}")
        print("Itulah yang perlu diperbaiki lebih dulu. Kirimkan output ini ke saya.")
    else:
        print("\nSemua lolos. Lanjutkan ke training bertahap (lihat PERBAIKAN.md).")