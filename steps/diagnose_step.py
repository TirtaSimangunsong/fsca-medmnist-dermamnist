"""
diagnose_step.py - Pelacak Presisi Kegagalan Training
=======================================================
Menggantikan diagnose.py untuk masalah ini.

diagnose.py menguji uji-overfit yang MELEWATI komponen yang berbeda dari
training sebenarnya (augmentasi, weighted loss + label smoothing, param
group, scheduler), sehingga lolos padahal training gagal.

Skrip ini mereplikasi konfigurasi training PERSIS seperti step5_training.py,
lalu menelusuri batch demi batch dan berhenti pada kejadian non-finite
pertama sambil melaporkan:
  - statistik input (min/maks/NaN)  -> apakah augmentasi yang bermasalah?
  - logits dan loss                  -> apakah forward pass yang meledak?
  - norm gradien PER PARAMETER       -> lapisan mana persisnya
  - perbandingan MPS vs CPU pada batch yang sama

Jalankan:
    python diagnose_step.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps"))

import json
import numpy as np
import torch
import torch.nn as nn

import common
from step4_model import build_model
from config import PATHS, TRAIN_HP, MIXUP

OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def tensor_report(name, t):
    """Ringkasan satu tensor, aman terhadap NaN/inf."""
    t = t.detach().float()
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    finite = t[torch.isfinite(t)]
    if finite.numel():
        lo, hi, mean, std = (finite.min().item(), finite.max().item(),
                             finite.mean().item(), finite.std().item())
    else:
        lo = hi = mean = std = float("nan")
    flag = FAIL if (n_nan or n_inf) else OK
    print(f"  {flag} {name:<14} min={lo:>10.3f} maks={hi:>10.3f} "
          f"mean={mean:>9.3f} std={std:>8.3f} NaN={n_nan} Inf={n_inf}")
    return n_nan == 0 and n_inf == 0


def inspect_gradients(model, top_n=12):
    """Laporkan norm gradien per parameter, yang non-finite lebih dulu."""
    rows = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        n_nan = torch.isnan(g).sum().item()
        n_inf = torch.isinf(g).sum().item()
        norm = g.float().norm(2).item()
        rows.append((name, norm, n_nan, n_inf, p.numel()))

    bad = [r for r in rows if r[2] or r[3] or not np.isfinite(r[1])]

    if bad:
        print(f"\n  {FAIL} {len(bad)} parameter punya gradien non-finite. "
              f"Yang PALING AWAL dalam urutan model adalah sumbernya:\n")
        print(f"    {'parameter':<44} {'norm':>12} {'NaN':>8} {'Inf':>8}")
        for name, norm, n_nan, n_inf, _ in bad[:top_n]:
            print(f"    {name:<44} {norm:>12.3e} {n_nan:>8} {n_inf:>8}")
        print(f"\n  >>> PARAMETER PERTAMA YANG RUSAK: {bad[0][0]}")
    else:
        finite = sorted(rows, key=lambda r: -r[1])
        print(f"\n  {OK} Semua gradien finite. Norm terbesar:\n")
        print(f"    {'parameter':<44} {'norm':>12}")
        for name, norm, _, _, _ in finite[:top_n]:
            print(f"    {name:<44} {norm:>12.3e}")
        total = float(np.sqrt(sum(r[1] ** 2 for r in rows)))
        print(f"\n    total norm (hitung manual) = {total:.4e}")
    return bad


def build_exact_training_setup(device, use_augmentation=True, use_weights=True,
                               use_label_smoothing=True, use_param_groups=True,
                               use_mixup=None):
    """Bangun setup PERSIS seperti step5_training.train_one_run()."""
    with open(PATHS["dataset_meta"]) as f:
        meta = json.load(f)
    with open(PATHS["preprocess_config"]) as f:
        pre = json.load(f)

    n_classes = meta["n_classes"]
    hp = TRAIN_HP
    common.set_seed(hp["seed"])

    strength = 1.0 if use_augmentation else 0.0
    train_tf = common.build_transforms(pre["mean"], pre["std"],
                                       train=use_augmentation, strength=strength)

    from medmnist import DermaMNIST
    from torch.utils.data import DataLoader
    ds = DermaMNIST(split="train", transform=train_tf, download=True, size=28)
    counts, labels = common.get_class_counts(ds, n_classes)

    g = torch.Generator(); g.manual_seed(hp["seed"])
    loader = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True,
                        drop_last=True, generator=g, num_workers=0)

    model = build_model(n_classes=n_classes).to(device)

    w = None
    if use_weights:
        lw, _ = common.compute_loss_weights(counts, "sqrt_weighted_loss")
        w = torch.tensor(lw, dtype=torch.float, device=device)

    crit = nn.CrossEntropyLoss(
        weight=w,
        label_smoothing=hp["label_smoothing"] if use_label_smoothing else 0.0,
    )

    if use_param_groups:
        opt = common.build_optimizer(model, lr=hp["lr"],
                                     weight_decay=hp["weight_decay"],
                                     backbone_lr_mult=hp["backbone_lr_mult"],
                                     log_fn=lambda *_: None)
        sched = common.WarmupCosine(opt, total_epochs=hp["epochs"],
                                    warmup_epochs=hp["warmup_epochs"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                                weight_decay=hp["weight_decay"])
        sched = None

    mix_on = MIXUP["enabled"] if use_mixup is None else use_mixup
    mixer = common.MixupCutmix(MIXUP["mixup_alpha"], MIXUP["cutmix_alpha"],
                               MIXUP["prob"], MIXUP["switch_prob"]) if mix_on else None

    return model, loader, crit, opt, sched, mixer


# =========================================================
# A. Replikasi persis, telusuri batch demi batch
# =========================================================
def trace_real_training(device_str=None, max_batches=60, **flags):
    label = ", ".join(f"{k}={v}" for k, v in flags.items()) or "konfigurasi penuh"
    print("\n" + "=" * 76)
    print(f"  TELUSUR TRAINING: {label}")
    print("=" * 76)

    device = torch.device(device_str) if device_str else common.resolve_device(log_fn=lambda *_: None)
    print(f"Device: {device}")

    model, loader, crit, opt, sched, mixer = build_exact_training_setup(device, **flags)
    model.train()

    worst_norm = 0.0
    for i, (imgs, labels) in enumerate(loader):
        if i >= max_batches:
            break

        imgs = imgs.to(device)
        labels = labels.squeeze(1).long().to(device)

        # --- cek input ---
        if not torch.isfinite(imgs).all():
            print(f"\n{FAIL} Batch {i}: INPUT sudah mengandung non-finite.")
            tensor_report("imgs", imgs)
            print("      -> Penyebabnya pipeline augmentasi, bukan model.")
            return False, i

        if mixer is not None:
            imgs_m, y_a, y_b, lam = mixer(imgs, labels)
            out = model(imgs_m)
            loss = common.mixup_criterion(crit, out, y_a, y_b, lam)
        else:
            out = model(imgs)
            loss = crit(out, labels)

        # --- cek forward ---
        if not torch.isfinite(out).all() or not torch.isfinite(loss):
            print(f"\n{FAIL} Batch {i}: FORWARD PASS menghasilkan non-finite.")
            tensor_report("imgs", imgs)
            tensor_report("logits", out)
            print(f"       loss = {loss.item()}")
            return False, i

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # --- cek gradien ---
        bad = [(n, p) for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            print(f"\n{FAIL} Batch {i}: GRADIEN non-finite.")
            print(f"       loss = {loss.item():.6f} (finite)")
            tensor_report("imgs", imgs)
            tensor_report("logits", out)
            inspect_gradients(model)
            return False, i

        # norm total, dihitung manual di CPU (hindari bug foreach di MPS)
        total = torch.stack([
            p.grad.detach().float().norm(2).cpu() for p in model.parameters()
            if p.grad is not None
        ]).norm(2).item()

        # bandingkan dengan clip_grad_norm_ bawaan
        builtin = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()

        if not np.isfinite(builtin) and np.isfinite(total):
            print(f"\n{FAIL} Batch {i}: gradien SEHAT (norm manual = {total:.4f}) "
                  f"tapi clip_grad_norm_ bawaan mengembalikan {builtin}.")
            print("       -> Ini BUG clip_grad_norm_ di MPS, bukan divergensi model.")
            print("       -> Perbaikannya: pakai common.safe_clip_grad_norm().")
            return False, i

        if not np.isfinite(total):
            print(f"\n{FAIL} Batch {i}: norm gradien total non-finite ({total}).")
            inspect_gradients(model)
            return False, i

        worst_norm = max(worst_norm, total)
        opt.step()

        if i % 10 == 0:
            print(f"  batch {i:>3}  loss {loss.item():.4f}  "
                  f"norm(manual) {total:>9.3f}  norm(bawaan) {builtin:>9.3f}")

    if sched:
        sched.step()
    print(f"\n{OK} {max_batches} batch selesai tanpa non-finite. "
          f"Norm gradien terbesar: {worst_norm:.3f}")
    return True, None


# =========================================================
# B. Isolasi: matikan komponen satu per satu
# =========================================================
def isolate():
    print("\n" + "=" * 76)
    print("  ISOLASI KOMPONEN (mundur dari konfigurasi penuh)")
    print("=" * 76)
    print("Tiap baris mematikan SATU komponen. Yang pertama LOLOS")
    print("menunjukkan komponen yang dimatikan itulah penyebabnya.\n")

    variants = [
        ("konfigurasi penuh (seperti step5)", {}),
        ("tanpa augmentasi",                  {"use_augmentation": False}),
        ("tanpa bobot kelas",                 {"use_weights": False}),
        ("tanpa label smoothing",             {"use_label_smoothing": False}),
        ("tanpa param group (lr seragam)",    {"use_param_groups": False}),
        ("tanpa mixup",                       {"use_mixup": False}),
        ("polos total",                       {"use_augmentation": False,
                                               "use_weights": False,
                                               "use_label_smoothing": False,
                                               "use_param_groups": False,
                                               "use_mixup": False}),
    ]

    out = {}
    for label, flags in variants:
        try:
            passed, batch = trace_real_training(max_batches=40, **flags)
            out[label] = (passed, batch)
        except Exception as e:
            import traceback
            traceback.print_exc()
            out[label] = (False, "exception")

    print("\n" + "=" * 76)
    print("  RINGKASAN ISOLASI")
    print("=" * 76)
    for label, (passed, batch) in out.items():
        detail = "lolos" if passed else f"gagal di batch {batch}"
        print(f"  {OK if passed else FAIL}  {label:<38} {detail}")

    lulus = [l for l, (p, _) in out.items() if p]
    gagal = [l for l, (p, _) in out.items() if not p]
    if gagal and lulus:
        print(f"\n  Komponen tersangka: bandingkan '{gagal[0]}' (gagal) "
              f"dengan '{lulus[0]}' (lolos).")
    return out


# =========================================================
# C. Kontrol CPU pada konfigurasi penuh
# =========================================================
def cpu_control():
    print("\n" + "=" * 76)
    print("  KONTROL CPU (konfigurasi penuh, 20 batch — sabar, ini lambat)")
    print("=" * 76)
    return trace_real_training(device_str="cpu", max_batches=20)


if __name__ == "__main__":
    print("DIAGNOSTIK PRESISI - PIPELINE FSCA-DERMAMNIST")
    print(f"PyTorch {torch.__version__}")

    for path, step in [(PATHS["dataset_meta"], "Step 1"),
                       (PATHS["preprocess_config"], "Step 2")]:
        if not os.path.exists(path):
            print(f"{FAIL} {os.path.basename(path)} tidak ada. Jalankan {step} dulu.")
            sys.exit(1)

    ok, _ = trace_real_training(max_batches=60)

    if not ok:
        isolate()
        print("\nMenjalankan kontrol CPU untuk memastikan ini bukan masalah backend...")
        cpu_ok, _ = cpu_control()
        if cpu_ok:
            print(f"\n{FAIL} CPU LOLOS tapi MPS GAGAL pada konfigurasi yang sama.")
            print("      -> Set DEVICE_CONFIG['device'] = 'cpu' di config.py")
            print("         sebagai solusi sementara, dan laporkan bug ke PyTorch.")
    else:
        print(f"\n{OK} Konfigurasi training berjalan bersih selama 60 batch.")
        print("      Kalau step5 masih gagal, kirimkan output ini ke saya.")

    print("\nKirimkan SELURUH output di atas.")