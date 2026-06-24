# Sistem CAD Klasifikasi Kanker Kulit — FSCA ResNet-18
**Tirta (01082230021) — Skripsi Tugas Akhir**

---

## Struktur Proyek

```
project/
├── gui.py                    ← Jalankan file ini
├── requirements.txt
├── steps/
│   ├── step1_extract.py      ← Download DermaMNIST
│   ├── step2_preprocess.py   ← Normalisasi & Augmentasi
│   ├── step3_split.py        ← Split data + chart distribusi
│   ├── step4_model.py        ← Arsitektur ResNet-18 + FSCA
│   ├── step5_training.py     ← Training loop (30 epoch)
│   ├── step6_finetune.py     ← Finetuning + simpan model .pth
│   ├── step7_evaluation.py   ← Confusion matrix, F1, AUC
│   ├── step8_gradcam.py      ← Grad-CAM heatmap per kelas
│   ├── step9_summary.py      ← Summary report & perbandingan
│   └── step10_inference.py   ← Prediksi citra baru
└── outputs/                  ← Generated (jangan edit manual)
    ├── dataset_meta.json
    ├── preprocess_config.json
    ├── split_info.json
    ├── training_history.json
    ├── finetune_history.json
    ├── evaluation_report.json
    ├── summary_report.json
    ├── checkpoint_best.pth
    ├── ResNet18_FSCA_DermaMNIST_Best.pth   ← model final
    ├── split_distribution.png
    ├── training_curve.png
    ├── finetune_curve.png
    ├── confusion_matrix.png
    ├── gradcam_grid.png
    ├── gradcam/                             ← per kelas
    └── summary_report.png
```

---

## Instalasi (Lakukan Sekali)

```bash
# 1. Buat virtual environment (opsional tapi dianjurkan)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install semua dependensi
pip install -r requirements.txt
```

---

## Cara Menjalankan

```bash
python gui.py
```

Lalu klik tab Step 1 → Step 2 → ... → Step 10 **secara berurutan**.

### Urutan yang WAJIB diikuti:
| Step | Fungsi | Output Penting |
|------|--------|----------------|
| Step 1 | Download DermaMNIST | dataset_meta.json |
| Step 2 | Hitung mean/std | preprocess_config.json |
| Step 3 | Split data | split_info.json + chart |
| Step 4 | Rakit model FSCA | model_summary.txt |
| Step 5 | Training awal | checkpoint_best.pth |
| Step 6 | Finetuning | ResNet18_FSCA_...Best.pth ← **model final** |
| Step 7 | Evaluasi test set | confusion_matrix.png |
| Step 8 | Grad-CAM (perlu Step 6 & 7) | gradcam_grid.png |
| Step 9 | Summary laporan | summary_report.png |
| Step 10 | Inference citra baru | inference_result.png |

---

## Estimasi Waktu (CPU)

| Step | Estimasi |
|------|----------|
| Step 1–4 | < 5 menit |
| Step 5 (Training 30 epoch) | 15–25 menit |
| Step 6 (Finetune 15 epoch) | 8–12 menit |
| Step 7–10 | < 5 menit |

---

## Troubleshooting

**Error: `medmnist` not found**
```bash
pip install medmnist
```

**Error: `torch` not found**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Chart tidak muncul di GUI**
```bash
pip install Pillow
```

**Step 8 dikunci (tombol tidak bisa diklik)**
→ Jalankan Step 6 dulu sampai selesai, lalu Step 7.

---

## Arsitektur FSCA

```
Input (B, 3, 28, 28)
    ↓
ResNet-18 Layer 0 (Conv + BN + ReLU + MaxPool)
    ↓
Layer 1 (64ch) → FSCA Module 1
    ↓
Layer 2 (128ch) → FSCA Module 2
    ↓
Layer 3 (256ch) → FSCA Module 3
    ↓
Layer 4 (512ch) → FSCA Module 4
    ↓
AdaptiveAvgPool → Dropout(0.4) → FC(512→7)
    ↓
Output (B, 7) — 7 kelas DermaMNIST

FSCA Module:
  X → [Channel Attention (MLP)]  ─┐
  X → [Spatial Attention (Conv)] ─┤ multiply simultan
                                  ↓
                              X * CA * SA
```
