# Sistem CAD Klasifikasi Kanker Kulit — FSCA ResNet-18
**Tirta (01082230021) — Skripsi Tugas Akhir**

Pengembangan *deep learning* berbasis **Fused Spatial-Channel Attention (FSCA)** untuk
klasifikasi kanker kulit citra resolusi rendah (DermaMNIST 28×28, 7 kelas).

- **Model usulan**: ResNet-18 + FSCA (modul atensi kanal dan spasial dihitung paralel,
  digabung dengan konvolusi 1×1, BatchNorm, dan koneksi residual) setelah layer1–layer4.
- **Pembanding**: ResNet-18 tanpa atensi (baseline), serta ResNet-18 + SE, ECA, dan CBAM
  pada slot dan resep pelatihan yang sama.
- Stem ResNet-18 memakai konvolusi 3×3 stride 1 tanpa max pooling, sama dengan baseline
  resmi MedMNIST v2 (bukan kontribusi baru, melainkan penyelarasan agar selisih performa
  bisa diatribusikan ke modul atensi).

## Instalasi

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Di Mac Apple Silicon, `config.py` otomatis menyalakan `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Menjalankan pipeline (GUI)

```bash
python gui.py
```

| Tab GUI | Berkas | Isi |
|---|---|---|
| Step 1: Ekstraksi | `steps/step1_extract.py` | Unduh DermaMNIST, simpan metadata |
| Step 2: Preprocessing | `steps/step2_preprocess.py` | Mean/std piksel untuk normalisasi |
| Step 3: Split Data | `steps/step3_split.py` | Split resmi 7007/1003/2005, bobot kelas |
| Step 4: Baseline ResNet-18 | `steps/step_baseline.py` | Latih + evaluasi ResNet-18 tanpa atensi |
| Step 5: Model FSCA | `steps/step4_model.py` | Bangun model, ringkasan parameter |
| Step 6: Training FSCA | `steps/step5_training.py` | Latih ResNet-18 + FSCA |
| Step 7: Evaluasi & Perbandingan | `steps/step7_evaluation.py` | Test set, bias melanoma, tabel baseline vs FSCA |
| Step 8: Grad-CAM | `steps/step8_gradcam.py` | Heatmap per kelas (fsca3/fsca4) |
| Step 9: Summary | `steps/step9_summary.py` | Ringkasan performa model utama |
| Step 10: Inference | `steps/step10_inference.py` | Prediksi satu citra + Grad-CAM |

Model utama untuk Step 8–10 diatur oleh `MAIN_MODEL` di `config.py`
(`"fsca"` = model usulan; ganti ke `"baseline"` bila diperlukan).

## Eksperimen dari terminal

Ablasi multi-seed (5 seed × 5 model, berjam-jam; run yang sudah selesai dilewati):

```bash
caffeinate -i python run_ablation.py --attentions none se eca cbam fsca
```

Efisiensi (parameter, FLOPs, ukuran berkas, latensi CPU/MPS) untuk kelima model:

```bash
python measure_efficiency.py
```

Kedua perintah mencetak baris tabel dalam format laporan (koma desimal) yang bisa langsung
disalin ke laporan: `run_ablation.py` untuk Tabel 4.6, 4.7, dan 4.8, sedangkan
`measure_efficiency.py` untuk Tabel 4.4 (rincian per slot) dan Tabel 4.13.

Baris "100 epoch" pada Tabel 4.5 (resep lama) dievaluasi ulang dari checkpoint yang sudah ada
dan disimpan terpisah ke `ablation_report_ep100.json`, sehingga laporan resep final tidak tertimpa:

```bash
python run_ablation.py --attentions none fsca --epochs 100
```

Gambar 4.2–4.7 dibuat ulang dari berkas JSON di `outputs/json/` (gaya laporan, koma desimal):

```bash
python make_report_figures.py      # hasil: outputs/report_figures/
```

## Resep pelatihan

| Komponen | Nilai |
|---|---|
| Epoch | 40 (checkpoint dipilih dari AUC validasi) |
| Batch size | 64 |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Penjadwal LR | Cosine annealing, T_max = 40 |
| Loss | Weighted cross-entropy, bobot ∝ 1/√n |
| Inisialisasi | Backbone pretrained ImageNet |
| Bias melanoma | Post-hoc, dipilih di data validasi (grid 0–3) |
| Seed | 42 (run tunggal), 0–4 (multi-seed) |

## Output utama (`outputs/`)

```
outputs/
├── models/          model final (.pth): FSCA dan Baseline
├── json/            metadata, history, laporan evaluasi/ablasi/efisiensi
├── png/             grafik dari tiap step
│   └── gradcam/     Grad-CAM per kelas
├── txt/             model_summary.txt
├── ablation/        checkpoint + history + log per run Step 11 (berpasangan)
└── report_figures/  Gambar 4.2–4.7 versi laporan
```

| Berkas | Dibuat oleh | Dipakai di laporan |
|---|---|---|
| `json/baseline_evaluation_report.json`, `json/evaluation_report.json` | Step 4, Step 7 | Tabel 4.10–4.12, Gambar 4.4–4.5 |
| `json/comparison_report.json` | Step 7 | Tabel perbandingan baseline vs FSCA |
| `json/training_history.json`, `json/baseline_training_history.json` | Step 4, Step 6 | Gambar 4.3 |
| `png/gradcam_grid.png`, `png/gradcam_fsca_effect.png` | Step 8 | Gambar 4.6–4.7 |
| `json/ablation_report.json` | `run_ablation.py` | Tabel 4.5–4.9, Gambar 4.2 |
| `json/ablation_report_ep100.json` | `run_ablation.py --epochs 100` | Tabel 4.5 (baris 100 epoch) |
| `json/efficiency_report.json` | `measure_efficiency.py` | Tabel 4.4, Tabel 4.13 |
| `report_figures/gambar_4_*.png` | `make_report_figures.py` | Gambar 4.2–4.7 |

## Struktur

```
config.py              konfigurasi terpusat (path, dataset, augmentasi, HP, model, imbalance)
gui.py                 antarmuka Tkinter (Step 1–10)
run_ablation.py        Step 11 dari terminal (log ke outputs/ablation/ablation_log.txt)
measure_efficiency.py  parameter, FLOPs, latensi, ukuran berkas
make_report_figures.py gambar Bab IV dari outputs/json/*.json
steps/common.py        dataloader, transformasi, metrik, bias melanoma
steps/step*.py         satu berkas per tahap pipeline
```
