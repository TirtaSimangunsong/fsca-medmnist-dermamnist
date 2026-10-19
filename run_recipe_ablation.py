"""
run_recipe_ablation.py - Menjalankan Step 12 (ablasi resep training) dari terminal.

Log tampil di layar SEKALIGUS disimpan ke outputs/recipe_ablation/recipe_ablation_log.txt.

Contoh (dari folder root repo, sejajar config.py):
    caffeinate -i python run_recipe_ablation.py
    caffeinate -i python run_recipe_ablation.py --variants no_pretrain no_aug
    caffeinate -i python run_recipe_ablation.py --variants official --seeds 0 1 2 3 4

Varian: full, no_pretrain, no_aug, no_wce, adam_step, official
("full" selalu ikut sebagai pembanding; checkpoint-nya dipakai ulang dari Step 11).

`caffeinate -i` (bawaan macOS) mencegah Mac tidur selama script berjalan.
Jika terhenti, jalankan perintah yang sama lagi: run yang sudah selesai
dilewati, run yang terputus diulang dari awal.
"""

import os
import sys
import argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "steps"))

from config import PATHS, RECIPE_ABLATION  # noqa: E402  (juga men-set env MPS fallback)


def main():
    ap = argparse.ArgumentParser(description="Ablasi resep training ResNet-18 (Step 12)")
    ap.add_argument("--variants", nargs="+", default=list(RECIPE_ABLATION["variants"]),
                    choices=list(RECIPE_ABLATION["variants"]))
    ap.add_argument("--seeds", nargs="+", type=int, default=RECIPE_ABLATION["seeds"])
    ap.add_argument("--epochs", type=int, default=RECIPE_ABLATION["epochs"])
    args = ap.parse_args()

    os.makedirs(PATHS["recipe_ablation_dir"], exist_ok=True)
    log_path = os.path.join(PATHS["recipe_ablation_dir"], "recipe_ablation_log.txt")
    log_file = open(log_path, "a", encoding="utf-8")

    def log_fn(msg=""):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log_fn(f"Log disimpan di {log_path}")
    import step12_recipe_ablation
    try:
        step12_recipe_ablation.run(log_fn, variants=args.variants,
                                   seeds=args.seeds, epochs=args.epochs)
    finally:
        log_file.close()


if __name__ == "__main__":
    main()