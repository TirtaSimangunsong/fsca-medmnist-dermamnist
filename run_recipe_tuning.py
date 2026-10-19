"""
run_recipe_tuning.py - Menjalankan Step 13 (uji resep, hanya data validasi) dari terminal.

Log tampil di layar SEKALIGUS disimpan ke outputs/recipe_tuning/recipe_tuning_log.txt.

Contoh (dari folder root repo, sejajar config.py):
    caffeinate -i python run_recipe_tuning.py
    caffeinate -i python run_recipe_tuning.py --candidates e40_wd1e-4 e40_wd5e-2

Kandidat (RECIPE_TUNING di config.py): e100_wd1e-4 (resep saat ini), e40_wd1e-4, e40_wd5e-2.
Resep saat ini memakai ulang history Step 11 (seed 0-2), jadi tidak dilatih ulang.

`caffeinate -i` (bawaan macOS) mencegah Mac tidur selama script berjalan.
Jika terhenti, jalankan perintah yang sama lagi: run yang sudah selesai dilewati.
"""

import os
import sys
import argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "steps"))

from config import PATHS, RECIPE_TUNING  # noqa: E402  (juga men-set env MPS fallback)


def main():
    ap = argparse.ArgumentParser(description="Uji resep ResNet-18 berbasis validasi (Step 13)")
    ap.add_argument("--candidates", nargs="+", default=list(RECIPE_TUNING["candidates"]),
                    choices=list(RECIPE_TUNING["candidates"]))
    ap.add_argument("--seeds", nargs="+", type=int, default=RECIPE_TUNING["seeds"])
    args = ap.parse_args()

    os.makedirs(PATHS["recipe_tuning_dir"], exist_ok=True)
    log_path = os.path.join(PATHS["recipe_tuning_dir"], "recipe_tuning_log.txt")
    log_file = open(log_path, "a", encoding="utf-8")

    def log_fn(msg=""):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log_fn(f"Log disimpan di {log_path}")
    import step13_recipe_tuning
    try:
        step13_recipe_tuning.run(log_fn, candidates=args.candidates, seeds=args.seeds)
    finally:
        log_file.close()


if __name__ == "__main__":
    main()