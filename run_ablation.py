"""
run_ablation.py - Menjalankan Step 11 (ablasi multi-seed) dari terminal.

Step 11 tidak ada di GUI karena berjalan berjam-jam. Script ini mencetak log
ke layar SEKALIGUS menyimpannya ke outputs/ablation/ablation_log.txt, jadi log
tidak hilang walau terminal ditutup.

Contoh (dari folder root repo):
    caffeinate -i python run_ablation.py
    caffeinate -i python run_ablation.py --attentions none fsca --seeds 0 1 2 3 4
    caffeinate -i python run_ablation.py --epochs 100

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

from config import PATHS, EXPERIMENT  # noqa: E402  (juga men-set env MPS fallback)


def main():
    ap = argparse.ArgumentParser(description="Ablasi multi-seed modul atensi (Step 11)")
    ap.add_argument("--attentions", nargs="+", default=EXPERIMENT["attentions"],
                    choices=["none", "se", "eca", "cbam", "fsca"])
    ap.add_argument("--seeds", nargs="+", type=int, default=EXPERIMENT["seeds"])
    ap.add_argument("--epochs", type=int, default=EXPERIMENT["epochs"])
    args = ap.parse_args()

    os.makedirs(PATHS["ablation_dir"], exist_ok=True)
    log_path = os.path.join(PATHS["ablation_dir"], "ablation_log.txt")
    log_file = open(log_path, "a", encoding="utf-8")

    def log_fn(msg=""):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log_fn(f"Log disimpan di {log_path}")
    import step11_ablation
    try:
        step11_ablation.run(log_fn, attentions=args.attentions,
                            seeds=args.seeds, epochs=args.epochs)
    finally:
        log_file.close()


if __name__ == "__main__":
    main()