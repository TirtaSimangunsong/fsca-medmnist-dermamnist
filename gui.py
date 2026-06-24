"""
gui.py — Sistem CAD Klasifikasi Kanker Kulit
=============================================
GUI terintegrasi penuh dengan 10 Step ML pipeline FSCA.
Author : Tirta (01082230021)
"""

import tkinter as tk
from tkinter import messagebox
import threading
import time
import os
import sys

# Tambahkan folder steps ke path
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
STEPS_DIR = os.path.join(BASE_DIR, "steps")
sys.path.insert(0, STEPS_DIR)

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ====================================================================
# CUSTOM COMPONENT: MAC-COMPATIBLE NAVIGATION & BUTTON TAB
# ====================================================================
class MacFriendlyTab(tk.Label):
    def __init__(self, master, text, bg, fg, font, command=None, is_active=False, **kwargs):
        super().__init__(master, text=text, bg=bg, fg=fg, font=font,
                         relief="flat", cursor="hand2", bd=0, **kwargs)
        self.command    = command
        self.is_active  = is_active
        self.state      = "normal"
        self.default_bg = bg
        self.default_fg = fg
        self.bind("<Enter>",    self._on_enter)
        self.bind("<Leave>",    self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _on_enter(self, event):
        if self.state == "normal":
            if self.is_active:               self.config(bg="#1a62d6")
            elif self.default_bg == "#333333": self.config(bg="#444444")
            elif self.default_bg == "#3b5998": self.config(bg="#4a6ea8")
            elif self.default_bg == "#444444": self.config(bg="#555555")

    def _on_leave(self, event):
        if self.state == "normal":
            self.config(bg="#0f52ba" if self.is_active else self.default_bg)

    def _on_click(self, event):
        if self.state == "normal" and self.command:
            self.command()

    def set_active(self, active):
        self.is_active = active
        if active:
            self.config(bg="#0f52ba", fg="white", font=("Arial", 9, "bold"))
            self.default_bg = "#0f52ba"
        else:
            self.config(bg="#333333", fg="#aaaaaa", font=("Arial", 8))
            self.default_bg = "#333333"

    def set_state(self, state, text=None):
        self.state = state
        if state == "disabled":
            self.config(bg="#555555", fg="#888888", cursor="arrow")
        else:
            self.config(bg=self.default_bg, fg=self.default_fg, cursor="hand2")
        if text:
            self.config(text=text)


# ====================================================================
# ANTARMUKA GUI UTAMA
# ====================================================================
class CancerSkinCADApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem CAD Klasifikasi Kanker Kulit (MedMNIST-FSCA) — Tirta 01082230021")
        self.root.geometry("1200x720")
        self.root.configure(bg="#2d2d2d")

        # State
        self.step_6_7_done  = False
        self.nav_tabs       = {}
        self.current_step   = ""
        self.chart_images   = {}   # simpan referensi PhotoImage agar tidak di-GC
        self.last_result    = {}   # hasil return dari setiap step

        self.create_top_navigation()
        self.create_main_content()
        self.create_terminal_output()
        self.create_execute_button()
        self.switch_step("Step 1: Ekstraksi")

        # Cek dependensi PIL sekali
        if not PIL_AVAILABLE:
            self.write_terminal("PERINGATAN: Pillow tidak terinstall. Chart tidak akan tampil di GUI.")
            self.write_terminal("Jalankan: pip install Pillow")

    # ------------------------------------------------------------------
    # NAVIGATION
    # ------------------------------------------------------------------
    def create_top_navigation(self):
        nav_frame = tk.Frame(self.root, bg="#1e1e1e", height=45)
        nav_frame.pack(fill="x", side="top")

        steps = [
            "Step 1: Ekstraksi",   "Step 2: Preprocessing", "Step 3: Split Data",
            "Step 4: Model FSCA",  "Step 5: Training",       "Step 6: Finetune",
            "Step 7: Evaluation",  "Step 8: Grad-CAM",       "Step 9: Summary",
            "Step 10: Inference",
        ]
        for step in steps:
            tab = MacFriendlyTab(
                nav_frame, text=step, bg="#333333", fg="#aaaaaa", font=("Arial", 8),
                padx=10, pady=10, is_active=False,
                command=lambda s=step: self.switch_step(s)
            )
            tab.pack(side="left", fill="y", padx=1, pady=2)
            self.nav_tabs[step] = tab

    # ------------------------------------------------------------------
    # MAIN CONTENT AREA
    # ------------------------------------------------------------------
    def create_main_content(self):
        self.main_frame = tk.Frame(self.root, bg="#2d2d2d", padx=20, pady=10)
        self.main_frame.pack(fill="both", expand=True)

        self.title_label = tk.Label(
            self.main_frame, text="🔸 Halaman Utama",
            bg="#2d2d2d", fg="white", font=("Arial", 12, "bold")
        )
        self.title_label.pack(anchor="w", pady=(0, 8))

        # Komponen eksklusif Step 8
        self.model_lf = tk.LabelFrame(
            self.main_frame, text="📁 Model to Analyze",
            bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 9)
        )
        self.model_status_label = tk.Label(
            self.model_lf, text="(Tidak ada model. Jalankan Step 6 & 7 dulu)",
            bg="#222222", fg="#ff6b6b", anchor="w", padx=10, pady=6, font=("Arial", 9, "italic")
        )
        self.model_status_label.pack(fill="x", padx=5, pady=4)

        self.files_lf = tk.LabelFrame(
            self.main_frame, text="📂 Grad-CAM Output Files",
            bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 9)
        )
        self.files_status_label = tk.Label(
            self.files_lf, text="(Mencari files...)",
            bg="#222222", fg="#aaaaaa", anchor="w", padx=10, pady=6, font=("Arial", 9, "italic")
        )
        self.files_status_label.pack(fill="x", padx=5, pady=4)

        # Dashboard Viewer
        self.results_lf = tk.LabelFrame(
            self.main_frame, text="🖼️ Dashboard Viewer",
            bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 9)
        )
        self.results_lf.pack(fill="both", expand=True, pady=5)

        # Canvas + scrollbar untuk chart
        self.canvas_frame = tk.Frame(self.results_lf, bg="#1a1a1a")
        self.canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.canvas_msg = tk.Label(
            self.canvas_frame, text="", bg="#1a1a1a",
            fg="#aaaaaa", font=("Arial", 10, "italic"), justify="center"
        )
        self.canvas_msg.place(relx=0.5, rely=0.5, anchor="center")

        # Label untuk menampilkan gambar chart
        self.img_label = tk.Label(self.canvas_frame, bg="#1a1a1a")
        self.img_label.place(relx=0.5, rely=0.5, anchor="center")

    # ------------------------------------------------------------------
    # TERMINAL
    # ------------------------------------------------------------------
    def create_terminal_output(self):
        term_bar = tk.Frame(self.root, bg="#2d2d2d", padx=20)
        term_bar.pack(fill="x")

        tk.Label(term_bar, text="📟 Terminal Output:", bg="#2d2d2d", fg="white",
                 font=("Arial", 10, "bold")).pack(side="left", pady=4)

        MacFriendlyTab(term_bar, text="Clear Logs", bg="#444444", fg="white",
                       font=("Arial", 8), command=self.clear_terminal,
                       padx=8, pady=3).pack(side="right", pady=4)

        self.terminal = tk.Text(
            self.root, bg="#000000", fg="#00ff00", font=("Consolas", 9),
            height=7, bd=0, padx=10, pady=5, wrap="word"
        )
        self.terminal.pack(fill="x", padx=20, pady=(0, 8))

    # ------------------------------------------------------------------
    # EXECUTE BUTTON
    # ------------------------------------------------------------------
    def create_execute_button(self):
        self.exec_btn = MacFriendlyTab(
            self.root, text="► Jalankan", bg="#3b5998", fg="white",
            font=("Arial", 11, "bold"), command=self.run_current_step, pady=11
        )
        self.exec_btn.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # SWITCH TAB
    # ------------------------------------------------------------------
    DESCRIPTIONS = {
        "Step 1: Ekstraksi":    "Unduh dan ekstrak dataset DermaMNIST (28×28) via library medmnist.",
        "Step 2: Preprocessing":"Hitung mean/std dan definisikan pipeline augmentasi data.",
        "Step 3: Split Data":   "Tampilkan distribusi kelas Train/Val/Test dengan chart.",
        "Step 4: Model FSCA":   "Rakit arsitektur ResNet-18 + 4 modul Fused Spatial-Channel Attention.",
        "Step 5: Training":     "Training 30 epoch dengan AdamW + CosineAnnealingLR + Early Stopping.",
        "Step 6: Finetune":     "Fine-tune layer akhir dengan LR kecil + Label Smoothing.",
        "Step 7: Evaluation":   "Evaluasi model pada test set: Accuracy, F1-W, AUC, Confusion Matrix.",
        "Step 9: Summary":      "Ringkasan performa akhir dan perbandingan dengan state-of-the-art.",
        "Step 10: Inference":   "Upload citra dermatoskopi baru dan dapatkan prediksi + Grad-CAM.",
    }

    def switch_step(self, target_step):
        self.current_step = target_step
        for name, tab in self.nav_tabs.items():
            tab.set_active(name == target_step)

        self.title_label.config(text=f"🔸 {target_step}")
        self.exec_btn.set_state("normal", text=f"► Jalankan {target_step}")

        # Reset viewer
        self.img_label.config(image="")
        self.img_label.image = None

        if "Step 8" in target_step:
            self.model_lf.pack(fill="x", before=self.results_lf, pady=4)
            self.files_lf.pack(fill="x", before=self.results_lf, pady=4)
            if self.step_6_7_done:
                final_path = os.path.join(BASE_DIR, "outputs", "ResNet18_FSCA_DermaMNIST_Best.pth")
                size_mb = os.path.getsize(final_path) / 1e6 if os.path.exists(final_path) else 0
                self.model_status_label.config(
                    text=f"✔ ResNet18_FSCA_DermaMNIST_Best.pth  ({size_mb:.1f} MB)",
                    fg="#2ecc71"
                )
                self.files_status_label.config(
                    text="✔ 7 Kelas Lesi — DermaMNIST. Klik Jalankan untuk generate Grad-CAM.",
                    fg="white"
                )
                self.canvas_msg.config(text="Model siap. Klik tombol eksekusi.", fg="#666666")
            else:
                self.model_status_label.config(
                    text="(Tidak ada model. Jalankan Step 6 dan 7 terlebih dahulu.)",
                    fg="#ff6b6b"
                )
                self.files_status_label.config(text="(Menunggu...)", fg="#aaaaaa")
                self.canvas_msg.config(text="Menunggu validasi model dari Step 6 & 7.", fg="#666666")
        else:
            self.model_lf.pack_forget()
            self.files_lf.pack_forget()
            desc = self.DESCRIPTIONS.get(target_step, "")
            self.canvas_msg.config(text=desc, fg="white")

            # Tampilkan chart terakhir jika ada
            chart_key = target_step
            if chart_key in self.chart_images:
                self._display_chart(self.chart_images[chart_key])

    # ------------------------------------------------------------------
    # TAMPILKAN CHART DI DASHBOARD
    # ------------------------------------------------------------------
    def _display_chart(self, chart_path):
        if not PIL_AVAILABLE or not chart_path or not os.path.exists(chart_path):
            return
        try:
            # Hitung ukuran frame
            self.root.update_idletasks()
            fw = self.canvas_frame.winfo_width()  - 20
            fh = self.canvas_frame.winfo_height() - 10
            fw = max(fw, 400); fh = max(fh, 200)

            img = Image.open(chart_path)
            img.thumbnail((fw, fh), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.img_label.config(image=photo)
            self.img_label.image = photo   # cegah GC
            self.canvas_msg.config(text="")
        except Exception as e:
            self.write_terminal(f"Gagal menampilkan chart: {e}")

    # ------------------------------------------------------------------
    # EXECUTE LOGIC
    # ------------------------------------------------------------------
    def run_current_step(self):
        step = self.current_step

        if "Step 8" in step and not self.step_6_7_done:
            messagebox.showwarning(
                "Akses Ditolak",
                "Model .pth belum tersedia!\nJalankan Step 6 dan Step 7 terlebih dahulu."
            )
            self.write_terminal("Error: Eksekusi Grad-CAM dibatalkan — model belum ada.")
            return

        self.exec_btn.set_state("disabled", text=f"⌛ Memproses {step}...")
        self.canvas_msg.config(text=f"⚙️ Menjalankan {step}...", fg="#aaaaaa")
        self.img_label.config(image="")
        self.img_label.image = None

        threading.Thread(target=self._worker, args=(step,), daemon=True).start()

    def _worker(self, step):
        self.write_terminal(f"--- Memulai {step} ---")
        result = None
        error  = None

        try:
            if "Step 1" in step:
                import step1_extract as m
                result = m.run(self.write_terminal)

            elif "Step 2" in step:
                import step2_preprocess as m
                result = m.run(self.write_terminal)

            elif "Step 3" in step:
                import step3_split as m
                result = m.run(self.write_terminal)

            elif "Step 4" in step:
                import step4_model as m
                result = m.run(self.write_terminal)

            elif "Step 5" in step:
                import step5_training as m
                result = m.run(self.write_terminal)

            elif "Step 6" in step:
                import step6_finetune as m
                result = m.run(self.write_terminal)

            elif "Step 7" in step:
                import step7_evaluation as m
                result = m.run(self.write_terminal)
                # Step 6 & 7 selesai → buka kunci Grad-CAM
                final = os.path.join(BASE_DIR, "outputs", "ResNet18_FSCA_DermaMNIST_Best.pth")
                if os.path.exists(final):
                    self.step_6_7_done = True

            elif "Step 8" in step:
                import step8_gradcam as m
                result = m.run(self.write_terminal)

            elif "Step 9" in step:
                import step9_summary as m
                result = m.run(self.write_terminal)

            elif "Step 10" in step:
                import step10_inference as m
                result = m.run(self.write_terminal)

        except Exception as e:
            import traceback
            error = traceback.format_exc()
            self.write_terminal(f"\nERROR: {e}")
            self.write_terminal("--- Traceback lengkap ada di terminal sistem ---")
            print(error)   # log ke konsol sistem

        # Kembali ke main thread
        self.root.after(0, self._on_complete, step, result, error)

    def _on_complete(self, step, result, error):
        self.exec_btn.set_state("normal", text=f"► Jalankan {step}")

        if error:
            self.canvas_msg.config(
                text=f"❌ {step} GAGAL\nLihat terminal untuk detail error.", fg="#ff6b6b"
            )
            return

        self.write_terminal(f"✓ {step} Selesai.")

        # Simpan hasil & chart
        if result and isinstance(result, dict):
            self.last_result[step] = result
            chart_path = result.get("chart_path")
            if chart_path and os.path.exists(chart_path):
                self.chart_images[step] = chart_path
                self._display_chart(chart_path)
                self.canvas_msg.config(text="")
                return

        self.canvas_msg.config(
            text=f"🌟 {step} Selesai!\nCek terminal untuk detail hasil.",
            fg="#2ecc71"
        )

    # ------------------------------------------------------------------
    # TERMINAL HELPERS
    # ------------------------------------------------------------------
    def write_terminal(self, text):
        def _write():
            self.terminal.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
            self.terminal.see(tk.END)
        # Thread-safe: panggil via main thread
        try:
            self.root.after(0, _write)
        except Exception:
            pass   # root mungkin sudah destroy

    def clear_terminal(self):
        self.terminal.delete("1.0", tk.END)
        self.write_terminal("Terminal logs dibersihkan.")


# ====================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = CancerSkinCADApp(root)
    root.mainloop()
