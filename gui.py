"""
gui.py — Klasifikasi Kanker Kulit
"""

import tkinter as tk
from tkinter import messagebox
import threading
import time
import os
import sys

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
STEPS_DIR = os.path.join(BASE_DIR, "steps")
sys.path.insert(0, STEPS_DIR)
sys.path.insert(0, BASE_DIR)

from config import PATHS

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ====================================================================
# CUSTOM COMPONENT
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
            if self.is_active:                 self.config(bg="#1a62d6")
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
        self.root.title("Klasifikasi Kanker Kulit (MedMNIST-FSCA) — Tirta 01082230021")
        self.root.geometry("1200x720")
        self.root.configure(bg="#2d2d2d")

        # State
        self.step_6_7_done = False
        self.nav_tabs      = {}
        self.current_step  = ""
        self.chart_images  = {}   # { step_name: path }
        self.last_result   = {}
        self.step_logs     = {}   # { step_name: [line, line, ...] }
        self._img_ref      = None # cegah GC PhotoImage

        self.create_top_navigation()
        self.create_main_content()
        self.create_terminal_output()
        self.create_execute_button()
        self.switch_step("Step 1: Ekstraksi")

        if not PIL_AVAILABLE:
            self.write_terminal("PERINGATAN: Pillow tidak terinstall. Jalankan: pip install Pillow")

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
        for col in range(len(steps)):
            nav_frame.columnconfigure(col, weight=1)

        for col, step in enumerate(steps):
            tab = MacFriendlyTab(
                nav_frame, text=step, bg="#333333", fg="#aaaaaa", font=("Arial", 8),
                padx=0, pady=10, is_active=False,
                command=lambda s=step: self.switch_step(s)
            )
            tab.grid(row=0, column=col, sticky="nsew", padx=1, pady=2)
            self.nav_tabs[step] = tab

    # ------------------------------------------------------------------
    # MAIN CONTENT AREA
    # ------------------------------------------------------------------
    def create_main_content(self):
        self.main_frame = tk.Frame(self.root, bg="#2d2d2d", padx=20, pady=10)
        self.main_frame.pack(fill="both", expand=True)

        self.title_label = tk.Label(
            self.main_frame, text="Halaman Utama",
            bg="#2d2d2d", fg="white", font=("Arial", 12, "bold")
        )
        self.title_label.pack(anchor="w", pady=(0, 8))

        # Komponen eksklusif Step 8
        self.model_lf = tk.LabelFrame(
            self.main_frame, text="Model to Analyze",
            bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 9)
        )
        self.model_status_label = tk.Label(
            self.model_lf, text="(Tidak ada model. Jalankan Step 6 & 7 dulu)",
            bg="#222222", fg="#ff6b6b", anchor="w", padx=10, pady=6, font=("Arial", 9, "italic")
        )
        self.model_status_label.pack(fill="x", padx=5, pady=4)

        self.files_lf = tk.LabelFrame(
            self.main_frame, text="Grad-CAM Output Files",
            bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 9)
        )
        self.files_status_label = tk.Label(
            self.files_lf, text="(Mencari files...)",
            bg="#222222", fg="#aaaaaa", anchor="w", padx=10, pady=6, font=("Arial", 9, "italic")
        )
        self.files_status_label.pack(fill="x", padx=5, pady=4)

        # Dashboard Viewer
        self.results_lf = tk.LabelFrame(
            self.main_frame, text="Dashboard Viewer",
            bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 9)
        )
        self.results_lf.pack(fill="both", expand=True, pady=5)

        self.canvas_frame = tk.Frame(self.results_lf, bg="#1a1a1a")
        self.canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # ── Text widget: log output kiri atas (untuk step tanpa chart) ──
        self.dashboard_text = tk.Text(
            self.canvas_frame,
            bg="#1a1a1a", fg="#cccccc",
            font=("Consolas", 10),
            bd=0, padx=12, pady=10,
            wrap="word",
            state="disabled",   # read-only
            cursor="arrow",
        )
        # Scrollbar untuk dashboard_text
        self.dash_scrollbar = tk.Scrollbar(
            self.canvas_frame, command=self.dashboard_text.yview,
            bg="#333333", troughcolor="#1a1a1a"
        )
        self.dashboard_text.config(yscrollcommand=self.dash_scrollbar.set)

        # ── Image label: chart (untuk step dengan chart) ──
        self.img_label = tk.Label(self.canvas_frame, bg="#1a1a1a")

        # Default: tampilkan mode teks (kosong)
        self._show_text_mode()

    # ------------------------------------------------------------------
    # HELPER: toggle mode dashboard
    # ------------------------------------------------------------------
    def _show_text_mode(self):
        """Tampilkan dashboard_text (kiri atas), sembunyikan img_label."""
        self.img_label.place_forget()
        self.dash_scrollbar.pack(side="right", fill="y")
        self.dashboard_text.pack(side="left", fill="both", expand=True)

    def _show_image_mode(self):
        """Tampilkan img_label (tengah), sembunyikan dashboard_text."""
        self.dashboard_text.pack_forget()
        self.dash_scrollbar.pack_forget()
        self.img_label.place(relx=0.5, rely=0.5, anchor="center")

    def _set_dashboard_text(self, lines, color="#cccccc"):
        """Tulis list of strings ke dashboard_text (read-only)."""
        self.dashboard_text.config(state="normal")
        self.dashboard_text.delete("1.0", tk.END)
        if lines:
            self.dashboard_text.insert(tk.END, "\n".join(lines))
        self.dashboard_text.config(state="disabled", fg=color)

    def _display_chart(self, chart_path):
        """Load dan tampilkan chart PNG di img_label."""
        if not PIL_AVAILABLE or not chart_path or not os.path.exists(chart_path):
            return
        try:
            self.root.update_idletasks()
            fw = max(self.canvas_frame.winfo_width()  - 20, 400)
            fh = max(self.canvas_frame.winfo_height() - 10, 200)

            img = Image.open(chart_path)
            img.thumbnail((fw, fh), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self._img_ref = photo          # cegah GC
            self.img_label.config(image=photo)
            self.img_label.image = photo
        except Exception as e:
            self.write_terminal(f"Gagal menampilkan chart: {e}")

    # ------------------------------------------------------------------
    # SWITCH TAB
    # ------------------------------------------------------------------
    def switch_step(self, target_step):
        self.current_step = target_step
        for name, tab in self.nav_tabs.items():
            tab.set_active(name == target_step)

        self.title_label.config(text=target_step)
        self.exec_btn.set_state("normal", text=f"Jalankan {target_step}")

        if "Step 8" in target_step:
            self.model_lf.pack(fill="x", before=self.results_lf, pady=4)
            self.files_lf.pack(fill="x", before=self.results_lf, pady=4)
            if self.step_6_7_done:
                final_path = PATHS["final_model"]
                size_mb = os.path.getsize(final_path) / 1e6 if os.path.exists(final_path) else 0
                self.model_status_label.config(
                    text=f"Model ditemukan: ResNet18_FSCA_DermaMNIST_Best.pth  ({size_mb:.1f} MB)",
                    fg="#2ecc71"
                )
                self.files_status_label.config(
                    text="7 Kelas Lesi tersedia — DermaMNIST. Klik Jalankan untuk generate Grad-CAM.",
                    fg="white"
                )
            else:
                self.model_status_label.config(
                    text="(Tidak ada model. Jalankan Step 6 dan 7 terlebih dahulu.)",
                    fg="#ff6b6b"
                )
                self.files_status_label.config(text="(Menunggu...)", fg="#aaaaaa")
        else:
            self.model_lf.pack_forget()
            self.files_lf.pack_forget()

        # Pulihkan tampilan terakhir step ini
        self._restore_dashboard(target_step)

    def _restore_dashboard(self, step):
        """Tampilkan chart jika ada, atau log terakhir jika ada, atau kosong."""
        if step in self.chart_images:
            self._show_image_mode()
            self._display_chart(self.chart_images[step])
        elif step in self.step_logs and self.step_logs[step]:
            self._show_text_mode()
            self._set_dashboard_text(self.step_logs[step])
        else:
            self._show_text_mode()
            self._set_dashboard_text([])

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

        # Reset log step ini
        self.step_logs[step] = []
        self.exec_btn.set_state("disabled", text=f"Memproses {step}...")

        # Kosongkan dashboard saat memproses
        self._show_text_mode()
        self._set_dashboard_text([f"Menjalankan {step}..."], color="#aaaaaa")

        self.img_label.config(image="")
        self.img_label.image = None

        threading.Thread(target=self._worker, args=(step,), daemon=True).start()

    def _worker(self, step):
        self.write_terminal(f"--- Memulai {step} ---")
        result = None
        error  = None

        try:
            if   step == "Step 1: Ekstraksi":     import step1_extract    as m; result = m.run(self.write_terminal)
            elif step == "Step 2: Preprocessing": import step2_preprocess as m; result = m.run(self.write_terminal)
            elif step == "Step 3: Split Data":    import step3_split      as m; result = m.run(self.write_terminal)
            elif step == "Step 4: Model FSCA":    import step4_model      as m; result = m.run(self.write_terminal)
            elif step == "Step 5: Training":      import step5_training   as m; result = m.run(self.write_terminal)
            elif step == "Step 6: Finetune":      import step6_finetune   as m; result = m.run(self.write_terminal)
            elif step == "Step 7: Evaluation":
                import step7_evaluation as m
                result = m.run(self.write_terminal)
                final = PATHS["final_model"]
                if os.path.exists(final):
                    self.step_6_7_done = True
            elif step == "Step 8: Grad-CAM":      import step8_gradcam    as m; result = m.run(self.write_terminal)
            elif step == "Step 9: Summary":       import step9_summary    as m; result = m.run(self.write_terminal)
            elif step == "Step 10: Inference":    import step10_inference as m; result = m.run(self.write_terminal)
            else:
                self.write_terminal(f"ERROR internal: step tidak dikenali -> '{step}'")

        except Exception as e:
            import traceback
            error = traceback.format_exc()
            self.write_terminal(f"\nERROR: {e}")
            self.write_terminal("--- Traceback lengkap ada di terminal sistem ---")
            print(error)

        self.root.after(0, self._on_complete, step, result, error)

    def _on_complete(self, step, result, error):
        self.exec_btn.set_state("normal", text=f"Jalankan {step}")

        if error:
            # Tampilkan log (termasuk pesan error) di dashboard
            logs = self.step_logs.get(step, [])
            self._show_text_mode()
            self._set_dashboard_text(logs, color="#ff9999")
            return

        self.write_terminal(f"{step} Selesai.")

        # Jika ada chart → tampilkan chart
        if result and isinstance(result, dict):
            self.last_result[step] = result
            chart_path = result.get("chart_path")
            if chart_path and os.path.exists(chart_path):
                self.chart_images[step] = chart_path
                self._show_image_mode()
                self._display_chart(chart_path)
                return

        # Tidak ada chart → tampilkan log terminal di dashboard
        logs = self.step_logs.get(step, [])
        self._show_text_mode()
        self._set_dashboard_text(logs, color="#cccccc")

    # ------------------------------------------------------------------
    # TERMINAL HELPERS
    # ------------------------------------------------------------------
    def write_terminal(self, text):
        def _write():
            self.terminal.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
            self.terminal.see(tk.END)
            # Akumulasi log untuk dashboard
            step = self.current_step
            if step:
                self.step_logs.setdefault(step, []).append(text)
        try:
            self.root.after(0, _write)
        except Exception:
            pass

    def clear_terminal(self):
        self.terminal.delete("1.0", tk.END)
        self.write_terminal("Terminal logs dibersihkan.")

    # ------------------------------------------------------------------
    # EXECUTE BUTTON
    # ------------------------------------------------------------------
    def create_terminal_output(self):
        term_bar = tk.Frame(self.root, bg="#2d2d2d", padx=20)
        term_bar.pack(fill="x")

        tk.Label(term_bar, text="Terminal Output:", bg="#2d2d2d", fg="white",
                 font=("Arial", 10, "bold")).pack(side="left", pady=4)

        MacFriendlyTab(term_bar, text="Clear Logs", bg="#444444", fg="white",
                       font=("Arial", 8), command=self.clear_terminal,
                       padx=8, pady=3).pack(side="right", pady=4)

        self.terminal = tk.Text(
            self.root, bg="#000000", fg="#00ff00", font=("Consolas", 9),
            height=7, bd=0, padx=10, pady=5, wrap="word"
        )
        self.terminal.pack(fill="x", padx=20, pady=(0, 8))

    def create_execute_button(self):
        self.exec_btn = MacFriendlyTab(
            self.root, text="Jalankan", bg="#3b5998", fg="white",
            font=("Arial", 11, "bold"), command=self.run_current_step, pady=11
        )
        self.exec_btn.pack(fill="x", side="bottom")


# ====================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = CancerSkinCADApp(root)
    root.mainloop()