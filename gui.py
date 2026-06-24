import tkinter as tk
from tkinter import messagebox
import threading
import time

# ====================================================================
# CUSTOM COMPONENT: MAC-COMPATIBLE NAVIGATION & BUTTON TAB
# ====================================================================
class MacFriendlyTab(tk.Label):
    """
    Menggantikan tombol biasa agar patuh pada skema warna Dark Mode macOS.
    Mendukung status Aktif (Selected) dan Tidak Aktif (Deselected).
    """
    def __init__(self, master, text, bg, fg, font, command=None, is_active=False, **kwargs):
        super().__init__(master, text=text, bg=bg, fg=fg, font=font, 
                         relief="flat", cursor="hand2", bd=0, **kwargs)
        self.command = command
        self.is_active = is_active
        self.state = "normal"
        self.default_bg = bg
        self.default_fg = fg
        
        # Binding event mouse
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _on_enter(self, event):
        if self.state == "normal":
            if self.is_active:
                self.config(bg="#1a62d6") # Biru terang
            elif self.default_bg == "#333333":
                self.config(bg="#444444") # Abu-abu terang (Nav)
            elif self.default_bg == "#3b5998":
                self.config(bg="#4a6ea8") # Biru terang (Execute btn)

    def _on_leave(self, event):
        if self.state == "normal":
            if self.is_active:
                self.config(bg="#0f52ba")
            else:
                self.config(bg=self.default_bg)

    def _on_click(self, event):
        if self.state == "normal" and self.command:
            self.command()

    def set_active(self, active):
        """Mengubah status interaktif tab menu secara dinamis"""
        self.is_active = active
        if active:
            self.config(bg="#0f52ba", fg="white", font=("Arial", 9, "bold"))
            self.default_bg = "#0f52ba"
        else:
            self.config(bg="#333333", fg="#aaaaaa", font=("Arial", 8))
            self.default_bg = "#333333"

    def set_state(self, state, text=None):
        """Mengubah status interaktif tombol eksekusi (Aktif / Loading)"""
        self.state = state
        if state == "disabled":
            self.config(bg="#555555", fg="#888888", cursor="arrow")
        else:
            self.config(bg=self.default_bg, fg=self.default_fg, cursor="hand2")
        if text:
            self.config(text=text)

# ====================================================================
# ANTARMUKA GUI UTAMA (SISTEM CAD EXPERIMENT KANKER KULIT)
# ====================================================================
class CancerSkinCADApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem CAD Klasifikasi Kanker Kulit (MedMNIST-FSCA) - Tirta 01082230021")
        self.root.geometry("1100x650")
        self.root.configure(bg="#2d2d2d")
        
        # State Data
        self.step_6_7_done = False
        self.nav_tabs = {} 
        self.current_step = ""
        
        self.create_top_navigation()
        self.create_main_content()
        self.create_terminal_output()
        self.create_execute_button()
        
        # Set default halaman pertama kali terbuka di Step 1
        self.switch_step("Step 1: Ekstraksi")

    def create_top_navigation(self):
        nav_frame = tk.Frame(self.root, bg="#1e1e1e", height=45)
        nav_frame.pack(fill="x", side="top")
        
        steps = [
            "Step 1: Ekstraksi", "Step 2: Preprocessing", "Step 3: Split Data",
            "Step 4: Model FSCA", "Step 5: Training", "Step 6: Finetune",
            "Step 7: Evaluation", "Step 8: Grad-CAM", "Step 9: Summary", "Step 10: Inference"
        ]
        
        for step in steps:
            tab = MacFriendlyTab(
                nav_frame, text=step, bg="#333333", fg="#aaaaaa", font=("Arial", 8), 
                padx=12, pady=10, is_active=False,
                command=lambda s=step: self.switch_step(s)
            )
            tab.pack(side="left", fill="y", padx=1, pady=2)
            self.nav_tabs[step] = tab

    def create_main_content(self):
        self.main_frame = tk.Frame(self.root, bg="#2d2d2d", padx=20, pady=15)
        self.main_frame.pack(fill="both", expand=True)
        
        self.title_label = tk.Label(self.main_frame, text="🔸 Halaman Utama", bg="#2d2d2d", fg="white", font=("Arial", 12, "bold"))
        self.title_label.pack(anchor="w", pady=(0, 10))
        
        # Komponen Spesifik Step 8 (Disembunyikan secara default)
        self.model_lf = tk.LabelFrame(self.main_frame, text="📁 Model to Analyze", bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 9))
        self.model_status_label = tk.Label(self.model_lf, text="(Tidak ada model tersedia. Jalankan Step 6 & 7 terlebih dahulu)", 
                                           bg="#222222", fg="#ff6b6b", anchor="w", padx=10, pady=8, font=("Arial", 9, "italic"))
        self.model_status_label.pack(fill="x", padx=5, pady=5)
        
        self.files_lf = tk.LabelFrame(self.main_frame, text="📂 Grad-CAM Output Files", bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 9))
        self.files_status_label = tk.Label(self.files_lf, text="(Mencari files...)", 
                                           bg="#222222", fg="#aaaaaa", anchor="w", padx=10, pady=8, font=("Arial", 9, "italic"))
        self.files_status_label.pack(fill="x", padx=5, pady=5)
        
        # Dashboard Viewer (Digunakan oleh semua Step)
        self.results_lf = tk.LabelFrame(self.main_frame, text="🖼️ Dashboard Viewer", bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 9))
        self.results_lf.pack(fill="both", expand=True, pady=5)
        
        self.results_canvas = tk.Frame(self.results_lf, bg="#1a1a1a")
        self.results_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.canvas_msg = tk.Label(self.results_canvas, text="", bg="#1a1a1a", fg="#aaaaaa", font=("Arial", 10, "italic"), justify="center")
        self.canvas_msg.place(relx=0.5, rely=0.5, anchor="center")

    def create_terminal_output(self):
        term_label_frame = tk.Frame(self.root, bg="#2d2d2d", padx=20)
        term_label_frame.pack(fill="x")
        
        term_label = tk.Label(term_label_frame, text="📟 Terminal Output:", bg="#2d2d2d", fg="white", font=("Arial", 10, "bold"))
        term_label.pack(side="left", pady=5)
        
        clear_btn = MacFriendlyTab(term_label_frame, text="Clear Logs", bg="#444444", fg="white", 
                                      font=("Arial", 8), command=self.clear_terminal, padx=8, pady=3)
        clear_btn.pack(side="right", pady=5)
        
        self.terminal = tk.Text(self.root, bg="#000000", fg="#00ff00", font=("Consolas", 10), height=6, bd=0, padx=10, pady=5)
        self.terminal.pack(fill="x", padx=20, pady=(0, 10))

    def create_execute_button(self):
        # Tombol eksekusi dinamis (Teks berubah sesuai Step)
        self.exec_btn = MacFriendlyTab(self.root, text="► Jalankan", bg="#3b5998", fg="white", 
                                          font=("Arial", 11, "bold"), command=self.run_current_step, pady=12)
        self.exec_btn.pack(fill="x", side="bottom")

    # ====================================================================
    # LOGIKA INTERAKSI BERPINDAH TAB (SWITCH STEP)
    # ====================================================================
    def switch_step(self, target_step):
        self.current_step = target_step
        
        # 1. Update status visual navigasi atas
        for step_name, tab_obj in self.nav_tabs.items():
            tab_obj.set_active(step_name == target_step)
                
        self.title_label.config(text=f"🔸 {target_step}")
        
        # 2. Update Teks Tombol Eksekusi
        self.exec_btn.set_state("normal", text=f"► Jalankan {target_step}")

        # 3. Kelola layout
        if "Step 8" in target_step:
            self.model_lf.pack(fill="x", before=self.results_lf, pady=5)
            self.files_lf.pack(fill="x", before=self.results_lf, pady=5)
            
            if self.step_6_7_done:
                self.model_status_label.config(text="✔ Model Ditemukan: ResNet18_FSCA_DermaMNIST_Best.pth (Size: 45.2 MB)", fg="#2ecc71")
                self.files_status_label.config(text="✔ 7 Kelas Lesi Terbaca dari Dataset DermaMNIST.", fg="white")
                self.canvas_msg.config(text="(Model Siap. Silakan klik tombol eksekusi di bawah)", fg="#666666")
            else:
                self.model_status_label.config(text="(Tidak ada model tersedia. Eksekusi Step 6 dan 7 terlebih dahulu)", fg="#ff6b6b")
                self.files_status_label.config(text="(Mencari files...)", fg="#aaaaaa")
                self.canvas_msg.config(text="(Menunggu validasi model dari tahap sebelumnya...)", fg="#666666")
        else:
            self.model_lf.pack_forget()
            self.files_lf.pack_forget()
            
            # Deskripsi default sebelum dieksekusi
            descriptions = {
                "Step 1: Ekstraksi": "Klik tombol di bawah untuk mengekstrak dataset MedMNIST (DermaMNIST 28x28).",
                "Step 2: Preprocessing": "Klik tombol di bawah untuk normalisasi & augmentasi data latih.",
                "Step 3: Split Data": "Klik tombol di bawah untuk membagi Training 70%, Val 10%, Testing 20%.",
                "Step 4: Model FSCA": "Klik tombol di bawah untuk merakit arsitektur ResNet-18 + Modul FSCA.",
                "Step 5: Training": "Klik tombol di bawah untuk memulai siklus Training (AdamW Optimizer).",
                "Step 6: Finetune": "Klik tombol di bawah untuk Hyperparameter Tuning dan menyimpan model (.pth).",
                "Step 7: Evaluation": "Klik tombol di bawah untuk menghitung akurasi akhir data uji.",
                "Step 9: Summary": "Klik tombol di bawah untuk menghasilkan ringkasan performa skripsi.",
                "Step 10: Inference": "Klik tombol di bawah untuk menguji prediksi pada citra kulit baru."
            }
            self.canvas_msg.config(text=descriptions.get(target_step, ""), fg="white")

    # ====================================================================
    # LOGIKA EKSEKUSI (JALANKAN MASING-MASING STEP)
    # ====================================================================
    def run_current_step(self):
        step = self.current_step
        
        # Validasi khusus Step 8: Harus lulus Step 6 & 7 dulu
        if "Step 8" in step and not self.step_6_7_done:
            messagebox.showwarning("Akses Ditolak", "Model .pth belum dibuat! Anda harus masuk ke tab Step 6 dan Step 7 lalu menekan tombol 'Jalankan' pada tahap tersebut.")
            self.write_terminal("Error: Eksekusi Grad-CAM dibatalkan.")
            return

        # Kunci tombol saat proses berjalan
        self.exec_btn.set_state("disabled", text=f"⌛ Memproses {step}...")
        self.canvas_msg.config(text=f"⚙️ Menjalankan perintah komputasi untuk {step}...", fg="#aaaaaa")
        
        # Lempar ke background thread agar GUI tidak freeze
        threading.Thread(target=self.worker_thread_general, args=(step,)).start()

    def worker_thread_general(self, step):
        # Simulasi log terminal berdasarkan step
        self.write_terminal(f"--- Memulai {step} ---")
        time.sleep(1) # Simulasi komputasi
        
        if "Step 1" in step:
            self.write_terminal("Mengunduh dan mengekstrak DermaMNIST.npz...")
            self.write_terminal("Ditemukan 10.015 gambar ukuran 28x28 piksel.")
        elif "Step 2" in step:
            self.write_terminal("Menerapkan RandomHorizontalFlip & ColorJitter...")
        elif "Step 3" in step:
            self.write_terminal("Split: 7.010 Train | 1.003 Val | 2.002 Test.")
        elif "Step 4" in step:
            self.write_terminal("Menyuntikkan layer Fused Spatial-Channel Attention ke ResNet-18...")
        elif "Step 5" in step:
            self.write_terminal("Epoch 1/50: Loss=1.24... Epoch 50/50: Loss=0.12...")
        elif "Step 6" in step or "Step 7" in step:
            self.write_terminal("Menghitung AUC dan F1-Score...")
            self.write_terminal("Menyimpan bobot terbaik ke file 'ResNet18_FSCA_DermaMNIST_Best.pth'")
            self.step_6_7_done = True # Membuka kunci Grad-CAM!
        elif "Step 8" in step:
            classes = ["Melanoma (mel)", "Benign Keratosis (bkl)"]
            for penyakit in classes:
                time.sleep(0.5)
                self.write_terminal(f"Memetakan Heatmap Atensi untuk: {penyakit}")
        elif "Step 10" in step:
            self.write_terminal("Menerima input citra eksternal...")
            self.write_terminal("Prediksi: Melanoma (Probabilitas 94.2%)")

        time.sleep(0.5)
        self.write_terminal(f"✓ {step} Selesai.")
        
        # Kembalikan ke main thread untuk update GUI
        self.root.after(0, self.process_complete_general, step)

    def process_complete_general(self, step):
        self.exec_btn.set_state("normal", text=f"► Jalankan {step}")
        self.canvas_msg.config(text=f"🌟 [EKSEKUSI SUKSES] 🌟\nProses komputasi untuk {step} telah berhasil diselesaikan.", fg="#2ecc71")

    def write_terminal(self, text):
        self.terminal.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self.terminal.see(tk.END)
        
    def clear_terminal(self):
        self.terminal.delete('1.0', tk.END)
        self.write_terminal("Terminal logs dibersihkan.")

if __name__ == "__main__":
    root = tk.Tk()
    app = CancerSkinCADApp(root)
    root.mainloop()