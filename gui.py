import tkinter as tk
from tkinter import messagebox
import threading
import time

# ====================================================================
# CUSTOM COMPONENT: MAC-COMPATIBLE NAVIGATION TAB
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
        
        # Binding event mouse
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _on_enter(self, event):
        if self.state == "normal":
            if self.is_active:
                self.config(bg="#1a62d6") # Biru agak terang saat di-hover
            else:
                self.config(bg="#444444") # Abu-abu agak terang saat di-hover

    def _on_leave(self, event):
        if self.state == "normal":
            # Kembali ke warna asalnya berdasarkan status aktif/tidak
            self.config(bg="#0f52ba" if self.is_active else "#333333")

    def _on_click(self, event):
        if self.state == "normal" and self.command:
            self.command()

    def set_active(self, active):
        """Mengubah status interaktif tab menu secara dinamis"""
        self.is_active = active
        if active:
            self.config(bg="#0f52ba", fg="white", font=("Arial", 9, "bold"))
        else:
            self.config(bg="#333333", fg="#aaaaaa", font=("Arial", 8))

# ====================================================================
# ANTARMUKA GUI UTAMA (SISTEM CAD EXPERIMENT KANKER KULIT)
# ====================================================================
class CancerSkinCADApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem CAD Klasifikasi Kanker Kulit (MedMNIST-FSCA) - Tirta 01082230021")
        self.root.geometry("1100x650")
        self.root.configure(bg="#2d2d2d")
        
        # State simulasi status kelayakan model (.pth)
        self.step_6_7_done = False
        self.nav_tabs = {} # Menyimpan objek tab navigasi
        
        # 1. TOP NAVIGATION BAR (10 Horizontal Steps)
        self.create_top_navigation()
        
        # 2. MAIN CONTENT AREA
        self.create_main_content()
        
        # 3. BOTTOM TERMINAL OUTPUT
        self.create_terminal_output()
        
        # 4. STEP 8 EXECUTE BUTTON (Akan disembunyikan jika di luar Step 8)
        self.create_execute_button()
        
        # Set default halaman pertama kali terbuka di Step 8 agar sesuai skenario awal Anda
        self.switch_step("Step 8: Grad-CAM")

    def create_top_navigation(self):
        nav_frame = tk.Frame(self.root, bg="#1e1e1e", height=45)
        nav_frame.pack(fill="x", side="top")
        
        steps = [
            "Step 1: Ekstraksi", "Step 2: Preprocessing", "Step 3: Split Data",
            "Step 4: Model FSCA", "Step 5: Training", "Step 6: Finetune",
            "Step 7: Evaluation", "Step 8: Grad-CAM", "Step 9: Summary", "Step 10: Inference"
        ]
        
        for step in steps:
            # Definisikan aksi klik dengan melemparkan nama step menggunakan default argumen lambda (s=step)
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
        
        # Label judul halaman dinamis
        self.title_label = tk.Label(self.main_frame, text="🔸 Halaman Utama", bg="#2d2d2d", fg="white", font=("Arial", 12, "bold"))
        self.title_label.pack(anchor="w", pady=(0, 10))
        
        # Komponen Konten Spesifik Step 8 (Grad-CAM)
        self.model_lf = tk.LabelFrame(self.main_frame, text="📁 Model to Analyze", bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 9))
        self.model_status_label = tk.Label(self.model_lf, text="(Tidak ada model tersedia. Jalankan Step 6 & 7 terlebih dahulu)", 
                                           bg="#222222", fg="#ff6b6b", anchor="w", padx=10, pady=8, font=("Arial", 9, "italic"))
        self.model_status_label.pack(fill="x", padx=5, pady=5)
        
        self.files_lf = tk.LabelFrame(self.main_frame, text="📂 Grad-CAM Output Files", bg="#2d2d2d", fg="#aaaaaa", font=("Arial", 9))
        self.files_status_label = tk.Label(self.files_lf, text="(Mencari files...)", 
                                           bg="#222222", fg="#aaaaaa", anchor="w", padx=10, pady=8, font=("Arial", 9, "italic"))
        self.files_status_label.pack(fill="x", padx=5, pady=5)
        
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
        self.exec_btn = MacFriendlyTab(self.root, text="► Jalankan Step 8 (Grad-CAM)", bg="#3b5998", fg="white", 
                                          font=("Arial", 11, "bold"), command=self.run_grad_cam_process, pady=12)

    # ====================================================================
    # LOGIKA INTERAKSI BERPINDAH TAB (SWITCH STEP)
    # ====================================================================
    def switch_step(self, target_step):
        """Mengatur perubahan visual tab atas & memperbarui isi dashboard utama"""
        # 1. Update status visual seluruh tab navigasi atas
        for step_name, tab_obj in self.nav_tabs.items():
            if step_name == target_step:
                tab_obj.set_active(True) # Aktif (Warna Biru)
            else:
                tab_obj.set_active(False) # Redup (Warna Abu-Abu gelap seperti draf foto)
                
        self.title_label.config(text=f"🔸 {target_step}")
        self.write_terminal(f"Berpindah navigasi ke halaman: {target_step}")
        
        # 2. Otomatis buka kunci model pth jika user mengklik tahap evaluasi (Step 6 / Step 7)
        if "Step 6" in target_step or "Step 7" in target_step:
            self.step_6_7_done = True
            self.write_terminal("Status: Bobot model 'ResNet18_FSCA_DermaMNIST_Best.pth' sukses diekspor & divalidasi!")

        # 3. Kelola penataan layout isi sesuai karakteristik masing-masing step
        if "Step 8" in target_step:
            # Tampilkan sub-panel khusus Grad-CAM
            self.model_lf.pack(fill="x", before=self.results_lf, pady=5)
            self.files_lf.pack(fill="x", before=self.results_lf, pady=5)
            self.exec_btn.pack(fill="x", side="bottom")
            
            # Sesuaikan teks status model pth
            if self.step_6_7_done:
                self.model_status_label.config(text="✔ Model Ditemukan: ResNet18_FSCA_DermaMNIST_Best.pth (Size: 45.2 MB)", fg="#2ecc71")
                self.files_status_label.config(text="✔ 7 Kelas Lesi Terbaca dari Dataset DermaMNIST.", fg="white")
                self.canvas_msg.config(text="(Model Siap. Silakan klik tombol eksekusi berwarna biru di bawah)", fg="#666666")
            else:
                self.model_status_label.config(text="(Tidak ada model tersedia. Silakan masuk ke Step 6 atau 7 terlebih dahulu)", fg="#ff6b6b")
                self.files_status_label.config(text="(Mencari files...)", fg="#aaaaaa")
                self.canvas_msg.config(text="(Mencari gradcam images... Silakan jalankan eksekusi)", fg="#666666")
        else:
            # Sembunyikan panel Grad-CAM jika berada di luar Step 8
            self.model_lf.pack_forget()
            self.files_lf.pack_forget()
            self.exec_btn.pack_forget()
            
            # Tampilkan ringkasan teks deskriptif bab skripsi Anda
            descriptions = {
                "Step 1: Ekstraksi": "Tahap ekstraksi file kompresi dataset MedMNIST (DermaMNIST).\nResolusi gambar asli direduksi menjadi matriks tensor 28x28 piksel.",
                "Step 2: Preprocessing": "Melakukan normalisasi nilai piksel (0-1) dan augmentasi data taktis:\nRandom Horizontal Flip, Random Vertical Flip, dan Color Jitter.",
                "Step 3: Split Data": "Membagi populasi data menjadi 3 repositori terpisah:\n- Training Set: 70%  |  - Validation Set: 10%  |  - Testing Set: 20%",
                "Step 4: Model FSCA": "Konfigurasi Arsitektur Jaringan: Mengintegrasikan blok baru\nFused Spatial-Channel Attention (FSCA) ke dalam interkoneksi layer ResNet-18.",
                "Step 5: Training": "Melakukan proses training model menggunakan kriteria Loss: Cross-Entropy\nserta algoritma optimasi bobot: AdamW Optimizer (LR = 1e-4).",
                "Step 6: Finetune": "🌟 SIMULASI AKTIF 🌟\nProses Hyperparameter Tuning bereskalasi tinggi.\nBobot ekstraksi fitur terbaik berhasil disimpan ke memori komputer.",
                "Step 7: Evaluation": "🌟 SIMULASI AKTIF 🌟\nMenghitung skor evaluasi akhir pada pengujian data uji.\nModel FSCA mencatatkan kestabilan akurasi klasifikasi multi-kelas yang tinggi.",
                "Step 9: Summary": "Rangkuman komparasi performa eksperimen skripsi.\nArsitektur ResNet-18 + Modul FSCA terbukti menghemat beban parameter komputasi.",
                "Step 10: Inference": "Uji klinik mandiri: Mengunggah berkas citra dermatoskopi lokal di luar dataset\nuntuk memprediksi probabilitas penyakit kanker kulit secara real-time."
            }
            msg = descriptions.get(target_step, "Halaman pengerjaan eksperimen tugas akhir.")
            self.canvas_msg.config(text=msg, fg="white")

    # ====================================================================
    # LOGIKA BACKEND GRAD-CAM (STEP 8)
    # ====================================================================
    def run_grad_cam_process(self):
        if not self.step_6_7_done:
            messagebox.showwarning("Akses Ditolak", "Model belum siap! Silakan klik tab 'Step 6' atau 'Step 7' di navigasi atas terlebih dahulu untuk memicu simulasi pembuatan berkas model .pth Anda.")
            self.write_terminal("Error: Eksekusi gagal karena file model .pth tidak terdeteksi.")
            return
            
        self.exec_btn.config(text="⌛ Sedang Memproses Atensi Spasial-Kanal...", bg="#555555")
        self.canvas_msg.config(text="⏳ Membaca Tensor Citra & Menghitung Gradien Atensi Layer Terakhir...")
        threading.Thread(target=self.worker_thread).start()

    def worker_thread(self):
        classes = ["Melanoma (mel)", "Benign Keratosis (bkl)", "Basal Cell Carcinoma (bcc)"]
        for penyakit in classes:
            time.sleep(1)
            self.write_terminal(f"Grad-CAM mengekstrak target layer: backbone.layer4.fsca_block ...")
            self.write_terminal(f"Sukses memetakan Heatmap Atensi Visual untuk kelas: {penyakit}")
            
        self.root.after(0, self.process_complete)

    def process_complete(self):
        self.exec_btn.config(text="► Jalankan Step 8 (Grad-CAM)", bg="#3b5998")
        self.canvas_msg.config(text="🌟 [SIMULASI SUKSES] 🌟\nHeatmap Grad-CAM berhasil di-render!\nAtensi Fused Spasial-Kanal terbukti fokus memetakan titik lokasi lesi kanker kulit.", fg="#2ecc71")
        messagebox.showinfo("Proses Selesai", "Kalkulasi Atensi Grad-CAM pada Citra DermaMNIST Sukses Berjalan!")

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