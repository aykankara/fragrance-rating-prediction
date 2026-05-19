import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix


# ── Model ve encoder'ları yükle ──────────────────────────────────────────────
try:
    model      = joblib.load('model.pkl')
    tfidf_top  = joblib.load('tfidf_top.pkl')
    tfidf_base = joblib.load('tfidf_base.pkl')
    le_brand   = joblib.load('le_brand.pkl')
except FileNotFoundError:
    print("HATA: model.pkl bulunamadı. Lütfen önce Jupyter Notebook'u çalıştırın.")
    exit(1)


# ── Tahmin fonksiyonu ─────────────────────────────────────────────────────────
def tahmin_et(ust_nota: str, alt_nota: str, marka: str) -> str:

    X_top  = tfidf_top.transform([ust_nota.lower()])
    X_base = tfidf_base.transform([alt_nota.lower()])

    # Marka encoding – veri setinde yoksa 0 ata
    marka_temiz = marka.lower().replace(' ', '-')
    if marka_temiz in le_brand.classes_:
        brand_enc = le_brand.transform([marka_temiz])[0]
    else:
        brand_enc = 0  # Bilinmeyen marka -> varsayılan

    X = hstack([X_top, X_base, csr_matrix([[brand_enc]])])
    puan = float(model.predict(X)[0])

    # Puanı [1, 5] aralığına sıkıştır
    puan = max(1.0, min(5.0, puan))
    return f"{puan:.2f} / 5"


# ── Tkinter Arayüzü ───────────────────────────────────────────────────────────
class UygulAmA:
    def __init__(self, root: tk.Tk):
        root.title("🌸 Parfüm Puan Tahmini")
        root.geometry("420x340")
        root.resizable(False, False)
        root.configure(bg="#fdf6f0")

        # Başlık
        tk.Label(root, text="Parfüm Puan Tahmini",
                 font=("Helvetica", 16, "bold"),
                 bg="#fdf6f0", fg="#6b3a5e").pack(pady=(20, 5))
        tk.Label(root, text="Üst nota, alt nota ve marka girin",
                 font=("Helvetica", 10), bg="#fdf6f0", fg="#888").pack()

        # Form çerçevesi
        form = tk.Frame(root, bg="#fdf6f0")
        form.pack(pady=15)

        labels = ["Üst Nota (Top):", "Alt Nota (Base):", "Marka:"]
        self.girişler = []
        for i, lbl in enumerate(labels):
            tk.Label(form, text=lbl, font=("Helvetica", 11),
                     bg="#fdf6f0", anchor='w', width=18).grid(row=i, column=0, pady=6, sticky='w')
            e = tk.Entry(form, font=("Helvetica", 11), width=22,
                         relief='flat', bd=2, highlightthickness=1,
                         highlightbackground="#d4a8c7", highlightcolor="#9b59b6")
            e.grid(row=i, column=1, pady=6)
            self.girişler.append(e)

        # Tahmin butonu
        tk.Button(root, text="Puanı Tahmin Et",
                  font=("Helvetica", 12, "bold"),
                  bg="#9b59b6", fg="white", relief='flat',
                  padx=15, pady=8, cursor='hand2',
                  command=self._tahmin_yap).pack(pady=8)

        # Sonuç etiketi
        self.sonuc_lbl = tk.Label(root, text="",
                                  font=("Helvetica", 18, "bold"),
                                  bg="#fdf6f0", fg="#2c3e50")
        self.sonuc_lbl.pack()

    def _tahmin_yap(self):
        ust   = self.girişler[0].get().strip()
        alt   = self.girişler[1].get().strip()
        marka = self.girişler[2].get().strip()

        if not all([ust, alt, marka]):
            messagebox.showwarning("Eksik Bilgi", "Lütfen tüm alanları doldurun.")
            return

        try:
            sonuc = tahmin_et(ust, alt, marka)
            self.sonuc_lbl.config(text=f"⭐  {sonuc}", fg="#6b3a5e")
        except Exception as e:
            messagebox.showerror("Hata", str(e))


# ── Uygulamayı başlat ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()
    UygulAmA(root)
    root.mainloop()
