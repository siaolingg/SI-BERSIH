# SI-BERSIH — Sistem Intelijen Berbasis Evaluasi Respons Sampah

LLM-Based Social Problem Analysis and Recommendation System · Pengelolaan Sampah di Sulawesi Utara

## Isi folder
| File | Keterangan |
|---|---|
| `app.py` | Seluruh kode: dataset generator, preprocessing, prompt, integrasi LLM, guardrail, batch, evaluasi, fishbone, dashboard |
| `requirements.txt` | Library yang dibutuhkan |
| `.env.example` | Template API key (salin menjadi `.env`) |
| `dataset_sampah.csv` | 600 laporan warga (sintetis) + label manual |
| `fishbone_sibersih.png` | Gambar fishbone untuk laporan |
| `.streamlit/config.toml` | Tema warna hijau untuk tampilan web |
| `Draft_Laporan_SI-BERSIH.md` | Draf teks laporan PDF |
| `Laporan_SI-BERSIH.docx` / `.pdf` | Laporan tugas (lengkapi bagian bertanda [ISI] lalu ekspor ulang ke PDF) |

## Menjalankan (Windows)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env            (lalu isi GEMINI_API_KEY dari https://aistudio.google.com/apikey)

python app.py generate            (buat ulang dataset, opsional)
python app.py batch --limit 50    (uji 50 laporan dulu, lalu jalankan lagi tanpa --limit untuk sisanya)
python app.py evaluate
streamlit run app.py
```
Aplikasi web punya 3 halaman: **Daur Ulang Sampah** (alur 4 langkah: jenis → isi → pilih ide → cara membuat), **Cek Laporan Warga** (teks atau **link website** yang teksnya diambil otomatis, plus **agen tindak lanjut**), dan **Dashboard**.

Tanpa API key, sistem otomatis memakai baseline rule-based (`python app.py batch --provider offline`).
Jika muncul error model tidak ditemukan, ganti `GEMINI_MODEL` di `.env` dengan nama model yang tersedia di AI Studio.

Yang dikumpulkan: `app.py`, `dataset_sampah.csv` (dan `hasil_analisis.csv`), serta laporan PDF.
Jangan ikut kumpulkan folder `.venv` dan file `.env`.

## Versi web (Streamlit Community Cloud)
API key **tidak** disimpan di GitHub. Isi di aplikasi Streamlit Cloud: **⋮ → Settings → Secrets**:
```
GEMINI_API_KEY = "isi-key-anda"
GEMINI_MODEL = "gemini-flash-latest"
```
Setiap `git push` ke branch `main` otomatis memperbarui aplikasi web (link tetap sama).
