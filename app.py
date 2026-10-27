# -*- coding: utf-8 -*-
"""
SI-BERSIH — Sistem Intelijen Berbasis Evaluasi Respons Sampah
LLM-Based Social Problem Analysis and Recommendation System
Studi kasus: Pengelolaan Sampah di Provinsi Sulawesi Utara

Cara pakai (jalankan dari folder proyek):
    python app.py generate                  -> membuat dataset_sampah.csv (600 laporan sintetis)
    python app.py batch --limit 50          -> analisis dataset dengan LLM -> hasil_analisis.csv
    python app.py batch --provider offline  -> baseline rule-based (tanpa API key)
    python app.py evaluate                  -> akurasi prediksi vs label manual
    python app.py fishbone                  -> membuat gambar fishbone_sibersih.png
    streamlit run app.py                    -> aplikasi web (3 halaman: Daur Ulang, Cek Laporan, Dashboard)

Struktur file:
    1.  Konfigurasi & placeholder API key
    2.  Knowledge base daur ulang (sumber RAG lokal)
    3.  Dataset generator
    4.  Preprocessing
    5.  Prompt engineering
    6.  Integrasi LLM (Gemini / GPT / DeepSeek / Grok)
    7.  Core AI logic + guardrail + baseline rule-based
    8.  Batch processing & evaluasi
    9.  Fishbone diagram
    10. Dashboard Streamlit
    11. CLI
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().with_name(".env"))
except ImportError:  # python-dotenv opsional
    pass


# =============================================================================
# 1. KONFIGURASI & PLACEHOLDER API KEY
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset_sampah.csv"
HASIL_PATH = BASE_DIR / "hasil_analisis.csv"
FISHBONE_PATH = BASE_DIR / "fishbone_sibersih.png"

NAMA_SISTEM = "SI-BERSIH"
KEPANJANGAN = "Sistem Intelijen Berbasis Evaluasi Respons Sampah"

# >>> PLACEHOLDER API KEY <<<
# Cara 1 (disarankan): salin .env.example menjadi .env lalu isi key di sana.
# Cara 2: ganti teks "MASUKKAN_..." di bawah ini (jangan unggah key ke publik/GitHub).
# Cara 3: tempel key di sidebar dashboard (Pengaturan LLM).
API_KEYS = {
    "Gemini": os.getenv("GEMINI_API_KEY", "MASUKKAN_API_KEY_GEMINI_DI_SINI"),
    "OpenAI (GPT)": os.getenv("OPENAI_API_KEY", "MASUKKAN_API_KEY_OPENAI_DI_SINI"),
    "DeepSeek": os.getenv("DEEPSEEK_API_KEY", "MASUKKAN_API_KEY_DEEPSEEK_DI_SINI"),
    "Grok (xAI)": os.getenv("XAI_API_KEY", "MASUKKAN_API_KEY_XAI_DI_SINI"),
}

OFFLINE = "Offline (rule-based)"

# Nama model bisa diganti lewat .env bila Google/penyedia lain mengganti versi model.
PROVIDERS = {
    "Gemini": {"model": os.getenv("GEMINI_MODEL", "gemini-flash-latest"), "base_url": None},
    "OpenAI (GPT)": {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "base_url": None},
    "DeepSeek": {"model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), "base_url": "https://api.deepseek.com"},
    "Grok (xAI)": {"model": os.getenv("XAI_MODEL", "grok-3-mini"), "base_url": "https://api.x.ai/v1"},
    OFFLINE: {"model": "rule-based", "base_url": None},
}

JENIS = ["Organik", "Anorganik", "Campuran"]
SENTIMEN = ["Positif", "Negatif", "Netral"]
MASALAH = [
    "Layanan Pengangkutan",
    "Fasilitas & TPS",
    "Perilaku Membuang Sampah",
    "Kebersihan Lingkungan",
    "Partisipasi & Edukasi",
]
AMBIGUITAS = ["Rendah", "Sedang", "Tinggi"]


@dataclass
class LLMConfig:
    provider: str = OFFLINE
    api_key: str = ""
    model: str = "rule-based"
    base_url: str | None = None
    temperature: float = 0.2

    @property
    def pakai_llm(self) -> bool:
        key = (self.api_key or "").strip()
        return self.provider != OFFLINE and bool(key) and not key.startswith("MASUKKAN")

    @property
    def label(self) -> str:
        return f"{self.provider} · {self.model}" if self.pakai_llm else "Rule-based (offline)"


def config_default(provider: str) -> LLMConfig:
    p = PROVIDERS.get(provider, PROVIDERS[OFFLINE])
    return LLMConfig(provider, API_KEYS.get(provider, ""), p["model"], p["base_url"])


# =============================================================================
# 2. KNOWLEDGE BASE DAUR ULANG (sumber RAG lokal)
# =============================================================================

# Kamus kata kunci benda sampah -> jenis. Dipakai untuk pra-klasifikasi,
# validasi input pengguna, retrieval (RAG), dan baseline offline.
KATA_ORGANIK = [
    "nasi", "sayur", "buah", "kulit buah", "kulit pisang", "pisang", "semangka", "kulit semangka",
    "ikan", "sisa ikan", "tulang ikan", "kepala ikan", "isi perut ikan", "sisik", "cakalang",
    "ampas kelapa", "sabut kelapa", "tempurung kelapa", "kelapa", "daun", "daun pisang",
    "ranting", "rumput", "kulit jagung", "tongkol jagung", "jagung", "ampas tahu", "kulit bawang",
    "kulit telur", "sisa makanan", "makanan", "minyak jelantah", "jelantah", "minyak goreng bekas",
    "kulit durian", "durian", "langsat", "kulit nanas", "nanas", "mangga", "pepaya", "jeruk",
    "ampas kopi", "roti", "mie", "tulang", "daging", "kotoran ternak", "batang pisang",
    "pelepah", "bunga", "kayu", "serbuk gergaji", "tomat", "cabai", "wortel", "kangkung",
    "alpukat", "rambutan", "salak", "sisa sayur",
]
KATA_ANORGANIK = [
    "plastik", "botol plastik", "botol", "kresek", "kantong plastik", "kantong kresek",
    "gelas plastik", "sedotan", "styrofoam", "sachet", "kemasan", "bungkus", "bungkus kopi",
    "kaleng", "besi", "logam", "aluminium", "botol kaca", "kaca", "toples kaca", "kardus",
    "karton", "kertas", "koran", "majalah", "jaring", "jaring nelayan", "tali", "tali tambang",
    "ban", "ban bekas", "karet", "sandal", "sepatu", "kain", "baju", "pakaian", "popok",
    "tutup botol", "galon", "ember", "pipa", "paralon", "masker", "barang bekas",
]
KATA_B3 = ["baterai", "aki", "lampu", "bohlam", "elektronik", "obat kedaluwarsa", "oli bekas", "pestisida"]

KAMUS_ITEM = {**{k: "Organik" for k in KATA_ORGANIK}, **{k: "Anorganik" for k in KATA_ANORGANIK},
              **{k: "B3" for k in KATA_B3}}

PRODUK_DAUR_ULANG = [
    # ----------------------------- ORGANIK -----------------------------
    {
        "nama": "Pakan Maggot BSF (Black Soldier Fly)", "jenis": "Organik", "kesulitan": "Mudah",
        "durasi": "± 2–3 minggu per siklus",
        "bahan_cocok": ["nasi", "sayur", "buah", "sisa makanan", "makanan", "ampas tahu", "ikan",
                        "roti", "mie", "kulit buah", "ampas kelapa", "semangka", "pisang", "sisa sayur"],
        "deskripsi": "Sampah organik basah diurai larva lalat BSF; maggot dipanen sebagai pakan ayam, bebek, dan ikan berprotein tinggi.",
        "nilai_guna": "Mengurangi biaya pakan ternak/ikan; maggot dan sisa uraiannya (kasgot) dapat dijual ke peternak.",
        "alat": ["Ember atau bak plastik (biopond) berpenutup jaring", "Wadah penetasan (nampan dangkal)",
                 "Pisau/talenan atau blender untuk mencacah", "Sarung tangan", "Ayakan/saringan kawat"],
        "bahan": ["Telur atau bibit BSF dari peternak maggot", "Sampah organik basah yang dicacah",
                  "Dedak/ampas tahu sebagai media awal", "Sedikit air"],
        "langkah": [
            "Siapkan biopond di tempat teduh, beri lubang drainase dan tutup jaring agar aman dari tikus dan lalat lain.",
            "Tetaskan telur BSF di atas media dedak lembap selama ± 4–5 hari hingga menjadi larva kecil.",
            "Cacah sampah organik (nasi, sayur, buah) dan tiriskan agar tidak terlalu berair.",
            "Pindahkan larva ke biopond dan beri pakan sampah organik sedikit demi sedikit setiap hari.",
            "Setelah 14–18 hari (larva gemuk berwarna krem), panen maggot dengan ayakan dan pisahkan kasgotnya.",
        ],
        "keamanan": "Jangan masukkan plastik, bahan kimia, atau minyak berlebihan; selalu pakai sarung tangan.",
    },
    {
        "nama": "Eco-Enzyme", "jenis": "Organik", "kesulitan": "Mudah", "durasi": "3 bulan fermentasi",
        "bahan_cocok": ["kulit buah", "buah", "sayur", "kulit pisang", "kulit nanas", "nanas", "semangka",
                        "kulit semangka", "jeruk", "pepaya", "mangga", "langsat", "sisa sayur", "tomat", "kangkung"],
        "deskripsi": "Cairan fermentasi serbaguna dari sisa buah/sayur segar + gula + air (1:3:10) untuk pembersih alami dan penyiram tanaman.",
        "nilai_guna": "Pengganti sabun pel dan cairan pembersih rumah tangga; bisa dikemas dan dijual.",
        "alat": ["Wadah plastik bertutup (bukan kaca)", "Timbangan", "Pisau dan talenan",
                 "Gayung/takaran", "Label dan spidol", "Saringan kain"],
        "bahan": ["1 bagian gula merah atau molase", "3 bagian kulit buah/sayur segar (tidak dimasak, tidak berminyak)",
                  "10 bagian air bersih"],
        "langkah": [
            "Isi wadah hanya ± 60% volume; larutkan 1 bagian gula merah ke dalam 10 bagian air.",
            "Masukkan 3 bagian potongan kulit buah/sayur segar, aduk hingga semua terendam.",
            "Tutup rapat, beri label tanggal, simpan di tempat sejuk yang tidak terkena matahari langsung.",
            "Buka tutup sebentar setiap hari pada 2 minggu pertama untuk membuang gas, lalu seminggu sekali.",
            "Panen setelah 3 bulan: saring cairannya ke botol; ampasnya dipakai lagi untuk fermentasi berikutnya.",
        ],
        "keamanan": "Jangan memakai wadah kaca (bisa pecah karena gas) dan jangan memakai sisa ikan/daging atau makanan berminyak.",
    },
    {
        "nama": "Biogas Rumah Tangga (Digester Drum)", "jenis": "Organik", "kesulitan": "Sulit",
        "durasi": "2–3 minggu sampai gas pertama",
        "bahan_cocok": ["sisa makanan", "makanan", "nasi", "sayur", "buah", "kotoran ternak", "ampas tahu", "sisa sayur"],
        "deskripsi": "Sampah organik difermentasi tanpa oksigen di dalam drum sehingga menghasilkan gas metana untuk memasak.",
        "nilai_guna": "Menghemat gas LPG rumah tangga atau warung makan.",
        "alat": ["Drum plastik 200 liter", "Pipa PVC, keran gas, dan selang gas", "Ban dalam bekas/kantong penampung gas",
                 "Lem pipa kedap udara", "Kompor biogas"],
        "bahan": ["Kotoran ternak (sapi/babi) sebagai starter", "Sampah organik yang dicacah", "Air"],
        "langkah": [
            "Rakit digester: buat lubang inlet, outlet ampas, dan keran gas di bagian atas drum; tutup semua sambungan dengan lem kedap udara.",
            "Isi awal dengan campuran kotoran ternak dan air 1:1 sampai ± 75% volume sebagai starter bakteri.",
            "Biarkan 2–3 minggu tanpa oksigen; buang gas pertama (masih banyak CO₂) di luar ruangan.",
            "Tambahkan 2–5 kg sampah organik cacah setiap hari lewat inlet; ampas cair keluar lewat outlet.",
            "Salurkan gas lewat selang ke kantong penampung lalu ke kompor biogas; uji nyala di area terbuka.",
        ],
        "keamanan": "Gas mudah terbakar: rakit jauh dari api, cek kebocoran dengan air sabun, letakkan digester di luar rumah.",
    },
    {
        "nama": "Briket Arang Tempurung/Tongkol Jagung", "jenis": "Organik", "kesulitan": "Sedang",
        "durasi": "± 1 minggu",
        "bahan_cocok": ["tempurung kelapa", "sabut kelapa", "kelapa", "tongkol jagung", "kayu", "ranting",
                        "serbuk gergaji", "kulit durian", "daun", "jagung"],
        "deskripsi": "Limbah keras/berserat dikarbonisasi menjadi arang lalu dicetak menjadi briket bahan bakar.",
        "nilai_guna": "Bahan bakar pengganti kayu/arang untuk rumah makan dan pemanggang; briket tempurung kelapa diminati pasar.",
        "alat": ["Drum/kaleng bekas berlubang untuk pengarangan", "Lesung atau alat penumbuk", "Ayakan",
                 "Cetakan dari pipa paralon", "Panci untuk lem kanji", "Tempat penjemuran"],
        "bahan": ["Tempurung kelapa/tongkol jagung kering", "Tepung tapioka (perekat ± 5–10%)", "Air panas"],
        "langkah": [
            "Jemur tempurung kelapa atau tongkol jagung 2–3 hari sampai benar-benar kering.",
            "Bakar di dalam drum dengan udara terbatas sampai menjadi arang hitam, lalu tutup rapat drum untuk memadamkan.",
            "Tumbuk arang sampai halus lalu ayak.",
            "Campurkan serbuk arang dengan lem kanji (tapioka + air panas) sampai bisa dikepal.",
            "Cetak dengan pipa paralon, lalu jemur 2–3 hari sampai keras dan siap dipakai/dikemas.",
        ],
        "keamanan": "Lakukan pengarangan di luar ruangan, jauh dari rumah padat, dan siapkan air pemadam.",
    },
    {
        "nama": "Sabun Cuci dari Minyak Jelantah", "jenis": "Organik", "kesulitan": "Sedang",
        "durasi": "1 hari pembuatan + 3–4 minggu pematangan",
        "bahan_cocok": ["minyak jelantah", "jelantah", "minyak goreng bekas"],
        "deskripsi": "Minyak goreng bekas dijernihkan lalu direaksikan dengan soda api menjadi sabun cuci piring/lantai.",
        "nilai_guna": "Mencegah jelantah dibuang ke saluran air; sabun bisa dipakai sendiri atau dijual.",
        "alat": ["Kacamata pelindung, masker, dan sarung tangan karet", "Wadah tahan panas (plastik PP/stainless, bukan aluminium)",
                 "Pengaduk kayu atau hand mixer", "Timbangan digital", "Cetakan sabun"],
        "bahan": ["Minyak jelantah yang sudah dijernihkan", "Soda api (NaOH)", "Air", "Pewangi alami (opsional)"],
        "langkah": [
            "Jernihkan jelantah: rendam dengan arang aktif selama 24 jam lalu saring.",
            "Larutkan soda api ke dalam air (masukkan soda ke air, bukan sebaliknya) di tempat berventilasi; takaran umum ± 130–140 g NaOH dan ± 300 ml air per 1 kg minyak — cek dengan kalkulator sabun.",
            "Tuang larutan soda perlahan ke minyak sambil terus diaduk sampai mengental seperti puding (trace).",
            "Tambahkan pewangi bila mau, lalu tuang ke cetakan.",
            "Diamkan 24–48 jam, keluarkan dari cetakan, lalu matangkan 3–4 minggu sebelum dipakai untuk mencuci (bukan untuk kulit).",
        ],
        "keamanan": "Soda api korosif: wajib sarung tangan dan kacamata, jauhkan dari anak-anak.",
    },
    {
        "nama": "Lilin Aromaterapi dari Minyak Jelantah", "jenis": "Organik", "kesulitan": "Mudah",
        "durasi": "± 1 hari",
        "bahan_cocok": ["minyak jelantah", "jelantah", "minyak goreng bekas"],
        "deskripsi": "Jelantah jernih dicampur stearin menjadi lilin; aroma serai wangi sekaligus mengusir nyamuk.",
        "nilai_guna": "Suvenir atau lilin pengusir nyamuk bernilai jual.",
        "alat": ["Panci dan kompor (metode tim/au bain-marie)", "Wadah gelas/kaleng bekas", "Sumbu lilin", "Pengaduk"],
        "bahan": ["Minyak jelantah jernih", "Stearin (asam stearat)", "Pewarna krayon", "Minyak atsiri serai wangi"],
        "langkah": [
            "Jernihkan jelantah dengan arang aktif lalu saring.",
            "Lelehkan stearin dengan api kecil, lalu campur minyak (± 2 bagian stearin : 1 bagian minyak).",
            "Tambahkan pewarna dan minyak serai wangi, aduk rata.",
            "Pasang sumbu di tengah wadah, lalu tuang campuran perlahan.",
            "Diamkan 4–6 jam sampai mengeras, lalu rapikan sumbu.",
        ],
        "keamanan": "Jangan memanaskan minyak langsung dengan api besar; awasi selama proses.",
    },
    {
        "nama": "Media Tanam Cocopeat & Cocofiber", "jenis": "Organik", "kesulitan": "Sedang",
        "durasi": "± 1 minggu",
        "bahan_cocok": ["sabut kelapa", "kelapa", "ampas kelapa"],
        "deskripsi": "Sabut kelapa diurai menjadi serbuk (cocopeat) dan serat (cocofiber) untuk media tanam hidroponik/tanaman hias.",
        "nilai_guna": "Sulawesi Utara kaya kelapa; cocopeat dibutuhkan pegiat tanaman hias dan hidroponik.",
        "alat": ["Golok/pisau", "Palu kayu atau alat pemukul", "Ayakan kawat", "Ember perendam", "Terpal penjemur", "Karung"],
        "bahan": ["Sabut kelapa kering", "Air bersih"],
        "langkah": [
            "Rendam sabut kelapa 1–3 hari untuk mengurangi tanin, ganti airnya setiap hari.",
            "Pukul/tumbuk sabut sampai seratnya terurai.",
            "Ayak: serbuk halus menjadi cocopeat, serat panjang menjadi cocofiber.",
            "Bilas cocopeat sekali lagi lalu jemur sampai kering.",
            "Kemas dalam karung atau padatkan menjadi blok untuk dijual sebagai media tanam.",
        ],
        "keamanan": "Gunakan sarung tangan saat memukul sabut; serat halus bisa mengiritasi.",
    },
    {
        "nama": "Silase Pakan Ikan dari Limbah Ikan", "jenis": "Organik", "kesulitan": "Sedang",
        "durasi": "7–14 hari fermentasi",
        "bahan_cocok": ["ikan", "sisa ikan", "tulang ikan", "kepala ikan", "isi perut ikan", "sisik", "cakalang"],
        "deskripsi": "Limbah ikan (kepala, tulang, isi perut) difermentasi menjadi silase lalu dicetak menjadi pelet pakan.",
        "nilai_guna": "Cocok untuk kota perikanan seperti Bitung; menekan biaya pakan budidaya ikan.",
        "alat": ["Penggiling daging/blender atau pisau", "Wadah plastik tertutup rapat", "Timbangan", "Pengaduk", "Cetakan/penggiling pelet"],
        "bahan": ["Limbah ikan segar (tidak busuk)", "Molase/gula merah 10–15%", "Starter bakteri asam laktat (mis. EM4) ± 5%",
                  "Dedak atau tepung jagung"],
        "langkah": [
            "Giling atau cacah limbah ikan segar.",
            "Campur dengan molase/gula merah 10–15% dan starter EM4 ± 5%, aduk rata.",
            "Masukkan ke wadah tertutup rapat dan fermentasi 7–14 hari sampai berbau asam segar (bukan busuk).",
            "Campurkan silase dengan dedak/tepung jagung sampai padat, lalu cetak menjadi pelet.",
            "Jemur sampai kering dan simpan di wadah kering sebagai pakan ikan atau ternak.",
        ],
        "keamanan": "Gunakan sarung tangan; buang batch yang berbau busuk menyengat atau berjamur hitam.",
    },
    {
        "nama": "Kerajinan Kulit Jagung & Pelepah Pisang", "jenis": "Organik", "kesulitan": "Mudah",
        "durasi": "2–3 hari",
        "bahan_cocok": ["kulit jagung", "jagung", "pelepah", "batang pisang", "daun pisang", "daun"],
        "deskripsi": "Kulit jagung atau pelepah pisang kering diwarnai dan dirangkai menjadi bunga hias, tatakan gelas, atau anyaman.",
        "nilai_guna": "Suvenir khas daerah untuk pariwisata dan acara adat.",
        "alat": ["Gunting", "Lem tembak/lem putih", "Kawat dan selotip bunga", "Setrika", "Ember untuk pewarnaan"],
        "bahan": ["Kulit jagung/pelepah pisang kering", "Pewarna tekstil atau alami", "Pernis"],
        "langkah": [
            "Pilih dan jemur kulit jagung/pelepah pisang sampai kering.",
            "Rendam dalam pewarna, lalu jemur kembali.",
            "Ratakan dengan setrika dan potong sesuai pola kelopak/anyaman.",
            "Rangkai menjadi bunga hias, tatakan gelas, atau anyaman tas kecil dengan lem dan kawat.",
            "Lapisi pernis tipis agar awet dan siap dijual.",
        ],
        "keamanan": "Hati-hati saat memakai setrika dan lem tembak panas.",
    },
    # ---------------------------- ANORGANIK ----------------------------
    {
        "nama": "Eco-brick", "jenis": "Anorganik", "kesulitan": "Mudah", "durasi": "1–3 hari per botol",
        "bahan_cocok": ["kresek", "kantong plastik", "kantong kresek", "sachet", "kemasan", "bungkus",
                        "sedotan", "styrofoam", "plastik", "bungkus kopi", "botol plastik"],
        "deskripsi": "Botol plastik dipadatkan dengan plastik lunak bersih-kering hingga menjadi 'bata' untuk kursi, meja, atau pagar taman.",
        "nilai_guna": "Mengunci plastik sekali pakai agar tidak mencemari laut; furnitur eco-brick dapat dipakai fasilitas umum.",
        "alat": ["Botol plastik sejenis (600 ml atau 1,5 L)", "Tongkat kayu/bambu pemadat", "Gunting", "Timbangan"],
        "bahan": ["Plastik lunak bersih dan kering (kresek, sachet, bungkus snack)", "Botol plastik bekas"],
        "langkah": [
            "Cuci dan keringkan semua plastik lunak (plastik basah memicu jamur).",
            "Gunting plastik menjadi potongan kecil.",
            "Masukkan sedikit demi sedikit ke botol dan padatkan dengan tongkat sampai keras.",
            "Timbang: minimal ± 200 g untuk botol 600 ml atau ± 500 g untuk botol 1,5 L.",
            "Tutup rapat, lalu rangkai beberapa eco-brick menjadi kursi, meja, atau pagar taman.",
        ],
        "keamanan": "Jangan masukkan kaca, logam, atau sampah organik ke dalam eco-brick.",
    },
    {
        "nama": "Paving Block Campuran Cacahan Plastik", "jenis": "Anorganik", "kesulitan": "Sulit",
        "durasi": "± 2 minggu (termasuk curing)",
        "bahan_cocok": ["botol plastik", "gelas plastik", "tutup botol", "galon", "ember", "plastik", "botol"],
        "deskripsi": "Cacahan plastik keras dicampur ke adukan semen-pasir lalu dicetak menjadi paving block untuk jalan setapak.",
        "nilai_guna": "Bahan bangunan murah untuk taman, halaman gereja/masjid, atau jalan lingkungan.",
        "alat": ["Gunting/mesin pencacah plastik", "Cetakan paving block", "Sekop dan ember adukan", "Palu karet/alat press", "Terpal"],
        "bahan": ["Cacahan plastik keras bersih (PET/PP/HDPE)", "Semen", "Pasir", "Air"],
        "langkah": [
            "Cuci dan keringkan plastik keras, lalu cacah ± 0,5–1 cm.",
            "Campur semen : pasir : cacahan plastik (contoh 1 : 3 : 0,5), sesuaikan dari hasil uji coba.",
            "Tambahkan air sedikit demi sedikit sampai adukan lembap (dikepal tidak hancur).",
            "Masukkan ke cetakan, padatkan/press, lalu lepaskan dari cetakan.",
            "Rawat (curing) 7–14 hari dengan disiram dan ditutupi sebelum dipasang.",
        ],
        "keamanan": "Gunakan untuk area pejalan kaki kecuali sudah diuji kuat tekan; jangan membakar plastik.",
    },
    {
        "nama": "Pot Tanaman & Vertical Garden dari Botol Plastik", "jenis": "Anorganik", "kesulitan": "Mudah",
        "durasi": "± 1 hari",
        "bahan_cocok": ["botol plastik", "botol", "galon", "gelas plastik", "ember"],
        "deskripsi": "Botol plastik dipotong dan digantung berjajar menjadi kebun vertikal sayur atau tanaman hias.",
        "nilai_guna": "Kebun sayur pekarangan (cabai, kangkung, daun bawang) untuk ketahanan pangan keluarga.",
        "alat": ["Cutter/gunting", "Paku yang dipanaskan untuk melubangi", "Tali atau kawat", "Cat akrilik dan kuas"],
        "bahan": ["Botol plastik bekas", "Tanah atau cocopeat", "Bibit tanaman"],
        "langkah": [
            "Bersihkan botol dan lepaskan labelnya.",
            "Potong sisi botol (posisi tidur) atau bagian atas (posisi berdiri).",
            "Buat lubang drainase di bawah dan lubang tali di sisi botol.",
            "Hias dengan cat, lalu isi media tanam dan bibit.",
            "Gantung berjajar di dinding atau pagar dengan tali/kawat.",
        ],
        "keamanan": "Hati-hati dengan cutter dan paku panas.",
    },
    {
        "nama": "Tas & Dompet Anyaman Kemasan Sachet", "jenis": "Anorganik", "kesulitan": "Sedang",
        "durasi": "3–5 hari",
        "bahan_cocok": ["sachet", "kemasan", "bungkus kopi", "bungkus", "kantong plastik"],
        "deskripsi": "Kemasan sachet dilipat menjadi bilah lalu dianyam dan dijahit menjadi tas, dompet, atau tempat pensil.",
        "nilai_guna": "Produk kerajinan bernilai jual untuk UMKM/PKK.",
        "alat": ["Gunting", "Penggaris", "Jarum dan benang nilon/mesin jahit", "Lem tembak"],
        "bahan": ["Kemasan sachet sejenis yang sudah dicuci", "Kain furing (opsional)", "Ritsleting/kancing"],
        "langkah": [
            "Cuci kemasan sachet dengan sabun lalu keringkan.",
            "Potong dengan ukuran seragam dan lipat menjadi bilah memanjang.",
            "Anyam/sambung bilah menjadi lembaran.",
            "Jahit lembaran menjadi tas, dompet, atau tempat pensil, lalu pasang ritsleting.",
            "Rapikan tepi dan tambahkan furing agar kuat.",
        ],
        "keamanan": "Aman untuk pemula; hati-hati saat memakai jarum dan lem tembak.",
    },
    {
        "nama": "Lampion & Tempat Alat Tulis dari Kaleng", "jenis": "Anorganik", "kesulitan": "Mudah",
        "durasi": "± 1 hari",
        "bahan_cocok": ["kaleng", "aluminium", "logam", "besi"],
        "deskripsi": "Kaleng bekas dilubangi berpola dan dicat menjadi lampion, tempat alat tulis, atau pot mini.",
        "nilai_guna": "Dekorasi acara (Natal, pengucapan syukur) dan suvenir.",
        "alat": ["Paku dan palu", "Kikir/amplas", "Tang", "Cat semprot dan kuas", "Sarung tangan"],
        "bahan": ["Kaleng bersih", "Cat", "Kawat", "Lilin/LED kecil (opsional)"],
        "langkah": [
            "Cuci kaleng, lepas label, dan haluskan tepi tajam dengan kikir/amplas.",
            "Isi kaleng dengan air lalu bekukan agar tidak penyok saat dilubangi.",
            "Gambar pola dan lubangi dengan paku dan palu mengikuti pola.",
            "Cat kaleng dan pasang pegangan dari kawat.",
            "Masukkan lilin/LED sebagai lampion atau pakai sebagai tempat alat tulis.",
        ],
        "keamanan": "Tepi kaleng tajam: wajib sarung tangan.",
    },
    {
        "nama": "Lampu Hias & Vas dari Botol Kaca", "jenis": "Anorganik", "kesulitan": "Sedang",
        "durasi": "± 1 hari",
        "bahan_cocok": ["botol kaca", "kaca", "toples kaca"],
        "deskripsi": "Botol kaca dipotong rapi atau dibiarkan utuh, dihias, lalu diberi lampu LED menjadi lampu hias/vas.",
        "nilai_guna": "Dekorasi kafe, homestay, dan objek wisata.",
        "alat": ["Pemotong botol kaca (bottle cutter)", "Amplas air", "Sarung tangan dan kacamata pelindung", "Lem kaca"],
        "bahan": ["Botol kaca bersih", "Lampu LED kawat (fairy light)", "Tali rami", "Cat kaca"],
        "langkah": [
            "Cuci botol dan rendam air hangat untuk melepas label.",
            "Gores botol dengan bottle cutter, lalu siram air panas–dingin bergantian sampai terbelah (atau biarkan utuh untuk vas).",
            "Amplas tepi potongan dengan amplas basah sampai tidak tajam.",
            "Hias dengan cat kaca atau lilitan tali rami.",
            "Masukkan lampu LED kawat atau rangkai menjadi kap lampu gantung.",
        ],
        "keamanan": "Pecahan kaca berbahaya: pakai sarung tangan dan kacamata; jangan pakai botol retak.",
    },
    {
        "nama": "Kertas Daur Ulang (Handmade Paper)", "jenis": "Anorganik", "kesulitan": "Mudah",
        "durasi": "2–3 hari",
        "bahan_cocok": ["kertas", "koran", "kardus", "karton", "majalah"],
        "deskripsi": "Kertas bekas dihancurkan menjadi bubur lalu dicetak ulang menjadi kertas seni.",
        "nilai_guna": "Kartu ucapan, map, dan kemasan ramah lingkungan.",
        "alat": ["Ember", "Blender", "Screen/kasa berbingkai", "Kain dan spons", "Papan penjemur"],
        "bahan": ["Kertas bekas", "Air", "Lem kanji (opsional)", "Daun kering untuk hiasan"],
        "langkah": [
            "Sobek kertas kecil-kecil dan rendam semalam.",
            "Blender menjadi bubur kertas (pulp), tambahkan sedikit lem kanji.",
            "Tuang pulp ke bak air, celupkan screen dan angkat perlahan agar lapisan rata.",
            "Balik ke atas kain dan tekan dengan spons untuk membuang air.",
            "Jemur sampai kering lalu lepaskan lembaran kertasnya.",
        ],
        "keamanan": "Aman; jangan memakai kertas berlapis plastik/termal (struk).",
    },
    {
        "nama": "Keset & Tas dari Jaring Nelayan Bekas", "jenis": "Anorganik", "kesulitan": "Sedang",
        "durasi": "2–4 hari",
        "bahan_cocok": ["jaring", "jaring nelayan", "tali", "tali tambang"],
        "deskripsi": "Jaring dan tali nelayan bekas dipilin lalu dianyam menjadi keset, tas belanja, atau gantungan pot.",
        "nilai_guna": "Mengurangi ghost net di laut Bitung, Sangihe, dan Talaud; produk kerajinan pesisir.",
        "alat": ["Gunting tajam", "Jarum anyam besar/hakpen", "Sarung tangan", "Ember dan sikat"],
        "bahan": ["Jaring/tali bekas yang sudah dicuci", "Tali kur tambahan"],
        "langkah": [
            "Cuci jaring dengan sikat dan air tawar, lalu jemur sampai kering.",
            "Lepaskan pemberat dan potong menjadi tali-tali panjang.",
            "Pilin 2–3 helai menjadi tali yang lebih tebal.",
            "Anyam/rajut menjadi keset, tas belanja, atau gantungan pot.",
            "Kunci ujung anyaman dengan simpul mati dan rapikan.",
        ],
        "keamanan": "Periksa kail/pancing yang tersangkut sebelum mengolah jaring.",
    },
    {
        "nama": "Pot & Kursi Taman dari Ban Bekas", "jenis": "Anorganik", "kesulitan": "Sedang",
        "durasi": "1–2 hari",
        "bahan_cocok": ["ban", "ban bekas", "karet"],
        "deskripsi": "Ban bekas dipotong/dicat menjadi pot bunga atau ditumpuk menjadi kursi taman.",
        "nilai_guna": "Perabot taman murah untuk sekolah, posyandu, dan ruang publik.",
        "alat": ["Cutter tajam/gerinda", "Bor", "Cat minyak dan kuas", "Sekrup dan papan/triplek"],
        "bahan": ["Ban bekas", "Cat", "Papan bundar", "Busa dan kain (untuk kursi)"],
        "langkah": [
            "Cuci ban sampai bersih lalu keringkan.",
            "Potong pola bergerigi/bunga untuk pot, atau biarkan utuh untuk kursi.",
            "Cat dengan warna cerah.",
            "Pot: balik ban dan isi tanah. Kursi: tumpuk 2 ban, baut papan bundar di atas, lapisi busa dan kain.",
            "Lubangi bagian bawah agar air tidak menggenang, lalu letakkan di taman.",
        ],
        "keamanan": "Genangan air di ban bisa menjadi sarang nyamuk DBD — pastikan ada lubang pembuangan.",
    },
]

PERINGATAN_B3 = ("Limbah B3 (baterai, lampu, elektronik, oli, obat) TIDAK boleh diolah sendiri. "
                 "Simpan terpisah dan serahkan ke dropbox/fasilitas pengelolaan limbah B3 DLH setempat.")


def _ada_kata(kata: str, teks: str) -> bool:
    return re.search(rf"\b{re.escape(kata)}\b", teks) is not None


def deteksi_item_dalam_teks(teks: str) -> list[str]:
    """Cari kata kunci benda sampah (pencocokan terpanjang dulu agar 'botol kaca' tidak terbaca 'botol' + 'kaca')."""
    t = str(teks).lower()
    ditemukan: list[str] = []
    for kw in sorted(KAMUS_ITEM, key=len, reverse=True):
        if _ada_kata(kw, t) and not any(kw in d for d in ditemukan):
            ditemukan.append(kw)
    return ditemukan


def gabung_jenis(daftar_jenis) -> str | None:
    s = set(daftar_jenis)
    if ("Organik" in s or "Campuran" in s) and ("Anorganik" in s or "Campuran" in s):
        return "Campuran"
    if "Organik" in s:
        return "Organik"
    if "Anorganik" in s:
        return "Anorganik"
    return None


def klasifikasi_item(item: str) -> str:
    jenis = [KAMUS_ITEM[k] for k in deteksi_item_dalam_teks(item)]
    if "B3" in jenis:
        return "B3"
    return gabung_jenis(jenis) or "Tidak dikenal"


def parse_items(teks: str) -> list[str]:
    """'nasi, sayur, dan buah semangka' -> ['nasi', 'sayur', 'buah semangka']"""
    bagian = re.split(r",|;|\n|\+|&|/|\bdan\b|\bserta\b", str(teks).lower())
    return [b.strip(" .-()") for b in bagian if b.strip(" .-()")]


def cari_produk_relevan(items: list[str], jenis: str, k: int = 6) -> list[dict]:
    """Retrieval sederhana (RAG lokal): skor = jumlah kata kunci bahan yang cocok dengan input."""
    teks = " ".join(items).lower()
    kandidat = []
    for p in PRODUK_DAUR_ULANG:
        if jenis in ("Organik", "Anorganik") and p["jenis"] != jenis:
            continue
        skor = sum(_ada_kata(kw, teks) for kw in p["bahan_cocok"])
        kandidat.append((skor, p))
    kandidat.sort(key=lambda x: -x[0])
    cocok = [p for s, p in kandidat if s > 0]
    if jenis == "Campuran":  # pastikan ada wakil organik dan anorganik
        for j in ("Organik", "Anorganik"):
            if not any(p["jenis"] == j for p in cocok):
                cocok.append(next(p for s, p in kandidat if p["jenis"] == j))
    sisa = [p for s, p in kandidat if p not in cocok]
    return (cocok + sisa)[:k] if len(cocok) < 3 else cocok[:k]


def cari_produk_kb(nama: str) -> dict | None:
    n = str(nama).lower()
    for p in PRODUK_DAUR_ULANG:
        inti = p["nama"].lower().split("(")[0].strip()
        if inti in n or n in p["nama"].lower():
            return p
    # cocokkan kata pertama yang khas, mis. "maggot", "eco-enzyme", "briket"
    for p in PRODUK_DAUR_ULANG:
        kata = [w for w in re.findall(r"[a-z\-]+", p["nama"].lower()) if len(w) > 4]
        if any(w in n for w in kata[:2]):
            return p
    return None


def kb_ringkas(produk: list[dict] | None = None) -> str:
    produk = produk or PRODUK_DAUR_ULANG
    return "\n".join(f"- {p['nama']} ({p['jenis']}, {p['kesulitan']}): cocok untuk {', '.join(p['bahan_cocok'][:8])}. "
                     f"{p['deskripsi']}" for p in produk)


def kb_detail(p: dict) -> str:
    return (f"Produk: {p['nama']} ({p['jenis']})\nAlat: {'; '.join(p['alat'])}\nBahan: {'; '.join(p['bahan'])}\n"
            f"Langkah referensi: {' | '.join(p['langkah'])}\nKeamanan: {p['keamanan']}")


# =============================================================================
# 3. DATASET GENERATOR
# =============================================================================

LOKASI_SULUT = {
    "Kota Manado": ["Pasar Bersehati", "Pantai Malalayang", "Kawasan Boulevard", "Kelurahan Tikala",
                    "Kelurahan Wanea", "Kelurahan Bahu", "Pasar Karombasan", "Kelurahan Paal Dua"],
    "Kota Bitung": ["Pasar Winenet", "Pelabuhan Perikanan Bitung", "Kelurahan Girian", "Kelurahan Madidir",
                    "Pantai Aertembaga"],
    "Kota Tomohon": ["Pasar Beriman Tomohon", "Kelurahan Kakaskasen", "Kelurahan Matani", "Kelurahan Talete"],
    "Kota Kotamobagu": ["Pasar Serasi", "Kelurahan Mogolaing", "Kelurahan Gogagoman"],
    "Kabupaten Minahasa": ["tepi Danau Tondano", "Pasar Langowan", "Pasar Kawangkoan", "Kelurahan Tataaran"],
    "Kabupaten Minahasa Utara": ["Airmadidi", "Pantai Likupang", "Dimembe"],
    "Kabupaten Minahasa Selatan": ["Pasar Amurang", "Tumpaan"],
    "Kabupaten Minahasa Tenggara": ["Ratahan", "Belang"],
    "Kabupaten Bolaang Mongondow": ["Lolak", "Inobonto"],
    "Kabupaten Kepulauan Sangihe": ["Pelabuhan Tahuna", "Kota Tahuna"],
    "Kabupaten Kepulauan Talaud": ["Melonguane"],
    "Kabupaten Kepulauan Siau Tagulandang Biaro": ["Ulu Siau"],
    "Kabupaten Bolaang Mongondow Utara": ["Boroko"],
    "Kabupaten Bolaang Mongondow Selatan": ["Molibagu"],
    "Kabupaten Bolaang Mongondow Timur": ["Tutuyan"],
}
BOBOT_WILAYAH = [30, 12, 8, 7, 9, 8, 5, 3, 4, 4, 2, 2, 2, 2, 2]

# Titik perkiraan ibu kota kab/kota (untuk peta, bukan koordinat presisi).
KOORDINAT = {
    "Kota Manado": (1.4748, 124.8421), "Kota Bitung": (1.4404, 125.1870), "Kota Tomohon": (1.3233, 124.8384),
    "Kota Kotamobagu": (0.7333, 124.3167), "Kabupaten Minahasa": (1.3045, 124.9126),
    "Kabupaten Minahasa Utara": (1.4214, 124.9784), "Kabupaten Minahasa Selatan": (1.1890, 124.5808),
    "Kabupaten Minahasa Tenggara": (1.0750, 124.8000), "Kabupaten Bolaang Mongondow": (0.8960, 124.0450),
    "Kabupaten Kepulauan Sangihe": (3.6067, 125.4930), "Kabupaten Kepulauan Talaud": (4.0094, 126.6744),
    "Kabupaten Kepulauan Siau Tagulandang Biaro": (2.7400, 125.4000),
    "Kabupaten Bolaang Mongondow Utara": (0.8930, 123.3450),
    "Kabupaten Bolaang Mongondow Selatan": (0.4170, 124.0700),
    "Kabupaten Bolaang Mongondow Timur": (0.7890, 124.6700),
}

ITEM_ORGANIK = ["sisa nasi", "sayur busuk", "kulit pisang", "kulit semangka", "sisa ikan", "tulang ikan",
                "kepala ikan cakalang", "ampas kelapa", "sabut kelapa", "tempurung kelapa", "daun kering",
                "kulit jagung", "tongkol jagung", "ampas tahu", "kulit buah langsat", "kulit durian",
                "sisa makanan pesta", "minyak jelantah", "kulit nanas", "sisa sayur kangkung", "batang pisang",
                "ampas kopi"]
ITEM_ANORGANIK = ["botol plastik", "kantong kresek", "gelas plastik air mineral", "sedotan plastik",
                  "kotak styrofoam", "kemasan sachet", "kaleng minuman", "botol kaca", "kardus bekas",
                  "kertas koran", "jaring nelayan bekas", "ban bekas", "tutup botol", "bungkus mi instan",
                  "galon pecah"]

# (kategori_teks, masalah, sentimen, template, jenis_tetap)
# jenis_tetap=None -> jenis diacak dan {items} diisi sesuai jenis.
TEMPLATE_LAPORAN = [
    # --- normal: negatif
    ("normal", "Layanan Pengangkutan", "Negatif", "Sampah {items} di {lokasi} sudah {hari} hari tidak diangkut, baunya sampai masuk ke rumah.", None),
    ("normal", "Layanan Pengangkutan", "Negatif", "Truk sampah tidak lewat lagi di {lokasi} sejak minggu lalu, {items} menumpuk di depan lorong.", None),
    ("normal", "Layanan Pengangkutan", "Negatif", "Jadwal angkut sampah di {lokasi} tidak jelas, {items} sudah dikerubungi lalat.", None),
    ("normal", "Fasilitas & TPS", "Negatif", "TPS di {lokasi} sudah penuh dan meluber ke jalan, isinya {items}.", None),
    ("normal", "Fasilitas & TPS", "Negatif", "Tidak ada tempat sampah terpilah di {lokasi}, akhirnya {items} dicampur jadi satu.", None),
    ("normal", "Fasilitas & TPS", "Negatif", "Kontainer sampah di {lokasi} rusak, {items} berserakan dibongkar anjing.", None),
    ("normal", "Perilaku Membuang Sampah", "Negatif", "Masih banyak warga buang {items} ke selokan di {lokasi}, kalau hujan langsung banjir.", None),
    ("normal", "Perilaku Membuang Sampah", "Negatif", "Pengunjung {lokasi} seenaknya buang {items} di pinggir jalan.", None),
    ("normal", "Perilaku Membuang Sampah", "Negatif", "Ada orang buang {items} dari atas motor di {lokasi}, benar-benar tidak tahu malu.", None),
    ("normal", "Kebersihan Lingkungan", "Negatif", "Sungai di dekat {lokasi} penuh {items}, airnya jadi hitam dan bau.", None),
    ("normal", "Kebersihan Lingkungan", "Negatif", "Bau busuk dari {items} di {lokasi} bikin anak-anak batuk dan mual.", None),
    # --- normal: netral
    ("normal", "Partisipasi & Edukasi", "Netral", "Mau tanya, {items} sebaiknya diolah jadi apa ya? Di {lokasi} belum ada sosialisasi pemilahan.", None),
    ("normal", "Fasilitas & TPS", "Netral", "Apakah bank sampah di {lokasi} menerima {items}? Mohon infonya.", None),
    ("normal", "Layanan Pengangkutan", "Netral", "Hari ini di {lokasi} ada {items} sekitar dua karung, jadwal truk berikutnya kapan ya?", None),
    ("normal", "Partisipasi & Edukasi", "Netral", "Info untuk warga {lokasi}: pengumpulan {items} dilakukan setiap hari Sabtu di balai kelurahan.", None),
    # --- normal: positif
    ("normal", "Fasilitas & TPS", "Positif", "Senang sekali bank sampah di {lokasi} sekarang menerima {items}, lingkungan jadi lebih bersih.", None),
    ("normal", "Layanan Pengangkutan", "Positif", "Terima kasih petugas DLH, {items} di {lokasi} sudah diangkut cepat pagi ini.", None),
    ("normal", "Partisipasi & Edukasi", "Positif", "Ibu-ibu PKK di {lokasi} mulai mengolah {items} jadi produk bernilai jual, keren!", None),
    ("normal", "Kebersihan Lingkungan", "Positif", "Setelah kerja bakti, {lokasi} bebas dari {items}. Semoga bisa dipertahankan.", None),
    # --- dialek Manado (uji multilingual)
    ("dialek_manado", "Layanan Pengangkutan", "Negatif", "Kiapa so {hari} hari ini sampah {items} di {lokasi} nyanda ta angka? Bau skali!", None),
    ("dialek_manado", "Perilaku Membuang Sampah", "Negatif", "Dorang di {lokasi} ja buang {items} di got, kong kalu ujang langsung banjir.", None),
    ("dialek_manado", "Fasilitas & TPS", "Positif", "Mantap no, sekarang di {lokasi} so ada tampa sampah pilah, {items} so nyanda baserak.", None),
    ("dialek_manado", "Partisipasi & Edukasi", "Netral", "Torang mo tanya, {items} itu bole bekeng apa? Di {lokasi} banyak skali.", None),
    ("dialek_manado", "Kebersihan Lingkungan", "Negatif", "Pante di {lokasi} so penuh deng {items}, malo kalu ada turis datang.", None),
    # --- sarkasme (label benar: Negatif)
    ("sarkasme", "Layanan Pengangkutan", "Negatif", "Wah hebat sekali, {items} di {lokasi} sudah {hari} hari jadi \"monumen\" baru. Terima kasih banyak atas pelayanannya!", None),
    ("sarkasme", "Fasilitas & TPS", "Negatif", "Mantap, sekarang {lokasi} punya objek wisata baru: gunung {items}. Pemandangannya luar biasa.", None),
    ("sarkasme", "Perilaku Membuang Sampah", "Negatif", "Bagus sekali warga {lokasi}, selokan dijadikan tempat koleksi {items}. Kreatif!", None),
    ("sarkasme", "Layanan Pengangkutan", "Negatif", "Salut buat truk sampah {lokasi}, saking jarangnya datang sampai {items} di sini bisa tumbuh jamur.", None),
    ("sarkasme", "Kebersihan Lingkungan", "Negatif", "Sungai di {lokasi} memang paling indah, airnya dihiasi {items} warna-warni.", None),
    # --- ambigu
    ("ambigu", "Layanan Pengangkutan", "Netral", "Sampah di {lokasi} ya begitulah, kadang diangkut kadang tidak.", "Campuran"),
    ("ambigu", "Fasilitas & TPS", "Negatif", "Ada kantong hitam besar di {lokasi}, entah isinya apa, sudah {hari} hari di situ.", "Campuran"),
    ("ambigu", "Kebersihan Lingkungan", "Netral", "Sisa-sisa acara pesta semalam di {lokasi} masih ada, nanti juga diangkut kok.", "Campuran"),
    ("ambigu", "Layanan Pengangkutan", "Netral", "Sampahnya di {lokasi} sudah diangkut, tapi ya gitu deh.", "Campuran"),
    ("ambigu", "Kebersihan Lingkungan", "Negatif", "Barang bekas di {lokasi} makin banyak saja.", "Anorganik"),
    ("ambigu", "Kebersihan Lingkungan", "Netral", "Ada yang menumpuk {items} di lahan kosong {lokasi}, mungkin mau diolah, mungkin juga dibuang.", None),
    ("ambigu", "Perilaku Membuang Sampah", "Netral", "Tetangga di {lokasi} rajin bakar {items} tiap sore, entah baik atau tidak.", None),
    # --- sentimen campuran (label = sentimen dominan)
    ("sentimen_campuran", "Perilaku Membuang Sampah", "Negatif", "Petugas kebersihan di {lokasi} rajin menyapu, tapi warganya masih buang {items} sembarangan.", None),
    ("sentimen_campuran", "Fasilitas & TPS", "Negatif", "Bank sampah di {lokasi} bagus, sayang cuma buka sebulan sekali jadi {items} menumpuk di rumah.", None),
    ("sentimen_campuran", "Layanan Pengangkutan", "Negatif", "Truk sampah sudah datang tepat waktu, tapi {items} tetap tercecer di jalan waktu diangkut.", None),
    ("sentimen_campuran", "Partisipasi & Edukasi", "Negatif", "Programnya bagus, tapi pelaksanaan di {lokasi} nol besar, {items} masih di mana-mana.", None),
    ("sentimen_campuran", "Kebersihan Lingkungan", "Positif", "Walaupun masih ada sedikit {items}, {lokasi} sekarang jauh lebih bersih dibanding tahun lalu.", None),
]
BOBOT_KATEGORI = {"normal": 0.58, "dialek_manado": 0.12, "sarkasme": 0.10, "ambigu": 0.08, "sentimen_campuran": 0.12}
SUMBER_SIMULASI = ["Facebook Grup Warga", "Instagram", "Aplikasi Pengaduan (simulasi)", "Komentar Berita Online",
                   "WhatsApp Grup RT"]
EMOJI_SENTIMEN = {"Negatif": ["😡", "🤮", "😤"], "Positif": ["👍", "🙏", "😊"], "Netral": ["🤔", "🙏"],
                  "sarkasme": ["😂", "👏", "🙂"]}
SINGKATAN = {"sudah": "sdh", "tidak": "tdk", "yang": "yg", "dengan": "dgn", "untuk": "utk"}
# Bobot bulan Nov 2025–Okt 2026: lonjakan Desember–Januari (musim hujan + Natal/Tahun Baru).
BULAN_BOBOT = {(2025, 11): 1.0, (2025, 12): 1.8, (2026, 1): 1.6, (2026, 2): 1.1, (2026, 3): 0.9, (2026, 4): 0.9,
               (2026, 5): 0.8, (2026, 6): 0.8, (2026, 7): 0.8, (2026, 8): 0.9, (2026, 9): 1.0, (2026, 10): 1.0}


def _gabung_items(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " dan " + items[-1]


def _pilih_items(rng: random.Random, jenis: str) -> list[str]:
    if jenis == "Organik":
        return rng.sample(ITEM_ORGANIK, rng.randint(1, 3))
    if jenis == "Anorganik":
        return rng.sample(ITEM_ANORGANIK, rng.randint(1, 3))
    items = rng.sample(ITEM_ORGANIK, rng.randint(1, 2)) + rng.sample(ITEM_ANORGANIK, rng.randint(1, 2))
    rng.shuffle(items)
    return items


def _tambah_noise(rng: random.Random, teks: str, sentimen: str, kategori: str) -> str:
    """Variasi gaya tulis warga di media sosial: singkatan, emoji, huruf kapital, tanda seru."""
    if rng.random() < 0.25:
        for kata, singkat in SINGKATAN.items():
            teks = re.sub(rf"\b{kata}\b", singkat, teks)
    if rng.random() < 0.30:
        teks += " " + rng.choice(EMOJI_SENTIMEN["sarkasme" if kategori == "sarkasme" else sentimen])
    if sentimen == "Negatif" and kategori == "normal" and rng.random() < 0.06:
        teks = teks.upper()
    if rng.random() < 0.08:
        teks = teks.rstrip(".") + "!!!"
    return teks


def _tanggal_acak(rng: random.Random) -> date:
    (tahun, bulan), = rng.choices(list(BULAN_BOBOT), weights=list(BULAN_BOBOT.values()))
    awal = date(tahun, bulan, 1)
    akhir = date(tahun + (bulan == 12), bulan % 12 + 1, 1) - timedelta(days=1)
    if (tahun, bulan) == (2026, 10):
        akhir = date(2026, 10, 25)
    return awal + timedelta(days=rng.randint(0, (akhir - awal).days))


def generate_dataset(n: int = 600, seed: int = 42, path: Path = DATASET_PATH) -> pd.DataFrame:
    """Membuat dataset laporan warga sintetis (dummy) beserta label manual untuk evaluasi."""
    rng = random.Random(seed)
    per_kategori = {k: [t for t in TEMPLATE_LAPORAN if t[0] == k] for k in BOBOT_KATEGORI}
    wilayah = list(LOKASI_SULUT)
    baris, teks_terpakai = [], set()
    while len(baris) < n:
        kategori = rng.choices(list(BOBOT_KATEGORI), weights=list(BOBOT_KATEGORI.values()))[0]
        _, masalah, sentimen, template, jenis_tetap = rng.choice(per_kategori[kategori])
        jenis = jenis_tetap or rng.choices(JENIS, weights=[0.38, 0.37, 0.25])[0]
        kab = rng.choices(wilayah, weights=BOBOT_WILAYAH)[0]
        lokasi = rng.choice(LOKASI_SULUT[kab])
        teks = template.format(items=_gabung_items(_pilih_items(rng, jenis)), lokasi=lokasi,
                               hari=rng.choice([2, 3, 4, 5, 7, 10]))
        teks = _tambah_noise(rng, teks, sentimen, kategori)
        if teks in teks_terpakai:
            continue
        teks_terpakai.add(teks)
        baris.append({
            "Tanggal": _tanggal_acak(rng).isoformat(), "Kabupaten_Kota": kab, "Lokasi": lokasi,
            "Sumber_Simulasi": rng.choice(SUMBER_SIMULASI), "Teks_Laporan": teks,
            "Label_Jenis_Sampah": jenis, "Label_Sentimen": sentimen, "Label_Masalah": masalah,
            "Kategori_Teks": kategori,
        })
    df = pd.DataFrame(baris).sort_values("Tanggal", kind="stable").reset_index(drop=True)
    df.insert(0, "ID_Laporan", [f"SB-{i:04d}" for i in range(1, len(df) + 1)])
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


# =============================================================================
# 4. PREPROCESSING
# =============================================================================

NORMALISASI = {
    # singkatan umum
    "sdh": "sudah", "udh": "sudah", "tdk": "tidak", "gak": "tidak", "nggak": "tidak", "yg": "yang",
    "dgn": "dengan", "utk": "untuk", "krn": "karena", "tp": "tapi", "bgt": "banget",
    # dialek Manado -> bahasa Indonesia
    "nyanda": "tidak", "so": "sudah", "dorang": "mereka", "torang": "kami", "kiapa": "kenapa",
    "skali": "sekali", "bekeng": "bikin", "kong": "lalu", "kalu": "kalau", "ujang": "hujan",
    "baserak": "berserakan", "tampa": "tempat", "bole": "boleh", "mo": "mau", "ja": "sering",
    "deng": "dengan", "pante": "pantai", "malo": "malu", "ta angka": "diangkut",
}
EMOJI_TOKEN = {"😡": "[emoji_marah]", "😤": "[emoji_kesal]", "🤮": "[emoji_jijik]", "👍": "[emoji_jempol]",
               "🙏": "[emoji_mohon]", "😊": "[emoji_senyum]", "🙂": "[emoji_senyum_tipis]",
               "😂": "[emoji_tertawa]", "👏": "[emoji_tepuk_tangan]", "🤔": "[emoji_berpikir]"}


def bersihkan_teks(teks: str) -> str:
    """Normalisasi teks: URL dibuang, emoji -> token makna, singkatan & dialek Manado -> baku.
    Huruf kapital dan tanda kutip dipertahankan karena bisa menjadi petunjuk emosi/sarkasme."""
    t = str(teks)
    for emoji, token in EMOJI_TOKEN.items():
        t = t.replace(emoji, f" {token} ")
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"!{2,}", "!", t)
    for kata, baku in sorted(NORMALISASI.items(), key=lambda x: -len(x[0])):
        t = re.sub(rf"\b{re.escape(kata)}\b", baku, t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def preprocess_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    awal = len(df)
    df = df.dropna(subset=["Teks_Laporan"]).copy()
    df["Teks_Laporan"] = df["Teks_Laporan"].astype(str).str.strip()
    df = df[df["Teks_Laporan"].str.len() > 0]
    kosong = awal - len(df)
    df = df.drop_duplicates(subset=["Teks_Laporan"])
    duplikat = awal - kosong - len(df)
    df["Teks_Bersih"] = df["Teks_Laporan"].map(bersihkan_teks)
    df["Jumlah_Kata"] = df["Teks_Bersih"].str.split().str.len()
    info = {"baris_awal": awal, "baris_kosong_dibuang": kosong, "duplikat_dibuang": duplikat, "baris_akhir": len(df)}
    return df.reset_index(drop=True), info


# =============================================================================
# 5. PROMPT ENGINEERING
# =============================================================================

SYSTEM_PROMPT = """Kamu adalah SI-BERSIH (Sistem Intelijen Berbasis Evaluasi Respons Sampah), asisten AI
untuk Dinas Lingkungan Hidup di Provinsi Sulawesi Utara. Kamu memahami bahasa Indonesia dan dialek Manado
(nyanda=tidak, so=sudah, dorang=mereka, torang=kami, bekeng=bikin, skali=sekali, baserak=berserakan).

ATURAN KETAT:
1. Jenis sampah hanya salah satu dari: Organik, Anorganik, Campuran.
2. DILARANG merekomendasikan pupuk kompos / kompos biasa / pengomposan dalam bentuk apa pun.
   Untuk organik gunakan olahan inovatif: pakan maggot BSF, eco-enzyme, biogas, briket arang,
   sabun/lilin minyak jelantah, cocopeat, silase pakan ikan, kerajinan kulit jagung.
3. Untuk anorganik utamakan: eco-brick, paving block plastik, pot/vertical garden, anyaman sachet,
   kerajinan kaleng/kaca, kertas daur ulang, kerajinan jaring nelayan, perabot ban bekas.
4. Limbah B3 (baterai, lampu, elektronik, oli, obat) tidak boleh diolah sendiri; arahkan ke fasilitas B3.
5. Setiap keputusan wajib memiliki confidence (0-100) dan confidence_reasoning yang menyebut kata kunci,
   konteks kalimat, dan sumber keraguan. Turunkan confidence bila teks ambigu.
6. Waspadai sarkasme: pujian berlebihan + fakta negatif, tanda kutip ironis, emoji tertawa/tepuk tangan.
   Sarkasme tentang kondisi sampah = sentimen Negatif.
7. Jawab HANYA dengan JSON valid sesuai skema yang diminta, tanpa teks lain."""

SYSTEM_PROMPT_V1 = "Kamu adalah asisten analisis teks. Jawab dalam format JSON."

SKEMA_LAPORAN = """{"hasil": [{
  "id": "id laporan",
  "jenis_sampah": "Organik|Anorganik|Campuran",
  "item_terdeteksi": ["benda sampah yang disebut"],
  "sentimen": "Positif|Negatif|Netral",
  "masalah_utama": "<<MASALAH>>",
  "sarkasme": true,
  "tingkat_ambiguitas": "Rendah|Sedang|Tinggi",
  "kata_kunci_dominan": ["2-5 kata/frasa paling berpengaruh"],
  "rekomendasi_produk": ["minimal 2 nama produk olahan inovatif"],
  "langkah_pembuatan": ["3-5 langkah singkat untuk rekomendasi pertama"],
  "confidence": 0,
  "confidence_reasoning": "1-3 kalimat alasan keputusan"
}]}""".replace("<<MASALAH>>", "|".join(MASALAH))

PROMPT_ANALISIS_V2 = """TUGAS: Analisis setiap laporan warga Sulawesi Utara tentang sampah berikut.
Langkah berpikir untuk setiap laporan:
1. Kenali benda sampah yang disebut -> tentukan jenis_sampah (Campuran bila ada organik DAN anorganik,
   atau bila isi sampah tidak jelas/kemungkinan tercampur).
2. Baca konteks seluruh kalimat (bukan kata per kata) -> tentukan sentimen terhadap kondisi/layanan sampah.
   Kalimat "X bagus, tapi Y buruk" -> pilih sentimen yang dominan (biasanya bagian setelah "tapi").
3. Tentukan masalah_utama (pilih satu): <<MASALAH>>.
4. Nilai sarkasme dan tingkat_ambiguitas.
5. Pilih minimal 2 rekomendasi_produk dari KNOWLEDGE BASE yang cocok dengan item (DILARANG kompos),
   lalu tulis 3-5 langkah_pembuatan untuk rekomendasi pertama.
6. Beri confidence 0-100 dan confidence_reasoning.

KNOWLEDGE BASE PRODUK DAUR ULANG (RAG):
<<KB>>

CONTOH (few-shot):
Laporan: {"id": "C1", "teks": "Wah keren, kresek dan sachet di depan pasar sudah jadi \\"gunung\\" baru [emoji_tertawa]"}
Jawaban: {"id": "C1", "jenis_sampah": "Anorganik", "item_terdeteksi": ["kresek", "sachet"], "sentimen": "Negatif",
"masalah_utama": "Layanan Pengangkutan", "sarkasme": true, "tingkat_ambiguitas": "Sedang",
"kata_kunci_dominan": ["wah keren", "\\"gunung\\" baru", "emoji_tertawa"],
"rekomendasi_produk": ["Eco-brick", "Tas & Dompet Anyaman Kemasan Sachet"],
"langkah_pembuatan": ["Cuci dan keringkan plastik", "Gunting kecil-kecil", "Padatkan ke botol dengan tongkat", "Timbang minimal 200 g/600 ml", "Rangkai menjadi kursi"],
"confidence": 72, "confidence_reasoning": "Kata 'keren' tampak positif, tetapi tanda kutip pada 'gunung' dan emoji tertawa menunjukkan sindiran atas sampah yang menumpuk, jadi Negatif. Confidence sedang karena sarkasme bergantung pada interpretasi."}

Laporan: {"id": "C2", "teks": "Petugas rajin menyapu, tapi warga masih buang sisa sayur dan botol plastik ke got."}
Jawaban: {"id": "C2", "jenis_sampah": "Campuran", "item_terdeteksi": ["sisa sayur", "botol plastik"], "sentimen": "Negatif",
"masalah_utama": "Perilaku Membuang Sampah", "sarkasme": false, "tingkat_ambiguitas": "Sedang",
"kata_kunci_dominan": ["rajin", "tapi", "buang ke got"],
"rekomendasi_produk": ["Pakan Maggot BSF (Black Soldier Fly)", "Pot Tanaman & Vertical Garden dari Botol Plastik"],
"langkah_pembuatan": ["Pilah sisa sayur dan botol plastik", "Cacah sisa sayur untuk biopond maggot", "Potong botol menjadi pot", "Isi media tanam dan bibit"],
"confidence": 80, "confidence_reasoning": "Ada pujian untuk petugas, tetapi fokus keluhan setelah 'tapi' adalah perilaku warga membuang sampah ke got. Item organik dan anorganik sama-sama disebut sehingga Campuran."}

LAPORAN YANG DIANALISIS (JSON):
<<LAPORAN>>

Kembalikan satu objek per laporan dengan id yang sama, format:
<<SKEMA>>""".replace("<<MASALAH>>", ", ".join(MASALAH))

PROMPT_ANALISIS_V1 = """Analisis laporan warga tentang sampah berikut. Tentukan jenis sampah, sentimen,
masalah utama, rekomendasi produk daur ulang, langkah pembuatan, dan confidence.

Laporan (JSON):
<<LAPORAN>>

Format jawaban JSON:
<<SKEMA>>"""

PROMPT_REKOMENDASI = """Pengguna memilih jenis sampah: <<JENIS>>
Pengguna menyebutkan isi sampahnya: "<<ITEMS>>"
Pra-klasifikasi kamus lokal SI-BERSIH (boleh kamu koreksi): <<PRA>>

KNOWLEDGE BASE — produk paling relevan hasil retrieval (RAG):
<<KB>>

TUGAS:
1. Validasi apakah jenis pilihan pengguna sesuai dengan isi sampahnya. Bila ada item yang tidak sesuai
   (mis. botol plastik pada pilihan Organik), jelaskan dan sebutkan jenis yang benar.
2. Klasifikasikan setiap item (Organik/Anorganik/B3).
3. Berikan 3-5 rekomendasi produk daur ulang inovatif yang bisa dibuat dari item tersebut, utamakan dari
   knowledge base. DILARANG kompos. Untuk Campuran, sertakan minimal 1 produk organik dan 1 anorganik.
4. Beri confidence dan confidence_reasoning (kata kunci apa yang menentukan, apa yang membuat ragu).

Format JSON:
{"validasi_jenis": {"jenis_dipilih": "...", "jenis_terdeteksi": "Organik|Anorganik|Campuran", "sesuai": true, "catatan": "..."},
 "klasifikasi_item": [{"item": "...", "jenis": "Organik|Anorganik|B3"}],
 "rekomendasi": [{"nama_produk": "...", "jenis": "Organik|Anorganik", "deskripsi": "...", "bahan_dari_input": ["..."],
   "tingkat_kesulitan": "Mudah|Sedang|Sulit", "estimasi_waktu": "...", "nilai_guna": "...", "alasan_pemilihan": "..."}],
 "kata_kunci_dominan": ["..."], "confidence": 0, "confidence_reasoning": "..."}"""

PROMPT_PANDUAN = """Pengguna memilih membuat produk: <<PRODUK>>
Jenis sampah: <<JENIS>>. Bahan sampah yang dimiliki pengguna: <<ITEMS>>
Referensi knowledge base: <<KB>>

Buat panduan praktis untuk rumah tangga di Sulawesi Utara:
- alat: 4-8 alat yang dibutuhkan (mudah didapat)
- bahan: bahan utama + bahan pendukung
- langkah: WAJIB 3-5 langkah, jelas dan berurutan
- estimasi_waktu, tips_keamanan (1-3), potensi_nilai_ekonomi
- confidence dan confidence_reasoning (seberapa yakin panduan ini cocok dengan bahan pengguna)
DILARANG mengarahkan menjadi kompos.

Format JSON:
{"nama_produk": "...", "alat": ["..."], "bahan": ["..."], "langkah": ["..."], "estimasi_waktu": "...",
 "tips_keamanan": ["..."], "potensi_nilai_ekonomi": "...", "confidence": 0, "confidence_reasoning": "..."}"""

PROMPT_KEBIJAKAN = """Berikut ringkasan statistik hasil analisis laporan warga tentang sampah di Sulawesi Utara:
<<STATISTIK>>

Sebagai analis kebijakan Dinas Lingkungan Hidup, susun 5 rekomendasi prioritas berbasis data untuk
pemerintah kab/kota, bank sampah, PKK, sekolah, dan komunitas gereja/masjid. Setiap rekomendasi harus
merujuk angka pada statistik. Jangan merekomendasikan kompos.

Format JSON:
{"ringkasan": "2-3 kalimat temuan utama",
 "rekomendasi": [{"prioritas": 1, "wilayah": "...", "masalah": "...", "tindakan": "...", "pihak_terlibat": "...",
   "indikator_keberhasilan": "...", "dasar_data": "..."}]}"""


def isi_prompt(template: str, **nilai) -> str:
    for kunci, isi in nilai.items():
        template = template.replace(f"<<{kunci.upper()}>>", str(isi))
    return template


# =============================================================================
# 6. INTEGRASI LLM
# =============================================================================


def ekstrak_json(teks: str):
    """Ambil JSON dari respons LLM (tahan terhadap ```json ... ``` atau teks tambahan)."""
    teks = re.sub(r"^```(?:json)?|```$", "", str(teks).strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(teks)
    except json.JSONDecodeError:
        awal = min([i for i in (teks.find("{"), teks.find("[")) if i >= 0], default=-1)
        akhir = max(teks.rfind("}"), teks.rfind("]"))
        if awal < 0 or akhir <= awal:
            raise
        return json.loads(teks[awal:akhir + 1])


def _panggil_gemini(cfg: LLMConfig, system: str, prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cfg.api_key)
    resp = client.models.generate_content(
        model=cfg.model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system, temperature=cfg.temperature,
                                           response_mime_type="application/json"),
    )
    return resp.text


def _panggil_openai_compatible(cfg: LLMConfig, system: str, prompt: str) -> str:
    """OpenAI, DeepSeek, dan Grok memakai format API yang sama (hanya base_url berbeda)."""
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    resp = client.chat.completions.create(
        model=cfg.model, temperature=cfg.temperature, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def panggil_llm(cfg: LLMConfig, system: str, prompt: str, percobaan: int = 3):
    """Panggil LLM dan kembalikan JSON hasil parsing. Retry dengan backoff untuk rate limit (429)."""
    fungsi = _panggil_gemini if cfg.provider == "Gemini" else _panggil_openai_compatible
    error_terakhir = None
    for i in range(percobaan):
        try:
            return ekstrak_json(fungsi(cfg, system, prompt))
        except Exception as e:  # noqa: BLE001 - error SDK berbeda-beda tiap provider
            error_terakhir = e
            pesan = str(e).lower()
            if any(s in pesan for s in ("api key", "api_key", "401", "403", "permission", "not found", "404")):
                break  # kesalahan konfigurasi: tidak ada gunanya mengulang
            time.sleep((20 if "429" in pesan or "quota" in pesan or "rate" in pesan else 3) * (i + 1))
    raise RuntimeError(f"Gagal memanggil {cfg.provider} ({cfg.model}): {error_terakhir}")


# =============================================================================
# 7. CORE AI LOGIC + GUARDRAIL + BASELINE RULE-BASED
# =============================================================================

POLA_TERLARANG = re.compile(r"kompos|composting|pengomposan", re.IGNORECASE)

KATA_POSITIF = ["senang", "terima kasih", "bagus", "bersih", "mantap", "keren", "hebat", "rajin", "cepat",
                "salut", "puas", "membantu", "rapi", "indah", "kreatif", "luar biasa", "semoga",
                "[emoji_jempol]", "[emoji_senyum]"]
KATA_NEGATIF = ["bau", "menumpuk", "tidak diangkut", "penuh", "meluber", "berserakan", "kotor", "banjir",
                "jorok", "rusak", "lalat", "busuk", "sembarangan", "kecewa", "parah", "tidak jelas", "hitam",
                "batuk", "mual", "tidak tahu malu", "nol besar", "jarang", "tercecer", "malu", "seenaknya",
                "tidak lewat", "[emoji_marah]", "[emoji_jijik]", "[emoji_kesal]"]
KATA_NETRAL = ["tanya", "mohon info", "infonya", "kapan", "apakah", "info untuk", "sebaiknya"]
KATA_MASALAH = {
    "Layanan Pengangkutan": ["truk", "diangkut", "angkut", "jadwal", "pengangkutan", "petugas dlh", "pelayanan"],
    "Fasilitas & TPS": ["tps", "tempat sampah", "kontainer", "bank sampah", "fasilitas", "objek wisata"],
    "Perilaku Membuang Sampah": ["buang", "selokan", "got", "sembarangan", "seenaknya", "bakar", "warga"],
    "Kebersihan Lingkungan": ["sungai", "bau", "kotor", "pantai", "air", "lalat", "bersih", "lahan kosong"],
    "Partisipasi & Edukasi": ["tanya", "sosialisasi", "pemilahan", "diolah", "pkk", "program", "bikin apa",
                              "info untuk"],
}


def _sebagai_list(nilai) -> list:
    if nilai is None or (isinstance(nilai, float) and pd.isna(nilai)):
        return []
    if isinstance(nilai, list):
        return [x for x in nilai if str(x).strip()]
    return [s.strip() for s in re.split(r"\s*\|\s*|\n|;", str(nilai)) if s.strip()]


def _pilih_opsi(nilai, opsi: list[str], default: str) -> str:
    s = str(nilai or "").strip().lower()
    for o in opsi:
        if s == o.lower():
            return o
    for o in sorted(opsi, key=len, reverse=True):  # "anorganik" dicek sebelum "organik"
        if o.lower() in s or o.lower().split()[0] in s:
            return o
    return default


def _confidence(nilai) -> int:
    try:
        x = float(str(nilai).replace("%", "").strip())
    except ValueError:
        return 50
    if 0 < x <= 1:
        x *= 100
    return int(max(0, min(100, round(x))))


def _rapikan_langkah(langkah: list[str]) -> list[str]:
    langkah = [re.sub(r"^\s*(langkah\s*)?\d+[.):]\s*", "", str(s), flags=re.IGNORECASE).strip() for s in langkah]
    langkah = [s for s in langkah if s]
    if len(langkah) > 5:  # aturan 3-5 langkah: langkah berlebih digabung ke langkah ke-5
        langkah = langkah[:4] + [" ".join(langkah[4:])]
    return langkah


def guardrail_produk(nama_produk: list[str], items: list[str], jenis: str, minimal: int) -> tuple[list[str], list[str]]:
    """Buang rekomendasi terlarang (kompos) dan lengkapi dari knowledge base sampai jumlah minimal."""
    catatan = []
    bersih = []
    for n in nama_produk:
        if POLA_TERLARANG.search(str(n)):
            catatan.append(f"Guardrail menghapus '{n}' (melanggar aturan: dilarang kompos).")
        elif str(n).strip() and str(n) not in bersih:
            bersih.append(str(n).strip())
    if len(bersih) < minimal:
        for p in cari_produk_relevan(items, jenis):
            if len(bersih) >= minimal:
                break
            if not any(cari_produk_kb(b) is p for b in bersih):
                bersih.append(p["nama"])
                catatan.append(f"Guardrail menambahkan '{p['nama']}' dari knowledge base.")
    return bersih, catatan


# ---------- baseline rule-based (dipakai bila API key kosong / LLM gagal) ----------

def analisis_offline(teks: str, id_laporan: str = "-") -> dict:
    t = bersihkan_teks(teks).lower()
    item = deteksi_item_dalam_teks(t)
    jenis = gabung_jenis(KAMUS_ITEM[k] for k in item if KAMUS_ITEM[k] != "B3") or "Campuran"
    pos = [k for k in KATA_POSITIF if k in t]
    neg = [k for k in KATA_NEGATIF if _ada_kata(k, t) or (k.startswith("[") and k in t)]
    net = [k for k in KATA_NETRAL if k in t] + (["?"] if "?" in t else [])
    if len(neg) > len(pos):
        sentimen = "Negatif"
    elif len(pos) > len(neg):
        sentimen = "Positif"
    else:
        sentimen = "Netral"
    if net and abs(len(pos) - len(neg)) <= 1 and sentimen != "Negatif":
        sentimen = "Netral"
    skor_masalah = {m: sum(_ada_kata(k, t) for k in kws) for m, kws in KATA_MASALAH.items()}
    masalah = max(skor_masalah, key=skor_masalah.get) if max(skor_masalah.values()) else "Kebersihan Lingkungan"
    selisih = abs(len(pos) - len(neg))
    conf = 40 + 12 * selisih + (10 if item else -10)
    produk = cari_produk_relevan(item, jenis, k=3)
    ambigu = "Tinggi" if not item else ("Sedang" if pos and neg else "Rendah")
    return {
        "id": id_laporan, "jenis_sampah": jenis, "item_terdeteksi": item, "sentimen": sentimen,
        "masalah_utama": masalah, "sarkasme": False, "tingkat_ambiguitas": ambigu,
        "kata_kunci_dominan": (neg + pos + item)[:5],
        "rekomendasi_produk": [p["nama"] for p in produk[:3]],
        "langkah_pembuatan": produk[0]["langkah"] if produk else [],
        "confidence": _confidence(min(conf, 90)),
        "confidence_reasoning": (
            f"[Baseline rule-based] Kata positif: {pos or '-'}; kata negatif: {neg or '-'}; "
            f"item terdeteksi: {item or '-'}. Metode ini menghitung kata kunci saja sehingga tidak mampu "
            "membaca sarkasme maupun konteks kalimat."),
    }


def _normalisasi_laporan(d: dict, teks: str, id_laporan: str) -> dict:
    cadangan = analisis_offline(teks, id_laporan)
    item = _sebagai_list(d.get("item_terdeteksi")) or cadangan["item_terdeteksi"]
    jenis = _pilih_opsi(d.get("jenis_sampah"), JENIS, cadangan["jenis_sampah"])
    produk, catatan = guardrail_produk(_sebagai_list(d.get("rekomendasi_produk")), item, jenis, minimal=2)
    langkah = _rapikan_langkah(_sebagai_list(d.get("langkah_pembuatan")))
    if len(langkah) < 3:
        p = cari_produk_kb(produk[0]) if produk else None
        langkah = p["langkah"] if p else langkah
        catatan.append("Guardrail melengkapi langkah pembuatan dari knowledge base.")
    sarkasme = d.get("sarkasme")
    reasoning = str(d.get("confidence_reasoning") or "").strip()
    if catatan:
        reasoning += " " + " ".join(catatan)
    return {
        "id": id_laporan, "jenis_sampah": jenis, "item_terdeteksi": item,
        "sentimen": _pilih_opsi(d.get("sentimen"), SENTIMEN, "Netral"),
        "masalah_utama": _pilih_opsi(d.get("masalah_utama"), MASALAH, cadangan["masalah_utama"]),
        "sarkasme": sarkasme if isinstance(sarkasme, bool) else str(sarkasme).lower() in ("true", "ya", "1"),
        "tingkat_ambiguitas": _pilih_opsi(d.get("tingkat_ambiguitas"), AMBIGUITAS, "Sedang"),
        "kata_kunci_dominan": _sebagai_list(d.get("kata_kunci_dominan")),
        "rekomendasi_produk": produk, "langkah_pembuatan": langkah,
        "confidence": _confidence(d.get("confidence")), "confidence_reasoning": reasoning.strip(),
    }


def analisis_laporan_batch(laporan: list[dict], cfg: LLMConfig, versi: str = "v2") -> tuple[list[dict], str]:
    """laporan = [{"id": ..., "teks": ...}]. Mengembalikan (hasil, metode). Gagal -> fallback rule-based."""
    if not cfg.pakai_llm:
        return [analisis_offline(x["teks"], x["id"]) for x in laporan], "rule-based"
    data = json.dumps(laporan, ensure_ascii=False, indent=0)
    if versi == "v1":
        system, prompt = SYSTEM_PROMPT_V1, isi_prompt(PROMPT_ANALISIS_V1, laporan=data, skema=SKEMA_LAPORAN)
    else:
        system, prompt = SYSTEM_PROMPT, isi_prompt(PROMPT_ANALISIS_V2, kb=kb_ringkas(), laporan=data,
                                                   skema=SKEMA_LAPORAN)
    respons = panggil_llm(cfg, system, prompt)
    daftar = respons.get("hasil", respons) if isinstance(respons, dict) else respons
    if isinstance(daftar, dict):
        daftar = [daftar]
    per_id = {str(d.get("id")): d for d in daftar if isinstance(d, dict)}
    hasil = []
    for i, x in enumerate(laporan):
        d = per_id.get(str(x["id"])) or (daftar[i] if len(laporan) == 1 and daftar else None)
        hasil.append(_normalisasi_laporan(d, x["teks"], x["id"]) if d else
                     {**analisis_offline(x["teks"], x["id"]), "_fallback": True})
    return hasil, cfg.label


def analisis_laporan(teks: str, cfg: LLMConfig, versi: str = "v2", mentah: bool = False) -> dict:
    """Analisis satu teks laporan. mentah=True juga mengembalikan respons asli LLM (untuk eksperimen prompt)."""
    teks_bersih = bersihkan_teks(teks)
    hasil = {"_teks_bersih": teks_bersih}
    try:
        if mentah and cfg.pakai_llm:
            data = json.dumps([{"id": "INPUT", "teks": teks_bersih}], ensure_ascii=False)
            if versi == "v1":
                system, prompt = SYSTEM_PROMPT_V1, isi_prompt(PROMPT_ANALISIS_V1, laporan=data, skema=SKEMA_LAPORAN)
            else:
                system, prompt = SYSTEM_PROMPT, isi_prompt(PROMPT_ANALISIS_V2, kb=kb_ringkas(), laporan=data,
                                                           skema=SKEMA_LAPORAN)
            respons = panggil_llm(cfg, system, prompt)
            daftar = respons.get("hasil", respons) if isinstance(respons, dict) else respons
            d = daftar[0] if isinstance(daftar, list) and daftar else daftar
            hasil.update(_normalisasi_laporan(d if isinstance(d, dict) else {}, teks_bersih, "INPUT"))
            hasil["_mentah"] = d
            hasil["_prompt"] = prompt
        else:
            daftar, metode = analisis_laporan_batch([{"id": "INPUT", "teks": teks_bersih}], cfg, versi)
            hasil.update(daftar[0])
        hasil["_metode"] = cfg.label
    except Exception as e:  # noqa: BLE001
        hasil.update(analisis_offline(teks_bersih, "INPUT"))
        hasil["_metode"] = "Rule-based (fallback)"
        hasil["_error"] = str(e)
    return hasil


def _rekomendasi_offline(jenis_user: str, items: list[str]) -> dict:
    klas = [{"item": i, "jenis": klasifikasi_item(i)} for i in items]
    terdeteksi = gabung_jenis(k["jenis"] for k in klas) or jenis_user
    produk = cari_produk_relevan(items, terdeteksi, k=4)
    teks = " ".join(items)
    rekom = []
    for p in produk:
        cocok = [i for i in items if any(_ada_kata(kw, i) for kw in p["bahan_cocok"])]
        rekom.append({
            "nama_produk": p["nama"], "jenis": p["jenis"], "deskripsi": p["deskripsi"],
            "bahan_dari_input": cocok, "tingkat_kesulitan": p["kesulitan"], "estimasi_waktu": p["durasi"],
            "nilai_guna": p["nilai_guna"],
            "alasan_pemilihan": (f"Input mengandung {', '.join(cocok)} yang termasuk bahan baku produk ini."
                                 if cocok else "Produk umum untuk jenis sampah ini (tidak ada kecocokan bahan spesifik)."),
        })
    tidak_dikenal = [k["item"] for k in klas if k["jenis"] == "Tidak dikenal"]
    rasio = 1 - len(tidak_dikenal) / max(len(items), 1)
    return {
        "validasi_jenis": {"jenis_dipilih": jenis_user, "jenis_terdeteksi": terdeteksi,
                           "sesuai": terdeteksi == jenis_user,
                           "catatan": "" if terdeteksi == jenis_user else
                           f"Isi sampah terdeteksi sebagai {terdeteksi}, bukan {jenis_user}."},
        "klasifikasi_item": klas, "rekomendasi": rekom,
        "kata_kunci_dominan": deteksi_item_dalam_teks(teks)[:5],
        "confidence": _confidence(35 + 50 * rasio),
        "confidence_reasoning": (f"[Baseline rule-based] {len(items) - len(tidak_dikenal)} dari {len(items)} item "
                                 f"dikenali kamus lokal{'; tidak dikenal: ' + ', '.join(tidak_dikenal) if tidak_dikenal else ''}. "
                                 "Rekomendasi dipilih berdasarkan kecocokan kata kunci bahan di knowledge base."),
    }


def rekomendasi_daur_ulang(jenis_user: str, items_text: str, cfg: LLMConfig) -> dict:
    """Langkah 2 alur Input Manual: validasi jenis + minimal 3 rekomendasi produk daur ulang."""
    items = parse_items(items_text)
    pra = {i: klasifikasi_item(i) for i in items}
    terdeteksi = gabung_jenis(pra.values()) or jenis_user
    relevan = cari_produk_relevan(items, terdeteksi, k=6)
    hasil = {"_items": items, "_rag": [p["nama"] for p in relevan]}
    if cfg.pakai_llm:
        prompt = isi_prompt(PROMPT_REKOMENDASI, jenis=jenis_user, items=items_text.strip(),
                            pra=json.dumps(pra, ensure_ascii=False), kb=kb_ringkas(relevan))
        hasil["_prompt"] = prompt
        try:
            hasil.update(panggil_llm(cfg, SYSTEM_PROMPT, prompt))
            hasil["_metode"] = cfg.label
        except Exception as e:  # noqa: BLE001
            hasil.update(_rekomendasi_offline(jenis_user, items))
            hasil["_metode"], hasil["_error"] = "Rule-based (fallback)", str(e)
    else:
        hasil.update(_rekomendasi_offline(jenis_user, items))
        hasil["_metode"] = cfg.label

    # --- guardrail: buang kompos, minimal 3 rekomendasi, normalisasi field
    rekom = [r for r in hasil.get("rekomendasi", []) if isinstance(r, dict) and r.get("nama_produk")]
    nama, catatan = guardrail_produk([r["nama_produk"] for r in rekom], items, terdeteksi, minimal=3)
    final = []
    for n in nama:
        r = next((r for r in rekom if r["nama_produk"] == n), None)
        if r is None:  # ditambahkan guardrail dari knowledge base
            p = cari_produk_kb(n)
            r = {"nama_produk": p["nama"], "jenis": p["jenis"], "deskripsi": p["deskripsi"], "bahan_dari_input": [],
                 "tingkat_kesulitan": p["kesulitan"], "estimasi_waktu": p["durasi"], "nilai_guna": p["nilai_guna"],
                 "alasan_pemilihan": "Ditambahkan guardrail dari knowledge base agar minimal 3 rekomendasi."}
        r["bahan_dari_input"] = _sebagai_list(r.get("bahan_dari_input"))
        final.append(r)
    hasil["rekomendasi"] = final[:5]
    hasil["_guardrail"] = catatan
    hasil["confidence"] = _confidence(hasil.get("confidence"))
    hasil["peringatan_b3"] = [i for i, j in pra.items() if j == "B3"]
    if not isinstance(hasil.get("klasifikasi_item"), list):
        hasil["klasifikasi_item"] = [{"item": i, "jenis": j} for i, j in pra.items()]
    return hasil


def _panduan_offline(produk: str) -> dict:
    p = cari_produk_kb(produk)
    if not p:
        return {"nama_produk": produk, "alat": [], "bahan": [], "langkah": [], "estimasi_waktu": "-",
                "tips_keamanan": [], "potensi_nilai_ekonomi": "-", "confidence": 0,
                "confidence_reasoning": "Produk ini tidak ada di knowledge base lokal; aktifkan LLM untuk panduan."}
    return {"nama_produk": p["nama"], "alat": p["alat"], "bahan": p["bahan"], "langkah": p["langkah"],
            "estimasi_waktu": p["durasi"], "tips_keamanan": [p["keamanan"]], "potensi_nilai_ekonomi": p["nilai_guna"],
            "confidence": 85, "confidence_reasoning": "[Knowledge base] Panduan diambil dari basis pengetahuan SI-BERSIH."}


def panduan_pembuatan(produk: str, jenis: str, items: list[str], cfg: LLMConfig) -> dict:
    """Langkah 3 alur Input Manual: alat, bahan, dan 3-5 langkah pembuatan produk pilihan."""
    if not cfg.pakai_llm:
        return {**_panduan_offline(produk), "_metode": cfg.label}
    p = cari_produk_kb(produk)
    prompt = isi_prompt(PROMPT_PANDUAN, produk=produk, jenis=jenis, items=", ".join(items),
                        kb=kb_detail(p) if p else "(tidak ada — gunakan pengetahuanmu sendiri)")
    try:
        d = panggil_llm(cfg, SYSTEM_PROMPT, prompt)
        hasil = {"nama_produk": d.get("nama_produk", produk), "alat": _sebagai_list(d.get("alat")),
                 "bahan": _sebagai_list(d.get("bahan")), "langkah": _rapikan_langkah(_sebagai_list(d.get("langkah"))),
                 "estimasi_waktu": d.get("estimasi_waktu", "-"), "tips_keamanan": _sebagai_list(d.get("tips_keamanan")),
                 "potensi_nilai_ekonomi": d.get("potensi_nilai_ekonomi", "-"),
                 "confidence": _confidence(d.get("confidence")),
                 "confidence_reasoning": d.get("confidence_reasoning", ""), "_metode": cfg.label, "_prompt": prompt}
        if len(hasil["langkah"]) < 3 and p:
            hasil["langkah"] = p["langkah"]
            hasil["confidence_reasoning"] += " Guardrail melengkapi langkah dari knowledge base."
        if POLA_TERLARANG.search(" ".join(hasil["langkah"])):
            hasil["confidence_reasoning"] += " Peringatan: langkah menyebut kompos — periksa ulang."
        return hasil
    except Exception as e:  # noqa: BLE001
        return {**_panduan_offline(produk), "_metode": "Rule-based (fallback)", "_error": str(e)}


def ringkas_statistik(df: pd.DataFrame) -> dict:
    neg = df[df["Pred_Sentimen"] == "Negatif"]
    pasangan = neg.groupby(["Kabupaten_Kota", "Pred_Masalah"]).size().sort_values(ascending=False).head(10)
    produk = df["Rekomendasi_Produk"].dropna().str.split(r" \| ").explode().value_counts().head(5)
    return {
        "total_laporan": int(len(df)),
        "distribusi_jenis": {k: int(v) for k, v in df["Pred_Jenis"].value_counts().items()},
        "distribusi_sentimen": {k: int(v) for k, v in df["Pred_Sentimen"].value_counts().items()},
        "distribusi_masalah": {k: int(v) for k, v in df["Pred_Masalah"].value_counts().items()},
        "laporan_negatif_per_wilayah_dan_masalah": [
            {"wilayah": w, "masalah": m, "jumlah": int(n)} for (w, m), n in pasangan.items()],
        "wilayah_terbanyak": {k: int(v) for k, v in df["Kabupaten_Kota"].value_counts().head(5).items()},
        "produk_rekomendasi_teratas": {k: int(v) for k, v in produk.items()},
        "rata_rata_confidence": round(float(df["Confidence"].mean()), 1),
    }


TINDAKAN_MASALAH = {
    "Layanan Pengangkutan": ("Tambah ritase truk dan umumkan jadwal angkut per kelurahan lewat grup WhatsApp/RT.", "DLH kab/kota, lurah, kepala lingkungan"),
    "Fasilitas & TPS": ("Perbaiki kontainer rusak, tambah tempat sampah terpilah dan TPS3R, perpanjang jam buka bank sampah.", "DLH, bank sampah, pemerintah kelurahan"),
    "Perilaku Membuang Sampah": ("Pasang papan larangan/CCTV di titik buang liar, kerja bakti rutin, dan tegakkan perda persampahan.", "Satpol PP, kelurahan, tokoh agama"),
    "Kebersihan Lingkungan": ("Pembersihan sungai/pantai berkala dan pemasangan trash boom di muara.", "DLH, BPBD, komunitas pecinta lingkungan"),
    "Partisipasi & Edukasi": ("Pelatihan maggot BSF, eco-enzyme, dan eco-brick lewat PKK, sekolah, serta komunitas gereja/masjid.", "PKK, dinas pendidikan, komunitas keagamaan"),
}


def rekomendasi_kebijakan(df: pd.DataFrame, cfg: LLMConfig) -> dict:
    """Rekomendasi decision making untuk lembaga (DLH) berdasarkan statistik hasil analisis."""
    stat = ringkas_statistik(df)
    if cfg.pakai_llm:
        try:
            hasil = panggil_llm(cfg, SYSTEM_PROMPT, isi_prompt(PROMPT_KEBIJAKAN,
                                                                statistik=json.dumps(stat, ensure_ascii=False, indent=1)))
            hasil["_metode"] = cfg.label
            return hasil
        except Exception as e:  # noqa: BLE001
            galat = str(e)
    else:
        galat = None
    rekom = []
    for i, x in enumerate(stat["laporan_negatif_per_wilayah_dan_masalah"][:5], start=1):
        tindakan, pihak = TINDAKAN_MASALAH[x["masalah"]]
        rekom.append({"prioritas": i, "wilayah": x["wilayah"], "masalah": x["masalah"], "tindakan": tindakan,
                      "pihak_terlibat": pihak,
                      "indikator_keberhasilan": f"Laporan negatif '{x['masalah']}' di {x['wilayah']} turun ≥ 30% dalam 3 bulan.",
                      "dasar_data": f"{x['jumlah']} laporan negatif."})
    jenis_top = max(stat["distribusi_jenis"], key=stat["distribusi_jenis"].get)
    hasil = {"ringkasan": (f"Dari {stat['total_laporan']} laporan, jenis sampah terbanyak adalah {jenis_top} dan "
                           f"{stat['distribusi_sentimen'].get('Negatif', 0)} laporan bernada negatif. "
                           "Prioritas disusun dari pasangan wilayah–masalah dengan laporan negatif terbanyak."),
             "rekomendasi": rekom, "_metode": "Rule-based" + (" (fallback)" if galat else "")}
    if galat:
        hasil["_error"] = galat
    return hasil


# =============================================================================
# 8. BATCH PROCESSING & EVALUASI
# =============================================================================

KOLOM_HASIL = ["Pred_Jenis", "Pred_Sentimen", "Pred_Masalah", "Sarkasme", "Ambiguitas", "Item_Terdeteksi",
               "Kata_Kunci", "Rekomendasi_Produk", "Langkah_Pembuatan", "Confidence", "Confidence_Reasoning",
               "Metode", "Waktu_Proses"]


def _ke_baris(h: dict, metode: str) -> dict:
    return {
        "Pred_Jenis": h["jenis_sampah"], "Pred_Sentimen": h["sentimen"], "Pred_Masalah": h["masalah_utama"],
        "Sarkasme": bool(h["sarkasme"]), "Ambiguitas": h["tingkat_ambiguitas"],
        "Item_Terdeteksi": " | ".join(map(str, h["item_terdeteksi"])),
        "Kata_Kunci": " | ".join(map(str, h["kata_kunci_dominan"])),
        "Rekomendasi_Produk": " | ".join(map(str, h["rekomendasi_produk"])),
        "Langkah_Pembuatan": " | ".join(map(str, h["langkah_pembuatan"])),
        "Confidence": h["confidence"], "Confidence_Reasoning": h["confidence_reasoning"],
        "Metode": "Rule-based (fallback)" if h.get("_fallback") else metode,
        "Waktu_Proses": datetime.now().isoformat(timespec="seconds"),
    }


def proses_batch(df: pd.DataFrame, cfg: LLMConfig, batch_size: int = 10, jeda: float = 7.0,
                 limit: int | None = None, output: Path = HASIL_PATH, lanjutkan: bool = True,
                 progress=None) -> pd.DataFrame:
    """Analisis dataset per kelompok (batch) agar hemat kuota API. Hasil disimpan bertahap sehingga
    proses yang terputus (mis. kuota habis) bisa dilanjutkan tanpa mengulang dari awal."""
    df, _ = preprocess_dataset(df)
    lama = pd.read_csv(output) if lanjutkan and output.exists() else pd.DataFrame()
    sudah = set(lama["ID_Laporan"]) if not lama.empty else set()
    target = df[~df["ID_Laporan"].isin(sudah)]
    if limit:
        target = target.head(limit)
    baru = []
    total = len(target)
    for mulai in range(0, total, batch_size):
        potong = target.iloc[mulai:mulai + batch_size]
        laporan = [{"id": r.ID_Laporan, "teks": r.Teks_Bersih} for r in potong.itertuples()]
        try:
            hasil, metode = analisis_laporan_batch(laporan, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  ! batch {mulai // batch_size + 1} gagal ({e}); memakai fallback rule-based")
            hasil = [{**analisis_offline(x["teks"], x["id"]), "_fallback": True} for x in laporan]
            metode = "Rule-based (fallback)"
        for r, h in zip(potong.to_dict("records"), hasil):
            baru.append({**r, **_ke_baris(h, metode)})
        gabungan = pd.concat([lama, pd.DataFrame(baru)], ignore_index=True)
        gabungan.to_csv(output, index=False, encoding="utf-8-sig")
        selesai = min(mulai + batch_size, total)
        if progress:
            progress(selesai, total)
        else:
            print(f"  {selesai}/{total} laporan dianalisis ({metode})")
        if cfg.pakai_llm and selesai < total:
            time.sleep(jeda)  # menjaga batas request per menit (free tier)
    return pd.concat([lama, pd.DataFrame(baru)], ignore_index=True)


def metrik_klasifikasi(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true, y_pred = y_true.astype(str), y_pred.astype(str)
    f1 = []
    for label in sorted(y_true.unique()):
        tp = ((y_pred == label) & (y_true == label)).sum()
        fp = ((y_pred == label) & (y_true != label)).sum()
        fn = ((y_pred != label) & (y_true == label)).sum()
        presisi = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * presisi * recall / (presisi + recall) if presisi + recall else 0.0)
    return {"akurasi": float((y_true == y_pred).mean()), "macro_f1": float(sum(f1) / len(f1)) if f1 else 0.0}


PASANGAN_EVAL = {"Jenis Sampah": ("Label_Jenis_Sampah", "Pred_Jenis"),
                 "Sentimen": ("Label_Sentimen", "Pred_Sentimen"),
                 "Masalah Sosial": ("Label_Masalah", "Pred_Masalah")}


def evaluasi(df: pd.DataFrame) -> dict:
    hasil = {nama: metrik_klasifikasi(df[a], df[b]) for nama, (a, b) in PASANGAN_EVAL.items()}
    per_kategori = (df.assign(Benar_Sentimen=df["Label_Sentimen"] == df["Pred_Sentimen"],
                              Benar_Jenis=df["Label_Jenis_Sampah"] == df["Pred_Jenis"])
                    .groupby("Kategori_Teks")[["Benar_Sentimen", "Benar_Jenis"]].mean().mul(100).round(1))
    return {"metrik": hasil, "per_kategori": per_kategori}


# =============================================================================
# 9. FISHBONE DIAGRAM
# =============================================================================

FISHBONE = {
    "MAN (Manusia)": ["Kesadaran memilah dari rumah rendah", "Kebiasaan buang ke sungai/selokan/pantai",
                      "Petugas kebersihan terbatas & kurang pelatihan"],
    "MACHINE (Sarana)": ["Armada truk terbatas & sering rusak", "Kontainer/TPS rusak dan cepat penuh",
                         "Belum ada mesin pencacah di tingkat kelurahan"],
    "MATERIAL (Bahan)": ["Dominasi plastik sekali pakai & sachet", "Organik (makanan, ikan, kelapa) tidak dipisah",
                         "Sampah kiriman dari laut & sungai"],
    "METHOD (Metode)": ["Pola kumpul–angkut–buang ke TPA", "Jadwal angkut tidak konsisten",
                        "Bank sampah/TPS3R belum terintegrasi"],
    "ENVIRONMENT (Lingkungan)": ["Wilayah kepulauan & topografi berbukit", "Curah hujan tinggi memicu banjir sampah",
                                 "Lonjakan sampah saat hari raya & pesta adat"],
}


def buat_fishbone(path: Path = FISHBONE_PATH) -> Path:
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(18, 9), dpi=150)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")
    y0 = 4.5
    ax.annotate("", xy=(14.2, y0), xytext=(0.6, y0), arrowprops=dict(arrowstyle="-|>", lw=3, color="#263238"))
    ax.text(15.9, y0, "Pengelolaan sampah\nbelum optimal di\nSulawesi Utara", ha="center", va="center",
            fontsize=13, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.6", fc="#C62828", ec="#8E0000"))
    warna = ["#2E7D32", "#1565C0", "#EF6C00", "#6A1B9A", "#00838F"]
    posisi = [(2.0, True), (6.4, True), (10.8, True), (4.2, False), (8.6, False)]
    for (kategori, sebab), (x, atas), w in zip(FISHBONE.items(), posisi, warna):
        y_ujung = 8.0 if atas else 1.0
        x_spine = x + 2.0
        ax.plot([x, x_spine], [y_ujung, y0], color=w, lw=2.5)
        ax.text(x, y_ujung + (0.35 if atas else -0.35), kategori, ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.4", fc=w, ec=w))
        for j, s in enumerate(sebab):
            y = y_ujung + (-(j + 1) if atas else (j + 1)) * 0.95
            xb = x + (y_ujung - y) / (y_ujung - y0) * 2.0
            ax.plot([xb - 0.35, xb], [y, y], color=w, lw=1.5)
            ax.text(xb - 0.4, y, textwrap.fill(s, 26), ha="right", va="center", fontsize=8.5, color="#212121")
    ax.set_title("Fishbone Diagram — Akar Masalah Pengelolaan Sampah di Sulawesi Utara (SI-BERSIH)",
                 fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# =============================================================================
# 10. DASHBOARD STREAMLIT
# =============================================================================

WARNA_JENIS = {"Organik": "#2E7D32", "Anorganik": "#1565C0", "Campuran": "#D98200"}
WARNA_SENTIMEN = {"Positif": "#2E7D32", "Negatif": "#C62828", "Netral": "#78909C"}
WARNA_KESULITAN = {"Mudah": "#2E7D32", "Sedang": "#EF6C00", "Sulit": "#C62828"}
EMOJI_SENTIMEN_UI = {"Positif": "😊", "Negatif": "😠", "Netral": "😐"}
INFO_JENIS = {
    "Organik": ("🍌", "Sisa makhluk hidup, mudah membusuk", "sisa makanan, sayur, buah, ikan, kelapa"),
    "Anorganik": ("🧴", "Buatan pabrik, sulit terurai", "plastik, kaleng, kaca, kertas, kardus"),
    "Campuran": ("🗑️", "Organik & anorganik jadi satu", "sisa makanan bercampur bungkus plastik"),
}
PILIHAN_ITEM = {
    "Organik": ["nasi", "sayur", "buah semangka", "kulit pisang", "sisa ikan", "ampas kelapa", "tempurung kelapa",
                "sabut kelapa", "kulit jagung", "minyak jelantah", "daun kering", "ampas tahu"],
    "Anorganik": ["botol plastik", "kantong kresek", "kemasan sachet", "gelas plastik", "kaleng minuman",
                  "botol kaca", "kardus", "kertas koran", "styrofoam", "jaring nelayan", "ban bekas"],
}
PILIHAN_ITEM["Campuran"] = PILIHAN_ITEM["Organik"][:6] + PILIHAN_ITEM["Anorganik"][:6]
IKON_PRODUK = [("maggot", "🪱"), ("enzyme", "🧴"), ("biogas", "🔥"), ("briket", "⚫"), ("sabun", "🧼"),
               ("lilin", "🕯️"), ("cocopeat", "🌱"), ("silase", "🐟"), ("jagung", "🌽"), ("eco-brick", "🧱"),
               ("paving", "🛤️"), ("pot tanaman", "🪴"), ("kertas", "📄"), ("tas", "👜"), ("kaleng", "🏮"),
               ("kaca", "💡"), ("jaring", "🎣"), ("ban", "🛞")]
CONTOH_TEKS = {
    "Keluhan biasa": "Sampah sisa ikan dan kantong kresek di Pasar Bersehati sudah 4 hari tidak diangkut, baunya parah.",
    "Sarkasme": "Wah hebat sekali, botol plastik di Kelurahan Tikala sudah seminggu jadi \"monumen\" baru. Terima kasih banyak atas pelayanannya! 😂",
    "Ambigu": "Ada kantong hitam besar di Pantai Malalayang, entah isinya apa, sudah 3 hari di situ.",
    "Perasaan campur": "Bank sampah di Bitung bagus, sayang cuma buka sebulan sekali jadi kaleng dan kardus menumpuk di rumah.",
    "Dialek Manado": "Kiapa so 5 hari ini sampah kulit jagung deng botol plastik di Tomohon nyanda ta angka? Bau skali!",
}

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1150px;}
.hero {background: linear-gradient(120deg, #1B5E20 0%, #2E7D32 50%, #66BB6A 100%); color: #fff;
       padding: 26px 30px; border-radius: 20px; margin-bottom: 18px; box-shadow: 0 10px 28px rgba(46,125,50,.25);}
.hero h1 {color: #fff; font-size: 2rem; margin: 2px 0 0; padding: 0;}
.hero p {color: #E8F5E9; margin: 6px 0 0; font-size: 1.05rem;}
.hero .kecil {font-size: .75rem; opacity: .85; letter-spacing: .06em; text-transform: uppercase;}
.stepper {display: flex; gap: 8px; margin: 4px 0 22px; flex-wrap: wrap;}
.step {flex: 1; min-width: 120px; text-align: center; padding: 10px 6px; border-radius: 12px;
       background: #EAF2EB; color: #7A8B7D; font-weight: 600; font-size: .9rem;}
.step.aktif {background: #2E7D32; color: #fff; box-shadow: 0 4px 14px rgba(46,125,50,.35);}
.step.selesai {background: #C8E6C9; color: #1B5E20;}
[class*="st-key-kartu"] {background: #fff; border: 1px solid #E0EAE1; border-radius: 18px; padding: 18px 18px 14px;
        box-sizing: border-box; box-shadow: 0 2px 10px rgba(0,0,0,.04);
        transition: box-shadow .2s, border-color .2s;}
[class*="st-key-kartu"]:hover {box-shadow: 0 8px 22px rgba(46,125,50,.14); border-color: #A5D6A7;}
.ikon {font-size: 2.6rem; line-height: 1.1;}
.judul-kartu {font-size: 1.15rem; font-weight: 700; line-height: 1.3; margin: 8px 0 4px; color: #1B2A1F;}
.redup {color: #5F6F62; font-size: .92rem; line-height: 1.45;}
.badge {display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .8rem; font-weight: 600;
         margin: 3px 5px 3px 0;}
div[data-testid="stHorizontalBlock"]:has([class*="st-key-kartu_jenis_"]),
div[data-testid="stHorizontalBlock"]:has([class*="st-key-kartu_ide_"]) {align-items: stretch;}
div[data-testid="stHorizontalBlock"]:has([class*="st-key-kartu_jenis_"]) > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"]:has([class*="st-key-kartu_ide_"]) > div[data-testid="stColumn"] {
        display: flex;}
div[data-testid="stHorizontalBlock"]:has([class*="st-key-kartu_jenis_"]) > div[data-testid="stColumn"] > div,
div[data-testid="stHorizontalBlock"]:has([class*="st-key-kartu_ide_"]) > div[data-testid="stColumn"] > div {
        display: flex; flex: 1; flex-direction: column;}
[class*="st-key-kartu_jenis_"], [class*="st-key-kartu_ide_"] {
        display: flex; flex: 1; flex-direction: column;}
[class*="st-key-kartu_jenis_"] > div[data-testid="stVerticalBlock"],
[class*="st-key-kartu_ide_"] > div[data-testid="stVerticalBlock"] {
        display: flex; flex: 1; flex-direction: column;}
[class*="st-key-kartu_jenis_"] > div[data-testid="stElementContainer"]:first-child,
[class*="st-key-kartu_ide_"] > div[data-testid="stElementContainer"]:first-child,
[class*="st-key-kartu_jenis_"] > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:first-child,
[class*="st-key-kartu_ide_"] > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:first-child {
        flex: 1;}
[class*="st-key-kartu_jenis_"] div[data-testid="stElementContainer"]:has(button),
[class*="st-key-kartu_ide_"] div[data-testid="stElementContainer"]:has(button) {margin-top: auto;}
.langkah {display: flex; gap: 14px; align-items: flex-start; margin: 10px 0;}
.nomor {flex: 0 0 34px; height: 34px; border-radius: 50%; background: #2E7D32; color: #fff; display: flex;
        align-items: center; justify-content: center; font-weight: 700;}
.isi-langkah {padding-top: 6px; color: #1B2A1F;}
.stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 6px 0 18px;}
.stat {background: #fff; border-radius: 14px; padding: 14px 16px; box-shadow: 0 2px 10px rgba(0,0,0,.05);
       border: 1px solid #E0EAE1;}
.stat-label {font-size: .75rem; color: #5F6F62; text-transform: uppercase; letter-spacing: .05em;}
.stat-nilai {font-size: 1.3rem; font-weight: 700; color: #1B2A1F; margin-top: 4px;}
.desk {min-height: 4.4em;}
.desk-ide {min-height: 6.2em;}
.dari {min-height: 3em;}
button[data-variant="pills"][data-selected="true"] {background: #2E7D32 !important; color: #fff !important;
       border-color: #2E7D32 !important;}
[class*="st-key-kartu_ide"] .judul-kartu {min-height: 2.6em;}
.judul-produk {font-size: 1.8rem; font-weight: 800; color: #1B5E20; margin: 4px 0 2px;}
</style>
"""


def _e(teks) -> str:
    return html.escape(str(teks))


def _ikon_produk(nama: str) -> str:
    n = str(nama).lower()
    return next((ikon for kunci, ikon in IKON_PRODUK if kunci in n), "♻️")


def _badge(teks, warna: str = "#2E7D32") -> str:
    return f'<span class="badge" style="background:{warna}1F;color:{warna}">{_e(teks)}</span>'


def _hero(st, judul: str, sub: str):
    st.markdown(f'<div class="hero"><div class="kecil">{NAMA_SISTEM} · {KEPANJANGAN}</div>'
                f'<h1>{judul}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def _stepper(st, aktif: int):
    nama = ["Jenis sampah", "Isi sampah", "Pilih ide", "Cara membuat"]
    isi = "".join(f'<div class="step {"aktif" if i == aktif else "selesai" if i < aktif else ""}">'
                  f'{"✓" if i < aktif else i} · {n}</div>' for i, n in enumerate(nama, start=1))
    st.markdown(f'<div class="stepper">{isi}</div>', unsafe_allow_html=True)


def _kartu_statistik(st, data: list[tuple[str, str, str]]):
    isi = "".join(f'<div class="stat" style="border-top:4px solid {w}"><div class="stat-label">{_e(l)}</div>'
                  f'<div class="stat-nilai">{_e(v)}</div></div>' for l, v, w in data)
    st.markdown(f'<div class="stats">{isi}</div>', unsafe_allow_html=True)


def _langkah_html(langkah: list[str]) -> str:
    return "".join(f'<div class="langkah"><div class="nomor">{i}</div><div class="isi-langkah">{_e(s)}</div></div>'
                   for i, s in enumerate(langkah, start=1))


def _status_metode(st, hasil: dict):
    if hasil.get("_error"):
        st.warning("AI online tidak bisa dihubungi, jadi hasil ini dibuat dengan mode offline.")
        st.caption(f"Detail: {hasil['_error'][:200]}")


def _penjelasan_ai(st, conf: int, alasan: str, kata_kunci=None, teknis: dict | None = None,
                   metode: str = "", key: str = "xai"):
    """Explainable AI: alasan keputusan + (opsional) detail teknis untuk dosen."""
    with st.expander(f"🤖 Kenapa SI-BERSIH memilih ini?  —  yakin {conf}%"):
        st.progress(conf / 100, text=f"Tingkat keyakinan AI: {conf}%")
        st.write(alasan or "-")
        if kata_kunci:
            st.markdown("Kata yang paling berpengaruh: " + "".join(_badge(k, "#1565C0") for k in kata_kunci),
                        unsafe_allow_html=True)
        if metode:
            st.caption(f"Diproses oleh: {metode}")
        if teknis and st.toggle("Tampilkan detail teknis (prompt, RAG, guardrail)", key=key):
            for judul, isi in teknis.items():
                st.markdown(f"**{judul}**")
                if isinstance(isi, str):
                    st.code(isi, language="markdown", wrap_lines=True)
                else:
                    st.json(isi, expanded=False)


def _gaya(fig, tinggi: int = 360):
    fig.update_layout(template="plotly_white", height=tinggi, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Segoe UI, Inter, sans-serif", size=13), title_font=dict(size=16),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.15, title=None))
    return fig


def main_ui():
    import streamlit as st

    st.set_page_config(page_title="SI-BERSIH", page_icon="♻️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    halaman = st.navigation([
        st.Page(halaman_daur_ulang, title="Daur Ulang Sampah", icon="♻️", url_path="daur-ulang", default=True),
        st.Page(halaman_cek_laporan, title="Cek Laporan Warga", icon="📝", url_path="cek-laporan"),
        st.Page(halaman_dashboard, title="Dashboard", icon="📊", url_path="dashboard"),
    ])
    with st.sidebar:
        st.session_state["cfg"] = _pengaturan_ai(st)
    halaman.run()


def _pengaturan_ai(st) -> LLMConfig:
    st.markdown(f"### ♻️ {NAMA_SISTEM}")
    st.caption(KEPANJANGAN)
    status = st.empty()
    with st.expander("⚙️ Pengaturan AI"):
        provider = st.selectbox("Model AI", list(PROVIDERS), key="provider")
        cfg = config_default(provider)
        if provider != OFFLINE:
            key_awal = "" if cfg.api_key.startswith("MASUKKAN") else cfg.api_key
            cfg.api_key = st.text_input("API key", value=key_awal, type="password", key=f"key_{provider}")
            cfg.model = st.text_input("Nama model", value=cfg.model, key=f"model_{provider}")
            cfg.temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="temperature")
    if cfg.pakai_llm:
        status.success(f"🟢 AI online · {cfg.provider}")
    else:
        status.info("🟡 Mode offline. Isi API key di **Pengaturan AI** agar memakai Gemini.")
    return cfg


# ---------------------------- Halaman 1: Daur Ulang ----------------------------

def halaman_daur_ulang():
    import streamlit as st

    ss = st.session_state
    ss.setdefault("du_langkah", 1)
    _hero(st, "♻️ Ubah Sampahmu Jadi Barang Berguna",
          "Pilih jenis sampah, sebutkan isinya, lalu SI-BERSIH memberi ide daur ulang beserta cara membuatnya.")
    _stepper(st, ss["du_langkah"])
    {1: _du_jenis, 2: _du_isi, 3: _du_ide, 4: _du_cara}[ss["du_langkah"]](st, ss["cfg"])


def _du_pindah(st, langkah: int):
    st.session_state["du_langkah"] = langkah
    st.rerun()


def _du_jenis(st, cfg):
    st.subheader("Sampahmu termasuk jenis apa?")
    for kolom, jenis in zip(st.columns(3), JENIS):
        ikon, arti, contoh = INFO_JENIS[jenis]
        with kolom.container(key=f"kartu_jenis_{jenis}"):
            st.markdown(f'<div class="ikon">{ikon}</div><div class="judul-kartu">{jenis}</div>'
                        f'<div class="redup desk">{arti}<br><i>Contoh: {contoh}</i></div>', unsafe_allow_html=True)
            if st.button(f"Pilih {jenis}", key=f"pilih_{jenis}", type="primary", width="stretch"):
                if st.session_state.get("du_jenis") != jenis:
                    for k in ("du_simpan_pil", "du_simpan_lain"):
                        st.session_state.pop(k, None)
                st.session_state["du_jenis"] = jenis
                _du_pindah(st, 2)
    st.caption("Belum yakin? Pilih **Campuran** — SI-BERSIH akan memeriksa dan memberi tahu jenis yang benar.")


def _du_isi(st, cfg):
    ss = st.session_state
    jenis = ss["du_jenis"]
    st.subheader(f"{INFO_JENIS[jenis][0]} Apa saja isi sampah {jenis.lower()}-mu?")
    opsi = PILIHAN_ITEM[jenis]
    with st.container(key="kartu_isi"):
        pilih = st.pills("Ketuk sampah yang kamu punya:", opsi, selection_mode="multi", key="du_pil",
                         default=[x for x in ss.get("du_simpan_pil", []) if x in opsi])
        lain = st.text_input("Ada yang lain? Tulis di sini, pisahkan dengan koma", key="du_lain",
                             value=ss.get("du_simpan_lain", ""), placeholder="contoh: kulit nanas, tutup botol")
    items = list(dict.fromkeys(list(pilih or []) + parse_items(lain)))
    if items:
        st.markdown("**Sampahmu:** " + "".join(_badge(i, WARNA_JENIS[jenis]) for i in items), unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3])
    if c1.button("← Kembali", width="stretch"):
        _du_pindah(st, 1)
    if c2.button("✨ Cari ide daur ulang", type="primary", disabled=not items, width="stretch"):
        ss["du_simpan_pil"], ss["du_simpan_lain"] = list(pilih or []), lain
        with st.spinner("SI-BERSIH sedang mencari ide..."):
            ss["du_rekom"] = rekomendasi_daur_ulang(jenis, ", ".join(items), cfg)
        _du_pindah(st, 3)


def _du_ide(st, cfg):
    ss = st.session_state
    jenis, rekom = ss["du_jenis"], ss["du_rekom"]
    _status_metode(st, rekom)
    v = rekom.get("validasi_jenis", {})
    terdeteksi = v.get("jenis_terdeteksi") or jenis
    if v.get("sesuai", True):
        st.success(f"👍 Cocok! Sampahmu memang termasuk **{terdeteksi}**.")
    else:
        st.warning(f"🤔 Kamu memilih **{jenis}**, tapi isinya terdeteksi **{terdeteksi}**. "
                   f"{v.get('catatan', '')} Pisahkan dulu sebelum diolah, ya.")
    klas = rekom.get("klasifikasi_item", [])
    if klas:
        st.markdown("".join(_badge(f"{k.get('item')} · {k.get('jenis')}",
                                   WARNA_JENIS.get(k.get("jenis"), "#C62828" if k.get("jenis") == "B3" else "#78909C"))
                            for k in klas if isinstance(k, dict)), unsafe_allow_html=True)
    if rekom.get("peringatan_b3"):
        st.error(f"☣️ **{', '.join(rekom['peringatan_b3'])}** — {PERINGATAN_B3}")

    st.subheader("💡 Ide daur ulang untukmu")
    st.caption("Pilih satu ide untuk melihat alat dan cara membuatnya.")
    daftar = rekom["rekomendasi"]
    for awal in range(0, len(daftar), 3):
        for kolom, (i, r) in zip(st.columns(3), list(enumerate(daftar))[awal:awal + 3]):
            with kolom.container(key=f"kartu_ide_{i}"):
                kesulitan = r.get("tingkat_kesulitan", "-")
                bahan = ", ".join(r.get("bahan_dari_input", []))
                st.markdown(
                    f'<div class="ikon">{_ikon_produk(r["nama_produk"])}</div>'
                    f'<div class="judul-kartu">{_e(r["nama_produk"])}</div>'
                    f'{_badge(kesulitan, WARNA_KESULITAN.get(kesulitan, "#78909C"))}'
                    f'{_badge("⏱ " + str(r.get("estimasi_waktu", "-")), "#5F6F62")}'
                    f'<p class="redup desk-ide" style="margin-top:8px">{_e(r.get("deskripsi", ""))}</p>'
                    f'<p class="redup dari"><b>Dari sampahmu:</b> {_e(bahan) if bahan else "–"}</p>',
                    unsafe_allow_html=True)
                if st.button("Lihat cara membuat →", key=f"buat_{i}", type="primary", width="stretch"):
                    with st.spinner("Menyusun panduan..."):
                        ss["du_panduan"] = panduan_pembuatan(r["nama_produk"], terdeteksi, rekom["_items"], cfg)
                    ss["du_produk"] = r
                    _du_pindah(st, 4)
    _penjelasan_ai(st, rekom["confidence"], rekom.get("confidence_reasoning", ""), rekom.get("kata_kunci_dominan"),
                   {"Konteks knowledge base (RAG)": rekom.get("_rag", []),
                    "Catatan guardrail": rekom.get("_guardrail") or ["Tidak ada koreksi."],
                    "Alasan tiap ide": {r["nama_produk"]: r.get("alasan_pemilihan", "-") for r in daftar},
                    **({"Prompt yang dikirim ke LLM": rekom["_prompt"]} if rekom.get("_prompt") else {})},
                   metode=rekom.get("_metode", ""), key="xai_ide")
    if st.button("← Ubah isi sampah"):
        _du_pindah(st, 2)


def _du_cara(st, cfg):
    ss = st.session_state
    p, r = ss["du_panduan"], ss["du_produk"]
    _status_metode(st, p)
    st.markdown(f'<div class="judul-produk">{_ikon_produk(p["nama_produk"])} {_e(p["nama_produk"])}</div>',
                unsafe_allow_html=True)
    _kartu_statistik(st, [("Waktu", p.get("estimasi_waktu", "-"), "#1565C0"),
                          ("Tingkat kesulitan", r.get("tingkat_kesulitan", "-"),
                           WARNA_KESULITAN.get(r.get("tingkat_kesulitan"), "#78909C")),
                          ("Jumlah langkah", f"{len(p['langkah'])} langkah", "#2E7D32")])
    c1, c2 = st.columns(2)
    with c1.container(key="kartu_alat"):
        st.markdown("#### 🛠️ Alat yang dibutuhkan")
        st.caption("Centang yang sudah kamu punya")
        for i, a in enumerate(p["alat"]):
            st.checkbox(a, key=f"alat_{i}_{a}")
    with c2.container(key="kartu_bahan"):
        st.markdown("#### 🧺 Bahan")
        for b in p["bahan"]:
            st.markdown(f"- {b}")
    with st.container(key="kartu_langkah"):
        st.markdown("#### 👣 Cara membuat")
        st.markdown(_langkah_html(p["langkah"]), unsafe_allow_html=True)
    for t in p.get("tips_keamanan", []):
        st.warning(f"⚠️ **Keamanan:** {t}")
    if p.get("potensi_nilai_ekonomi") not in (None, "", "-"):
        st.success(f"💰 **Manfaat:** {p['potensi_nilai_ekonomi']}")
    _penjelasan_ai(st, p["confidence"], p.get("confidence_reasoning", ""),
                   teknis={"Prompt yang dikirim ke LLM": p["_prompt"]} if p.get("_prompt") else None,
                   metode=p.get("_metode", ""), key="xai_cara")
    c1, c2 = st.columns(2)
    if c1.button("← Pilih ide lain", width="stretch"):
        _du_pindah(st, 3)
    if c2.button("🔄 Mulai dari awal", type="primary", width="stretch"):
        for k in ("du_jenis", "du_rekom", "du_panduan", "du_produk", "du_simpan_pil", "du_simpan_lain"):
            ss.pop(k, None)
        _du_pindah(st, 1)


# ---------------------------- Halaman 2: Cek Laporan ----------------------------

def halaman_cek_laporan():
    import streamlit as st

    ss, cfg = st.session_state, st.session_state["cfg"]
    _hero(st, "📝 Cek Laporan Warga",
          "Tempel keluhan warga tentang sampah. SI-BERSIH membaca jenis sampah, perasaan warga, dan masalahnya.")

    def _isi_contoh():
        if ss.get("cl_contoh"):
            ss["cl_teks"] = CONTOH_TEKS[ss["cl_contoh"]]

    with st.container(key="kartu_laporan"):
        st.pills("Coba contoh:", list(CONTOH_TEKS), key="cl_contoh", on_change=_isi_contoh)
        teks = st.text_area("Teks laporan", key="cl_teks", height=110, label_visibility="collapsed",
                            placeholder="contoh: Sampah di pasar sudah 3 hari tidak diangkut, baunya sampai ke rumah...")
        if st.button("🔍 Analisis laporan", type="primary", disabled=not teks.strip(), width="stretch"):
            with st.spinner("SI-BERSIH sedang membaca laporan..."):
                ss["cl_hasil"] = analisis_laporan(teks, cfg, mentah=True)

    h = ss.get("cl_hasil")
    if h:
        _status_metode(st, h)
        _kartu_statistik(st, [
            ("Jenis sampah", f"{INFO_JENIS[h['jenis_sampah']][0]} {h['jenis_sampah']}", WARNA_JENIS[h["jenis_sampah"]]),
            ("Perasaan warga", f"{EMOJI_SENTIMEN_UI[h['sentimen']]} {h['sentimen']}", WARNA_SENTIMEN[h["sentimen"]]),
            ("Masalah utama", h["masalah_utama"], "#6A1B9A"),
            ("Keyakinan AI", f"{h['confidence']}%", "#1565C0")])
        tanda = []
        if h["sarkasme"]:
            tanda.append(_badge("🎭 Sarkasme terdeteksi", "#C62828"))
        tanda.append(_badge(f"Ambiguitas: {h['tingkat_ambiguitas']}",
                            {"Rendah": "#2E7D32", "Sedang": "#EF6C00", "Tinggi": "#C62828"}[h["tingkat_ambiguitas"]]))
        tanda += [_badge(i, "#5F6F62") for i in h["item_terdeteksi"]]
        st.markdown("".join(tanda), unsafe_allow_html=True)
        with st.container(key="kartu_hasil_ide"):
            st.markdown("#### 💡 Ide daur ulang")
            st.markdown("".join(_badge(f"{_ikon_produk(p)} {p}", "#2E7D32") for p in h["rekomendasi_produk"]),
                        unsafe_allow_html=True)
            st.markdown(f"**Cara membuat {h['rekomendasi_produk'][0]}:**" if h["rekomendasi_produk"] else "")
            st.markdown(_langkah_html(h["langkah_pembuatan"]), unsafe_allow_html=True)
        _penjelasan_ai(st, h["confidence"], h["confidence_reasoning"], h["kata_kunci_dominan"],
                       {k: v for k, v in (("Teks setelah preprocessing", h.get("_teks_bersih")),
                                          ("Respons mentah LLM", h.get("_mentah")),
                                          ("Prompt yang dikirim", h.get("_prompt"))) if v},
                       metode=h.get("_metode", ""), key="xai_laporan")

    st.write("")
    with st.expander("🧪 Lab eksperimen (untuk dosen): bandingkan prompt & model", expanded=bool(ss.get("ep"))):
        _lab_eksperimen(st, cfg)


def _lab_eksperimen(st, cfg):
    ss = st.session_state
    st.caption("Prompt v1 = perintah singkat tanpa aturan. Prompt v2 = aturan ketat + contoh (few-shot) + knowledge base.")
    teks = st.text_area("Teks uji", value=CONTOH_TEKS["Sarkasme"], height=80, key="ep_teks")
    c1, c2 = st.columns(2)
    banding = c1.selectbox("Bandingkan dengan", [p for p in PROVIDERS if p != cfg.provider], key="ep_banding",
                           help="Provider lain memakai API key dari file .env")
    ulang = c2.checkbox("Uji konsistensi (Prompt v2 dijalankan 3×)", key="ep_ulang")
    if st.button("Jalankan eksperimen", key="ep_jalan"):
        with st.spinner("Menjalankan..."):
            ss["ep"] = {"v1": analisis_laporan(teks, cfg, versi="v1", mentah=True),
                        "v2": analisis_laporan(teks, cfg, versi="v2", mentah=True),
                        "banding": analisis_laporan(teks, config_default(banding), versi="v2", mentah=True),
                        "ulang": [analisis_laporan(teks, cfg, versi="v2") for _ in range(3)] if ulang else []}
    ep = ss.get("ep")
    if not ep:
        return
    baris = []
    for judul, h in [("Prompt v1", ep["v1"]), ("Prompt v2", ep["v2"]), (f"Pembanding ({banding})", ep["banding"])]:
        mentah = json.dumps(h.get("_mentah", ""), ensure_ascii=False)
        baris.append({"Versi": judul, "Diproses oleh": h["_metode"], "Jenis": h["jenis_sampah"],
                      "Sentimen": h["sentimen"], "Sarkasme": "Ya" if h["sarkasme"] else "Tidak",
                      "Confidence": h["confidence"], "Menyebut kompos?": "Ya ⚠️" if POLA_TERLARANG.search(mentah) else "Tidak",
                      "Alasan": h["confidence_reasoning"]})
    st.dataframe(pd.DataFrame(baris), hide_index=True, width="stretch")
    if ep["ulang"]:
        tabel = pd.DataFrame([{"Run": i + 1, "Jenis": h["jenis_sampah"], "Sentimen": h["sentimen"],
                               "Masalah": h["masalah_utama"], "Confidence": h["confidence"]}
                              for i, h in enumerate(ep["ulang"])])
        st.dataframe(tabel, hide_index=True, width="stretch")
        konsisten = all(tabel[c].nunique() == 1 for c in ["Jenis", "Sentimen", "Masalah"])
        (st.success if konsisten else st.warning)(
            "Label konsisten di ketiga run." if konsisten else "Label berubah antar-run → keputusan model belum stabil untuk teks ini.")


# ---------------------------- Halaman 3: Dashboard ----------------------------

def _jalankan_batch(st, cfg, jumlah: int, ukuran: int = 10, jeda: float | None = None):
    bar = st.progress(0.0, text="Memulai analisis...")
    proses_batch(pd.read_csv(DATASET_PATH), cfg, ukuran, (7.0 if cfg.pakai_llm else 0.0) if jeda is None else jeda,
                 jumlah, progress=lambda s, t: bar.progress(s / t, text=f"{s} dari {t} laporan selesai"))
    st.rerun()


def halaman_dashboard():
    import plotly.express as px
    import streamlit as st

    cfg = st.session_state["cfg"]
    _hero(st, "📊 Dashboard Sampah Sulawesi Utara", "Gambaran laporan warga yang sudah dianalisis SI-BERSIH.")
    if not DATASET_PATH.exists():
        generate_dataset()

    if not HASIL_PATH.exists():
        with st.container(key="kartu_kosong"):
            st.markdown("### 🗂️ Belum ada laporan yang dianalisis")
            st.write(f"Dataset berisi **{len(pd.read_csv(DATASET_PATH))} laporan warga**. "
                     f"Klik tombol di bawah untuk menganalisisnya dengan **{cfg.label}**.")
            jumlah = st.select_slider("Jumlah laporan", [10, 50, 100, 300, 600], value=50)
            if st.button("▶️ Analisis sekarang", type="primary"):
                _jalankan_batch(st, cfg, jumlah)
        return

    df = pd.read_csv(HASIL_PATH)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    wilayah = st.multiselect("📍 Wilayah", sorted(df["Kabupaten_Kota"].unique()), placeholder="Semua kabupaten/kota")
    if wilayah:
        df = df[df["Kabupaten_Kota"].isin(wilayah)]
    if df.empty:
        st.warning("Tidak ada data untuk wilayah ini.")
        return
    tab = st.tabs(["📈 Ringkasan", "🗺️ Peta & Tren", "🎯 Akurasi AI", "🏛️ Saran untuk Pemerintah", "⚙️ Kelola Data"])

    with tab[0]:
        _kartu_statistik(st, [
            ("Total laporan", f"{len(df)}", "#2E7D32"),
            ("Sampah organik", f"{(df['Pred_Jenis'] == 'Organik').mean():.0%}", WARNA_JENIS["Organik"]),
            ("Sampah anorganik", f"{(df['Pred_Jenis'] == 'Anorganik').mean():.0%}", WARNA_JENIS["Anorganik"]),
            ("Keluhan negatif", f"{(df['Pred_Sentimen'] == 'Negatif').mean():.0%}", WARNA_SENTIMEN["Negatif"])])
        c1, c2 = st.columns(2)
        for kolom, (kol, warna, judul) in zip((c1, c2), [("Pred_Jenis", WARNA_JENIS, "Jenis sampah"),
                                                        ("Pred_Sentimen", WARNA_SENTIMEN, "Perasaan warga")]):
            fig = px.pie(df, names=kol, hole=0.55, color=kol, color_discrete_map=warna, title=judul)
            fig.update_traces(textinfo="percent+label", showlegend=False)
            kolom.plotly_chart(_gaya(fig), width="stretch")
        masalah = df["Pred_Masalah"].value_counts().sort_values()
        st.plotly_chart(_gaya(px.bar(x=masalah.values, y=masalah.index, orientation="h", text=masalah.values,
                                     title="Masalah yang paling sering dikeluhkan", labels={"x": "", "y": ""},
                                     color_discrete_sequence=["#2E7D32"]), 320), width="stretch")
        produk = df["Rekomendasi_Produk"].dropna().str.split(r" \| ").explode().value_counts().head(8).sort_values()
        st.plotly_chart(_gaya(px.bar(x=produk.values, y=[f"{_ikon_produk(p)} {p}" for p in produk.index],
                                     orientation="h", text=produk.values, labels={"x": "", "y": ""},
                                     title="Ide daur ulang yang paling sering direkomendasikan",
                                     color_discrete_sequence=["#66BB6A"]), 360), width="stretch")

    with tab[1]:
        agg = (df.groupby("Kabupaten_Kota")
               .agg(Jumlah=("ID_Laporan", "count"),
                    Negatif=("Pred_Sentimen", lambda s: round((s == "Negatif").mean() * 100)),
                    Masalah=("Pred_Masalah", lambda s: s.mode().iat[0]))
               .reset_index())
        agg["lat"] = agg["Kabupaten_Kota"].map(lambda k: KOORDINAT[k][0])
        agg["lon"] = agg["Kabupaten_Kota"].map(lambda k: KOORDINAT[k][1])
        peta = px.scatter_map(agg, lat="lat", lon="lon", size="Jumlah", color="Negatif", hover_name="Kabupaten_Kota",
                              hover_data={"Masalah": True, "Jumlah": True, "lat": False, "lon": False},
                              color_continuous_scale="RdYlGn_r", size_max=40, zoom=6.2,
                              center={"lat": 2.0, "lon": 124.9}, map_style="open-street-map",
                              labels={"Negatif": "% negatif"},
                              title="Sebaran laporan (lingkaran besar = banyak laporan, merah = banyak keluhan)")
        st.plotly_chart(_gaya(peta, 560), width="stretch")
        bulanan = df.groupby([df["Tanggal"].dt.to_period("M").dt.to_timestamp(), "Pred_Jenis"]).size().reset_index(name="Jumlah")
        st.plotly_chart(_gaya(px.line(bulanan, x="Tanggal", y="Jumlah", color="Pred_Jenis", markers=True,
                                      color_discrete_map=WARNA_JENIS, title="Jumlah laporan per bulan",
                                      labels={"Tanggal": "", "Jumlah": ""})), width="stretch")
        total = df.groupby(df["Tanggal"].dt.to_period("M")).size()
        z = (total - total.mean()) / (total.std() or 1)
        anomali = z[z > 1.5]
        if len(anomali):
            st.warning("🚨 **Lonjakan laporan terdeteksi:** " + ", ".join(
                f"{p.strftime('%B %Y')} ({total[p]} laporan)" for p in anomali.index))
        else:
            st.success("Tidak ada lonjakan laporan yang tidak wajar.")

    with tab[2]:
        st.caption("Membandingkan jawaban AI dengan label yang benar di dataset.")
        ev = evaluasi(df)
        _kartu_statistik(st, [(f"Akurasi {n}", f"{m['akurasi']:.0%}", "#1565C0") for n, m in ev["metrik"].items()])
        pk = ev["per_kategori"]["Benar_Sentimen"].sort_values()
        fig = px.bar(x=pk.values, y=[k.replace("_", " ") for k in pk.index], orientation="h", text=[f"{v:.0f}%" for v in pk.values],
                     title="Akurasi sentimen per jenis kalimat (makin pendek = AI makin sering salah)",
                     labels={"x": "", "y": ""}, color=pk.values, color_continuous_scale="RdYlGn", range_color=[0, 100])
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(_gaya(fig, 320), width="stretch")
        kal = df.assign(Benar=df["Label_Sentimen"] == df["Pred_Sentimen"]).groupby("Benar")["Confidence"].mean()
        st.info(f"Saat jawabannya **benar**, AI rata-rata yakin **{kal.get(True, float('nan')):.0f}%**; "
                f"saat **salah**, yakin **{kal.get(False, float('nan')):.0f}%**.")
        with st.expander("Lihat confusion matrix"):
            c1, c2 = st.columns(2)
            for kolom, (nama, (a, b)) in zip((c1, c2), list(PASANGAN_EVAL.items())[:2]):
                cm = pd.crosstab(df[a], df[b])
                kolom.plotly_chart(_gaya(px.imshow(cm, text_auto=True, color_continuous_scale="Greens",
                                                   labels={"x": "Jawaban AI", "y": "Label benar"}, title=nama)),
                                   width="stretch")
        st.markdown("#### Contoh laporan yang salah dibaca AI")
        kat = st.pills("Jenis kalimat", sorted(df["Kategori_Teks"].unique()), selection_mode="multi",
                       default=[k for k in ("sarkasme", "ambigu", "sentimen_campuran") if k in set(df["Kategori_Teks"])])
        gagal = df[(df["Label_Sentimen"] != df["Pred_Sentimen"]) | (df["Label_Jenis_Sampah"] != df["Pred_Jenis"])]
        if kat:
            gagal = gagal[gagal["Kategori_Teks"].isin(kat)]
        st.dataframe(gagal[["Kategori_Teks", "Teks_Laporan", "Label_Sentimen", "Pred_Sentimen", "Confidence",
                            "Confidence_Reasoning"]].rename(columns={
                                "Kategori_Teks": "Jenis kalimat", "Teks_Laporan": "Laporan", "Label_Sentimen": "Seharusnya",
                                "Pred_Sentimen": "Jawaban AI", "Confidence_Reasoning": "Alasan AI"}),
                     hide_index=True, width="stretch")
        st.caption("Metode yang dipakai: " + ", ".join(f"{m} ({n})" for m, n in df["Metode"].value_counts().items()))

    with tab[3]:
        st.write("Saran tindakan untuk Dinas Lingkungan Hidup dan pemerintah kab/kota, disusun dari data laporan.")
        if st.button("🏛️ Susun saran", type="primary"):
            with st.spinner("Menyusun saran..."):
                st.session_state["kebijakan"] = rekomendasi_kebijakan(df, cfg)
        kb = st.session_state.get("kebijakan")
        if kb:
            _status_metode(st, kb)
            st.info(kb.get("ringkasan", ""))
            for r in kb.get("rekomendasi", []):
                with st.container(key=f"kartu_saran_{r.get('prioritas')}"):
                    st.markdown(f"**#{r.get('prioritas')} · {r.get('wilayah', '-')}** "
                                + _badge(r.get("masalah", "-"), "#6A1B9A"), unsafe_allow_html=True)
                    st.markdown(f"**Tindakan:** {r.get('tindakan', '-')}")
                    st.caption(f"👥 {r.get('pihak_terlibat', '-')} · 🎯 {r.get('indikator_keberhasilan', '-')} · "
                               f"📊 {r.get('dasar_data', '-')}")

    with tab[4]:
        ds, info = preprocess_dataset(pd.read_csv(DATASET_PATH))
        _kartu_statistik(st, [("Laporan di dataset", f"{info['baris_awal']}", "#2E7D32"),
                              ("Duplikat dibuang", f"{info['duplikat_dibuang']}", "#EF6C00"),
                              ("Sudah dianalisis", f"{len(pd.read_csv(HASIL_PATH))}", "#1565C0"),
                              ("Rata-rata kata", f"{ds['Jumlah_Kata'].mean():.1f}", "#6A1B9A")])
        with st.expander("Contoh hasil preprocessing (teks asli → teks bersih)"):
            st.dataframe(ds[["Teks_Laporan", "Teks_Bersih"]].head(20), hide_index=True, width="stretch")
        st.markdown(f"**Analisis laporan berikutnya** dengan `{cfg.label}` (melanjutkan yang belum dianalisis)")
        c1, c2 = st.columns(2)
        jumlah = c1.number_input("Jumlah laporan", 1, 1000, 50, step=10)
        ukuran = c2.number_input("Laporan per request", 1, 25, 10)
        c1, c2, c3 = st.columns(3)
        if c1.button("▶️ Lanjutkan analisis", type="primary", width="stretch"):
            _jalankan_batch(st, cfg, int(jumlah), int(ukuran))
        c2.download_button("⬇️ Unduh hasil CSV", pd.read_csv(HASIL_PATH).to_csv(index=False).encode("utf-8-sig"),
                           "hasil_analisis.csv", "text/csv", width="stretch")
        if c3.button("🗑️ Hapus hasil", width="stretch"):
            HASIL_PATH.unlink()
            st.rerun()
        if st.button("🔄 Buat ulang dataset (600 laporan)"):
            generate_dataset()
            st.rerun()


# =============================================================================
# 11. CLI
# =============================================================================

PILIHAN_PROVIDER = {"gemini": "Gemini", "openai": "OpenAI (GPT)", "deepseek": "DeepSeek", "grok": "Grok (xAI)",
                    "offline": OFFLINE}


def main_cli():
    if hasattr(sys.stdout, "reconfigure"):  # konsol Windows (cp1252) tidak bisa mencetak emoji/simbol
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser =argparse.ArgumentParser(description=f"{NAMA_SISTEM} — {KEPANJANGAN}")
    sub = parser.add_subparsers(dest="perintah")
    g = sub.add_parser("generate", help="buat dataset_sampah.csv")
    g.add_argument("--n", type=int, default=600)
    g.add_argument("--seed", type=int, default=42)
    b = sub.add_parser("batch", help="analisis dataset -> hasil_analisis.csv")
    b.add_argument("--provider", choices=PILIHAN_PROVIDER, default="gemini")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--batch-size", type=int, default=10)
    b.add_argument("--jeda", type=float, default=7.0)
    b.add_argument("--ulang", action="store_true", help="hapus hasil lama dan mulai dari awal")
    sub.add_parser("evaluate", help="hitung akurasi hasil_analisis.csv")
    sub.add_parser("fishbone", help="buat fishbone_sibersih.png")
    args = parser.parse_args()

    if args.perintah == "generate":
        df = generate_dataset(args.n, args.seed)
        print(f"✔ {len(df)} laporan disimpan ke {DATASET_PATH.name}")
        print(df["Kategori_Teks"].value_counts().to_string())
    elif args.perintah == "batch":
        if not DATASET_PATH.exists():
            generate_dataset()
        cfg = config_default(PILIHAN_PROVIDER[args.provider])
        if args.provider != "offline" and not cfg.pakai_llm:
            sys.exit(f"API key {cfg.provider} belum diisi (lihat .env.example). Pakai --provider offline untuk baseline.")
        if args.ulang and HASIL_PATH.exists():
            HASIL_PATH.unlink()
        print(f"Menganalisis dengan {cfg.label} ...")
        hasil = proses_batch(pd.read_csv(DATASET_PATH), cfg, args.batch_size, args.jeda, args.limit)
        print(f"✔ {len(hasil)} baris tersimpan di {HASIL_PATH.name}")
    elif args.perintah == "evaluate":
        df = pd.read_csv(HASIL_PATH)
        ev = evaluasi(df)
        for nama, m in ev["metrik"].items():
            print(f"{nama:<15} akurasi={m['akurasi']:.1%}  macro-F1={m['macro_f1']:.3f}")
        print("\nAkurasi (%) per kategori teks:\n" + ev["per_kategori"].to_string())
    elif args.perintah == "fishbone":
        print(f"✔ {buat_fishbone()}")
    else:
        parser.print_help()
        print("\nDashboard: streamlit run app.py")


def _jalan_di_streamlit() -> bool:
    try:
        from streamlit.runtime import exists

        return exists()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    if _jalan_di_streamlit():
        main_ui()
    else:
        main_cli()
