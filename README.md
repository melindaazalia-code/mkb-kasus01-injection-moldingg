# Prediksi Cacat *Injection Molding* · Tim [nama tim]

**Kasus:** 01 · Pengendalian mutu *injection molding* · **Aplikasi:** [https://mkb-kasus01-injection-moldingg-syjvmepvrmsdcq4khzpory.streamlit.app/](https://mkb-kasus01-injection-moldingg-syjvmepvrmsdcq4khzpory.streamlit.app/) · **Video:** [tautan video]

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| Navanza Varel | [NIM] | [misalnya EDA dan pembersihan data] |
| Melinda Azalia Fausta | 2407608 | [misalnya model dan evaluasi] |
| [nama] | [NIM] | [misalnya API dan aplikasi] |

## Masalah dan pengguna

Kepala QC pabrik komponen plastik di Cikarang baru mengetahui komponen cacat saat inspeksi visual di akhir *shift*, sehingga ratusan komponen lain bisa ikut cacat. Model ini menerima parameter satu siklus cetak dan menghitung probabilitas komponen cacat. Operator mesin memakainya untuk menyisihkan komponen yang dicurigai dan memperbaiki pengaturan mesin lebih awal.

## Data dan pembersihan

`data/raw.csv` berisi 25.350 siklus cetak dari 4 mesin (Maret–Juni 2026). Setelah pembersihan tersisa 25.000 baris (`data/data_bersih.csv`) dengan 7,76% cacat.

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| Baris duplikat persis | 350 | `drop_duplicates()` sebelum data dibagi, supaya tidak bocor ke *test* |
| Ejaan `bahan` tidak seragam (`P.P`, `pp`, `Abs`, ...) | 750 | strip, huruf besar, hapus titik: PP / ABS / PC |
| Ejaan `shift` tidak seragam (`SIANG`, `Pagi`, `malam␣`) | 500 | strip dan huruf kecil |
| `suhu_leleh_C` ternyata dalam Fahrenheit (nilai 384–615) | 579 | dikonversi `(F−32)×5/9`; sebaran per bahan jatuh di 230/240/300 °C |
| `tekanan_injeksi_bar` = 0 (sensor rusak) | 125 | diganti kosong (NaN) |
| `waktu_siklus_s` = −1 (penanda hilang) | 75 | diganti kosong (NaN) |
| `waktu_pendinginan_s` kosong | 250 | imputasi median di dalam *pipeline* (statistik dari *train*) |
| `kelembaban_bahan_pct` kosong | 1.000 | imputasi median di dalam *pipeline* |

Catatan: nilai `shift` tidak cocok dengan jam `waktu_produksi` (tiap label tersebar rata di semua jam); label dipakai apa adanya dan perlu dikonfirmasi ke klien.

## Hasil

Data dibagi *stratified* 60/20/20 (15.000 / 5.000 / 5.000, masing-masing 7,76% cacat). Metrik utama PR-AUC; tebakan acak = 0,078. Angka di bawah berasal dari notebook yang dijalankan di Google Colab.

| Model | PR-AUC *validation* | PR-AUC *test* |
|---|---|---|
| *Baseline*: regresi logistik, fitur asli | 0,483 | 0,536 |
| Regresi logistik + fitur turunan | 0,629 | n/a |
| ***Gradient boosting* + fitur turunan (final)** | **0,649** | **0,698** |

*Threshold* yang dipakai: **4,25%**, karena komponen cacat yang lolos merugikan Rp500.000 sedangkan komponen bagus yang tersisih Rp25.000 (titik impas teoretis 25.000/525.000 = 4,76%; pemindaian biaya di *validation* memberi 4,25%). Pada *test*: 91% komponen cacat tertangkap dengan 25% komponen disisihkan, biaya sekitar Rp8,0 juta per 1.000 komponen (baseline Rp12,0 juta; menyisihkan semua Rp23,1 juta).

> Angka bisa bergeser di belakang koma bila dijalankan dengan versi scikit-learn lain (di versi 1.9 hasilnya PR-AUC *test* 0,696 dan *threshold* 3,25%). Kesimpulannya tidak berubah.

## Aplikasi

- **Aplikasi operator (Streamlit), sudah di-*deploy*** (tautan di bagian atas). Operator mengisi mesin, bahan, shift, dan parameter proses; keluarannya probabilitas cacat, keputusan (lolos / sisihkan), peringatan, dan saran tindakan beserta perkiraan probabilitas bila tindakan dilakukan. Aplikasi memanggil model **langsung** lewat `app/predictor.py` (validasi masukan, prediksi, saran), tanpa melalui API. Masukan kosong, bukan angka, atau mustahil ditolak dengan pesan yang jelas, tidak membuat aplikasi mati. Ada pilihan "Contoh masukan" untuk mencoba.
- **API (FastAPI), hanya dijalankan lokal, tidak di-*deploy*.** Disertakan sesuai struktur repositori pada templat tugas (`app/api/main.py`). Memakai `app/predictor.py` yang sama: `POST /predict`, `GET /info`, `GET /health`, dokumentasi otomatis di `/docs`; masukan tidak valid mengembalikan status 422 berisi daftar galat.

## Cara menjalankan

```bash
pip install -r requirements.txt
python -m src.train                                  # (opsional) latih ulang, tulis artifacts/
uvicorn app.api.main:app --reload                    # API  -> http://localhost:8000/docs
streamlit run app/ui/streamlit_app.py                # aplikasi -> http://localhost:8501
python tests/test_predictor.py                       # uji logika prediksi
```

Untuk membuat aplikasi memakai API (bukan memanggil model langsung), jalankan API dulu lalu `API_URL=http://localhost:8000 streamlit run app/ui/streamlit_app.py`.

Notebook `notebooks/01_kasus01_injection_molding.ipynb` dijalankan di Google Colab (`raw.csv`, `features.py`, dan `train.py` diambil dari Google Drive di sel awal).

### Memakai model hasil Colab

Notebook menyimpan `model.joblib` dan `metadata.json` di `/content/artifacts/` pada Colab. Agar angka aplikasi sama dengan laporan: unduh dua berkas itu, timpa isi `artifacts/`, lalu di Colab jalankan `import sklearn; print(sklearn.__version__)` dan tulis versinya di `requirements.txt` (mis. `scikit-learn==1.6.1`). Bila versinya tidak cocok, aplikasi tidak error: ia melatih ulang model dari `data/raw.csv` saat dinyalakan (beberapa detik) dan memakai *threshold* hasil pelatihan ulang itu.

### *Deploy* agar punya tautan

Cara termudah memakai Streamlit Community Cloud: dorong repositori ke GitHub (publik), buka share.streamlit.io, pilih repositori ini, isi *Main file path* `app/ui/streamlit_app.py`. Aplikasi sudah di-*deploy* dengan cara ini (tautan di bagian atas README). API tidak wajib di-*deploy* karena aplikasi memanggil model langsung.

## Struktur repositori

```
notebooks/      01_kasus01_injection_molding.ipynb: dari data mentah sampai penyimpanan model
data/           raw.csv (mentah) dan data_bersih.csv
src/features.py clean() dan feats(): dipakai saat pelatihan dan saat prediksi
src/train.py    melatih dan menyimpan model + metadata
artifacts/      model.joblib + metadata.json (versi, fitur masukan, threshold, metrik, rentang data latih)
app/predictor.py  validasi, prediksi, saran tindakan (dipakai API dan Streamlit)
app/api/main.py   API FastAPI
app/ui/streamlit_app.py  aplikasi operator
tests/          uji logika prediksi
laporan/        LAPORAN.md dan gambar
```

## Keterbatasan

- Tiap bagian data hanya berisi 388 komponen cacat, sehingga PR-AUC *test* bergeser sekitar ±0,04–0,05 karena sampel saja.
- Model paling lemah pada PP (cacat hanya 3,7% dan pemicunya sedikit).
- Data hanya dari 4 mesin dan 4 bulan; mesin, bahan, atau musim baru dapat menggeser pola. Aplikasi memberi peringatan bila masukan di luar rentang data latih.
- Model memprediksi, tidak membuktikan sebab; saran tindakan dibuat dari aturan sederhana berdasarkan pola data dan perlu divalidasi tim proses.
- Suhu leleh IM-03 rata-rata +6 °C di atas standar dan cacatnya dua kali mesin lain; belum diketahui apakah sensor atau suhunya yang menyimpang.

