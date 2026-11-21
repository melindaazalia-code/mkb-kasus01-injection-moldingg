"""Logika prediksi untuk API dan aplikasi: validasi masukan, probabilitas cacat, keputusan, dan saran tindakan.
Tidak bergantung pada FastAPI/Streamlit sehingga mudah diuji."""
import json, sys, math
from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd, joblib, sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.features import clean, RAWCOLS, STD_LELEH, STD_CETAK  # noqa: E402

ART = ROOT / "artifacts"
MESIN = ["IM-01", "IM-02", "IM-03", "IM-04"]
BAHAN = ["PP", "ABS", "PC"]
SHIFT = ["pagi", "siang", "malam"]
LABEL = {"suhu_leleh_C": "Suhu leleh", "suhu_cetakan_C": "Suhu cetakan", "tekanan_injeksi_bar": "Tekanan injeksi",
         "kecepatan_injeksi_mm_s": "Kecepatan injeksi", "waktu_pendinginan_s": "Waktu pendinginan",
         "waktu_siklus_s": "Waktu siklus", "kelembaban_bahan_pct": "Kelembaban bahan"}
SATUAN = {"suhu_leleh_C": "°C", "suhu_cetakan_C": "°C", "tekanan_injeksi_bar": "bar", "kecepatan_injeksi_mm_s": "mm/s",
          "waktu_pendinginan_s": "detik", "waktu_siklus_s": "detik", "kelembaban_bahan_pct": "%"}
# batas yang masuk akal secara fisik (setelah konversi °F); di luar ini masukan ditolak
BATAS_FISIK = {"suhu_leleh_C": (150, 350), "suhu_cetakan_C": (5, 150), "tekanan_injeksi_bar": (1, 2500),
               "kecepatan_injeksi_mm_s": (1, 300), "waktu_pendinginan_s": (1, 120), "waktu_siklus_s": (2, 300),
               "kelembaban_bahan_pct": (0, 5)}
WAJIB = ["suhu_leleh_C", "suhu_cetakan_C", "kecepatan_injeksi_mm_s"]
OPSIONAL = ["tekanan_injeksi_bar", "waktu_pendinginan_s", "waktu_siklus_s", "kelembaban_bahan_pct"]


# ----------------------------------------------------------------------------- model
class Bundle:
    def __init__(self, model, meta, ranges, sumber):
        self.model, self.meta, self.ranges, self.sumber = model, meta, ranges, sumber
        self.threshold = float(meta["threshold"])


@lru_cache(maxsize=1)
def get_bundle() -> Bundle:
    """Muat artifacts/model.joblib. Jika tidak ada atau dibuat dengan versi scikit-learn lain, latih ulang dari data/raw.csv."""
    meta, model, sumber = {}, None, "artifacts"
    mp, jp = ART / "metadata.json", ART / "model.joblib"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    if jp.exists() and meta and meta.get("sklearn") == sklearn.__version__:
        try:
            model = joblib.load(jp)
        except Exception:
            model = None
    df = None
    if model is None:
        from src.train import fit_final, rentang_latih
        df = clean(pd.read_csv(ROOT / "data" / "raw.csv"))
        model, thr, _, _ = fit_final(df)
        meta = {**meta, "threshold": thr, "sklearn": sklearn.__version__, "versi_model": meta.get("versi_model", "1.0.0") + " (dilatih ulang saat start)",
                "rentang_latih": rentang_latih(df)}
        sumber = "dilatih ulang dari data/raw.csv (artifact tidak cocok dengan versi scikit-learn ini)"
    ranges = meta.get("rentang_latih")
    if not ranges and (ROOT / "data" / "raw.csv").exists():
        from src.train import rentang_latih
        ranges = rentang_latih(df if df is not None else clean(pd.read_csv(ROOT / "data" / "raw.csv")))
    return Bundle(model, meta, ranges or {}, sumber)


# ----------------------------------------------------------------------------- validasi
def _angka(v):
    """None/kosong -> None; '0,18' -> 0.18; teks non-angka -> ValueError."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip().replace(",", ".")
        if v == "" or v.lower() in {"nan", "na", "n/a", "-", "none", "null"}:
            return None
    x = float(v)
    if math.isnan(x):
        return None
    if math.isinf(x):
        raise ValueError("tak hingga")
    return x


def validasi(masukan: dict):
    """Kembalikan (baris_mentah, galat, peringatan). Jika ada galat, jangan prediksi."""
    galat, warn, baris = [], [], {}
    mesin = str(masukan.get("mesin") or "").strip().upper()
    bahan = str(masukan.get("bahan") or "").strip().upper().replace(".", "")
    shift = str(masukan.get("shift") or "").strip().lower()
    if mesin not in MESIN: galat.append(f"Mesin '{masukan.get('mesin')}' tidak dikenal. Pilih salah satu: {', '.join(MESIN)}.")
    if bahan not in BAHAN: galat.append(f"Bahan '{masukan.get('bahan')}' tidak dikenal. Pilih salah satu: {', '.join(BAHAN)}.")
    if shift not in SHIFT:
        if shift == "":
            warn.append("Shift tidak diisi; dipakai 'tidak diketahui' sehingga prediksi sedikit kurang pasti.")
            shift = "tidak diketahui"
        else:
            galat.append(f"Shift '{masukan.get('shift')}' tidak dikenal. Pilih salah satu: {', '.join(SHIFT)}.")
    baris.update(mesin=mesin, bahan=bahan, shift=shift)

    for k in WAJIB + OPSIONAL:
        nama = LABEL[k]
        try:
            v = _angka(masukan.get(k))
        except (ValueError, TypeError):
            galat.append(f"{nama}: '{masukan.get(k)}' bukan angka yang valid.")
            baris[k] = np.nan
            continue
        if v is None:
            if k in WAJIB:
                galat.append(f"{nama} wajib diisi.")
            else:
                warn.append(f"{nama} dikosongkan; diisi nilai tengah data latih, sehingga prediksi kurang pasti.")
            baris[k] = np.nan
            continue
        nilai = v
        if k == "suhu_leleh_C" and bahan in STD_LELEH and v > 1.35 * STD_LELEH[bahan]:
            konv = (v - 32) * 5 / 9
            if BATAS_FISIK[k][0] <= konv <= BATAS_FISIK[k][1]:      # hanya dianggap °F bila hasil konversinya masuk akal
                nilai = konv
                warn.append(f"Suhu leleh {v:g} tampak dalam °F; dikonversi menjadi {nilai:.1f} °C.")
        if k == "tekanan_injeksi_bar" and v <= 0:
            warn.append("Tekanan injeksi 0 atau negatif dianggap sensor rusak dan diperlakukan sebagai kosong.")
            baris[k] = v; continue
        if k == "waktu_siklus_s" and v <= 0:
            warn.append("Waktu siklus ≤ 0 dianggap penanda hilang dan diperlakukan sebagai kosong.")
            baris[k] = v; continue
        lo, hi = BATAS_FISIK[k]
        if not (lo <= nilai <= hi):
            galat.append(f"{nama} {nilai:g} {SATUAN[k]} tidak masuk akal (batas {lo:g}–{hi:g} {SATUAN[k]}). Periksa satuan atau salah ketik.")
        baris[k] = v
    if galat:
        return baris, galat, warn
    # kewajaran lintas-kolom dan rentang data latih
    pend, sik = baris.get("waktu_pendinginan_s"), baris.get("waktu_siklus_s")
    if pend is not None and sik is not None and not (isinstance(pend, float) and math.isnan(pend)) and not (isinstance(sik, float) and math.isnan(sik)) and sik > 0 and sik < pend:
        warn.append("Waktu siklus lebih pendek daripada waktu pendinginan; periksa kembali kedua angka.")
    rg = get_bundle().ranges
    for k in WAJIB + OPSIONAL:
        v = baris.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)) or k not in rg:
            continue
        if k == "suhu_leleh_C" and bahan in STD_LELEH and v > 1.35 * STD_LELEH[bahan]:
            v = (v - 32) * 5 / 9
        if k in ("tekanan_injeksi_bar", "waktu_siklus_s") and v <= 0:
            continue
        if v < rg[k]["min"] or v > rg[k]["max"]:
            warn.append(f"{LABEL[k]} {v:g} {SATUAN[k]} di luar rentang data latih ({rg[k]['min']:g}–{rg[k]['max']:g}); prediksi kurang dapat dipercaya.")
    return baris, galat, warn


# ----------------------------------------------------------------------------- prediksi
def _proba(model, baris: dict) -> float:
    x = clean(pd.DataFrame([baris]), drop_duplicates=False)
    return float(model.predict_proba(x[RAWCOLS])[0, 1])


def _bersih(baris: dict) -> dict:
    x = clean(pd.DataFrame([baris]), drop_duplicates=False).iloc[0]
    return {k: (None if (isinstance(x[k], float) and math.isnan(x[k])) else x[k]) for k in RAWCOLS}


def saran(baris: dict, p0: float):
    """Aturan sederhana dari pola EDA, masing-masing diuji dengan model: berapa probabilitasnya bila diperbaiki."""
    b = _bersih(baris); model = get_bundle().model
    hig = b["bahan"] in ("ABS", "PC")
    kandidat = []   # (teks, {kolom: nilai_baru})
    lb = b["kelembaban_bahan_pct"]
    if hig and lb is not None and lb > 0.10:
        kandidat.append((f"Keringkan bahan {b['bahan']} sebelum dicetak: kelembaban {lb:.3f}% di atas 0,10%.", {"kelembaban_bahan_pct": 0.05}))
    tk, tmin = b["tekanan_injeksi_bar"], (900 if hig else 800)
    if tk is not None and tk < tmin:
        kandidat.append((f"Naikkan tekanan injeksi ke ≥ {tmin + 50} bar (sekarang {tk:.0f} bar).", {"tekanan_injeksi_bar": float(tmin + 50)}))
    dl = b["suhu_leleh_C"] - STD_LELEH[b["bahan"]]
    if abs(dl) > 8:
        kandidat.append((f"Kembalikan suhu leleh ke sekitar {STD_LELEH[b['bahan']]} °C (sekarang {b['suhu_leleh_C']:.0f} °C, {'terlalu panas' if dl > 0 else 'terlalu dingin'}).", {"suhu_leleh_C": float(STD_LELEH[b["bahan"]])}))
    dc = b["suhu_cetakan_C"] - STD_CETAK[b["bahan"]]
    if abs(dc) > 8:
        kandidat.append((f"Kembalikan suhu cetakan ke sekitar {STD_CETAK[b['bahan']]} °C (sekarang {b['suhu_cetakan_C']:.0f} °C).", {"suhu_cetakan_C": float(STD_CETAK[b["bahan"]])}))
    pd_ = b["waktu_pendinginan_s"]
    if pd_ is not None and pd_ < 12:
        kandidat.append((f"Perpanjang waktu pendinginan ke ≥ 14 detik (sekarang {pd_:.1f} detik).", {"waktu_pendinginan_s": 14.0}))
    hasil = []
    for teks, ubah in kandidat:
        p1 = _proba(model, {**baris, **ubah})
        hasil.append({"tindakan": teks, "probabilitas_jika_diperbaiki": p1})
    hasil.sort(key=lambda r: r["probabilitas_jika_diperbaiki"])
    semua = None
    if len(kandidat) > 1:
        ub = {}
        for _, u in kandidat: ub.update(u)
        semua = _proba(model, {**baris, **ub})
    catatan = []
    if b["mesin"] == "IM-03":
        catatan.append("IM-03 cacatnya sekitar dua kali mesin lain dan suhu lelehnya rata-rata +6 °C di atas standar; minta teknisi memeriksa sensor suhu dan pemanas barrel.")
    return hasil, semua, catatan


def prediksi(masukan: dict) -> dict:
    bundle = get_bundle()
    baris, galat, warn = validasi(masukan)
    if galat:
        return {"valid": False, "galat": galat, "peringatan": warn}
    p = _proba(bundle.model, baris)
    thr = bundle.threshold
    if p >= thr:
        status = "bahaya" if p >= 0.30 else "waspada"
        keputusan = "SISIHKAN komponen ini dan periksa pengaturan mesin."
    else:
        status = "aman"
        keputusan = "LOLOS: komponen boleh dilanjutkan."
    tindakan, semua, catatan = saran(baris, p)
    if status != "aman" and not tindakan:
        catatan.append("Tidak ada parameter yang jelas menyimpang menurut aturan sederhana; periksa komponen secara visual dan parameter lain.")
    if status == "aman" and not tindakan:
        catatan.append("Parameter berada dalam rentang yang aman.")
    return {"valid": True, "probabilitas_cacat": p, "threshold": thr, "status": status, "keputusan": keputusan,
            "peringatan": warn, "saran": tindakan, "probabilitas_jika_semua_diperbaiki": semua, "catatan": catatan,
            "dinormalkan": {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in _bersih(baris).items()}}


def info() -> dict:
    b = get_bundle()
    return {"versi_model": b.meta.get("versi_model"), "algoritma": b.meta.get("algoritma"), "threshold": b.threshold,
            "metrik": b.meta.get("metrik"), "asumsi_biaya_rp": b.meta.get("asumsi_biaya_rp"), "sumber_model": b.sumber,
            "sklearn": sklearn.__version__}
