"""Pembersihan data dan pembuatan fitur. Dipakai saat pelatihan DAN saat prediksi (API/aplikasi)."""
import numpy as np
import pandas as pd

STD_LELEH = {"PP": 230, "ABS": 240, "PC": 300}   # suhu leleh standar (C), dari tim proses
STD_CETAK = {"PP": 40, "ABS": 60, "PC": 90}      # suhu cetakan standar (C)
NUM = ["suhu_leleh_C", "suhu_cetakan_C", "tekanan_injeksi_bar", "kecepatan_injeksi_mm_s",
       "waktu_pendinginan_s", "waktu_siklus_s", "kelembaban_bahan_pct"]
CAT = ["mesin", "shift", "bahan"]
RAWCOLS = NUM + CAT


def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Bersihkan data mentah (atau satu masukan operator). Lihat laporan bagian 2 untuk alasannya."""
    df = df.copy()
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    if "waktu_produksi" in df:
        df["waktu_produksi"] = pd.to_datetime(df["waktu_produksi"])
    df["bahan"] = df["bahan"].astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
    df["shift"] = df["shift"].astype(str).str.strip().str.lower()
    # sebagian suhu leleh tercatat dalam Fahrenheit: nilai > 1,35 x suhu standar bahan
    f = df["suhu_leleh_C"] > 1.35 * df["bahan"].map(STD_LELEH)
    df.loc[f, "suhu_leleh_C"] = ((df["suhu_leleh_C"] - 32) * 5 / 9).round(1)
    # penanda sensor rusak -> kosong
    df.loc[df["tekanan_injeksi_bar"] <= 0, "tekanan_injeksi_bar"] = np.nan
    df.loc[df["waktu_siklus_s"] <= 0, "waktu_siklus_s"] = np.nan
    return df


def feats(d: pd.DataFrame) -> pd.DataFrame:
    """Fitur turunan dari pengetahuan proses."""
    d = d.copy()
    b = d["bahan"]
    d["dev_leleh"] = d.suhu_leleh_C - b.map(STD_LELEH)
    d["dev_cetak"] = d.suhu_cetakan_C - b.map(STD_CETAK)
    d["abs_dev_leleh"] = d.dev_leleh.abs()
    d["abs_dev_cetak"] = d.dev_cetak.abs()
    hig = b.isin(["ABS", "PC"]).astype(float)
    d["higroskopis"] = hig
    d["lembab_x_higro"] = d.kelembaban_bahan_pct * hig
    d["lembab_lebih"] = np.maximum(0, d.kelembaban_bahan_pct - 0.1) * hig
    d["tekanan_kurang"] = np.maximum(0, 900 - d.tekanan_injeksi_bar)
    d["tekanan_kurang_higro"] = d.tekanan_kurang * hig
    d["pendinginan_kurang"] = np.maximum(0, 12 - d.waktu_pendinginan_s)
    d["overhead_siklus"] = d.waktu_siklus_s - d.waktu_pendinginan_s
    d["tek_tercatat"] = d.tekanan_injeksi_bar.notna().astype(float)
    d["im03"] = (d.mesin == "IM-03").astype(float)
    return d[["dev_leleh", "dev_cetak", "abs_dev_leleh", "abs_dev_cetak", "higroskopis", "lembab_x_higro",
              "lembab_lebih", "tekanan_kurang", "tekanan_kurang_higro", "pendinginan_kurang", "overhead_siklus",
              "tek_tercatat", "im03", *NUM, *CAT]]


ENG_NUM = [c for c in feats(pd.DataFrame({
    "bahan": ["PP"], "mesin": ["IM-01"], "shift": ["pagi"], **{c: [1.0] for c in NUM}})).columns if c not in CAT]
