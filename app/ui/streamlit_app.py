"""Aplikasi operator. Jalankan dari akar repositori:  streamlit run app/ui/streamlit_app.py
Secara bawaan memanggil model langsung (cocok untuk Streamlit Community Cloud).
Jika variabel lingkungan API_URL diisi (mis. http://localhost:8000), aplikasi memanggil API FastAPI."""
import json, os, sys, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app import predictor as P

st.set_page_config(page_title="Cek Risiko Cacat Cetak", page_icon="🏭", layout="centered")

FIELD_NUM = [  # (kunci, label, bantuan, wajib)
    ("suhu_leleh_C", "Suhu leleh (°C)", "Suhu plastik cair di barrel", True),
    ("suhu_cetakan_C", "Suhu cetakan (°C)", "", True),
    ("tekanan_injeksi_bar", "Tekanan injeksi (bar)", "Boleh kosong bila sensor rusak", False),
    ("kecepatan_injeksi_mm_s", "Kecepatan injeksi (mm/s)", "", True),
    ("waktu_pendinginan_s", "Waktu pendinginan (detik)", "Boleh kosong", False),
    ("waktu_siklus_s", "Waktu siklus (detik)", "Boleh kosong", False),
    ("kelembaban_bahan_pct", "Kelembaban bahan (%)", "Kadar air sebelum dicetak; boleh kosong", False),
]
CONTOH = {
    "Normal (PP)": dict(mesin="IM-01", bahan="PP", shift="pagi", suhu_leleh_C="230", suhu_cetakan_C="40", tekanan_injeksi_bar="950",
                        kecepatan_injeksi_mm_s="80", waktu_pendinginan_s="18", waktu_siklus_s="32", kelembaban_bahan_pct="0.02"),
    "ABS lembab dan tekanan rendah (IM-03)": dict(mesin="IM-03", bahan="ABS", shift="malam", suhu_leleh_C="255", suhu_cetakan_C="62",
                        tekanan_injeksi_bar="760", kecepatan_injeksi_mm_s="80", waktu_pendinginan_s="11", waktu_siklus_s="26", kelembaban_bahan_pct="0.18"),
    "Suhu leleh dalam °F": dict(mesin="IM-02", bahan="PC", shift="siang", suhu_leleh_C="572", suhu_cetakan_C="90",
                        tekanan_injeksi_bar="900", kecepatan_injeksi_mm_s="80", waktu_pendinginan_s="18", waktu_siklus_s="32", kelembaban_bahan_pct="0.05"),
    "Masukan tidak wajar": dict(mesin="IM-02", bahan="PC", shift="siang", suhu_leleh_C="9999", suhu_cetakan_C="abc",
                        tekanan_injeksi_bar="", kecepatan_injeksi_mm_s="80", waktu_pendinginan_s="", waktu_siklus_s="", kelembaban_bahan_pct="-3"),
    "Kosongkan semua": dict(mesin="IM-01", bahan="PP", shift="pagi", **{k: "" for k, *_ in FIELD_NUM}),
}


def terapkan_contoh():
    for k, v in CONTOH[st.session_state["contoh"]].items():
        st.session_state["f_" + k] = v


if "f_mesin" not in st.session_state:
    st.session_state["contoh"] = "Normal (PP)"
    terapkan_contoh()


@st.cache_resource(show_spinner="Memuat model...")
def muat():
    P.get_bundle()
    return P.info()


def panggil(masukan: dict) -> dict:
    api = os.environ.get("API_URL", "").rstrip("/")
    if not api:
        return P.prediksi(masukan)
    req = urllib.request.Request(api + "/predict", data=json.dumps(masukan).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())          # 422 berisi daftar galat


def persen(x):
    return f"{x * 100:.1f}%".replace(".", ",")


info = muat()
with st.sidebar:
    st.subheader("Tentang model")
    st.write(f"Model versi {info['versi_model']}")
    st.write(f"Threshold keputusan: **{persen(info['threshold'])}**")
    if info.get("metrik"):
        st.write(f"PR-AUC test: {info['metrik'].get('final_pr_test', 0):.3f} (baseline {info['metrik'].get('baseline_pr_test', 0):.3f})")
    st.caption("Threshold dipilih dari biaya: komponen cacat lolos Rp500.000, komponen bagus tersisih Rp25.000. "
               "Karena melewatkan cacat 20× lebih mahal, komponen sudah disisihkan pada probabilitas yang rendah.")
    st.caption("Model memprediksi, tidak membuktikan sebab. Saran tindakan perlu divalidasi tim proses.")

st.title("🏭 Cek Risiko Cacat Cetak")
st.write("Isi parameter siklus cetak, lalu tekan **Hitung risiko**. Aplikasi memberi probabilitas cacat dan saran tindakan.")
st.selectbox("Contoh masukan (untuk mencoba)", list(CONTOH), key="contoh", on_change=terapkan_contoh)

c1, c2, c3 = st.columns(3)
mesin = c1.selectbox("Mesin", P.MESIN, key="f_mesin")
bahan = c2.selectbox("Bahan", P.BAHAN, key="f_bahan")
shift = c3.selectbox("Shift", P.SHIFT, key="f_shift")
kiri, kanan = st.columns(2)
nilai = {}
for i, (k, label, bantuan, wajib) in enumerate(FIELD_NUM):
    nilai[k] = (kiri if i % 2 == 0 else kanan).text_input(label + (" *" if wajib else ""), key="f_" + k, help=bantuan or None)
st.caption("* wajib diisi. Gunakan titik atau koma untuk desimal.")

if st.button("Hitung risiko", type="primary"):
    masukan = {"mesin": mesin, "bahan": bahan, "shift": shift, **nilai}
    try:
        hasil = panggil(masukan)
    except Exception as e:
        st.error(f"Tidak dapat menghubungi layanan prediksi ({type(e).__name__}). Coba lagi.")
        st.stop()
    if not hasil.get("valid"):
        st.error("Masukan belum bisa diproses. Perbaiki hal berikut:")
        for g in hasil.get("galat", []):
            st.markdown(f"- {g}")
        for w in hasil.get("peringatan", []):
            st.caption("ℹ️ " + w)
    else:
        p, thr = hasil["probabilitas_cacat"], hasil["threshold"]
        m1, m2 = st.columns(2)
        m1.metric("Probabilitas cacat", persen(p))
        m2.metric("Threshold", persen(thr))
        {"aman": st.success, "waspada": st.warning, "bahaya": st.error}[hasil["status"]](f"**{hasil['keputusan']}**")
        for w in hasil["peringatan"]:
            st.warning(w)
        if hasil["saran"]:
            st.subheader("Saran tindakan")
            st.caption("Urut dari yang paling menurunkan risiko. Kolom kanan: perkiraan probabilitas bila hanya tindakan itu yang dilakukan.")
            st.table(pd.DataFrame([{"Tindakan": s["tindakan"], "Probabilitas bila dilakukan": persen(s["probabilitas_jika_diperbaiki"])} for s in hasil["saran"]]))
            if hasil.get("probabilitas_jika_semua_diperbaiki") is not None:
                st.write(f"Bila **semua** tindakan dilakukan: perkiraan probabilitas cacat **{persen(hasil['probabilitas_jika_semua_diperbaiki'])}**.")
        for c in hasil["catatan"]:
            st.info(c)
