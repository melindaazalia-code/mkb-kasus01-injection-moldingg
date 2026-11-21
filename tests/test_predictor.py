"""Uji logika prediksi: jalankan dengan `python -m pytest tests` atau `python tests/test_predictor.py`."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.predictor import prediksi, info

NORMAL_PP = dict(mesin="IM-01", bahan="PP", shift="pagi", suhu_leleh_C=230, suhu_cetakan_C=40, tekanan_injeksi_bar=950,
                 kecepatan_injeksi_mm_s=80, waktu_pendinginan_s=18, waktu_siklus_s=32, kelembaban_bahan_pct=0.02)
ABS_LEMBAB = dict(mesin="IM-03", bahan="ABS", shift="malam", suhu_leleh_C=255, suhu_cetakan_C=62, tekanan_injeksi_bar=760,
                  kecepatan_injeksi_mm_s=80, waktu_pendinginan_s=11, waktu_siklus_s=26, kelembaban_bahan_pct=0.18)


def test_normal_lolos():
    r = prediksi(NORMAL_PP)
    assert r["valid"] and r["status"] == "aman" and r["probabilitas_cacat"] < r["threshold"]


def test_berisiko_disisihkan_dan_ada_saran():
    r = prediksi(ABS_LEMBAB)
    assert r["valid"] and r["status"] in ("waspada", "bahaya")
    assert len(r["saran"]) >= 2
    assert r["probabilitas_jika_semua_diperbaiki"] < r["probabilitas_cacat"]


def test_masukan_kosong_wajib_ditolak():
    r = prediksi({})
    assert not r["valid"] and len(r["galat"]) >= 4


def test_mustahil_ditolak():
    r = prediksi({**NORMAL_PP, "suhu_leleh_C": 9999})
    assert not r["valid"] and any("tidak masuk akal" in g for g in r["galat"])
    r = prediksi({**NORMAL_PP, "kelembaban_bahan_pct": -3})
    assert not r["valid"]


def test_bukan_angka_ditolak_tanpa_crash():
    r = prediksi({**NORMAL_PP, "tekanan_injeksi_bar": "abc"})
    assert not r["valid"] and any("bukan angka" in g for g in r["galat"])


def test_opsional_kosong_tetap_diprediksi_dengan_peringatan():
    r = prediksi({**NORMAL_PP, "kelembaban_bahan_pct": "", "tekanan_injeksi_bar": None})
    assert r["valid"] and len(r["peringatan"]) >= 2


def test_fahrenheit_dikonversi():
    r = prediksi({**NORMAL_PP, "suhu_leleh_C": 446})
    assert r["valid"] and abs(r["dinormalkan"]["suhu_leleh_C"] - 230) < 0.5
    assert any("°F" in w for w in r["peringatan"])


def test_tekanan_nol_dianggap_sensor_rusak():
    r = prediksi({**NORMAL_PP, "tekanan_injeksi_bar": 0})
    assert r["valid"] and any("sensor rusak" in w for w in r["peringatan"])


def test_koma_desimal_dan_ejaan_bahan():
    r = prediksi({**NORMAL_PP, "bahan": " p.p ", "kelembaban_bahan_pct": "0,02"})
    assert r["valid"]


def test_di_luar_rentang_latih_diberi_peringatan():
    r = prediksi({**NORMAL_PP, "kecepatan_injeksi_mm_s": 250})
    assert r["valid"] and any("rentang data latih" in w for w in r["peringatan"])


def test_kategori_tak_dikenal_ditolak():
    assert not prediksi({**NORMAL_PP, "mesin": "IM-09"})["valid"]
    assert not prediksi({**NORMAL_PP, "bahan": "PVC"})["valid"]


def test_info():
    assert 0 < info()["threshold"] < 1


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("ok", k)
    print(n, "uji lulus")
