"""API prediksi cacat. Jalankan dari akar repositori:  uvicorn app.api.main:app --reload
Dokumentasi interaktif otomatis di /docs."""
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import predictor as P

app = FastAPI(title="API Prediksi Cacat Injection Molding", version="1.0.0",
              description="Menerima parameter satu siklus cetak dan mengembalikan probabilitas cacat, keputusan, dan saran tindakan.")


class Masukan(BaseModel):
    # Sengaja bertipe longgar (Any): validasi dilakukan oleh app.predictor supaya masukan aneh
    # (teks, kosong, mustahil) mendapat pesan galat yang jelas, bukan galat 500.
    mesin: Optional[Any] = None            # IM-01 .. IM-04
    bahan: Optional[Any] = None            # PP, ABS, PC
    shift: Optional[Any] = None            # pagi, siang, malam
    suhu_leleh_C: Optional[Any] = None
    suhu_cetakan_C: Optional[Any] = None
    tekanan_injeksi_bar: Optional[Any] = None
    kecepatan_injeksi_mm_s: Optional[Any] = None
    waktu_pendinginan_s: Optional[Any] = None
    waktu_siklus_s: Optional[Any] = None
    kelembaban_bahan_pct: Optional[Any] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    """Versi model, threshold, dan metrik dari artifacts/metadata.json."""
    return P.info()


@app.post("/predict")
def predict(m: Masukan):
    data = m.model_dump() if hasattr(m, "model_dump") else m.dict()
    try:
        hasil = P.prediksi(data)
    except Exception as e:  # tidak boleh membuat API mati
        return JSONResponse(status_code=500, content={"valid": False, "galat": [f"Kesalahan internal: {type(e).__name__}"], "peringatan": []})
    if not hasil["valid"]:
        return JSONResponse(status_code=422, content=hasil)
    return hasil
