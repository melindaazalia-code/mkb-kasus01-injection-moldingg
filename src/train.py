"""Latih baseline + model final, pilih threshold di validation, buka test SATU kali, simpan artifacts.
Jalankan dari akar repositori:  python -m src.train
"""
import json, datetime, platform
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from src.features import clean, feats, NUM, CAT, RAWCOLS, ENG_NUM

SEED, C_FN, C_FP = 42, 500_000, 25_000
HGB_PARAMS = dict(learning_rate=0.06, max_leaf_nodes=8, min_samples_leaf=60, l2_regularization=3.0)  # dipilih dari grid di validation (notebook Colab)


def make_baseline():
    return Pipeline([("prep", ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
        ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])), ("m", LogisticRegression(max_iter=2000))])


def make_final():
    cat_idx = list(range(len(ENG_NUM), len(ENG_NUM) + len(CAT)))
    prep = ColumnTransformer([("n", "passthrough", ENG_NUM),
                              ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)])
    return Pipeline([("f", FunctionTransformer(feats)), ("prep", prep), ("m", HistGradientBoostingClassifier(
        max_iter=800, early_stopping=True, validation_fraction=0.15, n_iter_no_change=40,
        categorical_features=cat_idx, random_state=SEED, **HGB_PARAMS))])


def split(df):
    tr, tmp = train_test_split(df, test_size=0.4, stratify=df.cacat, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.cacat, random_state=SEED)
    return tr, va, te


def cost(y, p, t):
    pred = p >= t
    return C_FN * ((~pred) & (y == 1)).sum() + C_FP * (pred & (y == 0)).sum()


def fit_final(df):
    """Latih model final pada train, pilih threshold di validation. Dipakai main() dan aplikasi (jika artifact tidak cocok)."""
    tr, va, te = split(df)
    final = make_final().fit(tr[RAWCOLS], tr.cacat)
    pv = final.predict_proba(va[RAWCOLS])[:, 1]
    ths = np.round(np.arange(0.005, 0.95, 0.0025), 4)
    thr = float(ths[np.argmin([cost(va.cacat.values, pv, t) for t in ths])])
    return final, thr, (tr, va, te), pv


def rentang_latih(df):
    """Rentang tiap fitur numerik di data latih, untuk peringatan 'di luar rentang' di aplikasi."""
    return {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                "p01": float(df[c].quantile(0.001)), "p99": float(df[c].quantile(0.999))} for c in NUM}


def main():
    df = clean(pd.read_csv("data/raw.csv"))
    final, thr, (tr, va, te), pv = fit_final(df)
    base = make_baseline().fit(tr[RAWCOLS], tr.cacat)
    # --- data test dibuka di sini, satu kali ---
    m = {"baseline_pr_val": average_precision_score(va.cacat, base.predict_proba(va[RAWCOLS])[:, 1]),
         "baseline_pr_test": average_precision_score(te.cacat, base.predict_proba(te[RAWCOLS])[:, 1]),
         "final_pr_val": average_precision_score(va.cacat, pv),
         "final_pr_test": average_precision_score(te.cacat, final.predict_proba(te[RAWCOLS])[:, 1])}
    print(json.dumps(m, indent=1), "threshold:", thr)
    joblib.dump(final, "artifacts/model.joblib")
    json.dump({"versi_model": "1.0.0", "dilatih": datetime.datetime.now().isoformat(timespec="seconds"),
               "algoritma": "HistGradientBoostingClassifier + fitur turunan (src/features.py)",
               "hyperparameter": HGB_PARAMS, "fitur_masukan": RAWCOLS, "target": "cacat",
               "threshold": thr, "asumsi_biaya_rp": {"cacat_lolos": C_FN, "bagus_tersisih": C_FP},
               "metrik": m, "rentang_latih": rentang_latih(df), "seed": SEED,
               "sklearn": sklearn.__version__, "python": platform.python_version(),
               "catatan": "Masukan mentah (kolom seperti raw.csv) harus lewat src.features.clean sebelum predict_proba."},
              open("artifacts/metadata.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
