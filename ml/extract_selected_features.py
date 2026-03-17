"""
extract_selected_features.py
-----------------------------
Her .pkl model dosyasından seçilen feature listesini çıkarır ve
yanına  <ModelAdı>_selected_features.csv  olarak kaydeder.

Kullanım (proje kökünden):
    python extract_selected_features.py

Gereksinim: pipeline_components.py'nin ml/ klasöründe olması.
"""

import glob
import os
import sys

import joblib
import pandas as pd

# ── Yollar ────────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
# Script ml/ içinde veya proje kökünde çalışabilir
ROOT = _here if not _here.endswith("ml") else os.path.dirname(_here)
_ml_dir = os.path.join(ROOT, "ml")
_app_dir = os.path.join(ROOT, "app")
MODELS_DIR = os.path.join(_ml_dir, "outputs", "saved_models")
sys.path.insert(0, _ml_dir)
sys.path.insert(0, _app_dir)

# FunctionTransformer, select_top_k_features'ı __main__ namespace'inde arar
# çünkü train_models.py __main__ olarak çalışırken pickle'landı.
import sys as _sys

import __main__

# pipeline_components app/ içinde — joblib deserializasyonu için gerekli
import pipeline_components  # noqa: F401
from pipeline_components import OptionalScaler, select_top_k_features, set_shap_order

__main__.select_top_k_features = select_top_k_features
__main__.OptionalScaler = OptionalScaler
__main__.set_shap_order = set_shap_order
_sys.modules["pipeline_components"] = pipeline_components

# SHAP importance sırası — preprocessing.py::IMPORTANCE_ORDER ile birebir aynı
# select_top_k_features inference'da X[:, :k] alır, X bu sırayla gelir
ALL_FEATURES = [
    "gama",
    "hw",
    "H",
    "sds",
    "fi",
    "q",
    "X5",
    "v2",
    "x1",
    "X8",
    "X2",
    "X1",
    "s1",
    "X7",
    "X6",
    "X4",
    "X3",
    "c",
]


def extract_features_from_pipeline(pipeline):
    """
    Sklearn Pipeline içindeki feature seçim adımından seçili feature
    isimlerini döndürür. Birkaç farklı yapıyı dener.
    """
    # 1. Pipeline adımlarında SelectKBest / feature selector ara
    for step_name, step in pipeline.steps:
        if hasattr(step, "get_support"):
            mask = step.get_support()
            if len(mask) == len(ALL_FEATURES):
                return [f for f, m in zip(ALL_FEATURES, mask) if m]

    # 2. select_top_k_features (custom transformer) — support_ attribute
    for step_name, step in pipeline.steps:
        if hasattr(step, "support_"):
            mask = step.support_
            if len(mask) == len(ALL_FEATURES):
                return [f for f, m in zip(ALL_FEATURES, mask) if m]

    # 3. feature_names_in_ — sadece gerçek isimler (Column_0 gibi otomatik isimler reddedilir)
    for step_name, step in pipeline.steps:
        if hasattr(step, "feature_names_in_"):
            names = list(step.feature_names_in_)
            if names and not any(
                n.startswith("Column_") or n.startswith("x") and n[1:].isdigit() for n in names
            ):
                return names

    # 4. Son adım (regressor) üzerinde dene
    final = pipeline.steps[-1][1]
    if hasattr(final, "feature_names_in_"):
        names = list(final.feature_names_in_)
        if names and not any(n.startswith("Column_") for n in names):
            return names

    return None


def extract_k_from_name(pkl_name):
    """Dosya adından k değerini çıkar: GPR_k8_StandardScaler_xxx.pkl → 8"""
    import re

    m = re.search(r"_k(\d+)_", pkl_name)
    return int(m.group(1)) if m else None


def main():
    pkl_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pkl")))
    if not pkl_files:
        print(f"HATA: {MODELS_DIR} içinde .pkl dosyası bulunamadı.")
        sys.exit(1)

    success, skipped = 0, 0

    for pkl_path in pkl_files:
        base = os.path.basename(pkl_path)
        out_path = pkl_path.replace(".pkl", "_selected_features.csv")

        if os.path.isfile(out_path):
            os.remove(out_path)  # IMPORTANCE_ORDER güncellemesi — yeniden üret

        try:
            obj = joblib.load(pkl_path)
        except Exception as e:
            print(f"  YÜKLENEMEDI: {base} — {e}")
            continue

        features = None

        # Pipeline ise
        if hasattr(obj, "steps"):
            features = extract_features_from_pipeline(obj)

        # Direkt model (Pipeline değil)
        if features is None and hasattr(obj, "feature_names_in_"):
            features = list(obj.feature_names_in_)

        # Son çare: dosya adındaki k sayısı kadar ilk feature'ı al
        if features is None:
            k = extract_k_from_name(base)
            if k and k <= len(ALL_FEATURES):
                print(
                    f"  UYARI: {base} — pipeline yapısından çıkarılamadı, "
                    f"k={k} varsayılan sırayla alındı (kontrol et!)"
                )
                features = ALL_FEATURES[:k]
            else:
                print(f"  ATLA: {base} — feature listesi çıkarılamadı.")
                continue

        df = pd.DataFrame({"selected_feature": features})
        df.to_csv(out_path, sep=";", index=False)
        print(f"  OK: {os.path.basename(out_path)}  ({len(features)} feature)")
        success += 1

    print(f"\nTamamlandı: {success} CSV üretildi, {skipped} zaten mevcuttu.")


if __name__ == "__main__":
    main()
