import sys
import os

# DOSYAYI TEK BAŞINA ÇALIŞTIRABİLMEK İÇİN YOL TANIMLAMASI (PATH HACK)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ml"))
sys.path.insert(0, os.path.join(ROOT, "app"))

import joblib
import numpy as np
import pandas as pd

# PICKLE MODEL YÜKLEME HATASI ÇÖZÜMÜ: Özel fonksiyonları modele tanıtıyoruz
from pipeline_components import select_top_k_features, OptionalScaler
import pipeline_components

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_FILE    = "catboost_model.pkl"
FEATURES_FILE = "catboost_selected_features.csv"



def load_model(model_file=MODEL_FILE, features_file=FEATURES_FILE):
    """
    Load the trained CatBoost model and its selected feature list from disk.
    """
    model = joblib.load(model_file)
    try:
        selected_features = (
            pd.read_csv(features_file, sep=";")["selected_feature"].tolist()
        )
    except FileNotFoundError:
        # Test ortamlarında dosya bulunamazsa modelin kendi özelliklerini al
        selected_features = getattr(model, "feature_names_", None)
        
    return model, selected_features


def predict_fss(inputs=None, input_values=None, model_path=None, model=None, selected_features=None):
    """
    Predict the global stability safety factor (Fss) for a single scenario.
    """
    # 1. Hangi girdinin geldiğini belirle
    data = inputs if inputs is not None else input_values
    if data is None:
        raise ValueError("Lütfen tahmin için 'inputs' veya 'input_values' sağlayın.")

    # 2. Pytest veya Bulk Predict için özel model yolu verildiyse yükle
    if model_path is not None:
        model = joblib.load(model_path)
        # Özellikleri modelin kendi içinden çekmeyi dene
        if selected_features is None:
            selected_features = getattr(model, "feature_names_", None)

    # 3. Model HALA yoksa, ancak o zaman varsayılan dosyaları aramaya çık
    if model is None:
        loaded_model, loaded_features = load_model()
        model = loaded_model
        if selected_features is None:
            selected_features = loaded_features

    # 4. Girdi tipini (Dict vs List) yönet
    if isinstance(data, dict):
        try:
            from preprocessing import preprocess_inputs
            processed_data = preprocess_inputs(data)
            input_array = np.array(processed_data).reshape(1, -1)
        except ImportError:
            input_df = pd.DataFrame([data])
            prediction = model.predict(input_df)
            return float(prediction[0])
    else:
        # Standart liste girişi
        input_array = np.array([data])

    # 5. DataFrame oluştur ve tahmin yap
    if selected_features is not None and len(selected_features) == input_array.shape[1]:
        input_df = pd.DataFrame(input_array, columns=selected_features)
    else:
        input_df = pd.DataFrame(input_array)

    prediction = model.predict(input_df)
    return float(prediction[0])


if __name__ == "__main__":
    import os, glob
    
    # Kök dizinden çalıştırıldığını varsayarak models klasörüne bak
    model_dir = os.path.join("ml", "outputs", "saved_models")
    pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    
    if pkl_files:
        test_model_path = pkl_files[0] # İlk bulduğun modeli al
        print(f"Hızlı Test Modeli: {test_model_path}")
        
        # Test senaryosu
        sample_input = [4, 1.80, 0.45, 0.32, 0.45, 1.26, 5, 0.9, 0.85, 1]
        
        # Hata vermemesi için predict_fss'e doğrudan model yolunu gönderiyoruz
        result = predict_fss(input_values=sample_input, model_path=test_model_path)
        print(f"Predicted Fss: {result:.4f}")
    else:
        print("Uyarı: 'ml/outputs/saved_models/' klasöründe test edilecek model (.pkl) bulunamadı!")