"""
tests/test_retaining_wall_ml.py
--------------------------------
Retaining Wall ML — pytest test suite
"""

import os
import sys
import glob
import pytest
import numpy as np
import pandas as pd


# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ml"))
sys.path.insert(0, os.path.join(ROOT, "app"))

# PICKLE HATASI ÇÖZÜMÜ: Modellerin test içinde yüklenebilmesi için özel fonksiyonları globale alıyoruz
from pipeline_components import select_top_k_features, OptionalScaler
import pipeline_components


# ── Constants ────────────────────────────────────────────────────────────────
def _find_dataset():
    p = os.path.join(ROOT, "ml", "data.csv")
    if os.path.isfile(p):
        return p
    return os.path.join(ROOT, "output.csv")

DATASET_PATH = _find_dataset()
SAVED_MODELS_DIR = os.path.join(ROOT, "ml", "outputs", "saved_models")

EXAMPLE_INPUTS = {
    "H": 7.0, "X1": 3.5, "X2": 0.6,  "X3": 0.45, "X4": 0.35,
    "X5": 0.42, "X6": 0.50, "X7": 0.35, "X8": 1.2,
    "q": 10, "sds": 1.2, "gama": 19, "c": 20, "fi": 30, "hw": 2
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_dataset():
    """Read dataset regardless of separator, handling decimal commas."""
    for sep in (";", ",", "\t"):
        try:
            # decimal=',' eklendi! Bu sayede 4,5 değerleri float'a dönüşür.
            df = pd.read_csv(DATASET_PATH, sep=sep, decimal=',')
            if len(df.columns) > 5:
                # Sütun isimlerindeki olası boşlukları temizle
                df.columns = df.columns.str.strip()
                return df
        except Exception:
            continue
    return pd.read_csv(DATASET_PATH, sep=";", decimal=',')

def _find_any_model():
    if not os.path.isdir(SAVED_MODELS_DIR):
        return None
    pkls = glob.glob(os.path.join(SAVED_MODELS_DIR, "*.pkl"))
    return sorted(pkls)[0] if pkls else None

def _find_gpr_model():
    if not os.path.isdir(SAVED_MODELS_DIR):
        return None
    pkls = glob.glob(os.path.join(SAVED_MODELS_DIR, "GPR*.pkl"))
    return sorted(pkls)[0] if pkls else None

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def example_inputs():
    return EXAMPLE_INPUTS.copy()

@pytest.fixture
def dataset():
    if not os.path.isfile(DATASET_PATH):
        pytest.skip(f"Dataset not found: {DATASET_PATH}")
    return _read_dataset()

@pytest.fixture
def any_model_path():
    p = _find_any_model()
    if p is None:
        pytest.skip("No .pkl model files found in ml/outputs/saved_models/")
    return p

@pytest.fixture
def gpr_model_path():
    p = _find_gpr_model()
    if p is None:
        pytest.skip("No GPR .pkl model found in ml/outputs/saved_models/")
    return p

# =============================================================================
# 1. Dataset tests
# =============================================================================
class TestDataset:

    def test_dataset_exists(self):
        assert os.path.isfile(DATASET_PATH), f"Dataset not found at {DATASET_PATH}."

    def test_dataset_columns(self, dataset):
        expected = {"H","X1","X2","X3","X4","X5","X6","X7","X8",
                    "q","sds","v2","x1","s1","gama","c","fi","hw","fss"}
        assert expected.issubset(set(dataset.columns)), f"Missing columns: {expected - set(dataset.columns)}"

    def test_dataset_row_count(self, dataset):
        assert len(dataset) >= 2000, f"Expected ≥2000 rows, got {len(dataset)}"

    def test_dataset_no_nulls(self, dataset):
        null_counts = dataset[["H","X1","sds","gama","c","fi","hw","fss"]].isnull().sum()
        assert null_counts.sum() == 0, "Null values in key columns"

    def test_fss_positive(self, dataset):
        assert (dataset["fss"].astype(float) > 0).all(), "Fss values must be positive"

    def test_fss_plausible_range(self, dataset):
        assert dataset["fss"].astype(float).between(0.3, 15.0).all(), "Fss range outside expected [0.3, 15.0]"

    def test_H_within_design_space(self, dataset):
        assert dataset["H"].astype(float).between(3.5, 10.5).all(), "H values outside design space [4, 10] m"

    def test_sds_within_design_space(self, dataset):
        assert dataset["sds"].astype(float).between(0.55, 1.85).all(), "sds values outside design space [0.6, 1.8] g"

    def test_hw_plausible_range(self, dataset):
        # hw sınıf değil gerçek derinlik olduğu için >= 0 testi yapıyoruz.
        assert (dataset["hw"].astype(float) >= 0).all(), "hw (water depth) must be positive or zero"

    def test_geometric_consistency_stem(self, dataset):
        assert (dataset["X4"].astype(float) <= dataset["X3"].astype(float) + 0.01).all(), "Geometric constraint violated: X4 > X3"

# =============================================================================
# 2. Pipeline components tests
# =============================================================================
class TestPipelineComponents:

    def test_importable(self):
        import pipeline_components

    def test_optional_scaler_exists(self):
        from pipeline_components import OptionalScaler
        assert OptionalScaler is not None

    def test_optional_scaler_passthrough(self):
        from pipeline_components import OptionalScaler
        scaler = OptionalScaler(scaler=None)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = scaler.fit_transform(X)
        np.testing.assert_array_equal(result, X)

    def test_optional_scaler_standard(self):
        from pipeline_components import OptionalScaler
        from sklearn.preprocessing import StandardScaler
        scaler = OptionalScaler(scaler=StandardScaler())
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = scaler.fit_transform(X)
        assert result.shape == X.shape

    def test_select_top_k_importable(self):
        from pipeline_components import select_top_k_features
        assert callable(select_top_k_features)

# =============================================================================
# 3. Metrics tests
# =============================================================================
class TestMetrics:

    def test_importable(self):
        from metrics import compute_metrics

    def test_returns_17_metrics(self):
        from metrics import compute_metrics
        y_true = np.array([1.2, 1.5, 1.8, 2.0, 1.3])
        y_pred = np.array([1.25, 1.45, 1.75, 2.05, 1.35])
        result = compute_metrics(y_true, y_pred)
        assert len(result) == 17

    def test_perfect_r2(self):
        from metrics import compute_metrics
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_metrics(y, y + 1e-12)
        assert abs(result["R2"] - 1.0) < 1e-6

    def test_required_keys_present(self):
        from metrics import compute_metrics
        y = np.random.rand(30) + 1.0
        result = compute_metrics(y, y + np.random.rand(30) * 0.05)
        for key in ("MAE", "RMSE", "R2", "MaxE", "NSE", "KGE", "CCC"):
            assert key in result

# =============================================================================
# 4. Inference tests
# =============================================================================
class TestInference:

    def test_importable(self):
        from inference import predict_fss

    def test_predict_with_model_path(self, any_model_path, example_inputs):
        from inference import predict_fss
        try:
            result = predict_fss(inputs=example_inputs, model_path=any_model_path)
            assert isinstance(result, (float, np.floating, int))
        except TypeError:
            pytest.fail("predict_fss() 'model_path' parametresini kabul etmiyor. Lütfen inference.py dosyasını güncelleyin.")

    def test_predict_positive(self, any_model_path, example_inputs):
        from inference import predict_fss
        try:
            result = predict_fss(inputs=example_inputs, model_path=any_model_path)
            assert float(result) > 0
        except TypeError:
            pytest.skip("model_path hatası yüzünden atlandı.")

# =============================================================================
# 5. Split dataset tests (Güvenli Import)
# =============================================================================
class TestSplitDataset:

    def test_split_output_files(self, tmp_path, dataset):
        try:
            from split_dataset import split_and_save
            split_and_save(dataset, output_dir=str(tmp_path), random_state=42)
            assert (tmp_path / "train.csv").exists()
        except ImportError:
            pytest.skip("split_and_save fonksiyonu split_dataset.py içinde bulunamadı.")

# =============================================================================
# 6. Preprocessing tests
# =============================================================================
class TestPreprocessing:

    def test_importable(self):
        from preprocessing import preprocess_inputs

    def test_output_has_18_features(self, example_inputs):
        from preprocessing import preprocess_inputs
        result = np.array(preprocess_inputs(example_inputs)).flatten()
        assert len(result) == 18

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])