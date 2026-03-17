"""
tests/test_retaining_wall_ml.py
--------------------------------
Retaining Wall ML — pytest test suite
SoftwareX reviewer requirement: automated tests exercising all main features.

Run from project root:
    pytest tests/ -v --tb=short

Or run directly:
    python tests/test_retaining_wall_ml.py
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
sys.path.insert(0, os.path.join(ROOT, "ml"))   # allow "from pipeline_components import ..."
sys.path.insert(0, os.path.join(ROOT, "app"))  # allow "from preprocessing import ..."

# ── Constants ────────────────────────────────────────────────────────────────
# Dataset may be output.csv or output.txt — find whichever exists
def _find_dataset():
    for name in ("output.csv", "output.txt", "dataset.csv"):
        p = os.path.join(ROOT, name)
        if os.path.isfile(p):
            return p
    return os.path.join(ROOT, "output.csv")  # default for error message

DATASET_PATH = _find_dataset()
SAVED_MODELS_DIR = os.path.join(ROOT, "ml", "outputs", "saved_models")

# Illustrative example from Section 4 of the paper
EXAMPLE_INPUTS = {
    "H": 7.0, "X1": 3.5, "X2": 0.6,  "X3": 0.45, "X4": 0.35,
    "X5": 0.42, "X6": 0.50, "X7": 0.35, "X8": 1.2,
    "q": 10, "sds": 1.2, "gama": 19, "c": 20, "fi": 30, "hw": 2
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_dataset():
    """Read dataset regardless of separator."""
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(DATASET_PATH, sep=sep)
            if len(df.columns) > 5:
                return df
        except Exception:
            continue
    return pd.read_csv(DATASET_PATH, sep=";")


def _find_any_model():
    """Return path to first .pkl in saved_models/, or None."""
    if not os.path.isdir(SAVED_MODELS_DIR):
        return None
    pkls = glob.glob(os.path.join(SAVED_MODELS_DIR, "*.pkl"))
    return sorted(pkls)[0] if pkls else None


def _find_gpr_model():
    """Return path to first GPR .pkl, or None."""
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
        assert os.path.isfile(DATASET_PATH), \
            f"Dataset not found at {DATASET_PATH}. " \
            f"Expected output.csv or output.txt in repository root."

    def test_dataset_columns(self, dataset):
        expected = {"H","X1","X2","X3","X4","X5","X6","X7","X8",
                    "q","sds","v2","x1","s1","gama","c","fi","hw","fss"}
        assert expected.issubset(set(dataset.columns)), \
            f"Missing columns: {expected - set(dataset.columns)}"

    def test_dataset_row_count(self, dataset):
        assert len(dataset) >= 2000, \
            f"Expected ≥2000 rows, got {len(dataset)}"

    def test_dataset_no_nulls(self, dataset):
        null_counts = dataset[["H","X1","sds","gama","c","fi","hw","fss"]].isnull().sum()
        assert null_counts.sum() == 0, \
            f"Null values in key columns:\n{null_counts[null_counts > 0]}"

    def test_fss_positive(self, dataset):
        assert (dataset["fss"] > 0).all(), "Fss values must be positive"

    def test_fss_plausible_range(self, dataset):
        assert dataset["fss"].between(0.3, 15.0).all(), \
            f"Fss range [{dataset['fss'].min():.3f}, {dataset['fss'].max():.3f}] " \
            f"outside expected [0.3, 15.0]"

    def test_H_within_design_space(self, dataset):
        assert dataset["H"].between(3.5, 10.5).all(), \
            "H values outside design space [4, 10] m"

    def test_sds_within_design_space(self, dataset):
        assert dataset["sds"].between(0.55, 1.85).all(), \
            "sds values outside design space [0.6, 1.8] g"

    def test_hw_discrete_values(self, dataset):
        assert set(dataset["hw"].unique()).issubset({0, 1, 2, 3, 4}), \
            f"hw contains unexpected values: {set(dataset['hw'].unique()) - {0,1,2,3,4}}"

    def test_geometric_consistency_stem(self, dataset):
        """Stem top width (X4) must not exceed stem bottom width (X3)."""
        assert (dataset["X4"] <= dataset["X3"] + 0.01).all(), \
            "Geometric constraint violated: X4 > X3"


# =============================================================================
# 2. Pipeline components tests
# =============================================================================
class TestPipelineComponents:

    def test_importable(self):
        import pipeline_components  # noqa: F401

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
        np.testing.assert_almost_equal(result.mean(axis=0), [0.0, 0.0], decimal=8)

    def test_select_top_k_importable(self):
        from pipeline_components import select_top_k_features
        assert callable(select_top_k_features)


# =============================================================================
# 3. Metrics tests
# =============================================================================
class TestMetrics:

    def test_importable(self):
        from metrics import compute_metrics  # noqa: F401

    def test_returns_17_metrics(self):
        from metrics import compute_metrics
        y_true = np.array([1.2, 1.5, 1.8, 2.0, 1.3])
        y_pred = np.array([1.25, 1.45, 1.75, 2.05, 1.35])
        result = compute_metrics(y_true, y_pred)
        assert len(result) == 17, \
            f"Expected 17 metrics, got {len(result)}: {list(result.keys())}"

    def test_perfect_r2(self):
        from metrics import compute_metrics
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_metrics(y, y + 1e-12)  # near-perfect
        assert abs(result["R2"] - 1.0) < 1e-6

    def test_zero_mae_perfect(self):
        from metrics import compute_metrics
        y = np.linspace(1.0, 2.0, 20)
        result = compute_metrics(y, y)
        assert result["MAE"] < 1e-10

    def test_required_keys_present(self):
        from metrics import compute_metrics
        y = np.random.rand(30) + 1.0
        result = compute_metrics(y, y + np.random.rand(30) * 0.05)
        for key in ("MAE", "RMSE", "R2", "MaxE", "NSE", "KGE", "CCC"):
            assert key in result, f"Key '{key}' missing from metrics output"


# =============================================================================
# 4. Inference tests
# =============================================================================
class TestInference:

    def test_importable(self):
        from inference import predict_fss  # noqa: F401

    def test_predict_with_model_path(self, any_model_path, example_inputs):
        """predict_fss should accept an explicit model path."""
        from inference import predict_fss
        result = predict_fss(model_path=any_model_path, inputs=example_inputs)
        assert isinstance(result, (float, np.floating, int)), \
            f"Expected numeric result, got {type(result)}"

    def test_predict_positive(self, any_model_path, example_inputs):
        from inference import predict_fss
        result = predict_fss(model_path=any_model_path, inputs=example_inputs)
        assert float(result) > 0, f"Predicted Fss must be positive, got {result}"

    def test_predict_plausible_range(self, any_model_path, example_inputs):
        from inference import predict_fss
        result = float(predict_fss(model_path=any_model_path, inputs=example_inputs))
        assert 0.3 < result < 15.0, \
            f"Predicted Fss {result:.4f} outside plausible range"

    def test_gpr_illustrative_scenario(self, gpr_model_path):
        """GPR should predict close to 1.4287 for the paper's Section 4 example."""
        from inference import predict_fss
        result = float(predict_fss(model_path=gpr_model_path,
                                   inputs=EXAMPLE_INPUTS))
        assert abs(result - 1.4287) < 0.30, \
            f"GPR prediction {result:.4f} far from paper value 1.4287"


# =============================================================================
# 5. Split dataset tests
# =============================================================================
class TestSplitDataset:

    def test_split_output_files(self, tmp_path, dataset):
        """After splitting, train/test/unseen CSVs must be created."""
        from split_dataset import split_and_save
        split_and_save(dataset, output_dir=str(tmp_path), random_state=42)
        for fname in ("train.csv", "test.csv", "unseen.csv"):
            assert (tmp_path / fname).exists(), f"{fname} not created"

    def test_split_ratios_approximate(self, tmp_path, dataset):
        from split_dataset import split_and_save
        split_and_save(dataset, output_dir=str(tmp_path), random_state=42)
        train  = pd.read_csv(tmp_path / "train.csv",  sep=";")
        test   = pd.read_csv(tmp_path / "test.csv",   sep=";")
        unseen = pd.read_csv(tmp_path / "unseen.csv", sep=";")
        total  = len(train) + len(test) + len(unseen)
        assert abs(len(train)  / total - 0.70) < 0.03
        assert abs(len(test)   / total - 0.20) < 0.03
        assert abs(len(unseen) / total - 0.10) < 0.03

    def test_combined_rows_equal_original(self, tmp_path, dataset):
        from split_dataset import split_and_save
        split_and_save(dataset, output_dir=str(tmp_path), random_state=42)
        train  = pd.read_csv(tmp_path / "train.csv",  sep=";")
        test   = pd.read_csv(tmp_path / "test.csv",   sep=";")
        unseen = pd.read_csv(tmp_path / "unseen.csv", sep=";")
        assert len(train) + len(test) + len(unseen) == len(dataset)


# =============================================================================
# 6. Preprocessing tests
# =============================================================================
class TestPreprocessing:

    def test_importable(self):
        from preprocessing import preprocess_inputs  # noqa: F401

    def test_output_is_array_like(self, example_inputs):
        from preprocessing import preprocess_inputs
        result = preprocess_inputs(example_inputs)
        arr = np.array(result).flatten()
        assert len(arr) > 0, "preprocess_inputs returned empty result"

    def test_output_has_18_features(self, example_inputs):
        from preprocessing import preprocess_inputs
        result = np.array(preprocess_inputs(example_inputs)).flatten()
        assert len(result) == 18, \
            f"Expected 18 features after preprocessing, got {len(result)}"

    def test_output_finite(self, example_inputs):
        from preprocessing import preprocess_inputs
        result = np.array(preprocess_inputs(example_inputs)).flatten()
        assert np.all(np.isfinite(result)), \
            "Preprocessed inputs contain NaN or Inf"


# =============================================================================
# Run directly
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])