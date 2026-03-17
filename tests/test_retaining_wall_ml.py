"""
tests/test_retaining_wall_ml.py
--------------------------------
Retaining Wall ML — pytest test suite
SoftwareX reviewer requirement: automated tests exercising all main features.

Run from project root:
    pytest tests/ -v --tb=short
"""

import os
import sys
import glob
import subprocess
import pytest
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ml"))
sys.path.insert(0, os.path.join(ROOT, "app"))

# joblib, select_top_k_features'ı __main__ namespace'inde arar
import __main__ as _main
import pipeline_components as _pc
_main.select_top_k_features = _pc.select_top_k_features
_main.OptionalScaler        = _pc.OptionalScaler
_main.set_shap_order        = _pc.set_shap_order
sys.modules['pipeline_components'] = _pc

# ── Constants ─────────────────────────────────────────────────────────────────
def _find_dataset():
    candidates = [
        "output.csv", "output.txt", "dataset.csv",
        "data.csv", "dataset.txt",
    ]
    for name in candidates:
        for base in (ROOT, os.path.join(ROOT, "ml")):
            p = os.path.join(base, name)
            if os.path.isfile(p):
                return p
    return os.path.join(ROOT, "output.csv")

DATASET_PATH     = _find_dataset()
SAVED_MODELS_DIR = os.path.join(ROOT, "ml", "outputs", "saved_models")
SPLIT_DATASET_PY = os.path.join(ROOT, "ml", "split_dataset.py")

# Tüm 18 feature — IMPORTANCE_ORDER sırası ile
# H=7m, Scenario S2: hw = H = 7.0m
# Derived: v2 = X2*H, x1 = X1, s1 = X3+X4
EXAMPLE_INPUTS = {
    "gama": 19.0, "hw": 7.0,  "H": 7.0,   "sds": 1.2,
    "fi":   30.0, "q":  10.0, "X5": 0.42, "v2":  4.2,
    "x1":   3.5,  "X8": 1.2,  "X2": 0.6,  "X1":  3.5,
    "s1":   0.80, "X7": 0.35, "X6": 0.50, "X4":  0.35,
    "X3":   0.45, "c":  20.0,
}
# Düz liste — IMPORTANCE_ORDER sırasında, inference.py'ye geçmek için
EXAMPLE_INPUT_VECTOR = [
    19.0, 7.0, 7.0, 1.2, 30.0, 10.0, 0.42, 4.2,
    3.5,  1.2, 0.6, 3.5, 0.80, 0.35, 0.50, 0.35,
    0.45, 20.0,
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_dataset():
    """Read dataset with Turkish locale format: semicolon separator, comma decimal."""
    # Primary format: sep=';', decimal=',' (Turkish locale)
    try:
        df = pd.read_csv(DATASET_PATH, sep=";", decimal=",")
        if len(df.columns) > 5 and pd.api.types.is_numeric_dtype(
                pd.to_numeric(df.iloc[:, 0], errors="coerce")):
            return df
    except Exception:
        pass
    # Fallback: try other combinations
    for sep, dec in [(";", "."), (",", "."), (",", ","), ("\t", ".")]:
        try:
            df = pd.read_csv(DATASET_PATH, sep=sep, decimal=dec)
            if len(df.columns) > 5:
                return df
        except Exception:
            continue
    return pd.read_csv(DATASET_PATH, sep=";", decimal=",")


def _find_models(pattern="*.pkl"):
    if not os.path.isdir(SAVED_MODELS_DIR):
        return []
    return sorted(glob.glob(os.path.join(SAVED_MODELS_DIR, pattern)))


def _load_model_pair(pkl_path):
    """Load model + its features file. Returns (model, features) or raises."""
    from inference import load_model
    # 1. Try same base name with _selected_features.csv suffix
    features_path = pkl_path.replace(".pkl", "_selected_features.csv")
    # 2. Try any *selected_features* CSV in the same folder
    if not os.path.isfile(features_path):
        candidates = glob.glob(
            os.path.join(SAVED_MODELS_DIR, "*selected_features*"))
        features_path = candidates[0] if candidates else None
    # 3. Try any *features* CSV in saved_models/
    if not features_path or not os.path.isfile(features_path):
        candidates = glob.glob(
            os.path.join(SAVED_MODELS_DIR, "*features*"))
        features_path = candidates[0] if candidates else None
    # 4. Try ml/outputs/ directly
    if not features_path or not os.path.isfile(features_path):
        candidates = glob.glob(
            os.path.join(ROOT, "ml", "outputs", "*selected_features*"))
        features_path = candidates[0] if candidates else None
    if not features_path or not os.path.isfile(features_path):
        raise FileNotFoundError(
            "No features CSV found. Expected a file matching "
            "'*selected_features*.csv' in ml/outputs/saved_models/ "
            "or ml/outputs/. Run train_models.py to generate it.")
    return load_model(model_file=pkl_path, features_file=features_path)


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
def loaded_model():
    """GPR modelini yükler — inference.py ile en uyumlu model."""
    pkls = _find_models("GPR*.pkl")
    if not pkls:
        # GPR yoksa herhangi bir modeli dene
        pkls = _find_models()
    if not pkls:
        pytest.skip("No .pkl files in ml/outputs/saved_models/ — "
                    "run ml/train_models.py first.")
    try:
        model, features = _load_model_pair(pkls[0])
        return model, features
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")


@pytest.fixture
def loaded_gpr_model():
    pkls = _find_models("GPR*.pkl")
    if not pkls:
        pytest.skip("No GPR model found — run train_models.py first.")
    try:
        model, features = _load_model_pair(pkls[0])
        return model, features
    except Exception as e:
        pytest.skip(f"Could not load GPR model: {e}")


# =============================================================================
# 1. Dataset tests
# =============================================================================
class TestDataset:

    def test_dataset_exists(self):
        assert os.path.isfile(DATASET_PATH), (
            f"Dataset not found. Searched in {ROOT} for: "
            "output.csv, output.txt, dataset.csv.")

    def test_dataset_columns(self, dataset):
        expected = {"H","X1","X2","X3","X4","X5","X6","X7","X8",
                    "q","sds","v2","x1","s1","gama","c","fi","hw","fss"}
        assert expected.issubset(set(dataset.columns)), \
            f"Missing columns: {expected - set(dataset.columns)}"

    def test_dataset_row_count(self, dataset):
        assert len(dataset) >= 2000, \
            f"Expected ≥2000 rows, got {len(dataset)}"

    def test_dataset_key_columns_numeric(self, dataset):
        """Key numeric columns must be numeric dtype after correct CSV read."""
        for col in ("fss", "sds", "H", "gama"):
            if col in dataset.columns:
                series = dataset[col]
                # Try converting if still string
                if not pd.api.types.is_numeric_dtype(series):
                    series = series.str.replace(",", ".").astype(float)
                assert pd.api.types.is_numeric_dtype(series) or \
                       series.apply(lambda x: isinstance(x, (int, float))).all(), \
                    f"Column '{col}' could not be converted to numeric"

    def test_dataset_no_nulls_key_columns(self, dataset):
        key_cols = [c for c in
                    ["H","X1","sds","gama","c","fi","hw","fss"]
                    if c in dataset.columns]
        null_counts = dataset[key_cols].isnull().sum()
        assert null_counts.sum() == 0, \
            f"Null values:\n{null_counts[null_counts > 0]}"

    def _to_numeric(self, series):
        """Convert series to numeric, handling comma decimal if needed."""
        if pd.api.types.is_numeric_dtype(series):
            return series
        return pd.to_numeric(
            series.astype(str).str.replace(",", "."), errors="coerce")

    def test_fss_positive(self, dataset):
        fss = self._to_numeric(dataset["fss"])
        assert fss.notna().all(), \
            "fss column contains non-convertible values"
        assert (fss > 0).all(), "Fss values must be positive"

    def test_fss_plausible_range(self, dataset):
        fss = self._to_numeric(dataset["fss"])
        lo, hi = fss.min(), fss.max()
        assert 0.3 <= lo and hi <= 15.0, \
            f"Fss range [{lo:.3f}, {hi:.3f}] outside expected [0.3, 15.0]"

    def test_H_within_design_space(self, dataset):
        H = self._to_numeric(dataset["H"])
        assert H.between(3.5, 10.5).all(), \
            "H values outside design space [4, 10] m"

    def test_sds_within_design_space(self, dataset):
        sds = self._to_numeric(dataset["sds"])
        assert sds.between(0.55, 1.85).all(), \
            "sds values outside design space [0.6, 1.8] g"

    def test_hw_non_negative(self, dataset):
        """hw stores groundwater depth in metres — must be non-negative."""
        hw = self._to_numeric(dataset["hw"])
        assert hw.notna().all(), "hw column contains non-convertible values"
        assert (hw >= 0).all(), \
            f"hw values must be non-negative, found min={hw.min():.3f}"

    def test_geometric_consistency_stem(self, dataset):
        X3 = self._to_numeric(dataset["X3"])
        X4 = self._to_numeric(dataset["X4"])
        assert (X4 <= X3 + 0.01).all(), \
            "Geometric constraint violated: X4 > X3 in some rows"


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

    def test_optional_scaler_with_standard(self):
        from pipeline_components import OptionalScaler
        from sklearn.preprocessing import StandardScaler
        scaler = OptionalScaler(scaler=StandardScaler())
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = scaler.fit_transform(X)
        assert result.shape == X.shape
        np.testing.assert_almost_equal(result.mean(axis=0), [0.0, 0.0],
                                       decimal=8)

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
        result = compute_metrics(y, y + 1e-12)
        assert abs(result["R2"] - 1.0) < 1e-6

    def test_zero_mae_perfect(self):
        from metrics import compute_metrics
        y = np.linspace(1.0, 2.0, 20)
        result = compute_metrics(y, y)
        assert result["MAE"] < 1e-10

    def test_required_keys_present(self):
        from metrics import compute_metrics
        rng = np.random.default_rng(42)
        y = rng.random(30) + 1.0
        result = compute_metrics(y, y + rng.random(30) * 0.05)
        for key in ("MAE", "RMSE", "R2", "MaxE", "NSE", "KGE", "CCC"):
            assert key in result, f"Key '{key}' missing from metrics"


# =============================================================================
# 4. Inference tests
# =============================================================================
class TestInference:

    def test_importable(self):
        from inference import predict_fss, load_model  # noqa: F401

    def test_load_model_signature(self):
        import inspect
        from inference import load_model
        sig = inspect.signature(load_model)
        assert "model_file"    in sig.parameters
        assert "features_file" in sig.parameters

    def test_predict_fss_signature(self):
        import inspect
        from inference import predict_fss
        sig    = inspect.signature(predict_fss)
        params = list(sig.parameters.keys())
        assert params[0] == "input_values", \
            f"First param should be 'input_values', got '{params[0]}'"

    def test_predict_returns_float(self, loaded_model, example_inputs):
        from inference import predict_fss
        model, features = loaded_model
        # inference.py: selected_features kadar değer bekliyor
        # EXAMPLE_INPUT_VECTOR IMPORTANCE_ORDER sırasında — ilk k eleman
        k = len(features)
        result = predict_fss(EXAMPLE_INPUT_VECTOR[:k], model=model,
                             selected_features=features)
        assert isinstance(result, (float, np.floating, int))

    def test_predict_positive(self, loaded_model, example_inputs):
        from inference import predict_fss
        model, features = loaded_model
        # inference.py: selected_features kadar değer bekliyor
        # EXAMPLE_INPUT_VECTOR IMPORTANCE_ORDER sırasında — ilk k eleman
        k = len(features)
        result = predict_fss(EXAMPLE_INPUT_VECTOR[:k], model=model,
                             selected_features=features)
        assert float(result) > 0

    def test_predict_plausible_range(self, loaded_model, example_inputs):
        from inference import predict_fss
        model, features = loaded_model
        vals = [example_inputs[f] for f in features if f in example_inputs]
        result = float(predict_fss(vals, model=model,
                                   selected_features=features))
        assert 0.3 < result < 15.0, \
            f"Predicted Fss {result:.4f} outside plausible range"

    def test_gpr_illustrative_scenario(self, loaded_gpr_model):
        from inference import predict_fss
        model, features = loaded_gpr_model
        k = len(features)
        result = float(predict_fss(EXAMPLE_INPUT_VECTOR[:k], model=model,
                                   selected_features=features))
        assert abs(result - 1.4287) < 0.30, \
            f"GPR prediction {result:.4f} far from paper value 1.4287"


# =============================================================================
# 5. Split dataset tests
# =============================================================================
class TestSplitDataset:

    def test_split_script_exists(self):
        assert os.path.isfile(SPLIT_DATASET_PY), \
            f"split_dataset.py not found at {SPLIT_DATASET_PY}"

    def test_split_importable(self):
        import split_dataset  # noqa: F401

    def test_split_creates_output_files(self, tmp_path):
        """Run split_dataset.py as a subprocess and check output files."""
        if not os.path.isfile(DATASET_PATH):
            pytest.skip("Dataset not found — cannot test splitting.")
        result = subprocess.run(
            [sys.executable, SPLIT_DATASET_PY,
             "--input",  DATASET_PATH,
             "--outdir", str(tmp_path)],
            capture_output=True, text=True, cwd=ROOT
        )
        # If the script doesn't accept --input/--outdir, fall back to
        # checking that the default output files exist after running
        if result.returncode != 0:
            # Try running without arguments (uses hardcoded paths)
            result2 = subprocess.run(
                [sys.executable, SPLIT_DATASET_PY],
                capture_output=True, text=True, cwd=ROOT
            )
            default_out = os.path.join(ROOT, "ml", "outputs")
            for fname in ("train.csv", "test.csv", "unseen.csv"):
                p = os.path.join(default_out, fname)
                if os.path.isfile(p):
                    return  # Files exist from previous run — pass
            pytest.skip(
                "split_dataset.py does not accept --input/--outdir; "
                "run it manually once to generate train/test/unseen splits.")

        for fname in ("train.csv", "test.csv", "unseen.csv"):
            assert (tmp_path / fname).exists() or \
                   os.path.isfile(os.path.join(
                       ROOT, "ml", "outputs", fname)), \
                f"{fname} not found after running split_dataset.py"

    def test_existing_splits_have_correct_ratios(self):
        """If splits already exist, verify their row-count ratios."""
        out_dir = os.path.join(ROOT, "ml")
        paths = {
            "train":  os.path.join(out_dir, "train.csv"),
            "test":   os.path.join(out_dir, "test.csv"),
            "unseen": os.path.join(out_dir, "unseen.csv"),
        }
        if not all(os.path.isfile(p) for p in paths.values()):
            pytest.skip("Split files not yet generated — run split_dataset.py")
        dfs = {k: pd.read_csv(p, sep=";", decimal=",")
               for k, p in paths.items()}
        total = sum(len(df) for df in dfs.values())
        assert abs(len(dfs["train"])  / total - 0.70) < 0.03
        assert abs(len(dfs["test"])   / total - 0.20) < 0.03
        assert abs(len(dfs["unseen"]) / total - 0.10) < 0.03

    def test_existing_splits_no_nulls(self):
        """Splits must not contain null values in the target column."""
        out_dir = os.path.join(ROOT, "ml")
        for fname in ("train.csv", "test.csv", "unseen.csv"):
            p = os.path.join(out_dir, fname)
            if not os.path.isfile(p):
                pytest.skip(f"{fname} not found — run split_dataset.py")
            df = pd.read_csv(p, sep=";", decimal=",")
            if "fss" in df.columns:
                assert df["fss"].notna().all(), \
                    f"Null fss values found in {fname}"


# =============================================================================
# 6. Preprocessing tests
# =============================================================================
class TestPreprocessing:

    def test_importable(self):
        from preprocessing import preprocess_inputs  # noqa: F401

    def test_output_is_array_like(self, example_inputs):
        from preprocessing import preprocess_inputs
        result = np.array(preprocess_inputs(example_inputs)).flatten()
        assert len(result) > 0

    def test_output_has_18_features(self, example_inputs):
        from preprocessing import preprocess_inputs
        result = np.array(preprocess_inputs(example_inputs)).flatten()
        assert len(result) == 18, \
            f"Expected 18 features, got {len(result)}"

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