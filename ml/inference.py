import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_FILE = "catboost_model.pkl"
FEATURES_FILE = "catboost_selected_features.csv"


def load_model(model_file=MODEL_FILE, features_file=FEATURES_FILE):
    """
    Load the trained CatBoost model and its selected feature list from disk.

    Parameters
    ----------
    model_file    : str — Path to the serialized model (.pkl)
    features_file : str — Path to the selected features CSV (semicolon-separated)

    Returns
    -------
    model             : trained CatBoost regressor
    selected_features : list[str] — feature names in the order expected by the model
    """
    model = joblib.load(model_file)
    selected_features = pd.read_csv(features_file, sep=";")["selected_feature"].tolist()
    return model, selected_features


def predict_fss(input_values, model=None, selected_features=None):
    """
    Predict the global stability safety factor (Fss) for a single scenario.

    This function is designed to be imported and called from other modules,
    such as the PyQt5 desktop application.

    Parameters
    ----------
    input_values      : list[float]
        Input feature values in the same order as selected_features.
    model             : trained model, optional
        If None, the model is loaded from MODEL_FILE on each call.
        For repeated calls (e.g. in a GUI), pass a pre-loaded model
        to avoid repeated disk reads.
    selected_features : list[str], optional
        Feature names corresponding to input_values.
        If None, loaded from FEATURES_FILE on each call.

    Returns
    -------
    float : predicted Fss value

    Example
    -------
    >>> from inference import predict_fss, load_model
    >>> model, features = load_model()
    >>> predict_fss([4, 1.80, 0.45, 0.32, 0.45, 1.26, 5, 0.9, 0.85, 1],
    ...             model=model, selected_features=features)
    1.0742
    """
    if model is None or selected_features is None:
        model, selected_features = load_model()

    input_array = np.array([input_values])
    input_df = pd.DataFrame(input_array, columns=selected_features)
    prediction = model.predict(input_df.values)
    return float(prediction[0])


if __name__ == "__main__":
    # Quick sanity check
    model, features = load_model()
    sample_input = [4, 1.80, 0.45, 0.32, 0.45, 1.26, 5, 0.9, 0.85, 1]
    result = predict_fss(sample_input, model=model, selected_features=features)
    print(f"Predicted Fss: {result:.4f}")
