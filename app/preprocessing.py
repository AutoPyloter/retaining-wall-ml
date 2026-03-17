# preprocessing.py
# Feature mapping and input preprocessing for model inference.
#
# New pipeline architecture: each saved model is a full sklearn Pipeline
# (feature selection + optional scaler + estimator). preprocessing.py
# only builds the raw full-length feature vector; the pipeline handles
# feature selection and scaling internally.

import logging
import os
import sys
from typing import Any, List

import numpy as np


def resource_path(relative_path: str) -> str:
    try:
        base = sys._MEIPASS
    except AttributeError:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SHAP importance ranking (XGBoost baseline, all 18 features)
# Must match the shap_order used during training in train_models.py
# ---------------------------------------------------------------------------
IMPORTANCE_ORDER: List[str] = [
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

# UI widget key → internal feature name
INPUT_MAP: dict[str, str] = {
    "k": "X4",
    "h": "H",
    "xx": "X5",
    "v1": "X2",
    "v2": "v2",
    "x1": "x1",
    "x2": "X7",
    "s1": "s1",
    "x3": "X8",
    "hw": "hw",
    "gama": "gama",
    "fi": "fi",
    "c": "c",
    "q": "q",
    "sds": "sds",
}

REVERSE_MAP: dict[str, str] = {v: k for k, v in INPUT_MAP.items()}

# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def preprocess_inputs(vals: dict[str, float]) -> np.ndarray:
    """Build the full raw feature vector (18 features) from UI values.

    The returned array is passed directly to the saved Pipeline, which
    handles feature selection (top-k SHAP) and scaling internally.

    Parameters
    ----------
    vals:
        Dictionary of UI widget values keyed by widget name.

    Returns
    -------
    np.ndarray
        Shape (1, 18) — full feature vector, unscaled.
    """
    arr: List[float] = []
    for feat in IMPORTANCE_ORDER:
        ui_key = REVERSE_MAP.get(feat)
        val = vals.get(ui_key, 0.0) if ui_key else 0.0
        arr.append(float(val))

    return np.array([arr])  # shape (1, 18)
