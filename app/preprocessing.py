# preprocessing.py
# Feature mapping constants and input preprocessing for model inference.
#
# IMPORTANCE_ORDER defines the global SHAP-ranked feature list (18 features).
# preprocess_inputs() selects the top-k features and optionally standardises
# them using pre-computed scaling factors.

import os
import logging
from typing import Any, List

import numpy as np
import pandas as pd

from config import resource_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SHAP importance ranking (XGBoost baseline, all 18 features)
IMPORTANCE_ORDER: List[str] = [
    "gama", "hw", "sds", "H", "fi", "q", "X5", "v2", "x1",
    "X8", "X2", "X7", "X1", "X6", "s1", "X4", "X3", "c",
]

# UI widget key → internal feature name
INPUT_MAP: dict[str, str] = {
    "k":    "X4",
    "h":    "H",
    "xx":   "X5",
    "v1":   "X2",
    "v2":   "v2",
    "x1":   "x1",
    "x2":   "X7",
    "s1":   "s1",
    "x3":   "X8",
    "hw":   "hw",
    "gama": "gama",
    "fi":   "fi",
    "c":    "c",
    "q":    "q",
    "sds":  "sds",
}

REVERSE_MAP: dict[str, str] = {v: k for k, v in INPUT_MAP.items()}

# ---------------------------------------------------------------------------
# Scaling factors (loaded once at import time)
# ---------------------------------------------------------------------------

_scaling_df = pd.read_csv(resource_path("scaling_factors.csv"), sep=";", decimal=",")
MEANS: dict[str, float] = _scaling_df.set_index("feature")["mean"].to_dict()
SCALES: dict[str, float] = _scaling_df.set_index("feature")["scale"].to_dict()

_model_scaling_df = pd.read_csv(resource_path("model_scaling_info.csv"), sep=";")
SCALE_REQUIRED: dict[str, bool] = (
    _model_scaling_df.set_index("model")["scale"].astype(bool).to_dict()
)

# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def preprocess_inputs(vals: dict[str, float], model_name: str, k: int) -> List[float]:
    """Build the feature vector for *model_name* from raw UI values.

    Parameters
    ----------
    vals:
        Dictionary of UI widget values keyed by widget name (e.g. ``'h'``, ``'gama'``).
    model_name:
        Identifier of the selected model (must exist in ``SCALE_REQUIRED``).
    k:
        Number of top-ranked SHAP features to include.

    Returns
    -------
    List[float]
        Feature vector of length *k*, optionally standardised.

    Raises
    ------
    KeyError
        If *model_name* is not found in the scaling info table.
    """
    if model_name not in SCALE_REQUIRED:
        raise KeyError(f"Unknown model: '{model_name}'. Check model_scaling_info.csv.")

    do_scale = SCALE_REQUIRED[model_name]

    # Derived geometric features
    s1 = vals.get("s1", 0)
    bottom_width = (vals.get("h", 0) / s1 if s1 else 0) + vals.get("k", 0)
    foundation_width = vals.get("xx", 0) + vals.get("v1", 0) + bottom_width
    key_thickness = vals.get("x1", 0) - vals.get("k", 0)

    arr: List[float] = []
    for feat in IMPORTANCE_ORDER[:k]:
        if feat == "alt_govde":
            val = bottom_width
        elif feat == "temel_genislik":
            val = foundation_width
        elif feat == "dis_kalinlik":
            val = key_thickness
        else:
            ui_key = REVERSE_MAP.get(feat)
            val = vals.get(ui_key, 0) if ui_key else 0

        if do_scale and feat in MEANS:
            val = (val - MEANS[feat]) / SCALES[feat]

        arr.append(val)

    return arr
