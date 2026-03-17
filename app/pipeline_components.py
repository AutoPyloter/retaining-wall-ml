# pipeline_components.py
# Shared sklearn-compatible transformer classes used in both
# train_models.py (training) and app.py (inference).
# Must be importable from both ml/ and app/ contexts.

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone


class OptionalScaler(BaseEstimator, TransformerMixin):
    """Wraps a scaler or None for use inside a sklearn Pipeline.

    When scaler=None the transformer is a no-op (pass-through).
    """

    def __init__(self, scaler=None):
        self.scaler = scaler

    def fit(self, X, y=None):
        if self.scaler is not None:
            self.scaler_ = clone(self.scaler)
            self.scaler_.fit(X)
        else:
            self.scaler_ = None
        return self

    def transform(self, X):
        if self.scaler_ is not None:
            return self.scaler_.transform(X)
        return X


import numpy as np

# ---------------------------------------------------------------------------
# SHAP-guided feature selector — must be importable from both
# train_models.py and app.py so joblib.load works in both contexts.
# shap_order is set at runtime by train_models.py after SHAP is computed.
# ---------------------------------------------------------------------------
_shap_order = None  # set via set_shap_order() before use


def set_shap_order(order: np.ndarray) -> None:
    """Register the SHAP feature order (called from train_models.py)."""
    global _shap_order
    _shap_order = order


def select_top_k_features(X: np.ndarray, k: int) -> np.ndarray:
    """Return the top-k SHAP-ranked columns from X.

    When called from a fitted pipeline (inference), the FunctionTransformer
    already has kw_args={'k': best_k} baked in, and the column order in X
    matches IMPORTANCE_ORDER from preprocessing.py — so we just take the
    first k columns.

    When called from train_models.py, set_shap_order() has been called first
    and _shap_order reorders the columns correctly.
    """
    if _shap_order is not None:
        return X[:, _shap_order[:k]]
    # Inference mode: X is already ordered by IMPORTANCE_ORDER, take first k
    return X[:, :k]
