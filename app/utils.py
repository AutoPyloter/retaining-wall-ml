# utils.py
# Shared utility functions used across the application.

import numpy as np


def select_top_k(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Return the first *k* columns of *X*.

    Used to align a full feature matrix with a model trained on the
    top-k SHAP-selected features.
    """
    return X[:, :k]
