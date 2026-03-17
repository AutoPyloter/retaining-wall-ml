# test_scaling.py
# Validates that scale_param.csv can be loaded correctly and that
# 'mean' and 'std' columns are present and readable.
#
# Run from the app/ directory:
#
#   python test_scaling.py

import os

import pandas as pd

SCALE_FILE = "scale_param.csv"


def load_scale_params(filepath: str = SCALE_FILE) -> tuple[dict, dict]:
    """Load mean and std scaling parameters from *filepath*.

    Returns
    -------
    means, stds : tuple[dict, dict]
        Dictionaries mapping feature name → mean and feature name → std.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    KeyError
        If 'mean' or 'std' columns are absent.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"'{filepath}' not found. Ensure it is present in the working directory."
        )

    df = pd.read_csv(filepath, sep=";", decimal=",", engine="python")

    # Treat the first column as the feature index if 'feature' is absent
    if "feature" not in df.columns:
        df = df.rename(columns={df.columns[0]: "feature"})

    # Case-insensitive column lookup
    mean_col = next((c for c in df.columns if c.lower() == "mean"), None)
    std_col = next((c for c in df.columns if c.lower() == "std"), None)

    if mean_col is None or std_col is None:
        raise KeyError(
            f"'mean' and 'std' columns not found in '{filepath}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.set_index("feature")
    means = df[mean_col].to_dict()
    stds = df[std_col].to_dict()

    return means, stds


if __name__ == "__main__":
    means, stds = load_scale_params()
    print(f"Loaded {len(means)} features from '{SCALE_FILE}'.")
    sample_key = next(iter(means))
    print(f"  Example — {sample_key}: mean={means[sample_key]:.4f}, std={stds[sample_key]:.4f}")
