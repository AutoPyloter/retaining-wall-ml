# compute_scaling.py
# One-time script: fits a StandardScaler on train.csv and saves the resulting
# mean / scale values to scaling_factors.csv and model_scaling_info.csv.
#
# Run this script once after generating the training split:
#
#   python compute_scaling.py
#
# Output files are read at runtime by preprocessing.py.

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Load training data and extract input features
# ---------------------------------------------------------------------------

df = pd.read_csv("train.csv", sep=";", decimal=",")
feature_df = df.iloc[:, :-1]
feature_names = feature_df.columns.tolist()

# ---------------------------------------------------------------------------
# 2. Fit StandardScaler and save mean / scale per feature
# ---------------------------------------------------------------------------

scaler = StandardScaler()
scaler.fit(feature_df)

scaling_df = pd.DataFrame({
    "feature": feature_names,
    "mean":    scaler.mean_,
    "scale":   scaler.scale_,
})
scaling_df.to_csv("scaling_factors.csv", sep=";", decimal=",", index=False)
print("Scaling factors saved → scaling_factors.csv")

# ---------------------------------------------------------------------------
# 3. Save per-model scaling flag (True = requires standardisation)
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, bool] = {
    "OLS":        True,
    "Ridge":      True,
    "Lasso":      True,
    "Elastic":    True,
    "Bayesian":   True,
    "ARD":        True,
    "Huber":      True,
    "RANSAC":     True,
    "TheilSen":   True,
    "PLS":        True,
    "MLP":        True,
    "SVM":        True,
    "kNN":        True,
    "KR":         True,
    "PolyR":      True,
    "GPR":        True,
    "Stack":      True,
    "Quantile":   True,
    "Poisson":    True,
    "Tweedie":    True,
    "Gamma":      True,
    "OMP":        True,
    "PA":         True,
    "DT":         False,
    "AdaBoost":   False,
    "RF":         False,
    "ET":         False,
    "ExtraTrees": False,
    "GBDT":       False,
    "HGB":        False,
    "XGBoost":    False,
    "LightGBM":   False,
    "CAT":        False,
}

model_scaling_info = pd.DataFrame([
    {"model": name, "scale": flag}
    for name, flag in MODEL_CONFIGS.items()
])
model_scaling_info.to_csv("model_scaling_info.csv", sep=";", index=False)
print("Model scaling flags saved → model_scaling_info.csv")