"""
train_models.py  (v2)
---------------------
SHAP-guided feature selection + scaler selection + RandomizedSearchCV
for global stability safety factor (Fss) prediction.

Pipeline stages
---------------
1.  Load train / test / unseen splits
2.  XGBoost grid-search  →  best baseline model for SHAP
3.  SHAP feature ranking  →  ordered feature importance
4.  RandomizedSearchCV (5000 iterations per model):
      • feature subset selection (k = 1 … n_features)
      • scaler selection (scalable models only):
          StandardScaler | MinMaxScaler | RobustScaler | MaxAbsScaler | None
      • hyperparameter search
5.  Final model training on full train set (Pipeline: select → scale → model)
6.  Save each trained Pipeline to outputs/saved_models/
7.  Compute Train / Test / Unseen metrics
8.  Save results to outputs/all_models_random_search_results.csv
9.  Save SHAP bar and summary plots
"""

import os
import sys
import warnings

import catboost as cb
import joblib
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    GammaRegressor,
    HuberRegressor,
    Lasso,
    LinearRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    PoissonRegressor,
    QuantileRegressor,
    RANSACRegressor,
    Ridge,
    TheilSenRegressor,
    TweedieRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Optional imports — gracefully skipped if not installed
try:
    from ngboost import NGBRegressor

    HAS_NGBOOST = True
except ImportError:
    HAS_NGBOOST = False
    print("[WARN] ngboost not installed — NGBoost model will be skipped.")

from metrics import compute_metrics

# ---------------------------------------------------------------------------
# Suppress warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_CV_FOLDS = 5
N_SEARCH_ITER = 5000
OUTPUT_DIR = "outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "all_models_random_search_results.csv")
LOG_FILE = os.path.join(LOGS_DIR, "training_log.txt")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, logfile_path):
        self.terminal = sys.stdout
        self.log = open(logfile_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(LOG_FILE)
sys.stderr = sys.stdout


import os as _os

# ---------------------------------------------------------------------------
# Shared pipeline components (OptionalScaler)
# ---------------------------------------------------------------------------
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "app"))
from pipeline_components import OptionalScaler

# ---------------------------------------------------------------------------
# Scaler lists
# ---------------------------------------------------------------------------
SCALERS_ON = [StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler(), None]
SCALERS_OFF = [None]  # tree-based models — no scaling


# ---------------------------------------------------------------------------
# Stage 1: Load data
# ---------------------------------------------------------------------------
print("Stage 1: Loading data...")
train = pd.read_csv("train.csv", sep=";", decimal=",")
test = pd.read_csv("test.csv", sep=";", decimal=",")
unseen = pd.read_csv("unseen.csv", sep=";", decimal=",")

feature_names = train.columns[:-1].tolist()
X_train, y_train = train.iloc[:, :-1].values, train.iloc[:, -1].values
X_test, y_test = test.iloc[:, :-1].values, test.iloc[:, -1].values
X_un, y_un = unseen.iloc[:, :-1].values, unseen.iloc[:, -1].values
print(f" • Train: {X_train.shape}, Test: {X_test.shape}, Unseen: {X_un.shape}")

kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


# ---------------------------------------------------------------------------
# Stage 2: XGBoost grid-search (SHAP baseline) — cached
# ---------------------------------------------------------------------------
XGB_CACHE = os.path.join(LOGS_DIR, "xgb_baseline_cache.json")
SHAP_CACHE = os.path.join(LOGS_DIR, "shap_order_cache.npy")
SHAP_VALS_CACHE = os.path.join(LOGS_DIR, "shap_vals_cache.npy")

if os.path.exists(XGB_CACHE) and os.path.exists(SHAP_CACHE):
    print("\nStage 2: Loading cached XGBoost baseline...")
    import json

    with open(XGB_CACHE) as f:
        cache = json.load(f)
    best_xgb_params = cache["params"]
    best_xgb_mse = cache["mse"]
    print(f" ✔ Cached XGBoost params: {best_xgb_params}, CV MSE={best_xgb_mse:.4f}")
    best_xgb = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=RANDOM_SEED, **best_xgb_params
    )
    best_xgb.fit(X_train, y_train)

    print("\nStage 3: Loading cached SHAP feature ranking...")
    shap_order = np.load(SHAP_CACHE)
    shap_vals = np.load(SHAP_VALS_CACHE)
    print(f" ✔ Top 10 features by SHAP: {[feature_names[i] for i in shap_order[:10]]}")

else:
    print("\nStage 2: XGBoost grid-search for SHAP baseline...")
    xgb_grid = [
        {
            "n_estimators": n,
            "max_depth": d,
            "learning_rate": lr,
            "subsample": ss,
            "colsample_bytree": cbt,
            "gamma": g,
            "reg_alpha": ra,
            "reg_lambda": rl,
        }
        for n in [100, 200]
        for d in [3, 5]
        for lr in [0.01, 0.1]
        for ss in [0.8, 1.0]
        for cbt in [0.8, 1.0]
        for g in [0, 0.1]
        for ra in [0, 0.1]
        for rl in [1, 10]
    ]

    best_xgb_mse, best_xgb_params = np.inf, None
    for params in xgb_grid:
        fold_mses = []
        for train_idx, val_idx in kf.split(X_train):
            model = xgb.XGBRegressor(
                objective="reg:squarederror", random_state=RANDOM_SEED, **params
            )
            model.fit(X_train[train_idx], y_train[train_idx])
            fold_mses.append(mean_squared_error(y_train[val_idx], model.predict(X_train[val_idx])))
        avg_mse = np.mean(fold_mses)
        if avg_mse < best_xgb_mse:
            best_xgb_mse, best_xgb_params = avg_mse, params

    print(f" ✔ Best XGBoost params: {best_xgb_params}, CV MSE={best_xgb_mse:.4f}")
    best_xgb = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=RANDOM_SEED, **best_xgb_params
    )
    best_xgb.fit(X_train, y_train)

    # Save XGBoost baseline cache
    import json

    with open(XGB_CACHE, "w") as f:
        json.dump({"params": best_xgb_params, "mse": best_xgb_mse}, f)

    # ---------------------------------------------------------------------------
    # Stage 3: SHAP feature ranking
    # ---------------------------------------------------------------------------
    print("\nStage 3: SHAP feature ranking...")
    explainer = shap.TreeExplainer(best_xgb)
    shap_vals = explainer.shap_values(X_train)
    shap_order = np.argsort(np.mean(np.abs(shap_vals), axis=0))[::-1]
    print(f" ✔ Top 10 features by SHAP: {[feature_names[i] for i in shap_order[:10]]}")

    # Save SHAP cache
    np.save(SHAP_CACHE, shap_order)
    np.save(SHAP_VALS_CACHE, shap_vals)
    print(" ✔ XGBoost baseline and SHAP order cached.")


from pipeline_components import select_top_k_features, set_shap_order

set_shap_order(shap_order)


# ---------------------------------------------------------------------------
# Stage 4: Model configurations
# ---------------------------------------------------------------------------
k_range = list(range(1, len(feature_names) + 1))

model_configs = {
    # ── Linear ──────────────────────────────────────────────────────────────
    "OLS": {
        "est": LinearRegression(),
        "grid": [{}],
        "scale": True,
    },
    "Ridge": {
        "est": Ridge(random_state=RANDOM_SEED),
        "grid": [
            {"alpha": a, "solver": s}
            for a in [0.001, 0.01, 0.1, 1, 10, 100]
            for s in ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]
        ],
        "scale": True,
    },
    "Lasso": {
        "est": Lasso(max_iter=5000, random_state=RANDOM_SEED),
        "grid": [
            {"alpha": a, "selection": sel}
            for a in [0.0001, 0.001, 0.01, 0.1, 1, 10]
            for sel in ["cyclic", "random"]
        ],
        "scale": True,
    },
    "Elastic": {
        "est": ElasticNet(max_iter=5000, random_state=RANDOM_SEED),
        "grid": [
            {"alpha": a, "l1_ratio": l, "selection": sel}
            for a in [0.0001, 0.001, 0.01, 0.1, 1]
            for l in [0.1, 0.3, 0.5, 0.7, 0.9]
            for sel in ["cyclic", "random"]
        ],
        "scale": True,
    },
    "Bayesian": {
        "est": BayesianRidge(),
        "grid": [
            {"alpha_1": a1, "alpha_2": a2, "lambda_1": l1, "lambda_2": l2}
            for a1 in [1e-7, 1e-6, 1e-5, 1e-4]
            for a2 in [1e-7, 1e-6, 1e-5, 1e-4]
            for l1 in [1e-7, 1e-6, 1e-5, 1e-4]
            for l2 in [1e-7, 1e-6, 1e-5, 1e-4]
        ],
        "scale": True,
    },
    "ARD": {
        "est": ARDRegression(),
        "grid": [
            {"max_iter": mi, "alpha_1": a1, "alpha_2": a2, "lambda_1": l1, "lambda_2": l2}
            for mi in [200, 300, 500]
            for a1 in [1e-7, 1e-6, 1e-4]
            for a2 in [1e-7, 1e-6, 1e-4]
            for l1 in [1e-7, 1e-6, 1e-4]
            for l2 in [1e-7, 1e-6, 1e-4]
        ],
        "scale": True,
    },
    "Huber": {
        "est": HuberRegressor(max_iter=500),
        "grid": [
            {"epsilon": e, "alpha": a}
            for e in [1.1, 1.2, 1.35, 1.5, 2.0]
            for a in [1e-5, 1e-4, 1e-3, 0.01, 0.1]
        ],
        "scale": True,
    },
    "RANSAC": {
        "est": RANSACRegressor(random_state=RANDOM_SEED),
        "grid": [
            {"max_trials": mt, "residual_threshold": rt, "min_samples": ms}
            for mt in [50, 100, 200]
            for rt in [0.5, 1.0, 2.0, 5.0]
            for ms in [0.5, 0.8, None]
        ],
        "scale": True,
    },
    "TheilSen": {
        "est": TheilSenRegressor(random_state=RANDOM_SEED),
        "grid": [
            {"max_subpopulation": ms, "n_subsamples": ns}
            for ms in [100, 200, 300, 500]
            for ns in [30, 50, 100, None]
        ],
        "scale": True,
    },
    "OMP": {
        "est": OrthogonalMatchingPursuit(),
        "grid": [{"n_nonzero_coefs": k} for k in [1, 2, 3, 5, 8, 10, 15, 20]],
        "scale": True,
    },
    "PA": {
        "est": PassiveAggressiveRegressor(random_state=RANDOM_SEED),
        "grid": [
            {"C": c, "epsilon": e, "max_iter": mi}
            for c in [0.001, 0.01, 0.1, 1.0, 10.0]
            for e in [0.01, 0.1, 0.5, 1.0]
            for mi in [100, 300, 500]
        ],
        "scale": True,
    },
    # ── Cross-decomposition ──────────────────────────────────────────────────
    "PLS": {
        "est": PLSRegression(),
        "grid": [{"n_components": n, "max_iter": mi} for n in range(1, 19) for mi in [500, 1000]],
        "scale": True,
    },
    # ── GLM family ───────────────────────────────────────────────────────────
    "Quantile": {
        "est": QuantileRegressor(),
        "grid": [
            {"quantile": q, "alpha": a}
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]
            for a in [0.0, 0.01, 0.1, 1.0]
        ],
        "scale": True,
    },
    "Poisson": {
        "est": PoissonRegressor(),
        "grid": [
            {"alpha": a, "max_iter": mi} for a in [0.0, 0.01, 0.1, 1.0] for mi in [100, 300, 500]
        ],
        "scale": True,
    },
    "Tweedie": {
        "est": TweedieRegressor(),
        "grid": [
            {"power": p, "alpha": a, "max_iter": mi}
            for p in [0, 1, 1.5, 2, 3]
            for a in [0.0, 0.01, 0.1, 1.0]
            for mi in [100, 300]
        ],
        "scale": True,
    },
    "Gamma": {
        "est": GammaRegressor(),
        "grid": [
            {"alpha": a, "max_iter": mi} for a in [0.0, 0.01, 0.1, 1.0] for mi in [100, 300, 500]
        ],
        "scale": True,
    },
    # ── Kernel / Distance ────────────────────────────────────────────────────
    "SVM": {
        "est": SVR(),
        "grid": [
            {"kernel": k, "C": C, "gamma": g, "epsilon": e}
            for k in ["rbf", "linear", "poly"]
            for C in [0.01, 0.1, 1, 10, 100]
            for g in ["scale", "auto"]
            for e in [0.01, 0.05, 0.1, 0.2, 0.5]
        ],
        "scale": True,
    },
    "kNN": {
        "est": KNeighborsRegressor(),
        "grid": [
            {"n_neighbors": n, "weights": w, "p": p, "algorithm": alg}
            for n in [2, 3, 5, 7, 10, 15]
            for w in ["uniform", "distance"]
            for p in [1, 2]
            for alg in ["auto", "ball_tree", "kd_tree"]
        ],
        "scale": True,
    },
    "KR": {
        "est": KernelRidge(),
        "grid": [
            {"alpha": a, "kernel": k, "gamma": g, "degree": d}
            for a in [0.001, 0.01, 0.1, 1, 10]
            for k in ["linear", "rbf", "polynomial", "laplacian"]
            for g in [0.01, 0.1, 1, 10]
            for d in [2, 3]
        ],
        "scale": True,
    },
    "GPR": {
        "est": GaussianProcessRegressor(random_state=RANDOM_SEED, n_restarts_optimizer=0),
        "grid": [
            {"kernel": k, "alpha": a}
            for k in [RBF(l) for l in [0.1, 0.5, 1.0, 2.0, 5.0]]
            for a in [1e-4, 1e-3, 1e-2]
        ],
        "scale": True,
    },
    # ── Neural network ───────────────────────────────────────────────────────
    "MLP": {
        "est": MLPRegressor(max_iter=1000, random_state=RANDOM_SEED),
        "grid": [
            {
                "hidden_layer_sizes": h,
                "activation": act,
                "alpha": a,
                "learning_rate": lr,
                "early_stopping": es,
                "validation_fraction": vf,
                "beta_1": b1,
                "beta_2": b2,
            }
            for h in [(64,), (100,), (128,), (64, 64), (50, 50), (100, 50)]
            for act in ["relu", "tanh", "logistic"]
            for a in [1e-5, 1e-4, 1e-3, 1e-2]
            for lr in ["constant", "adaptive", "invscaling"]
            for es in [True, False]
            for vf in [0.1, 0.2]
            for b1 in [0.9, 0.95]
            for b2 in [0.999, 0.9999]
        ],
        "scale": True,
    },
    # ── Polynomial ───────────────────────────────────────────────────────────
    "PolyR": {
        "est": Pipeline(
            [("poly", PolynomialFeatures()), ("ridge", Ridge(random_state=RANDOM_SEED))]
        ),
        "grid": [
            {"poly__degree": d, "poly__interaction_only": io, "ridge__alpha": a}
            for d in [2, 3]
            for io in [False, True]
            for a in [0.001, 0.01, 0.1, 1, 10]
        ],
        "scale": True,
    },
    # ── Tree-based ───────────────────────────────────────────────────────────
    "DT": {
        "est": DecisionTreeRegressor(random_state=RANDOM_SEED),
        "grid": [
            {
                "max_depth": d,
                "min_samples_split": ms,
                "min_samples_leaf": ml,
                "max_features": mf,
                "criterion": cr,
            }
            for d in [None, 3, 5, 10, 15]
            for ms in [2, 5, 10, 20]
            for ml in [1, 2, 4, 8]
            for mf in [None, "sqrt", "log2"]
            for cr in ["squared_error", "friedman_mse"]
        ],
        "scale": False,
    },
    "AdaBoost": {
        "est": AdaBoostRegressor(random_state=RANDOM_SEED),
        "grid": [
            {"n_estimators": n, "learning_rate": lr, "loss": loss}
            for n in [50, 100, 200, 300]
            for lr in [0.001, 0.01, 0.1, 0.5, 1.0]
            for loss in ["linear", "square", "exponential"]
        ],
        "scale": False,
    },
    "RF": {
        "est": RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        "grid": [
            {
                "n_estimators": n,
                "max_depth": d,
                "min_samples_split": ms,
                "min_samples_leaf": ml,
                "max_features": mf,
                "max_samples": ms2,
            }
            for n in [100, 200, 300, 500]
            for d in [None, 5, 10, 20]
            for ms in [2, 5, 10]
            for ml in [1, 2, 4]
            for mf in ["sqrt", "log2", None]
            for ms2 in [None, 0.6, 0.8]
        ],
        "scale": False,
    },
    "ExtraTrees": {
        "est": ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        "grid": [
            {
                "n_estimators": n,
                "max_depth": d,
                "min_samples_split": ms,
                "min_samples_leaf": ml,
                "max_features": mf,
                "max_samples": ms2,
            }
            for n in [100, 200, 300, 500]
            for d in [None, 5, 10, 20]
            for ms in [2, 5, 10]
            for ml in [1, 2, 4]
            for mf in ["sqrt", "log2", None]
            for ms2 in [None, 0.6, 0.8]
        ],
        "scale": False,
    },
    "GBDT": {
        "est": GradientBoostingRegressor(random_state=RANDOM_SEED),
        "grid": [
            {
                "n_estimators": n,
                "learning_rate": lr,
                "max_depth": d,
                "subsample": ss,
                "max_leaf_nodes": mln,
                "min_impurity_decrease": mid,
                "min_samples_leaf": ml,
            }
            for n in [100, 200, 300]
            for lr in [0.005, 0.01, 0.05, 0.1, 0.2]
            for d in [3, 4, 5, 6]
            for ss in [0.6, 0.8, 1.0]
            for mln in [None, 10, 20, 50]
            for mid in [0.0, 0.001, 0.01]
            for ml in [1, 5, 10]
        ],
        "scale": False,
    },
    "HGB": {
        "est": HistGradientBoostingRegressor(random_state=RANDOM_SEED),
        "grid": [
            {
                "learning_rate": lr,
                "max_iter": it,
                "max_leaf_nodes": mln,
                "l2_regularization": l2,
                "max_depth": d,
                "min_samples_leaf": ml,
            }
            for lr in [0.005, 0.01, 0.05, 0.1, 0.2]
            for it in [100, 200, 300]
            for mln in [15, 20, 31, 50, None]
            for l2 in [0.0, 0.01, 0.1, 1.0]
            for d in [None, 5, 10]
            for ml in [10, 20, 50]
        ],
        "scale": False,
    },
    # ── Boosting ─────────────────────────────────────────────────────────────
    "XGBoost": {
        "est": xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
        "grid": [
            {
                "n_estimators": n,
                "max_depth": d,
                "learning_rate": lr,
                "subsample": ss,
                "colsample_bytree": cbt,
                "gamma": g,
                "reg_alpha": ra,
                "reg_lambda": rl,
                "min_child_weight": mcw,
            }
            for n in [100, 200, 300, 500]
            for d in [3, 4, 5, 6]
            for lr in [0.005, 0.01, 0.05, 0.1, 0.2]
            for ss in [0.6, 0.8, 1.0]
            for cbt in [0.6, 0.8, 1.0]
            for g in [0, 0.05, 0.1, 0.3]
            for ra in [0, 0.01, 0.1]
            for rl in [0.5, 1, 5, 10]
            for mcw in [1, 3, 5, 10]
        ],
        "scale": False,
    },
    "XGBoost_RF": {
        "est": xgb.XGBRFRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
        "grid": [
            {
                "n_estimators": n,
                "max_depth": d,
                "subsample": ss,
                "colsample_bytree": cbt,
                "reg_alpha": ra,
                "reg_lambda": rl,
                "num_parallel_tree": npt,
            }
            for n in [100, 200, 300]
            for d in [3, 5, 7]
            for ss in [0.6, 0.8, 1.0]
            for cbt in [0.6, 0.8, 1.0]
            for ra in [0, 0.1, 1.0]
            for rl in [1, 5, 10]
            for npt in [4, 8, 16]
        ],
        "scale": False,
    },
    "LightGBM": {
        "est": lgb.LGBMRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
        ),
        "grid": [
            {
                "n_estimators": n,
                "learning_rate": lr,
                "max_depth": d,
                "num_leaves": nl,
                "subsample": ss,
                "colsample_bytree": cbt,
                "reg_alpha": ra,
                "reg_lambda": rl,
                "min_child_samples": mcs,
            }
            for n in [100, 200, 300, 500]
            for lr in [0.005, 0.01, 0.05, 0.1, 0.2]
            for d in [-1, 5, 10, 15]
            for nl in [20, 31, 50, 100]
            for ss in [0.6, 0.8, 1.0]
            for cbt in [0.6, 0.8, 1.0]
            for ra in [0, 0.01, 0.1]
            for rl in [0, 0.1, 1]
            for mcs in [5, 10, 20, 50]
        ],
        "scale": False,
    },
    "CatBoost": {
        "est": cb.CatBoostRegressor(
            random_state=RANDOM_SEED,
            silent=True,
            thread_count=-1,
            allow_writing_files=False,
            task_type="CPU",
        ),
        "grid": [
            {
                "iterations": it,
                "learning_rate": lr,
                "depth": d,
                "l2_leaf_reg": l2,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": bt,
                "min_data_in_leaf": mdl,
                "max_bin": mb,
                "random_strength": rs,
                "nan_mode": nm,
            }
            for it in [100, 150, 200, 250, 300, 500]
            for lr in [0.003, 0.005, 0.01, 0.02, 0.05, 0.1]
            for d in [3, 4, 5, 6, 7, 8]
            for l2 in [1, 2, 3, 5, 7, 10]
            for bt in [0, 0.25, 0.5, 0.75, 1, 1.5, 2]
            for mdl in [1, 5, 10, 20]
            for mb in [32, 64, 128, 256]
            for rs in [0.5, 1.0, 5.0, 10.0, 20.0]
            for nm in ["Min", "Max"]
        ]
        + [
            {
                "iterations": it,
                "learning_rate": lr,
                "depth": d,
                "l2_leaf_reg": l2,
                "bootstrap_type": "Bernoulli",
                "min_data_in_leaf": mdl,
                "max_bin": mb,
                "random_strength": rs,
                "nan_mode": nm,
            }
            for it in [100, 150, 200, 250, 300, 500]
            for lr in [0.003, 0.005, 0.01, 0.02, 0.05, 0.1]
            for d in [3, 4, 5, 6, 7, 8]
            for l2 in [1, 2, 3, 5, 7, 10]
            for mdl in [1, 5, 10, 20]
            for mb in [32, 64, 128, 256]
            for rs in [0.5, 1.0, 5.0, 10.0, 20.0]
            for nm in ["Min", "Max"]
        ],
        "scale": False,
    },
    # ── Ensemble ─────────────────────────────────────────────────────────────
    "Stack": {
        "est": StackingRegressor(
            estimators=[
                ("xgb", xgb.XGBRegressor(objective="reg:squarederror", random_state=RANDOM_SEED)),
                ("svr", SVR()),
                ("rf", RandomForestRegressor(random_state=RANDOM_SEED)),
            ],
            final_estimator=Ridge(random_state=RANDOM_SEED),
            passthrough=True,
            n_jobs=-1,
        ),
        "grid": [
            {"final_estimator__alpha": a, "passthrough": pt}
            for a in [0.001, 0.01, 0.1, 1, 10]
            for pt in [True, False]
        ],
        "scale": True,
    },
    "Voting": {
        "est": VotingRegressor(
            estimators=[
                (
                    "xgb",
                    xgb.XGBRegressor(
                        objective="reg:squarederror", random_state=RANDOM_SEED, n_jobs=1
                    ),
                ),
                ("lgb", lgb.LGBMRegressor(random_state=RANDOM_SEED, verbosity=-1)),
                (
                    "cat",
                    cb.CatBoostRegressor(
                        random_state=RANDOM_SEED, silent=True, allow_writing_files=False
                    ),
                ),
                ("rf", RandomForestRegressor(random_state=RANDOM_SEED)),
            ],
            n_jobs=-1,
        ),
        "grid": [
            {
                "xgb__n_estimators": n,
                "xgb__learning_rate": lr,
                "lgb__n_estimators": n,
                "lgb__learning_rate": lr,
                "cat__iterations": n,
                "cat__learning_rate": lr,
                "rf__n_estimators": rf_n,
            }
            for n in [100, 200, 300]
            for lr in [0.05, 0.1, 0.2]
            for rf_n in [100, 200]
        ],
        "scale": False,
    },
}

# ── Optional: NGBoost ────────────────────────────────────────────────────────
if HAS_NGBOOST:
    model_configs["NGBoost"] = {
        "est": NGBRegressor(random_state=RANDOM_SEED, verbose=False),
        "grid": [
            {"n_estimators": n, "learning_rate": lr, "minibatch_frac": mf, "col_sample": cs}
            for n in [100, 200, 500]
            for lr in [0.01, 0.05, 0.1]
            for mf in [0.5, 0.8, 1.0]
            for cs in [0.5, 0.8, 1.0]
        ],
        "scale": False,
    }

# TabNet removed — incompatible with sklearn Pipeline (requires 2D y)


# ---------------------------------------------------------------------------
# Per-model iteration overrides — slow models get fewer iterations
# ---------------------------------------------------------------------------
MODEL_ITER_OVERRIDE = {
    "SVM": 500,
    "KR": 500,
    "GPR": 200,
    "TheilSen": 500,
    "RANSAC": 500,
    "Stack": 500,
    "Voting": 500,
    "MLP": 1000,
    "PolyR": 1000,
}

# ---------------------------------------------------------------------------
# Stage 5: RandomizedSearchCV with SHAP-guided feature + scaler selection
# ---------------------------------------------------------------------------
print(f"\nStage 5: RandomizedSearchCV (up to {N_SEARCH_ITER} iterations per model)...")

# Resume: detect completed models from saved_models/ pkl files AND existing CSV
results = []
completed_models = set()

# 1) Load existing CSV results if present
if os.path.exists(RESULTS_FILE):
    df_existing = pd.read_csv(RESULTS_FILE, sep=";", decimal=",")
    results = df_existing.to_dict("records")
    counts = df_existing.groupby("Model")["Dataset"].count()
    completed_models = set(counts[counts >= 3].index.tolist())

# 2) Also detect from pkl files in saved_models/ (catches runs without incremental save)
if os.path.exists(MODELS_DIR):
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith(".pkl"):
            model_name = fname.split("_k")[0]  # e.g. "OLS_k18_..." -> "OLS"
            completed_models.add(model_name)

if completed_models:
    print(f" ✔ Resuming — skipping {len(completed_models)} completed: {sorted(completed_models)}")
else:
    print(" • Starting fresh — no completed models found.")

for model_name, cfg in model_configs.items():
    if model_name in completed_models:
        print(f"\n→ Model: {model_name}  [SKIPPED — already complete]")
        continue

    n_iter = MODEL_ITER_OVERRIDE.get(model_name, N_SEARCH_ITER)
    print(f"\n→ Model: {model_name}  [n_iter={n_iter}]")
    base_estimator = cfg["est"]
    base_grid = cfg["grid"]
    use_scale = cfg["scale"]
    scalers = SCALERS_ON if use_scale else SCALERS_OFF

    # Pipeline: feature selection → optional scaler → model
    pipe = Pipeline(
        [
            ("select", FunctionTransformer(select_top_k_features, validate=False)),
            ("scale", OptionalScaler()),
            ("model", base_estimator),
        ]
    )

    # Build parameter distributions
    param_dist = {
        "select__kw_args": [{"k": k} for k in k_range],
        "scale__scaler": scalers,
    }
    for param_set in base_grid:
        for param, value in param_set.items():
            key = f"model__{param}"
            param_dist.setdefault(key, []).append(value)

    # De-duplicate
    for key, values in param_dist.items():
        unique = []
        for v in values:
            if v not in unique:
                unique.append(v)
        param_dist[key] = unique

    # RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=N_CV_FOLDS,
        scoring="neg_mean_squared_error",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    # Convergence data
    neg_mses = search.cv_results_["mean_test_score"]
    iter_mses = -neg_mses
    best_so_far = np.minimum.accumulate(iter_mses)
    iter_df = pd.DataFrame(
        {
            "iteration": np.arange(1, len(iter_mses) + 1),
            "cv_mse": iter_mses,
            "best_so_far_mse": best_so_far,
        }
    )
    iter_df.to_csv(
        os.path.join(PLOTS_DIR, f"{model_name}_random_search_iters.csv"),
        sep=";",
        decimal=",",
        index=False,
    )

    plt.figure()
    plt.plot(iter_df["iteration"], iter_df["best_so_far_mse"])
    plt.title(f"{model_name} — CV MSE Convergence")
    plt.xlabel("Random Search Iteration")
    plt.ylabel("Best CV MSE so far")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_convergence.png"))
    plt.close()

    # Best params
    best_k = search.best_params_["select__kw_args"]["k"]
    best_scaler = search.best_params_["scale__scaler"]
    best_model_params = {
        k.replace("model__", ""): v
        for k, v in search.best_params_.items()
        if k.startswith("model__")
    }
    scaler_name = type(best_scaler).__name__ if best_scaler is not None else "None"
    print(
        f" ✔ Best CV MSE: {-search.best_score_:.4f}, k={best_k}, "
        f"scaler={scaler_name}, params={best_model_params}"
    )
    print(f"   Selected features: {[feature_names[i] for i in shap_order[:best_k]]}")

    # Final Pipeline: fit on full train set
    final_pipe = Pipeline(
        [
            (
                "select",
                FunctionTransformer(select_top_k_features, validate=False, kw_args={"k": best_k}),
            ),
            ("scale", OptionalScaler(scaler=best_scaler)),
            ("model", clone(base_estimator).set_params(**best_model_params)),
        ]
    )
    final_pipe.fit(X_train, y_train)

    # Save full pipeline (includes scaler — no separate preprocessing needed)
    # Use short hash of params to keep filename within OS/Git limits (< 200 chars)
    import hashlib as _hl

    params_str = "_".join(f"{k}-{v}" for k, v in best_model_params.items())
    params_hash = _hl.md5(params_str.encode()).hexdigest()[:8]
    model_filename = f"{model_name}_k{best_k}_{scaler_name}_{params_hash}.pkl"
    joblib.dump(final_pipe, os.path.join(MODELS_DIR, model_filename))
    print(f" ✔ Pipeline saved: {model_filename}  (params: {params_str})")

    # Metrics on Train / Test / Unseen
    for split_name, X_split, y_split in [
        ("Train", X_train, y_train),
        ("Test", X_test, y_test),
        ("Unseen", X_un, y_un),
    ]:
        y_pred = final_pipe.predict(X_split)
        metrics = compute_metrics(y_split, y_pred)
        results.append(
            {
                "Model": model_name,
                "Dataset": split_name,
                "Num_Features": best_k,
                "Scaler": scaler_name,
                **{m: round(v, 4) for m, v in metrics.items()},
                "Best_Params": str(best_model_params),
            }
        )

    # Save incrementally after each model — safe to interrupt
    pd.DataFrame(results).to_csv(RESULTS_FILE, sep=";", decimal=",", index=False)
    print(f" ✔ Results saved ({len(results)//3} models complete)")


# ---------------------------------------------------------------------------
# Stage 6: Save results
# ---------------------------------------------------------------------------
df_results = pd.DataFrame(results)
print(f"\nStage 6: Saving results to '{RESULTS_FILE}'...")
print(df_results.to_string(index=False))
df_results.to_csv(RESULTS_FILE, sep=";", decimal=",", index=False)


# ---------------------------------------------------------------------------
# Stage 7: SHAP plots
# ---------------------------------------------------------------------------
print("\nStage 7: Saving SHAP plots...")
shap.summary_plot(shap_vals, X_train, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_bar.png"))
plt.close()

shap.summary_plot(shap_vals, X_train, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"))
plt.close()

print("\nDone.")
sys.stdout.log.close()
