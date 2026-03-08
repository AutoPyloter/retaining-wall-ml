"""
train_models.py
---------------
SHAP-guided feature selection and RandomizedSearchCV training pipeline
for global stability safety factor (Fss) prediction.

Pipeline stages
---------------
1. Load train / test / unseen splits
2. XGBoost grid-search  →  best baseline model for SHAP
3. SHAP feature ranking  →  ordered feature importance
4. RandomizedSearchCV (2000 iterations per model) with simultaneous
   feature subset selection (k = 1 … n_features)
5. Final model training on full train set
6. Save each trained model to saved_models/
7. Compute Train / Test / Unseen metrics
8. Save results to all_models_random_search_results.csv
9. Save SHAP bar and summary plots
"""

import os
import sys
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

from sklearn.linear_model import (
    ARDRegression, BayesianRidge, ElasticNet, HuberRegressor,
    Lasso, LinearRegression, OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor, PoissonRegressor, QuantileRegressor,
    RANSACRegressor, Ridge, TheilSenRegressor, TweedieRegressor,
    GammaRegressor, SGDRegressor,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor,
    VotingRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from metrics import compute_metrics

# ---------------------------------------------------------------------------
# Suppress convergence and verbosity warnings for cleaner logs
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED    = 42
N_CV_FOLDS     = 5
N_SEARCH_ITER  = 2000
MODELS_DIR     = "saved_models"
RESULTS_FILE   = "all_models_random_search_results.csv"
LOG_FILE       = "training_log.txt"

os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Logger — mirrors stdout/stderr to a log file
# ---------------------------------------------------------------------------
class Logger:
    """Redirect stdout and stderr to both the console and a log file."""

    def __init__(self, logfile_path):
        self.terminal = sys.stdout
        self.log = open(logfile_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(LOG_FILE)
sys.stderr = sys.stdout


# ---------------------------------------------------------------------------
# Stage 1: Load data
# ---------------------------------------------------------------------------
print("Stage 1: Loading data...")
train  = pd.read_csv("train.csv",  sep=";", decimal=",")
test   = pd.read_csv("test.csv",   sep=";", decimal=",")
unseen = pd.read_csv("unseen.csv", sep=";", decimal=",")

feature_names = train.columns[:-1].tolist()
X_train, y_train = train.iloc[:, :-1].values,  train.iloc[:, -1].values
X_test,  y_test  = test.iloc[:, :-1].values,   test.iloc[:, -1].values
X_un,    y_un    = unseen.iloc[:, :-1].values,  unseen.iloc[:, -1].values
print(f" • Train: {X_train.shape}, Test: {X_test.shape}, Unseen: {X_un.shape}")

kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


# ---------------------------------------------------------------------------
# Stage 2: XGBoost grid-search (baseline model for SHAP)
# ---------------------------------------------------------------------------
print("\nStage 2: XGBoost grid-search for SHAP baseline...")
xgb_grid = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr,
     "subsample": ss, "colsample_bytree": cbt,
     "gamma": g, "reg_alpha": ra, "reg_lambda": rl}
    for n   in [100, 200]
    for d   in [3, 5]
    for lr  in [0.01, 0.1]
    for ss  in [0.8, 1.0]
    for cbt in [0.8, 1.0]
    for g   in [0, 0.1]
    for ra  in [0, 0.1]
    for rl  in [1, 10]
]

best_xgb_mse, best_xgb_params = np.inf, None
for params in xgb_grid:
    fold_mses = []
    for train_idx, val_idx in kf.split(X_train):
        model = xgb.XGBRegressor(
            objective="reg:squarederror", random_state=RANDOM_SEED, **params
        )
        model.fit(X_train[train_idx], y_train[train_idx])
        fold_mses.append(
            mean_squared_error(y_train[val_idx], model.predict(X_train[val_idx]))
        )
    avg_mse = np.mean(fold_mses)
    if avg_mse < best_xgb_mse:
        best_xgb_mse, best_xgb_params = avg_mse, params

print(f" ✔ Best XGBoost params: {best_xgb_params}, CV MSE={best_xgb_mse:.4f}")

best_xgb = xgb.XGBRegressor(
    objective="reg:squarederror", random_state=RANDOM_SEED, **best_xgb_params
)
best_xgb.fit(X_train, y_train)


# ---------------------------------------------------------------------------
# Stage 3: SHAP feature ranking
# ---------------------------------------------------------------------------
print("\nStage 3: SHAP feature ranking...")
explainer  = shap.TreeExplainer(best_xgb)
shap_vals  = explainer.shap_values(X_train)
shap_order = np.argsort(np.mean(np.abs(shap_vals), axis=0))[::-1]
print(f" ✔ Top 10 features by SHAP: {[feature_names[i] for i in shap_order[:10]]}")


def select_top_k_features(X, k):
    """Return the top-k SHAP-ranked features from X."""
    return X[:, shap_order[:k]]


# ---------------------------------------------------------------------------
# Stage 4: Model configurations
# ---------------------------------------------------------------------------
k_range = list(range(1, len(feature_names) + 1))

model_configs = {
    "OLS": {
        "est":   LinearRegression(),
        "grid":  [{}],
        "scale": True,
    },
    "Ridge": {
        "est":   Ridge(random_state=RANDOM_SEED),
        "grid":  [{"alpha": a, "solver": s}
                  for a in [0.01, 0.1, 1, 10]
                  for s in ["auto", "svd", "cholesky", "lsqr"]],
        "scale": True,
    },
    "Lasso": {
        "est":   Lasso(max_iter=5000, random_state=RANDOM_SEED),
        "grid":  [{"alpha": a, "selection": sel}
                  for a in [0.001, 0.01, 0.1, 1]
                  for sel in ["cyclic", "random"]],
        "scale": True,
    },
    "ElasticNet": {
        "est":   ElasticNet(max_iter=5000, random_state=RANDOM_SEED),
        "grid":  [{"alpha": a, "l1_ratio": l, "selection": sel}
                  for a   in [0.001, 0.01, 0.1, 1]
                  for l   in [0.2, 0.5, 0.8]
                  for sel in ["cyclic", "random"]],
        "scale": True,
    },
    "BayesianRidge": {
        "est":   BayesianRidge(),
        "grid":  [{}],
        "scale": True,
    },
    "HuberRegressor": {
        "est":   HuberRegressor(max_iter=300),
        "grid":  [{"epsilon": e, "alpha": a}
                  for e in [1.1, 1.35, 1.5]
                  for a in [0.0001, 0.01, 0.1]],
        "scale": True,
    },
    "SVM": {
        "est":   SVR(),
        "grid":  [{"kernel": k, "C": C, "gamma": g, "epsilon": e}
                  for k in ["rbf", "linear"]
                  for C in [0.1, 1, 10]
                  for g in ["scale", "auto"]
                  for e in [0.1, 0.2]],
        "scale": True,
    },
    "KNN": {
        "est":   KNeighborsRegressor(),
        "grid":  [{"n_neighbors": n, "weights": w, "p": p}
                  for n in [3, 5, 7]
                  for w in ["uniform", "distance"]
                  for p in [1, 2]],
        "scale": True,
    },
    "DecisionTree": {
        "est":   DecisionTreeRegressor(random_state=RANDOM_SEED),
        "grid":  [{"max_depth": d, "min_samples_split": ms,
                   "min_samples_leaf": ml, "max_features": mf}
                  for d  in [None, 5, 10]
                  for ms in [2, 5, 10]
                  for ml in [1, 2, 4]
                  for mf in [None, "sqrt", "log2"]],
        "scale": False,
    },
    "RandomForest": {
        "est":   RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        "grid":  [{"n_estimators": n, "max_depth": d,
                   "min_samples_split": ms, "min_samples_leaf": ml,
                   "max_features": mf}
                  for n  in [100, 200]
                  for d  in [None, 5, 10]
                  for ms in [2, 5]
                  for ml in [1, 2]
                  for mf in ["sqrt", "log2"]],
        "scale": False,
    },
    "ExtraTrees": {
        "est":   ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        "grid":  [{"n_estimators": n, "max_depth": d,
                   "min_samples_split": ms, "min_samples_leaf": ml,
                   "max_features": mf}
                  for n  in [100, 200]
                  for d  in [None, 5, 10]
                  for ms in [2, 5]
                  for ml in [1, 2]
                  for mf in ["sqrt", "log2"]],
        "scale": False,
    },
    "GBDT": {
        "est":   GradientBoostingRegressor(random_state=RANDOM_SEED),
        "grid":  [{"n_estimators": n, "learning_rate": lr, "max_depth": d,
                   "subsample": ss, "max_leaf_nodes": mln,
                   "min_impurity_decrease": mid}
                  for n   in [100, 200]
                  for lr  in [0.01, 0.1]
                  for d   in [3, 5]
                  for ss  in [0.8, 1.0]
                  for mln in [None, 20, 50]
                  for mid in [0.0, 0.01]],
        "scale": False,
    },
    "AdaBoost": {
        "est":   AdaBoostRegressor(random_state=RANDOM_SEED),
        "grid":  [{"n_estimators": n, "learning_rate": lr, "loss": loss}
                  for n    in [50, 100]
                  for lr   in [0.01, 0.1, 1.0]
                  for loss in ["linear", "square", "exponential"]],
        "scale": False,
    },
    "XGBoost": {
        "est":   xgb.XGBRegressor(
                     objective="reg:squarederror",
                     random_state=RANDOM_SEED,
                     n_jobs=1,
                 ),
        "grid":  [{"n_estimators": n, "max_depth": d, "learning_rate": lr,
                   "subsample": ss, "colsample_bytree": cbt,
                   "gamma": g, "reg_alpha": ra, "reg_lambda": rl}
                  for n   in [100, 200]
                  for d   in [3, 5]
                  for lr  in [0.01, 0.1]
                  for ss  in [0.8, 1.0]
                  for cbt in [0.8, 1.0]
                  for g   in [0, 0.1]
                  for ra  in [0, 0.1]
                  for rl  in [1, 10]],
        "scale": False,
    },
    "LightGBM": {
        "est":   lgb.LGBMRegressor(
                     random_state=RANDOM_SEED, n_jobs=-1,
                     verbosity=-1, silent=True,
                 ),
        "grid":  [{"n_estimators": n, "learning_rate": lr, "max_depth": d,
                   "subsample": ss, "colsample_bytree": cbt,
                   "reg_alpha": ra, "reg_lambda": rl}
                  for n   in [100, 200]
                  for lr  in [0.01, 0.1]
                  for d   in [3, 5]
                  for ss  in [0.8, 1.0]
                  for cbt in [0.8, 1.0]
                  for ra  in [0, 0.1]
                  for rl  in [0, 1]],
        "scale": False,
    },
    "CatBoost": {
        "est":   cb.CatBoostRegressor(
                     random_state=RANDOM_SEED,
                     silent=True,
                     thread_count=-1,
                     allow_writing_files=False,
                     task_type="CPU",
                 ),
        "grid":  [
            # Bayesian bootstrap — bagging_temperature controls randomness
            {"iterations": it, "learning_rate": lr, "depth": d,
             "l2_leaf_reg": l2, "bootstrap_type": "Bayesian",
             "bagging_temperature": bt, "min_data_in_leaf": mdl,
             "max_bin": mb, "random_strength": rs, "nan_mode": nm}
            for it  in [100, 150, 200, 250, 300]
            for lr  in [0.005, 0.01, 0.02, 0.05, 0.1]
            for d   in [3, 4, 5, 6, 7, 8]
            for l2  in [1, 2, 3, 5, 7, 10]
            for bt  in [0, 0.25, 0.5, 0.75, 1, 1.5]
            for mdl in [1, 5, 10, 20]
            for mb  in [32, 64, 128, 256]
            for rs  in [0.5, 1.0, 5.0, 10.0, 20.0]
            for nm  in ["Min", "Max"]
        ] + [
            # Bernoulli bootstrap — no bagging_temperature
            {"iterations": it, "learning_rate": lr, "depth": d,
             "l2_leaf_reg": l2, "bootstrap_type": "Bernoulli",
             "min_data_in_leaf": mdl, "max_bin": mb,
             "random_strength": rs, "nan_mode": nm}
            for it  in [100, 150, 200, 250, 300]
            for lr  in [0.005, 0.01, 0.02, 0.05, 0.1]
            for d   in [3, 4, 5, 6, 7, 8]
            for l2  in [1, 2, 3, 5, 7, 10]
            for mdl in [1, 5, 10, 20]
            for mb  in [32, 64, 128, 256]
            for rs  in [0.5, 1.0, 5.0, 10.0, 20.0]
            for nm  in ["Min", "Max"]
        ],
        "scale": False,
    },
}


# ---------------------------------------------------------------------------
# Stage 5: RandomizedSearchCV with SHAP-guided feature selection
# ---------------------------------------------------------------------------
print(f"\nStage 5: RandomizedSearchCV ({N_SEARCH_ITER} iterations per model)...")
results = []

for model_name, cfg in model_configs.items():
    print(f"\n→ Model: {model_name}")
    base_estimator, base_grid, use_scale = cfg["est"], cfg["grid"], cfg["scale"]

    # Build pipeline: feature selection → optional scaling → model
    steps = [("select", FunctionTransformer(select_top_k_features, validate=False))]
    if use_scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", base_estimator))
    pipe = Pipeline(steps)

    # Build parameter distributions from base grid + k range
    param_dist = {"select__kw_args": [{"k": k} for k in k_range]}
    for param_set in base_grid:
        for param, value in param_set.items():
            key = f"model__{param}"
            param_dist.setdefault(key, []).append(value)

    # De-duplicate parameter values
    for key, values in param_dist.items():
        unique = []
        for v in values:
            if v not in unique:
                unique.append(v)
        param_dist[key] = unique

    # Run randomized search
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=N_SEARCH_ITER,
        cv=N_CV_FOLDS,
        scoring="neg_mean_squared_error",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    # Save per-iteration convergence data
    neg_mses    = search.cv_results_["mean_test_score"]
    iter_mses   = -neg_mses
    best_so_far = np.minimum.accumulate(iter_mses)
    iter_df = pd.DataFrame({
        "iteration":       np.arange(1, len(iter_mses) + 1),
        "cv_mse":          iter_mses,
        "best_so_far_mse": best_so_far,
    })
    iter_df.to_csv(f"{model_name}_random_search_iters.csv", sep=";", decimal=",", index=False)

    plt.figure()
    plt.plot(iter_df["iteration"], iter_df["best_so_far_mse"])
    plt.title(f"{model_name} — CV MSE Convergence")
    plt.xlabel("Random Search Iteration")
    plt.ylabel("Best CV MSE so far")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_name}_convergence.png")
    plt.close()

    # Extract best parameters
    best_k = search.best_params_["select__kw_args"]["k"]
    best_model_params = {
        k.replace("model__", ""): v
        for k, v in search.best_params_.items()
        if k.startswith("model__")
    }
    print(f" ✔ Best CV MSE: {-search.best_score_:.4f}, k={best_k}, params={best_model_params}")
    print(f"   Selected features: {[feature_names[i] for i in shap_order[:best_k]]}")

    # Final training on full train set
    final_model = clone(base_estimator).set_params(**best_model_params)
    X_train_sel = (
        StandardScaler().fit_transform(X_train) if use_scale else X_train
    )[:, shap_order[:best_k]]
    final_model.fit(X_train_sel, y_train)

    # Save trained model
    params_str = "_".join(f"{k}-{v}" for k, v in best_model_params.items())
    model_filename = f"{model_name}_k{best_k}_{params_str}.pkl"
    joblib.dump(final_model, os.path.join(MODELS_DIR, model_filename))
    print(f" ✔ Model saved: {model_filename}")

    # Compute metrics on Train / Test / Unseen
    for split_name, X_split, y_split in [
        ("Train",  X_train, y_train),
        ("Test",   X_test,  y_test),
        ("Unseen", X_un,    y_un),
    ]:
        X_sel = (
            StandardScaler().fit_transform(X_split) if use_scale else X_split
        )[:, shap_order[:best_k]]
        y_pred = final_model.predict(X_sel)
        metrics = compute_metrics(y_split, y_pred)

        results.append({
            "Model":        model_name,
            "Dataset":      split_name,
            "Num_Features": best_k,
            **{m: round(v, 4) for m, v in metrics.items()},
            "Best_Params":  best_model_params,
        })


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
plt.savefig("shap_bar.png")
plt.close()

shap.summary_plot(shap_vals, X_train, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()

print("Done.")
sys.stdout.log.close()
