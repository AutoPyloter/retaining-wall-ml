import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def compute_metrics(y_true, y_pred):
    """
    Compute a comprehensive set of regression performance metrics.

    Includes standard sklearn metrics as well as hydrological and
    geotechnical goodness-of-fit indicators commonly used in the
    ML literature for physical system modelling.

    Parameters
    ----------
    y_true : array-like — Observed (ground truth) values
    y_pred : array-like — Model-predicted values

    Returns
    -------
    dict with the following keys:

    Standard metrics:
        MAE        — Mean Absolute Error
        MSE        — Mean Squared Error
        RMSE       — Root Mean Squared Error
        RSR        — RMSE-to-standard-deviation Ratio (0 = perfect)
        MAPE       — Mean Absolute Percentage Error (%)
        sMAPE      — Symmetric MAPE (%)
        R2         — Coefficient of Determination
        EVS        — Explained Variance Score
        MBE        — Mean Bias Error
        CV(RMSE)%  — Coefficient of Variation of RMSE (%)
        MdAE       — Median Absolute Error
        MaxE       — Maximum Absolute Error

    Advanced goodness-of-fit indicators:
        NSE        — Nash-Sutcliffe Efficiency (1 = perfect)
        KGE        — Kling-Gupta Efficiency (1 = perfect)
        CCC        — Concordance Correlation Coefficient (1 = perfect)
        VAF(%)     — Variance Accounted For (%)
        PI         — Performance Index (R2 / RMSE)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    rsr = rmse / np.std(y_true, ddof=1)

    # MAPE — exclude zero-valued observations to avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)
    cvrmse = rmse / np.mean(y_true) * 100
    md_ae = median_absolute_error(y_true, y_pred)
    maxe = max_error(y_true, y_pred)

    # Nash-Sutcliffe Efficiency
    nse = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    # Kling-Gupta Efficiency
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred, ddof=1) / np.std(y_true, ddof=1)
    beta = np.mean(y_pred) / np.mean(y_true)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    # Concordance Correlation Coefficient
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true, ddof=1), np.var(y_pred, ddof=1)
    cov = np.cov(y_true, y_pred, ddof=1)[0, 1]
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)

    # Variance Accounted For
    vaf = (1 - np.var(y_true - y_pred, ddof=1) / np.var(y_true, ddof=1)) * 100

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "RSR": rsr,
        "MAPE": mape,
        "sMAPE": smape,
        "R2": r2,
        "EVS": evs,
        "MBE": mbe,
        "CV(RMSE)%": cvrmse,
        "MdAE": md_ae,
        "MaxE": maxe,
        "NSE": nse,
        "KGE": kge,
        "CCC": ccc,
        "VAF(%)": vaf,
        "PI": r2 / rmse,
    }
