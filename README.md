# Retaining Wall ML

**An open-source machine learning framework for global stability safety factor prediction of cantilever retaining walls.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

---

## Overview

**Retaining Wall ML** predicts the global stability safety factor (*F*ss) of reinforced concrete cantilever retaining walls using an ensemble of 35 regression algorithms. The framework covers the full pipeline from dataset ingestion to interactive desktop inference:

- **SHAP-guided feature selection** — XGBoost baseline + TreeExplainer ranks 18 input features by global importance
- **35-model benchmark** — linear models, kernel methods, neural networks, gradient boosting (XGBoost, LightGBM, CatBoost), and meta-learners, all tuned jointly via `RandomizedSearchCV`
- **Desktop application** — bilingual (EN/TR) `customtkinter` GUI with real-time wall cross-section rendering, single-scenario prediction, and bulk uncertainty analysis

Gaussian Process Regression (GPR) achieves the lowest maximum absolute error on the unseen hold-out set (**MaxE = 0.184**, R² = 0.986).

> ⚠️ The trained models are valid within the parameter ranges of the dataset. Predictions outside these bounds should be verified against independent analyses.

---

## Repository Structure

```
retaining-wall-ml/
│
├── ml/
│   ├── split_dataset.py          # Train / test / unseen split (70/20/10)
│   ├── train_models.py           # Full training pipeline (SHAP + RandomizedSearchCV)
│   ├── metrics.py                # 17 regression metrics (MAE, RMSE, R², NSE, KGE, CCC …)
│   ├── predict.py                # Batch prediction on CSV datasets
│   ├── inference.py              # Single-scenario prediction (importable)
│   ├── pipeline_components.py    # OptionalScaler and SHAP feature selector
│   └── outputs/
│       ├── saved_models/         # Trained pipeline binaries (.pkl)
│       ├── plots/                # SHAP and convergence plots
│       ├── logs/                 # Training log and cache files
│       └── all_models_random_search_results.csv
│
├── app/
│   ├── main.py                   # Application entry point
│   ├── app.py                    # StabilityApp GUI (two-tab layout)
│   ├── model_info.py             # Metadata for all 35 regression models
│   ├── preprocessing.py          # SHAP feature ordering and input preparation
│   ├── pipeline_components.py    # Shared pipeline classes (joblib compatibility)
│   ├── config.py                 # Config file helpers
│   ├── language.py               # Translation loader
│   ├── utils.py                  # Shared utilities
│   ├── Language/
│   │   ├── EN.json               # English UI strings
│   │   └── TR.json               # Turkish UI strings
│   └── config.cfg                # Persisted user preferences
│
├── figs/
│   └── generate_figures.py       # Reproduces all paper figures and LaTeX tables
│
├── requirements.txt
└── README.md
```

---

## Dataset

A labelled dataset of **more than 2,000 independent design scenarios** is provided in the repository root. Each scenario is fully characterised by **18 input parameters** spanning four categories:

| Category | Parameters |
|---|---|
| Geometry | H, x₁–x₈, v₂, x₁ (derived), s₁ |
| Seismic loading | S_DS ∈ [0.6, 1.8] g |
| Soil properties | γ, c, φ — five discrete soil classes (ZA–ZE) |
| Hydrogeology | h_w — five discrete groundwater scenarios |

The target variable is the global stability safety factor *F*ss = *M*p / *M*a, where *M*p is the resisting moment and *M*a is the overturning moment of the critical Bishop slip circle.

The dataset is split into **training (70%) / test (20%) / unseen hold-out (10%)** by `split_dataset.py`.

---

## Installation

```bash
git clone https://github.com/AutoPyloter/retaining-wall-ml.git
cd retaining-wall-ml
pip install -r requirements.txt
```

**Requirements:** Python 3.11+, scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, customtkinter, matplotlib, joblib

---

## Usage

### 1 — Train all models

```bash
cd ml/
python train_models.py
```

Trains all 35 pipelines via `RandomizedSearchCV` (up to 5,000 iterations per model). A resume mechanism skips already-completed models — interrupted runs continue cleanly.

Results are saved to `ml/outputs/all_models_random_search_results.csv`.

### 2 — Run the desktop application

```bash
cd app/
python main.py
```

**Input & Visualisation tab** — enter 15 parameter values; the wall cross-section renders in real time.

**Model Selection tab** — choose a model, press **Predict** to get *F*ss ± MaxE. Press **Bulk Predict** to run all 35 models and view the KDE + box plot uncertainty window.

### 3 — Single-scenario inference (programmatic)

```python
from ml.inference import predict_fss

result = predict_fss(
    model_path="ml/outputs/saved_models/GPR_k14_Standard_a3f7c2b1.pkl",
    inputs={
        "H": 7.0, "X1": 3.5, "X2": 0.6, "X3": 0.45, "X4": 0.35,
        "X5": 0.42, "X6": 0.50, "X7": 0.35, "X8": 1.2,
        "q": 10, "sds": 1.2, "gama": 19, "c": 20, "fi": 30, "hw": 2
    }
)
print(f"Predicted Fss: {result:.4f}")
```

### 4 — Reproduce paper figures

```bash
cd figs/
python generate_figures.py \
  --csv ../ml/outputs/all_models_random_search_results.csv \
  --shap ../ml/outputs/plots/shap_bar.png \
  --outdir ./output
```

---

## Results

Top-5 models ranked by MaxE on the unseen hold-out set:

| Model | MAE | RMSE | R² | MaxE |
|---|---|---|---|---|
| GPR | 0.0404 | 0.0531 | 0.9855 | **0.184** |
| HGB | 0.0494 | 0.0657 | 0.9778 | 0.240 |
| Kernel Ridge | 0.0459 | 0.0602 | 0.9813 | 0.250 |
| Polynomial Ridge | 0.0448 | 0.0593 | 0.9819 | 0.252 |
| MLP | 0.0447 | 0.0604 | 0.9812 | 0.255 |

All 35 model results are in `ml/outputs/all_models_random_search_results.csv`.

---

## Citation

If you use this framework or dataset in your research, please cite:

```bibtex
@article{ozcan2025retainingwallml,
  author  = {Özcan, Abdulkadir and Uray, Esra},
  title   = {Retaining Wall ML: An Open-Source Machine Learning Framework
             for Global Stability Safety Factor Prediction of
             Cantilever Retaining Walls},
  journal = {SoftwareX},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT © 2025 Abdulkadir Özcan, Esra Uray — KTO Karatay University