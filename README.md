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
│   ├── split_dataset.py               # Train / test / unseen split (70/20/10)
│   ├── train_models.py                # Full training pipeline (SHAP + RandomizedSearchCV)
│   ├── metrics.py                     # 17 regression metrics (MAE, RMSE, R², NSE, KGE, CCC …)
│   ├── predict.py                     # Batch prediction on CSV datasets
│   ├── inference.py                   # Single-scenario prediction (importable)
│   ├── pipeline_components.py         # OptionalScaler and SHAP feature selector
│   └── outputs/
│       ├── saved_models/              # Trained pipeline binaries (.pkl)
│       ├── plots/                     # SHAP bar and beeswarm plots
│       ├── logs/                      # training_log.txt and cache files
│       ├── train.csv                  # Training split (70 %)
│       ├── test.csv                   # Test split (20 %)
│       ├── unseen.csv                 # Hold-out split (10 %)
│       └── all_models_random_search_results.csv   # Full results (35 models × 3 splits)
│
├── app/
│   ├── main.py                        # Application entry point
│   ├── app.py                         # StabilityApp GUI (two-tab layout)
│   ├── model_info.py                  # Metadata for all 35 regression models
│   ├── preprocessing.py               # SHAP feature ordering and input preparation
│   ├── pipeline_components.py         # Shared pipeline classes (joblib compatibility)
│   ├── config.py                      # Config file helpers and path resolver
│   ├── language.py                    # Translation loader
│   ├── utils.py                       # Shared utility functions
│   ├── Language/
│   │   ├── EN.json                    # English UI strings
│   │   └── TR.json                    # Turkish UI strings
│   └── config.cfg                     # Persisted user preferences
│
├── figs/
│   └── generate_figures.py            # Reproduces all paper figures and LaTeX tables
│
├── output.csv                         # Full labelled dataset (>2 000 scenarios)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/AutoPyloter/retaining-wall-ml.git
cd retaining-wall-ml
pip install -r requirements.txt
```

> **Note:** `numpy<2` is pinned due to OpenCV compatibility. `scikit-learn==1.6.1` is pinned to match the serialised model binaries in `ml/outputs/saved_models/` — using a different version will cause joblib deserialisation errors.

---

## Dataset

A labelled dataset of **more than 2,000 independent design scenarios** is provided as `output.csv` in the repository root. Each scenario is fully characterised by **18 input parameters** spanning four categories:

| Category | Parameters |
|---|---|
| Geometry | H, x₁–x₈, v₂, x₁ (derived), s₁ |
| Seismic loading | S_DS ∈ [0.6, 1.8] g |
| Soil properties | γ, c, φ — five discrete soil classes (ZA–ZE) |
| Hydrogeology | h_w — five discrete groundwater scenarios |

**Soil classes:**

| Class | γ (kN/m³) | c (kPa) | φ (°) |
|---|---|---|---|
| ZA | 20 | 0 | 40 |
| ZB | 20 | 0 | 36 |
| ZC | 19 | 20 | 30 |
| ZD | 18 | 30 | 26 |
| ZE | 17 | 40 | 20 |

**Groundwater scenarios:**

| h_w | Water-table depth | Condition |
|---|---|---|
| 0 | 0 | At ground surface (fully saturated) |
| 1 | 0.5H | At mid-height of wall stem |
| 2 | H | At foundation base level |
| 3 | H + 0.5x₁ | Below foundation base |
| 4 | H + x₁ | Well below foundation (effectively dry) |

**Target variable:** *F*ss = *M*p / *M*a (Bishop circular-slip method). Pre-computed and stored as the last column `fss`.

**Dataset format** — semicolon-separated CSV with a one-row header:

```
H;X1;X2;X3;X4;X5;X6;X7;X8;q;sds;v2;x1;s1;gama;c;fi;hw;fss
```

---

## Usage

### 1 — Split the dataset

```bash
cd ml/
python split_dataset.py
```

Produces `train.csv` (70%), `test.csv` (20%), `unseen.csv` (10%) in `ml/outputs/` using random sampling (`random_state=42`).

### 2 — Train all models

```bash
python train_models.py
```

Runs the full 8-stage training pipeline. A resume mechanism skips already-completed models — interrupted runs continue cleanly. All outputs are saved to `ml/outputs/`.

### 3 — Run batch predictions

```bash
python predict.py
```

### 4 — Run the desktop application

```bash
cd app/
python main.py
```

### 5 — Single-scenario inference (programmatic)

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

### 6 — Reproduce paper figures

```bash
cd figs/
python generate_figures.py \
  --csv ../ml/outputs/all_models_random_search_results.csv \
  --shap ../ml/outputs/plots/shap_bar.png \
  --outdir ./output
```

---

## ML Pipeline

### Stage 1 — Data preparation
`split_dataset.py` reads `output.csv` directly (header included, `fss` is the target column) and produces the three splits.

### Stage 2 — XGBoost baseline
Grid search over 256 hyperparameter combinations with 5-fold cross-validation identifies the best XGBoost baseline, used exclusively for SHAP computation.

### Stage 3 — SHAP feature ranking
`shap.TreeExplainer` computes mean absolute SHAP values across the training set. All 18 features are ranked by global importance. This ranking is frozen and reused by every downstream model.

### Stage 4 — Model configurations
All 35 algorithms are wrapped in a three-step sklearn `Pipeline`:

```
FunctionTransformer (SHAP top-k selector) → OptionalScaler → Estimator
```

`OptionalScaler` applies one of five scalers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, or no scaling). Feature count `k` and scaler are treated as hyperparameters and tuned jointly.

### Stage 5 — Joint hyperparameter search
`RandomizedSearchCV` jointly optimises feature count, scaler, and model hyperparameters (up to 5,000 iterations, 5-fold CV, negative MSE scoring).

### Stage 6 — Final training and serialisation
The best pipeline per model is re-fitted on the full training set and saved with `joblib`. Filenames embed feature count, scaler name, and an 8-character MD5 hash of the hyperparameter string.

### Stage 7 — Evaluation
Each pipeline is evaluated on all three splits using **17 performance indicators**:

`MAE`, `MSE`, `RMSE`, `RSR`, `MAPE`, `sMAPE`, `R²`, `EVS`, `MBE`, `CV(RMSE)%`, `MdAE`, `MaxE`, `NSE`, `KGE`, `CCC`, `VAF(%)`, `PI`

Results are consolidated in `ml/outputs/all_models_random_search_results.csv` (35 models × 3 splits = 105 rows).

### Stage 8 — SHAP visualisations
`shap_bar.png` (mean absolute importance) and `shap_summary.png` (beeswarm) are saved to `ml/outputs/plots/`.

---

## Desktop Application

Built with `customtkinter`, the application provides two tabs.

### Input & Visualisation tab
- **Input panel** — 15 editable parameter fields (3 further geometric parameters are derived automatically)
- **Wall canvas** — renders a scaled cross-section of the cantilever wall in real time as values are typed, including the backfill wedge, groundwater line, and surcharge arrows
- **Model selector** — lists all loadable pipelines sorted by MaxE on the unseen set
- **Predict** — runs the selected pipeline and displays *F*ss ± MaxE
- **Get Info** — shows a popup table with all 17 metrics across Train / Test / Unseen splits

### Bulk prediction window
- **Model list** — scrollable checkbox list with individual *F*ss predictions
- **KDE chart** — density curve, per-model error boxes (*F*ss ± MaxE), box plot, and jittered swarm dots on a shared axis
- Chart updates instantly when model checkboxes are toggled

### Language support
Interface is available in **English** and **Turkish**. Language preference is persisted in `config.cfg`.

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

Full results for all 35 models across all three splits are in `ml/outputs/all_models_random_search_results.csv`.

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

> Citation details will be updated upon publication.

---

## License

MIT © 2025 Abdulkadir Özcan, Esra Uray — KTO Karatay University