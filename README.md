# Retaining Wall ML — Global Stability Prediction for Cantilever Retaining Walls

A machine learning framework for instant prediction of the global stability safety factor (F<sub>ss</sub>) of reinforced concrete cantilever retaining walls. The project covers the full pipeline: automated dataset generation, SHAP-guided feature selection, and an ensemble of 35 trained regression models deployed in a customtkinter desktop application.

---

## Motivation

Conventional global stability analysis (Bishop circular-slip method) requires dedicated geotechnical software and significant computation time for each design scenario. This project replaces that process with trained ML models capable of predicting F<sub>ss</sub> in real time, directly from wall geometry and site parameters — enabling rapid design screening and sensitivity analysis.

---

## Repository Structure

```
retaining-wall-ml/
│
├── data_generation/
│   ├── geo5_2018_11_en_cantilever_wall_automation.py  # GEO5 GUI automation (CantileverWall class)
│   ├── geo5_interface.py                              # Single-scenario analysis orchestrator
│   ├── design_space_sampler.py                        # Discrete sampler & ScenarioGenerator
│   ├── generate_dataset.py                            # Entry point — generates output.txt
│   ├── geo5_setup.py                                  # One-time automated GEO5 configuration
│   ├── annotate_screenshots.py                        # Overlays metadata on scenario screenshots
│   ├── SETUP.md                                       # GEO5 setup instructions
│   └── timelapse.py                                   # Compiles screenshots into MP4
│
├── ml/
│   ├── split_dataset.py                               # Train / test / unseen split (70/20/10)
│   ├── train_models.py                                # Full training pipeline (SHAP + RandomizedSearchCV)
│   ├── metrics.py                                     # Regression metrics module (17 indicators)
│   ├── predict.py                                     # Batch prediction on CSV datasets
│   ├── inference.py                                   # Single-scenario prediction (importable module)
│   ├── pipeline_components.py                         # Shared OptionalScaler and feature selector
│   └── outputs/
│       ├── saved_models/                              # Trained pipeline binaries (.pkl)
│       ├── plots/                                     # SHAP and convergence plots
│       ├── logs/                                      # training_log.txt and cache files
│       └── all_models_random_search_results.csv       # Full evaluation results (35 models × 3 splits)
│
├── app/
│   ├── main.py                      # Application entry point
│   ├── app.py                       # StabilityApp GUI (customtkinter, two-tab layout)
│   ├── model_info.py                # Metadata for all 35 regression models
│   ├── preprocessing.py             # SHAP feature ordering and input preparation
│   ├── pipeline_components.py       # Shared pipeline classes (joblib compatibility)
│   ├── config.py                    # Config file helpers and path resolver
│   ├── language.py                  # Translation loader
│   ├── utils.py                     # Shared utility functions
│   ├── Language/
│   │   ├── EN.json                  # English UI translations
│   │   └── TR.json                  # Turkish UI translations
│   └── config.cfg                   # Persisted user preferences
│
├── figs/
│   └── generate_figures.py          # Script to produce all paper figures and LaTeX tables
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/retaining-wall-ml.git
cd retaining-wall-ml
pip install -r requirements.txt
```

> **Note:** `numpy<2` is pinned due to OpenCV compatibility. `scikit-learn==1.6.1` is pinned to match the serialised model binaries in `ml/outputs/saved_models/`.

---

## Usage

### 1. Generate the dataset

The labelled dataset (`output.txt`, 2 048 scenarios) is included in the repository root and can be used directly. If you wish to regenerate or extend it, a data generation pipeline is provided in `data_generation/` (requires a licensed installation of the relevant geotechnical analysis software).

### 2. Split the dataset

```bash
cd ml
python split_dataset.py
```

Produces `train.csv` (70 %), `test.csv` (20 %), `unseen.csv` (10 %) with stratified random sampling (`random_state=42`). F<sub>ss</sub> = M<sub>p</sub> / M<sub>a</sub> is computed and appended as the target column.

### 3. Train models

```bash
python train_models.py
```

Runs the full 8-stage training pipeline (see [ML Pipeline](#ml-pipeline-overview)). All outputs are saved to `ml/outputs/`.

### 4. Run batch predictions

```bash
python predict.py
```

### 5. Launch the desktop application

```bash
cd app
python main.py
```

---

## Dataset Description

### Design Space — Input Parameters (13)

| Column | Description | Notes |
|---|---|---|
| H | Wall height | 4 – 10 m |
| X1 | Foundation total width | 0.3H – 10.0 m |
| X2 | Front overhang (toe projection) | 0.15X1 – 0.45X1 m |
| X3 | Stem bottom width | 0.3 – 0.6 m |
| X4 | Stem top width | 0.3 – X3 m |
| X5 | Foundation thickness | 0.06H – 0.18H m |
| X6 | Key thickness | 0 – 1.2X5 m |
| X7 | Key width | 0 – 0.3X1 m |
| X8 | Key offset from heel | 0 – 0.7X1 m |
| q | Surcharge load | 0 – 20 kN/m² |
| sds | Design spectral acceleration | 0.6 – 1.8 g |
| v2 | Rear overhang | geometric |
| x1 | Foundation + key thickness (X5 + X6) | derived |
| s1 | Wall batter slope | geometric |
| gama | Soil unit weight (kN/m³) | numerical |
| c | Cohesion (kPa) | numerical |
| fi | Internal friction angle (°) | numerical |
| hw | Water level scenario index | 0 – 4 |
| **fss** | **Global stability safety factor (target)** | Fss = Mp / Ma |

**Soil properties** (`gama`, `c`, `fi`) are stored as numerical values in the dataset. The original sampling drew from five soil classes (dense sand/gravel through soft clay); the class index is not retained in the final CSV.

**Water level scenarios:**

| Index | Water level |
|---|---|
| 0 | Dry (hw = 0) |
| 1 | Mid-stem (hw = H / 2) |
| 2 | Top of stem (hw = H) |
| 3 | Mid-heel slab (hw = H + x1 / 2) |
| 4 | Top of backfill (hw = H + x1) |

> **Target variable:** F<sub>ss</sub> = M<sub>p</sub> / M<sub>a</sub> (global stability factor of safety, Bishop method). Pre-computed and stored as the last column `fss`.

### Dataset Format

Semicolon-separated CSV with a one-row header:

```
H;X1;X2;X3;X4;X5;X6;X7;X8;q;sds;v2;x1;s1;gama;c;fi;hw;fss
```

- 18 input features + 1 target column (`fss`)
- `fss` is pre-computed (Fss = Mp / Ma); no post-processing required
- Total: **>2 000 scenarios** (outliers removed)

---

## ML Pipeline Overview

### Stage 1 — Data preparation (`split_dataset.py`)

The dataset CSV is read directly (header included, `fss` column is the target). The dataset is split into 70 % train / 20 % test / 10 % unseen hold-out using stratified random sampling.

### Stage 2 — XGBoost baseline for SHAP

An XGBoost grid search (256 combinations, 5-fold cross-validation) identifies the best baseline model. This model is used exclusively to compute SHAP feature importances.

### Stage 3 — SHAP feature ranking

`shap.TreeExplainer` computes mean absolute SHAP values across the training set. Features are ranked by global importance. This ranking is frozen and used by all downstream models.

### Stage 4 — Model configurations (35 algorithms)

All models are wrapped in a three-step sklearn `Pipeline`:

```
FunctionTransformer (SHAP top-k selector) → OptionalScaler → Estimator
```

The `OptionalScaler` is a custom sklearn-compatible wrapper that applies one of five scalers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, or no scaling) depending on the model configuration. Both the feature count `k` and the scaler are treated as hyperparameters and tuned jointly with the model.

### Stage 5 — RandomizedSearchCV with simultaneous feature selection

For each model, `RandomizedSearchCV` jointly optimises feature count, scaler, and model hyperparameters (up to 5 000 iterations, 5-fold CV, negative MSE scoring). A resume mechanism skips already-completed models.

### Stage 6 — Final pipeline training and saving

The best pipeline is re-fitted on the full training set and serialised with joblib. Filenames use an 8-character MD5 hash of the hyperparameter string.

### Stage 7 — Evaluation metrics (`metrics.py`)

Each model is evaluated on Train, Test, and Unseen splits with 17 indicators: MAE, MSE, RMSE, RSR, MAPE, sMAPE, R², EVS, MBE, CV(RMSE)%, MdAE, MaxE, NSE, KGE, CCC, VAF(%), PI.

Results are saved to `outputs/all_models_random_search_results.csv` (35 models × 3 splits = 105 rows).

### Stage 8 — SHAP visualisations

`shap_bar.png` (mean absolute importance) and `shap_summary.png` (beeswarm) are saved to `outputs/plots/`.

---

## Desktop Application

The application is built with `customtkinter` and provides two tabs.

### Prediction tab

- **Wall cross-section canvas** — renders a scaled diagram of the cantilever wall geometry in real time as parameters are entered.
- **Input panel** — 18 parameter fields organised by category (geometry, soil, loads).
- **Model selector** — lists all loadable models sorted by MaxE on the unseen set.
- **Predict** — runs the selected pipeline and displays F<sub>ss</sub> ± MaxE.
- **Model info** — shows full Train / Test / Unseen metrics for the selected model.
- **Bulk Predict** — runs all models simultaneously and opens a separate results window.

### Bulk prediction window

- Left panel: scrollable checkbox list of all models with individual predictions.
- Right panel: matplotlib figure with KDE density curve, error boxes, box plot, and jittered swarm dots.
- Chart updates instantly when checkboxes are toggled.

### Language support

The interface is available in English and Turkish. Language preference is persisted in `config.cfg`.

---

## Citation

If you use this dataset or codebase in your research, please cite:

```bibtex
@article{,
  title   = {},
  author  = {},
  journal = {SoftwareX},
  year    = {},
  doi     = {}
}
```

> Citation details will be updated upon publication.

---

## License

MIT License. See `LICENSE` for details.