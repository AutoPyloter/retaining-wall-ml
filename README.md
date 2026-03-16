# Retaining Wall ML — Global Stability Prediction for Cantilever Retaining Walls

A machine learning framework for instant prediction of the global stability safety factor (F<sub>ss</sub>) of reinforced concrete cantilever retaining walls. The project covers the full pipeline: automated dataset generation via GEO5, SHAP-guided feature selection, and an ensemble of 35 trained regression models deployed in a customtkinter desktop application.

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

### 1. Configure GEO5 (one-time)

Run the automated setup script before the first data generation session:

```bash
cd data_generation
python geo5_setup.py
```

This will launch GEO5, configure all required settings (analysis methods, soil definitions, backfill, water, earthquake), and exit. See `data_generation/SETUP.md` for manual fallback instructions.

### 2. Generate the dataset

Ensure GEO5 is open with a Cantilever Wall project named **`guz`**, then run:

```bash
python generate_dataset.py
```

Results are appended to `output.txt` after each scenario. Press **`i`** at any time for a clean interrupt.

### 3. Annotate screenshots (optional)

```bash
python annotate_screenshots.py
```

Overlays wall parameters and computed F<sub>ss</sub> onto each scenario screenshot saved during the run. Annotated images are written to `screenshots/annotated/`.

### 4. Compile timelapse (optional)

```bash
python timelapse.py
```

Produces `timelapse.mp4` from the scenario screenshots.

### 5. Split the dataset

```bash
cd ml
python split_dataset.py
```

Produces `train.csv` (70 %), `test.csv` (20 %), `unseen.csv` (10 %) with stratified random sampling (`random_state=42`). F<sub>ss</sub> = M<sub>p</sub> / M<sub>a</sub> is computed and appended as the target column.

### 6. Train models

```bash
python train_models.py
```

Runs the full 8-stage training pipeline (see [ML Pipeline](#ml-pipeline-overview)). All outputs are saved to `ml/outputs/`.

### 7. Run batch predictions

```bash
python predict.py
```

### 8. Launch the desktop application

```bash
cd app
python main.py
```

---

## Dataset Description

### Design Space — Input Parameters (13)

| Parameter | Description | Range |
|---|---|---|
| H | Wall height | 4 – 10 m |
| x1 | Heel slab width | 0.3H – 10.0 m |
| x2 | Toe slab width | 0.15x1 – 0.45x1 m |
| x3 | Base slab thickness | 0.3 – 0.6 m |
| x4 | Stem bottom width | 0.3 – x3 m |
| x5 | Stem top width | 0.06H – 0.18H m |
| x6 | Key thickness | 0 – 1.2x5 m |
| x7 | Key width | 0 – 0.3x1 m |
| x8 | Key offset from heel | 0 – 0.7x1 m |
| q | Surcharge load | 0 – 20 kN/m² |
| SDS | Design spectral acceleration | 0.6 – 1.8 g |
| Soil_Class | Soil class index | 0 – 4 |
| hw | Water level scenario index | 0 – 4 |

**Soil classes:**

| Index | Description | γ (kN/m³) | c (kPa) | φ (°) |
|---|---|---|---|---|
| 0 | Dense sand / gravel | 20 | 0 | 40 |
| 1 | Medium-dense sand | 20 | 0 | 36 |
| 2 | Silty sand | 19 | 20 | 30 |
| 3 | Silt / low-plasticity clay | 18 | 30 | 26 |
| 4 | Soft clay | 17 | 40 | 20 |

**Water level scenarios:**

| Index | Water level |
|---|---|
| 0 | Dry (hw = 0) |
| 1 | Mid-stem (hw = H / 2) |
| 2 | Top of stem (hw = H) |
| 3 | Mid-heel slab (hw = H + x1 / 2) |
| 4 | Top of backfill (hw = H + x1) |

### Output Variables (7)

| Variable | Description |
|---|---|
| Fa | Sum of active forces (kN/m) |
| Fp | Sum of passive forces (kN/m) |
| Ma | Sliding moment (kN·m/m) |
| Mp | Resisting moment (kN·m/m) |
| x | Slip circle centre x-coordinate (m) |
| z | Slip circle centre z-coordinate (m) |
| R | Slip circle radius (m) |

> **Target variable:** F<sub>ss</sub> = M<sub>p</sub> / M<sub>a</sub> (global stability factor of safety, Bishop method)

### output.txt Format

Each row corresponds to one scenario (comma-separated, no header):

```
H, x1, x2, x3, x4, x5, x6, x7, x8, q, SDS, Soil_Class, hw, Fa, Fp, Ma, Mp, x, z, R
```

Total dataset: **2048 scenarios**

---

## ML Pipeline Overview

### Stage 1 — Data preparation (`split_dataset.py`)

F<sub>ss</sub> = M<sub>p</sub> / M<sub>a</sub> is computed from `output.txt`. The dataset is split into 70 % train / 20 % test / 10 % unseen hold-out using stratified random sampling.

### Stage 2 — XGBoost baseline for SHAP

An XGBoost grid search (256 combinations, 5-fold cross-validation) identifies the best baseline model. This model is used exclusively to compute SHAP feature importances.

```
Grid: n_estimators ∈ {100, 200} × max_depth ∈ {3, 5} × learning_rate ∈ {0.01, 0.1}
      × subsample ∈ {0.8, 1.0} × colsample_bytree ∈ {0.8, 1.0}
      × gamma ∈ {0, 0.1} × reg_alpha ∈ {0, 0.1} × reg_lambda ∈ {1, 10}
```

### Stage 3 — SHAP feature ranking

`shap.TreeExplainer` computes mean absolute SHAP values across the training set. Features are ranked by global importance. This ranking is frozen and used by all downstream models.

### Stage 4 — Model configurations (35 algorithms)

All models are wrapped in a three-step sklearn `Pipeline`:

```
FunctionTransformer (SHAP top-k selector) → OptionalScaler → Estimator
```

The `OptionalScaler` is a custom sklearn-compatible wrapper that applies one of five scalers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, or no scaling) depending on the model configuration. Both the feature count `k` and the scaler are treated as hyperparameters and tuned jointly with the model.

**Model families:**

| Family | Models |
|---|---|
| Linear | OLS, Ridge, Lasso, ElasticNet, Bayesian, ARD, Huber, RANSAC, TheilSen, OMP, PA |
| Cross-decomposition | PLS |
| GLM | Quantile, Poisson, Tweedie, Gamma |
| Kernel / Distance | SVM, kNN, KernelRidge, GPR |
| Neural | MLP |
| Polynomial | PolyR (PolynomialFeatures + Ridge) |
| Tree | DT, AdaBoost, RF, ExtraTrees, GBDT, HGB |
| Gradient Boosting | XGBoost, XGBoost_RF, LightGBM, CatBoost |
| Ensemble | StackingRegressor, VotingRegressor, NGBoost |

**CatBoost search space** includes both Bayesian bootstrap (`bagging_temperature`) and Bernoulli bootstrap variants with the following ranges:

```
iterations ∈ {100…500}, learning_rate ∈ {0.005…0.1}, depth ∈ {3…8},
l2_leaf_reg ∈ {1…10}, min_data_in_leaf ∈ {1…20}, max_bin ∈ {32…256},
random_strength ∈ {0.5…20}, nan_mode ∈ {Min, Max}
```

### Stage 5 — RandomizedSearchCV with simultaneous feature selection

For each model, `RandomizedSearchCV` jointly optimises model hyperparameters and the number of SHAP-ranked features `k ∈ {1, …, 18}`. The search uses 5-fold cross-validation with negative MSE as the scoring criterion.

| Model group | Iterations |
|---|---|
| SVM, KernelRidge, GPR, TheilSen, RANSAC, Stack, Voting | 500 |
| MLP, PolyR | 1 000 |
| All others | 5 000 |

A resume mechanism detects already-completed models from `saved_models/` and skips them, enabling interrupted runs to continue cleanly.

### Stage 6 — Final pipeline training and saving

The best pipeline (feature count + scaler + model hyperparameters) is re-fit on the full training set and serialised with joblib. Filenames use an 8-character MD5 hash of the hyperparameter string to stay within OS filename length limits:

```
CatBoost_k10_None_a3f7c2b1.pkl
```

### Stage 7 — Evaluation metrics (`metrics.py`)

Each model is evaluated on Train, Test, and Unseen splits with 17 indicators:

| Category | Metrics |
|---|---|
| Standard | MAE, MSE, RMSE, RSR, MAPE, sMAPE, R², EVS, MBE, CV(RMSE)%, MdAE, MaxE |
| Advanced | NSE, KGE, CCC, VAF(%), PI |

Results are saved to `outputs/all_models_random_search_results.csv` (35 models × 3 splits = 105 rows).

### Stage 8 — SHAP visualisations

`shap_bar.png` (mean absolute importance) and `shap_summary.png` (beeswarm) are saved to `outputs/plots/`. Per-model convergence plots and iteration logs are also written there.

---

## Desktop Application

The application is built with `customtkinter` and provides two tabs.

### Prediction tab

- **Wall cross-section canvas** — renders a scaled diagram of the cantilever wall geometry in real time as parameters are entered. Includes the backfill wedge, groundwater line (extends to canvas edges; left edge reaches the canvas boundary when hw exceeds wall height), and surcharge arrows.
- **Input panel** — 18 parameter fields organised by category (geometry, soil, loads).
- **Model selector** — lists all loadable models sorted by MaxE on the unseen set. Selecting a model enables the Predict and Info buttons.
- **Predict** — runs the selected pipeline and displays F<sub>ss</sub> ± MaxE.
- **Model info** — shows full Train / Test / Unseen metrics for the selected model in a popup table.
- **Bulk Predict** — runs all models simultaneously and opens a separate results window.

### Bulk prediction window

- Left panel: scrollable checkbox list of all models with individual predictions; **All / None** toggle buttons.
- Right panel: matplotlib figure containing:
  - KDE density curve of all predictions
  - Semi-transparent error boxes (prediction ± MaxE per model)
  - Box plot (Q1 / median / Q3 / whiskers / outliers)
  - Jittered swarm dots
  - Vernier-style axis with major and minor ticks
  - Summary statistics in the title (n, min, Q1, median, Q3, max)
- Chart updates instantly when checkboxes are toggled.

### Language support

The interface is available in English and Turkish. Language preference is persisted in `config.cfg`.

---

## GEO5 Automation

The data generation module automates GEO5 2018 — Cantilever Wall using `pywinauto` for GUI control and `pyperclip` for clipboard-based value entry.

### geo5_setup.py

A standalone one-time configuration script that:
1. Locates the GEO5 executable via Windows registry, `where` command, and `os.walk` scan across Program Files directories.
2. Launches GEO5 and dismisses the startup dialog.
3. Configures Settings (Coulomb earth pressure, Mononobe-Okabe earthquake, ASD verification), Profile (two-layer depth table), Soils (soil1 cohesive + backfill cohesionless), Assign, Backfill (45° slope), Water (behind wall), FF Resistance (passive, soil1), and Earthquake (analysis enabled).
4. All mouse clicks are resolved via `pywinauto` element lookup (`child_window(class_name=..., found_index=...)`) and relative coordinates expressed as fractions of element dimensions — resolution and DPI independent.

### generate_dataset.py

Iterates over the design space, invokes GEO5 for each scenario, reads results from the Stability analysis window, and appends one row to `output.txt`. A screenshot of the GEO5 stability view is saved to `screenshots/` for each completed scenario.

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

This repository is currently private. License will be added upon publication.