# Retaining Wall ML — Global Stability Prediction for Cantilever Retaining Walls

A machine learning framework for instant prediction of the global stability safety factor (F<sub>ss</sub>) of reinforced concrete cantilever retaining walls. The project covers the full pipeline: automated dataset generation via GEO5, SHAP-guided feature selection, and a CatBoost model deployed in a PyQt5 desktop application.

---

## Motivation

Conventional global stability analysis (Bishop circular-slip method) requires dedicated geotechnical software and significant computation time for each design scenario. This project replaces that process with a trained ML model capable of predicting F<sub>ss</sub> in real time, directly from wall geometry and site parameters — enabling rapid design screening and sensitivity analysis.

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
│   └── timelapse.py                                   # Compiles screenshots into MP4
│
├── ml/
│   └── (ML training, SHAP analysis, model evaluation)
│
├── app/
│   └── (PyQt5 desktop application)
│
├── requirements.txt
└── README.md
```

---

## GEO5 Prerequisites

Before running the data generation pipeline, the following must be configured **manually** in GEO5 (Cantilever Wall module):

### Settings → Materials and Standards
| Setting | Value |
|--------|-------|
| Active earth pressure | Coulomb |
| Passive earth pressure | Coulomb |
| Earthquake analysis | Mononobe-Okabe |
| Shape of earth wedge | Consider always vertical |
| Base key | Inclined footing bottom |
| Allowable eccentricity | 0.333 |
| Verification methodology | Safety Factors (ASD) |
| SF_o = SF_s = SF_b | 1.50 |

### Settings → Slope Stability
| Setting | Value |
|--------|-------|
| Verification methodology | Safety Factors (ASD) |
| SF | 1.50 |

### Profile
Enter two depth values. The second value is not critical — the automation script updates it at runtime.

### Soils
Define a soil named **`soil1`** with the following properties:
- Poisson's ratio: `0.33`
- Saturated unit weight: ≥ 20 kN/m³
- γ, φ, c, δ: any values — the script overwrites them at runtime

Define a second soil named **`backfill`** with:
- φ = 40°, c = 0, δ = 26.67°, cohesionless

### Backfill Tab
- Select option 3
- Slope = 45°
- Assigned soil = `backfill`

### FF Resistance
- Resistance type: Passive
- Soil: `backfill`
- All other values are updated by the script at runtime

### Earthquake
- Enable **"Analyze earthquake"** checkbox
- k<sub>h</sub> and k<sub>v</sub> values are entered by the script at runtime

---

## Installation

```bash
git clone https://github.com/<your-username>/retaining-wall-ml.git
cd retaining-wall-ml
pip install -r requirements.txt
```

> **Note:** NumPy 2.x is not compatible with the current OpenCV build.  
> `requirements.txt` pins `numpy<2` to avoid this issue.

---

## Usage

### 1. Generate the dataset

Ensure GEO5 is open with a Cantilever Wall project named **`guz`**, then run:

```bash
cd data_generation
python generate_dataset.py
```

Results are appended to `output.txt` after each scenario.  
Press **`i`** at any time for a clean interrupt.

### 2. Compile timelapse (optional)

```bash
python timelapse.py
```

Produces `timelapse.mp4` from screenshots saved during the generation run.

---

## Dataset Description

### Design Space — Input Parameters (13)

| Parameter | Description | Range |
|-----------|-------------|-------|
| H | Wall height (m) | 4 – 10 |
| x1 | Heel slab width (m) | 0.3H – 10.0 |
| x2 | Toe slab width (m) | 0.15x1 – 0.45x1 |
| x3 | Base slab thickness (m) | 0.3 – 0.6 |
| x4 | Stem bottom width (m) | 0.3 – x3 |
| x5 | Stem top width (m) | 0.06H – 0.18H |
| x6 | Key thickness (m) | 0 – 1.2x5 |
| x7 | Key width (m) | 0 – 0.3x1 |
| x8 | Key offset from heel (m) | 0 – 0.7x1 |
| q | Surcharge load (kN/m²) | 0 – 20 |
| SDS | Design spectral acceleration (g) | 0.6 – 1.8 |
| Soil_Class | Soil class index | 0 – 4 |
| hw | Water level scenario index | 0 – 4 |

**Soil classes:**

| Index | Description | γ (kN/m³) | c (kPa) | φ (°) |
|-------|-------------|-----------|---------|-------|
| 0 | Dense sand/gravel | 20 | 0 | 40 |
| 1 | Medium-dense sand | 20 | 0 | 36 |
| 2 | Silty sand | 19 | 20 | 30 |
| 3 | Silt/low-plasticity clay | 18 | 30 | 26 |
| 4 | Soft clay | 17 | 40 | 20 |

**Water level scenarios:**

| Index | Water level |
|-------|-------------|
| 0 | Dry (hw = 0) |
| 1 | Mid-stem (hw = H/2) |
| 2 | Top of stem (hw = H) |
| 3 | Mid-heel (hw = H/2 + x1/2) |
| 4 | Top of backfill (hw = H + x1) |

### Output Variables (7)

| Variable | Description |
|----------|-------------|
| Fa | Sum of active forces (kN/m) |
| Fp | Sum of passive forces (kN/m) |
| Ma | Sliding moment (kN·m/m) |
| Mp | Resisting moment (kN·m/m) |
| x | Slip circle center x-coordinate (m) |
| z | Slip circle center z-coordinate (m) |
| R | Slip circle radius (m) |

> **Target variable:** F<sub>ss</sub> = Mp / Ma (factor of safety against sliding, Bishop method)

### output.txt Format

Each row corresponds to one scenario (comma-separated, no header):

```
H, x1, x2, x3, x4, x5, x6, x7, x8, q, SDS, Soil_Class, hw, Fa, Fp, Ma, Mp, x, z, R
```

Total dataset: **2048 scenarios**

---

## ML Pipeline Overview

The ML pipeline (see `ml/`) consists of the following stages:

1. **Data preparation** — F<sub>ss</sub> = Mp / Ma is computed from `output.txt`; dataset split into 70% train / 20% test / 10% unseen hold-out
2. **Baseline model** — XGBoost trained on all features; SHAP values computed to rank feature importance
3. **SHAP-guided feature selection** — For each candidate algorithm, Randomized Search CV (2000 iterations) selects the optimal feature subset based on SHAP ranking
4. **Algorithm comparison** — 24+ algorithms evaluated using five metrics: NSE, KGE, sMAPE, CCC, VAF
5. **Final model** — CatBoost with Bayesian bootstrap, trained on the SHAP-selected feature subset

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
