# GEO5 Pre-Run Setup

Before running the data generation pipeline, GEO5 2018 – Cantilever Wall module must be configured manually. The automation script controls geometry, soil parameters, loads, and water table at runtime — but the settings below must be in place **once**, before the first run.

---

## Step 1 — Settings: Analysis Methods

`Settings` frame → **Edit** button

**Materials and standards** tab:

| Parameter | Value |
|---|---|
| Active earth pressure calculation | Coulomb |
| Passive earth pressure calculation | Coulomb |
| Earthquake analysis | Mononobe-Okabe |
| Shape of earth wedge | Consider always vertical |
| Base key | The base key is considered as inclined footing bottom |
| Allowable eccentricity | 0.333 |
| Verification methodology | Safety factors (ASD) |

**Wall analysis** tab → **Permanent design situation** → Safety factors:

| Factor | Value |
|---|---|
| Safety factor for overturning (SF₀) | 1.50 |
| Safety factor for sliding resistance (SFs) | 1.50 |
| Safety factor for bearing capacity (SFb) | 1.50 |

---

## Step 2 — Settings: Slope Stability

From the same Settings panel → **Change analysis settings for program** → **Slope Stability**

**Materials and standards** tab:

| Parameter | Value |
|---|---|
| Earthquake analysis | Standard |
| Verification methodology | Safety factors (ASD) |

**Stability analysis** tab — analysis methods:
- Methods of analysis for polygonal slip surface: *(default)*
- Methods of analysis for circular slip surface: *(default)*

**Permanent design situation** → Safety Factor:

| Factor | Value |
|---|---|
| Safety Factor (SFs) | 1.50 |

---

## Step 3 — Profile

`Profile` frame → **Add** button → add a second depth entry.

The depth value does not matter — the script overwrites it at runtime.

| Interface # | Depth (m) |
|---|---|
| 1 | 0.00 |
| 2 | *(any value, e.g. 10.00)* |

---

## Step 4 — Soils: Define Two Soil Types

`Soils` frame → **Add** (twice)

### Soil 1 — `soil1`

| Parameter | Value |
|---|---|
| Name | soil1 |
| Unit weight γ | 20.00 kN/m³ |
| Stress-state | effective |
| Angle of internal friction φef | *(any — overwritten by script)* |
| Cohesion cef | *(any — overwritten by script)* |
| Angle of friction struct.-soil δ | *(any — overwritten by script)* |
| Soil (pressure at rest) | cohesive |
| Poisson's ratio ν | **0.33** |
| Saturated unit weight γsat | **≥ 20.00 kN/m³** |

> **Note:** γ and γsat must be ≥ 20 kN/m³ because the design space includes dry unit weights up to 20 kN/m³.

### Soil 2 — `backfill`

| Parameter | Value |
|---|---|
| Name | backfill |
| Unit weight γ | 20.00 kN/m³ |
| Stress-state | effective |
| Angle of internal friction φef | 40.00° |
| Cohesion cef | 0.00 kPa |
| Angle of friction struct.-soil δ | 26.67° |
| Soil (pressure at rest) | cohesionless |
| Poisson's ratio ν | *(default)* |
| Saturated unit weight γsat | **≥ 20.00 kN/m³** |

---

## Step 5 — Assign

`Assign` frame → assign **both layers** to `soil1`.

| Layer | Thickness (m) | Assigned soil |
|---|---|---|
| 1 | 10.00 | soil1 |
| 2 | *(auto)* | soil1 |

---

## Step 6 — Backfill

`Backfill` frame → select the **third icon** (slope backfill option)

| Parameter | Value |
|---|---|
| Assigned soil | backfill |
| Slope α | 45.00° |

---

## Step 7 — FF Resistance

`FF resistance` frame → select the **second icon**

| Parameter | Value |
|---|---|
| Resistance type | passive |
| Soil | backfill |
| Angle of friction struct.-soil δ | 0.00° |
| Thickness h | 0.50 m |
| Terrain surcharge f | 0.00 kN/m² |

> Other fields are updated by the script at runtime.

---

## Step 8 — Earthquake

`Earthquake` frame → check **Analyze earthquake**

| Parameter | Value |
|---|---|
| Factor of horizontal acceleration kh | 0.0000 |
| Factor of vertical acceleration kv | 0.0000 |
| Water influence | Confined water |

> Kh and Kv are set by the script according to the SDS parameter.

---

## Ready

After completing all steps above, save the `.guz` file and run the pipeline:

```bash
cd data_generation
python generate_dataset.py
```

The script will open GEO5 automatically, load the template file, and iterate through all 2048 design scenarios.
