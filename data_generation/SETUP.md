# GEO5 Pre-Run Setup

Before running the data generation pipeline, GEO5 2018 – Cantilever Wall module
must be configured. This can be done **automatically** using the setup script,
or **manually** by following the steps below.

---

## Automated Setup (Recommended)

Run the following command once on each machine before starting data generation:

```bash
cd data_generation
python geo5_setup.py
```

The script will:
1. Automatically locate and launch GEO5
2. Dismiss the startup dialog
3. Configure all required settings, frames and parameters in the correct order
4. Print progress to the console at each step

After the script completes, GEO5 is ready. You can proceed with:

```bash
python generate_dataset.py
```

> **Note:** Run `geo5_setup.py` only once per machine. Re-running it on an
> already-configured project file may cause unexpected behaviour.

---

## Manual Setup (Alternative)

If the automated script fails, configure GEO5 manually by following the steps below.

---

### Step 1 — Settings: Analysis Methods

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

### Step 2 — Settings: Slope Stability

Settings panel → **Change analysis settings for program** → **Slope Stability**

**Permanent design situation** → Safety Factor:

| Factor | Value |
|---|---|
| Safety Factor (SFs) | 1.50 |

---

### Step 3 — Profile

`Profile` frame → **Add** → enter any depth value (e.g. 10.00 m).

The script overwrites this value at runtime.

| Interface # | Depth (m) |
|---|---|
| 1 | 0.00 |
| 2 | 10.00 (any) |

---

### Step 4 — Soils: Define Two Soil Types

`Soils` frame → **Add** (twice)

#### Soil 1 — `soil1`

| Parameter | Value |
|---|---|
| Name | soil1 |
| Unit weight γ | 20.00 kN/m³ |
| Stress-state | effective |
| Soil (pressure at rest) | cohesive |
| Poisson's ratio ν | **0.33** |
| Saturated unit weight γsat | **≥ 20.00 kN/m³** |

#### Soil 2 — `backfill`

| Parameter | Value |
|---|---|
| Name | backfill |
| Unit weight γ | 20.00 kN/m³ |
| Angle of internal friction φef | 40.00° |
| Cohesion cef | 0.00 kPa |
| Angle of friction struct.-soil δ | 26.67° |
| Soil (pressure at rest) | cohesionless |
| Saturated unit weight γsat | **≥ 20.00 kN/m³** |

---

### Step 5 — Assign

`Assign` frame → assign **both layers** to `soil1`.

---

### Step 6 — Backfill

`Backfill` frame (right panel) → select the **third icon** (slope backfill)

| Parameter | Value |
|---|---|
| Assigned soil | backfill |
| Slope α | 45.00° |

---

### Step 7 — Water

`Water` frame → enable **water behind wall**.

---

### Step 8 — FF Resistance

`FF resistance` frame → enable resistance

| Parameter | Value |
|---|---|
| Resistance type | passive |
| Soil | soil1 |

---

### Step 9 — Earthquake

`Earthquake` frame → check **Analyze earthquake**

---

## Ready

After setup (automated or manual), save the `.guz` file and run:

```bash
cd data_generation
python generate_dataset.py
```