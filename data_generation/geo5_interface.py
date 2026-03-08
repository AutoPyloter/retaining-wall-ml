from geo5_2018_11_en_cantilever_wall_automation import CantileverWall
import time

# ---------------------------------------------------------------------------
# Soil class definitions (index 0–4)
# Each entry: [unit_weight (kN/m³), cohesion (kPa), friction_angle (°)]
# ---------------------------------------------------------------------------
SOIL_PROPERTIES = [
    [20,  0, 40],   # Class 1 — dense sand / gravel
    [20,  0, 36],   # Class 2 — medium-dense sand
    [19, 20, 30],   # Class 3 — silty sand / sandy silt
    [18, 30, 26],   # Class 4 — silt / low-plasticity clay
    [17, 40, 20],   # Class 5 — soft clay
]

# ---------------------------------------------------------------------------
# Water level scenarios mapped to hw index (0–4)
# Levels are expressed as fractions of wall geometry:
#   0 → dry (hw = 0)
#   1 → mid-stem (hw = H/2)
#   2 → top of stem (hw = H)
#   3 → mid-heel (hw = H/2 + H)   [x1/2 + H in original]
#   4 → top of backfill (hw = H + x1)
# ---------------------------------------------------------------------------
WATER_LEVEL_COUNT = 5


def perform_design_and_extract_safety_factors(
    H, x1, x2, x3, x4, x5, x6, x7, x8,
    q, SDS, Soil_Class, hw
):
    """
    Automates a single GEO5 Cantilever Wall analysis and returns the results.

    Drives the GEO5 GUI through geometry, soil, water, surcharge, foundation
    resistance, and earthquake inputs, then runs a Bishop circular-slip
    stability analysis and collects the output forces and moments.

    Parameters
    ----------
    H  : float  — Wall height (m)
    x1 : float  — Heel slab width (m)
    x2 : float  — Base slab thickness (m)
    x3 : float  — Stem bottom width (m)
    x4 : float  — Stem top width (m)
    x5 : float  — Toe slab width (m)
    x6 : float  — Key width (m)
    x7 : float  — Key depth (m)
    x8 : float  — Batter offset (m)
    q  : float  — Uniform surcharge load (kN/m²)
    SDS: float  — Design spectral acceleration parameter (g)
    Soil_Class : int  — Soil class index (0–4); see SOIL_PROPERTIES
    hw : int    — Water level scenario index (0–4); see WATER_LEVEL_COUNT

    Returns
    -------
    results : list[float]
        [Fa, Fp, Ma, Mp, slip_center_x, slip_center_z, slip_radius]
        where Fss (factor of safety against sliding) = Mp / Ma
    """
    wall = CantileverWall()
    wall.connect_to_application()
    wall.focus_on_window()

    # --- Geometry ---
    geo5_params = wall.map_to_geo5_params([H, x1, x2, x3, x4, x5, x6, x7, x8])
    print("GEO5 geometry params:", geo5_params)
    wall.geometry(geo5_params)

    # --- Profile (soil depth = wall height + toe width) ---
    soil_depth = round(H + x5, 2)
    print("Soil depth:", soil_depth)
    wall.profile([soil_depth])

    # --- Soil ---
    unit_weight, cohesion, friction_angle = SOIL_PROPERTIES[Soil_Class]
    print("Soil [unit_weight, friction_angle, cohesion]:", [unit_weight, friction_angle, cohesion])
    wall.soil([unit_weight, friction_angle, cohesion])

    # --- Water level ---
    water_level_options = [0, H / 2, H, x1 / 2 + H, H + x1]
    water_level = water_level_options[hw]
    print("Water level (m):", water_level)
    wall.water([water_level])

    # --- Surcharge ---
    print("Surcharge load (kN/m²):", q)
    wall.surcharge(q)

    # --- Foundation resistance ---
    wall_friction_angle = round(friction_angle * 2 / 3, 2)
    base_thickness = geo5_params[2]
    print("FF resistance [wall_friction_angle, base_thickness]:", [wall_friction_angle, base_thickness])
    wall.ff_resistance([wall_friction_angle, base_thickness])

    # --- Earthquake (Mononobe-Okabe coefficients from SDS) ---
    kh = round(0.4 * SDS / 2, 4)
    kv = round(0.5 * kh, 4)
    print("Seismic coefficients [kh, kv]:", [kh, kv])
    wall.earthquake([kh, kv])

    # --- Stability analysis (Bishop circular slip) ---
    results = wall.stability()
    print("Stability results [Fa, Fp, Ma, Mp, x, z, R]:", results)
    return results
