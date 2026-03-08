from design_space_sampler import Discrete, ScenarioGenerator
from geo5_interface import perform_design_and_extract_safety_factors
from functools import lru_cache
import time
import keyboard


# ---------------------------------------------------------------------------
# Design space definition
# Each parameter is a Discrete sampler with (min, step, max) expressions.
# Expressions may reference previously sampled parameters by name,
# enabling geometrically consistent dependent sampling.
# ---------------------------------------------------------------------------
design_space = {
    # Wall height (m)
    'H':          Discrete('4', '1', '10'),

    # Heel slab width (m)
    'x1':         Discrete('0.3*H', '0.05*H', '10.0'),

    # Toe slab width (m)
    'x2':         Discrete('0.15*x1', '0.05*x1', 'min(round(x1-0.3,2), 0.45*x1)'),

    # Base slab thickness (m)
    'x3':         Discrete('0.3', '0.05', 'min(round(x1-x2,2), 0.6)'),

    # Stem bottom width (m)
    'x4':         Discrete('0.3', '0.05', 'x3'),

    # Stem top width (m)
    'x5':         Discrete('0.06*H', '0.01*H', '0.18*H'),

    # Key thickness — excluding base slab (m); 0 means no key
    'x6':         Discrete('0', '0.05*x5', '1.2*x5'),

    # Key width (m); 0 if no key
    'x7':         Discrete(
                      '0.05*x1 if x6 > 0 else 0',
                      '0.05*x1',
                      '0.3*x1  if x6 > 0 else 0'
                  ),

    # Key offset from heel (m); 0 if no key
    'x8':         Discrete(
                      '0',
                      '0.05*x1',
                      'min(round(x1-x7-0.01,2), 0.7*x1) if x6 > 0 else 0'
                  ),

    # Uniform surcharge load (kN/m²)
    'q':          Discrete('0', '5', '20.0'),

    # Design spectral acceleration parameter SDS (g)
    'SDS':        Discrete('0.6', '0.1', '1.8'),

    # Soil class index (0: Class A → 4: Class E)
    'Soil_Class': Discrete('0', '1', '4'),

    # Water level scenario index (0: dry → 4: full backfill height)
    'hw':         Discrete('0', '1', '4'),
}

OUTPUT_FILE = "output.txt"


@lru_cache(maxsize=None)
def _run_analysis_cached(H, x1, x2, x3, x4, x5, x6, x7, x8, q, SDS, Soil_Class, hw):
    """
    Cached wrapper around the GEO5 analysis function.

    Identical parameter combinations are skipped — GEO5 is not re-invoked
    and the output file is not written again. This avoids redundant GUI
    automation when the sampler draws a repeated scenario.

    Returns
    -------
    list[float] : [Fa, Fp, Ma, Mp, slip_center_x, slip_center_z, slip_radius]
    """
    results = perform_design_and_extract_safety_factors(
        H, x1, x2, x3, x4, x5, x6, x7, x8, q, SDS, Soil_Class, hw
    )

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{H},{x1},{x2},{x3},{x4},{x5},{x6},{x7},{x8},"
            f"{q},{SDS},{Soil_Class},{hw},"
            f"{','.join(map(str, results))}\n"
        )

    return results


def evaluate_scenario(scenario):
    """
    Unpack a scenario dict, check for interrupt, and call the cached analysis.

    Pressing 'i' during execution raises KeyboardInterrupt to allow a clean
    stop without corrupting the output file mid-write.

    Parameters
    ----------
    scenario : dict[str, float]
        Parameter dict produced by ScenarioGenerator.
    """
    if keyboard.is_pressed('i'):
        print("Interrupt key 'i' pressed. Stopping generation.")
        raise KeyboardInterrupt

    result = _run_analysis_cached(
        scenario['H'],   scenario['x1'], scenario['x2'], scenario['x3'],
        scenario['x4'],  scenario['x5'], scenario['x6'], scenario['x7'],
        scenario['x8'],  scenario['q'],  scenario['SDS'],
        scenario['Soil_Class'], scenario['hw']
    )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
try:
    start_time = time.time()
    generator = ScenarioGenerator(design_space, evaluate_scenario)
    generator.generate(n_scenarios=2048)
    end_time = time.time()
except KeyboardInterrupt:
    end_time = time.time()
    print("Generation stopped manually.")

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
