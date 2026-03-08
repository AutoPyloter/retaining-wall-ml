import random
from abc import ABC, abstractmethod


def frange(start, step, stop):
    """
    Generate a list of evenly spaced float values from start to stop (inclusive).

    Parameters
    ----------
    start : float — First value in the range
    step  : float — Increment between consecutive values
    stop  : float — Last value in the range (included if not already present)

    Returns
    -------
    list[float]
    """
    values = []
    current = start
    while current <= stop:
        values.append(round(current, 2))
        current += step
    if round(stop, 2) not in values:
        values.append(round(stop, 2))
    return values


class Sampler(ABC):
    """Abstract base class for parameter samplers."""

    @abstractmethod
    def sample(self, dependency_values=None):
        """
        Draw a single sample value.

        Parameters
        ----------
        dependency_values : dict, optional
            Previously sampled parameter values that this sampler may depend on.

        Returns
        -------
        float or None
        """
        pass


class Discrete(Sampler):
    """
    Samples a random value from a discrete uniform grid.

    The grid boundaries and step size are defined as expression strings,
    allowing them to depend dynamically on previously sampled parameters.

    Parameters
    ----------
    min_expr  : str — Expression string for the minimum value (e.g. "0.3*H")
    step_expr : str — Expression string for the step size  (e.g. "0.01")
    max_expr  : str — Expression string for the maximum value (e.g. "0.5*H")

    Example
    -------
    >>> s = Discrete("0.3*H", "0.05", "0.6*H")
    >>> s.sample({"H": 6.0})
    1.65
    """

    def __init__(self, min_expr, step_expr, max_expr):
        self.min_expr = min_expr
        self.step_expr = step_expr
        self.max_expr = max_expr

    def sample(self, dependency_values=None):
        """
        Evaluate boundary expressions and return a random grid point.

        Returns None if the resulting grid is empty.
        """
        if dependency_values is None:
            dependency_values = {}

        min_val  = eval(self.min_expr,  {}, dependency_values)
        step_val = eval(self.step_expr, {}, dependency_values)
        max_val  = eval(self.max_expr,  {}, dependency_values)

        grid = frange(min_val, step_val, max_val)
        if not grid:
            return None
        return random.choice(grid)


class ScenarioGenerator:
    """
    Generates a dataset of cantilever wall design scenarios.

    Each scenario is produced by sampling all design parameters in declaration
    order, passing previously sampled values as context so that dependent
    parameters (e.g. x2 as a fraction of H) remain geometrically consistent.

    Parameters
    ----------
    design    : dict[str, Sampler]
        Ordered mapping of parameter names to their Sampler instances.
        Insertion order determines sampling order (Python 3.7+).
    objective : callable
        Function called with each sampled scenario dict, e.g. to run a
        GEO5 analysis and write results to disk.

    Example
    -------
    >>> gen = ScenarioGenerator(design=design_space, objective=run_analysis)
    >>> gen.generate(n_scenarios=2048)
    """

    def __init__(self, design, objective):
        self.design = design
        self.objective = objective

    def generate(self, n_scenarios=1000):
        """
        Sample and evaluate n_scenarios independent design scenarios.

        Parameters
        ----------
        n_scenarios : int — Number of scenarios to generate (default: 1000)
        """
        for _ in range(n_scenarios):
            scenario = {}
            dependency_values = {}
            for param_name, sampler in self.design.items():
                sampled_value = sampler.sample(dependency_values)
                scenario[param_name] = sampled_value
                dependency_values[param_name] = sampled_value
            self.objective(scenario)
