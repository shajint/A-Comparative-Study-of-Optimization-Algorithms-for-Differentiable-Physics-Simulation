"""Optimizer adapters for the ``new_fem_engine`` objective.

All optimizers share one contract: ``run_<name>(core_spec, ...) -> dict`` with
keys ``optimizer, core, x_opt, params, L_opt, loss, history, nit, time,
success, message``.  Optimizers are imported lazily so fragile optional
dependencies (``pymoo``, ``scipy``, ``optax``) only load when used.
"""

from optimizers import objective

__all__ = [
    "objective",
    "run_lbfgs",
    "run_adam",
    "run_bayesian",
    "run_cmaes",
    "run_nsga3",
]


def _load(name):
    import importlib
    return importlib.import_module(f"optimizers.{name}")


def run_lbfgs(core_spec, **kwargs):
    return _load("lbfgs").run_lbfgs(core_spec, **kwargs)


def run_adam(core_spec, **kwargs):
    return _load("adam").run_adam(core_spec, **kwargs)


def run_bayesian(core_spec, **kwargs):
    return _load("bayesian").run_bayesian(core_spec, **kwargs)


def run_cmaes(core_spec, **kwargs):
    return _load("cmaes").run_cmaes(core_spec, **kwargs)


def run_nsga3(core_spec, **kwargs):
    return _load("nsga3").run_nsga3(core_spec, **kwargs)


OPTIMIZERS = {
    "lbfgs": run_lbfgs,
    "adam": run_adam,
    "bayesian": run_bayesian,
    "cmaes": run_cmaes,
    "nsga3": run_nsga3,
}


def run_optimizer(name, core_spec, **kwargs):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {name!r}; choices: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name](core_spec, **kwargs)
