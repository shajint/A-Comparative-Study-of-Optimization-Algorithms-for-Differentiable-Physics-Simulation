"""Bayesian Optimization family on the ``new_fem_engine`` objective.

Single home for every Bayesian-family member in the project (paper-fidelity
requirement: the optimizer machinery is never written from scratch):

- :data:`BAYESIAN_FAMILY` — ``gp_skopt`` (skopt ``gp_minimize``: Gaussian-
  process surrogate with a Matern 5/2 kernel + Expected-Improvement
  acquisition) and ``tpe_optuna`` (optuna's Tree-structured Parzen Estimator
  sampler).  Both are maintained packages and both are gradient-free global
  searches: a surrogate refits from every evaluated candidate, so the design
  space is explored without gradients (one forward solve per candidate).
- :func:`_bayesian_driver` — the generic Bayesian loop (GEOM_KEYS-free,
  testable on toy objectives); the FEM wrapper :func:`run_bayesian` builds
  the FEM objective and enriches the driver history.
- :func:`main` — sweeps the whole family on one core; each member's optimized
  design is reported run_forward-style (console blocks + geometry/psi/bfield
  PNGs) under ``output/bayesian/<member>/``.

The result follows the objective contract (``optimizer, core, x_opt, params,
L_opt, loss, history, nit, time, success, message``).
"""

import os
import sys
import time

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from optimizers import driver, objective, backends

# Name -> (package, short description).  Both accept a plain scalar objective
# over the raw geometry vector, so the objective ``eval(x) -> (loss, L, vol)``
# drops straight in.
BAYESIAN_FAMILY = {
    "gp_skopt": "skopt.gp_minimize (GP + EI, Matern 5/2)",
    "tpe_optuna": "optuna TPE sampler",
}


def _make_eval(core_spec, mesh_size, w_vol, backend):
    """Forward ``eval(x) -> (loss, L, vol)``; jitted on jax-native backends.

    Gradient-free methods only need the forward solve, not gradients, so the
    fused traceable ``eval`` (one compiled program) is used on the lineax
    backend; spsolve/feax fall back to the eager path.
    """
    traceable = backend is None or backend in backends.TRACEABLE_BACKENDS
    return objective.make_eval(core_spec, w_vol=w_vol, mesh_size=mesh_size,
                             backend=backend, traceable=traceable)


def _bayesian_driver(fn, x0, lo, hi, name, n_init=5, max_iters=40, seed=42,
                     verbose=False):
    """Generic Bayesian-family loop — GEOM_KEYS-free, testable on toy objectives.

    ``fn(x) -> float`` (scalar objective).  ``n_init`` initial points (x0
    anchors the initialization) are followed by ``max_iters`` surrogate-guided
    candidates, so ``len(history) == n_init + max_iters`` (skopt/optuna record
    the initialization points too).  Returns the uniform driver dict (see
    ``driver.py``).
    """
    if name == "gp_skopt":
        return _skopt_driver(fn, x0, lo, hi, n_init, max_iters, seed, verbose)
    if name == "tpe_optuna":
        return _optuna_driver(fn, x0, lo, hi, n_init, max_iters, seed, verbose)
    raise ValueError(
        f"Unknown Bayesian optimizer {name!r}; choices: {sorted(BAYESIAN_FAMILY)}")


def _skopt_driver(fn, x0, lo, hi, n_init, max_iters, seed, verbose):
    """GP + Expected-Improvement via ``skopt.gp_minimize`` (Matern 5/2)."""
    from skopt import gp_minimize

    dim = len(lo)
    n_initial = max(n_init - 1, 0)  # x0 anchors the initialization
    n_calls = n_init + max_iters
    history = []

    def record(res):
        history.append({
            "iteration": len(res.func_vals) - 1,
            "loss": float(res.func_vals[-1]),
            "x": [float(v) for v in res.x_iters[-1]],
        })
        if verbose:
            i = len(res.func_vals)
            print(f"  gp_skopt iter {i}/{n_calls}: loss={res.func_vals[-1]:.4e}")

    res = gp_minimize(
        fn,
        dimensions=[(float(lo[i]), float(hi[i])) for i in range(dim)],
        x0=[x0.tolist()],
        n_initial_points=n_initial,
        n_calls=n_calls,
        base_estimator="GP",
        acq_func="EI",
        acq_optimizer="lbfgs",
        n_restarts_optimizer=5,
        random_state=seed,
        callback=record,
    )
    x_opt = np.clip(np.asarray(res.x, dtype=float), lo, hi)
    return {
        "x_best": x_opt,
        "best_loss": float(res.fun),
        "history": history,
        "nit": len(history),
        "diagnostics": {"n_calls": n_calls},
    }


def _optuna_driver(fn, x0, lo, hi, n_init, max_iters, seed, verbose):
    """Tree-structured Parzen Estimator via optuna's ``TPESampler``."""
    import optuna
    from optuna.samplers import TPESampler

    dim = len(lo)
    n_trials = n_init + max_iters
    history = []

    def objective(trial):
        x = [trial.suggest_float(f"x{j}", float(lo[j]), float(hi[j]))
             for j in range(dim)]
        return fn(x)

    def record(study, trial):
        history.append({
            "iteration": trial.number,
            "loss": float(trial.value),
            "x": [trial.params[f"x{j}"] for j in range(dim)],
        })
        if verbose:
            print(f"  tpe_optuna trial {trial.number + 1}/{n_trials}: "
                  f"loss={trial.value:.4e}")

    study = optuna.create_study(
        sampler=TPESampler(seed=seed), direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[record])

    x_opt = np.clip(np.asarray([study.best_params[f"x{j}"] for j in range(dim)],
                               dtype=float), lo, hi)
    return {
        "x_best": x_opt,
        "best_loss": float(study.best_value),
        "history": history,
        "nit": len(history),
        "diagnostics": {"n_trials": n_trials},
    }


def run_bayesian(core_spec, name="gp_skopt", x0=None, max_iters=40, w_vol=0.0,
                 mesh_size=None, n_init=5, backend=None, seed=42, verbose=True):
    """Run a Bayesian-family member on the objective.

    ``n_init`` points are evaluated first (a grid-free random initialization,
    anchored on ``x0``), then ``max_iters`` surrogate-guided candidates.
    Returns the objective result-dict contract.
    """
    mesh_size = mesh_size or core_spec["mesh_size"]
    target_L = core_spec["target_L"]
    lo, hi = objective.bounds_arrays(core_spec)

    eval_fn = _make_eval(core_spec, mesh_size, w_vol, backend)
    fn = lambda x: float(eval_fn(jnp.asarray(x, dtype=jnp.float64))[0])

    x0 = objective.default_x0(core_spec) if x0 is None else np.asarray(x0, dtype=float)
    x0 = np.clip(x0, lo, hi)

    t_start = time.time()
    out = _bayesian_driver(fn, x0, lo, hi, name, n_init=n_init,
                           max_iters=max_iters, seed=seed, verbose=verbose)
    opt_time = time.time() - t_start

    x_opt = np.clip(out["x_best"], lo, hi)
    best_loss = out["best_loss"]
    _, L_opt, _ = eval_fn(jnp.asarray(x_opt, dtype=jnp.float64))
    L_opt = float(L_opt)
    params_dict = objective.make_params_dict(x_opt, core_spec)
    history = driver.enrich_history(out["history"])

    if verbose:
        print(f"\n  {name}: completed, {opt_time:.2f}s")
        print(f"  Final L = {L_opt * 1e9:.1f} nH, Target = {target_L * 1e9:.1f} nH")

    return {
        "optimizer": name,
        "core": core_spec["name"],
        "x_opt": x_opt,
        "params": params_dict,
        "L_opt": L_opt,
        "loss": best_loss,
        "history": history,
        "nit": len(history),
        "time": opt_time,
        "success": best_loss < 1.0,
        "message": "completed",
    }


def main():
    """Sweep the Bayesian family on one core; report + plots per member.

    Usage::

        python optimizers/bayesian.py [core] [mesh_size] [budget]
    """
    from new_fem_engine.report import report_design, OUTPUT_DIR

    core_name = sys.argv[1] if len(sys.argv) > 1 else "pq_40x40"
    mesh_size = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 100  # n_init + iters

    spec = objective.load_core_spec(core_name)
    target_nH = spec["target_L"] * 1e9

    print(f"=== Bayesian family | core {core_name} | target {target_nH:.0f} nH | "
          f"mesh {mesh_size} m | budget ~{budget} evals/member ===")
    for k, desc in BAYESIAN_FAMILY.items():
        print(f"  {k}: {desc}")
    print()

    out_root = os.path.join(OUTPUT_DIR, "bayesian")
    for name in sorted(BAYESIAN_FAMILY):
        print(f"--- {name} ---")
        r = run_bayesian(spec, name=name, max_iters=budget, mesh_size=mesh_size,
                         n_init=5, verbose=False)
        print(f"  L_opt = {r['L_opt'] * 1e9:.1f} nH | loss = {r['loss']:.3e} | "
              f"{r['time']:.1f}s\n")
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        report_design(spec, mesh_size, r["x_opt"], out_dir, label=name)
        driver.save_convergence_plot(
            r["history"], os.path.join(out_dir, "convergence.png"),
            title=f"{name} (mesh {mesh_size * 1e3:.3f} mm)", target_nH=target_nH)
        print(f"  report + plots saved under {out_dir}/")
    print(f"\nDone. Bayesian-family reports under {out_root}/")


if __name__ == "__main__":
    main()
