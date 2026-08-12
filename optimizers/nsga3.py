"""NSGA3 optimizer adapter on the ``new_fem_engine`` objective.

Many-objective evolutionary search via ``pymoo``.  The problem is genuinely
multi-objective: minimise ``(L - target)^2`` AND the axisymmetric core volume
-- i.e. hit the target inductance with as little core material as possible.
The chosen ``x_opt`` is the population member whose primary objective
(distance to target) is best.

Entry point (run_forward-style report + plots under ``output/nsga3/NSGA3/``)::

    python optimizers/nsga3.py [core] [mesh_size] [max_iters]
"""

import os
import sys
import time

import numpy as np
import jax.numpy as jnp

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.util.ref_dirs import get_reference_directions

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from optimizers import driver, objective


class _DriverProblem(Problem):
    """pymoo problem wrapper over a generic 2-objective ``fn`` (GEOM_KEYS-free).

    ``fn(x) -> (f1, f2)`` with the primary objective first; an optional third
    value is stored per-entry as the auxiliary ``inductance`` field (one FEM
    eval, no extra solves).
    """

    def __init__(self, fn, lo, hi):
        self.fn = fn
        self.n_evals = 0
        self.history = []
        super().__init__(n_var=len(lo), n_obj=2, n_constr=0, xl=lo, xu=hi)

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = np.empty(X.shape[0])
        f2 = np.empty(X.shape[0])
        for k in range(X.shape[0]):
            out_k = self.fn(np.asarray(X[k], dtype=float))
            if isinstance(out_k, tuple):
                a, b = float(out_k[0]), float(out_k[1])
            else:
                a = b = float(out_k)
            f1[k] = a
            f2[k] = b
            self.n_evals += 1
            entry = {
                "iteration": self.n_evals - 1,
                "loss": a,
                "x": np.asarray(X[k], dtype=float).tolist(),
            }
            if isinstance(out_k, tuple) and len(out_k) > 2:
                entry["inductance"] = float(out_k[2])
            self.history.append(entry)
        out["F"] = np.column_stack([f1, f2])


def _nsga3_driver(fn, x0, lo, hi, max_iters=30, pop_size=40, seed=42,
                  verbose=False):
    """Generic NSGA3 loop — GEOM_KEYS-free, testable on toy objectives.

    ``fn(x) -> (f1, f2)`` with the primary objective first (see
    :class:`_DriverProblem`).  ``x_best`` is the population member with the
    best primary objective.  Returns the uniform driver dict (see
    ``driver.py``): one history entry per evaluated member (``pop_size`` per
    generation); ``diagnostics`` carry the final Pareto set.
    """
    problem = _DriverProblem(fn, np.asarray(lo), np.asarray(hi))
    ref_dirs = get_reference_directions("uniform", 2, n_points=pop_size)

    sampling = FloatRandomSampling()
    crossover = SBX(prob=0.9, eta=15)
    mutation = PM(eta=20)
    algorithm = NSGA3(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
        ref_dirs=ref_dirs,
    )
    termination = MaximumGenerationTermination(n_max_gen=max_iters)

    res = pymoo_minimize(problem, algorithm, termination, verbose=verbose, seed=seed)

    F = res.F
    f1 = F[:, 0]
    k_best = int(np.argmin(f1))
    best_x = np.array(res.X[k_best], dtype=float)

    return {
        "x_best": best_x,
        "best_loss": float(f1[k_best]),
        "history": problem.history,
        "nit": problem.n_evals,
        "diagnostics": {"pareto": F, "n_evals": problem.n_evals},
    }


def run_nsga3(core_spec, x0=None, max_iters=30, w_vol=0.0, pop_size=40,
              mesh_size=None, fwd_opts=None, adj_opts=None, backend=None,
              seed=42, verbose=True):
    mesh_size = mesh_size or core_spec["mesh_size"]
    target_L = core_spec["target_L"]
    lo, hi = objective.bounds_arrays(core_spec)

    eval_fn = objective.make_eval(
        core_spec, w_vol=w_vol, mesh_size=mesh_size,
        fwd_opts=fwd_opts or objective.SP, adj_opts=adj_opts or objective.ADJ,
        backend=backend,
    )

    def fn(x):
        loss, L, V = eval_fn(jnp.asarray(x))
        return float(loss), float(V), float(L)

    t_start = time.time()
    out = _nsga3_driver(fn, objective.default_x0(core_spec), lo, hi,
                        max_iters=max_iters, pop_size=pop_size, seed=seed,
                        verbose=verbose)
    opt_time = time.time() - t_start

    best_x = np.clip(out["x_best"], lo, hi)
    best_loss = out["best_loss"]
    params_dict = objective.make_params_dict(best_x, core_spec)
    L_opt, _ = objective.solve_forward(params_dict, mesh_size, backend=backend)
    L_opt = float(L_opt)
    history = driver.enrich_history(out["history"])

    if verbose:
        print(f"\n  NSGA3 completed, {opt_time:.2f}s, "
              f"{out['diagnostics']['n_evals']} evaluations")
        print(f"  Final L = {L_opt * 1e9:.1f} nH, Target = {target_L * 1e9:.1f} nH")
        print(f"  Pareto size: {len(out['diagnostics']['pareto'])}")

    return {
        "optimizer": "NSGA3",
        "core": core_spec["name"],
        "x_opt": best_x,
        "params": params_dict,
        "L_opt": L_opt,
        "loss": best_loss,
        "history": history,
        "nit": out["nit"],
        "time": opt_time,
        "success": best_loss < 1.0,
        "message": "completed",
        "pareto": out["diagnostics"]["pareto"],
    }


if __name__ == "__main__":
    from new_fem_engine.report import report_design, OUTPUT_DIR

    core_name = sys.argv[1] if len(sys.argv) > 1 else "pq_40x40"
    mesh_size = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
    max_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    spec = objective.load_core_spec(core_name)
    target_nH = spec["target_L"] * 1e9
    print(f"=== NSGA3 | core {core_name} | target {target_nH:.0f} nH | "
          f"mesh {mesh_size} m | {max_iters} gens ===")

    r = run_nsga3(spec, max_iters=max_iters, mesh_size=mesh_size, verbose=True)

    out_dir = os.path.join(OUTPUT_DIR, "nsga3", "NSGA3")
    os.makedirs(out_dir, exist_ok=True)
    report_design(spec, mesh_size, r["x_opt"], out_dir, label="NSGA3")
    driver.save_convergence_plot(
        r["history"], os.path.join(out_dir, "convergence.png"),
        title=f"NSGA3 (mesh {mesh_size * 1e3:.3f} mm)", target_nH=target_nH)
    print(f"\nReport + plots saved under {out_dir}/")
