"""CMA-ES optimizer adapter on the ``new_fem_engine`` objective.

Evolutionary, gradient-free global search via the maintained ``cma`` package
(the reference implementation, ``cma.CMAEvolutionStrategy``): (mu/mu, lambda)
covariance-matrix adaptation with rank-1 + rank-mu updates, box bounds handled
natively by the package.  Population points are evaluated in a Python loop —
each evaluation runs one forward solve through the objective, so the loss
function is NOT vmappable (the implicit-differentiation solve is an eager
custom_vjp).

Entry point (run_forward-style report + plots under ``output/cmaes/CMA-ES/``)::

    python optimizers/cmaes.py [core] [mesh_size] [max_iters]
"""

import os
import sys
import time

import numpy as np
import jax.numpy as jnp
import cma

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from optimizers import driver, objective


def _cmaes_driver(fn, x0, lo, hi, max_iters=40, pop_size=None, seed=42,
                  verbose=False):
    """Generic CMA-ES loop — GEOM_KEYS-free, testable on toy objectives.

    ``fn(x) -> float`` (or ``(loss, L)``; the second element, when present, is
    stored per-entry as the auxiliary ``inductance`` field — one FEM eval, no
    extra solves).  Box bounds are handled natively by the ``cma`` package;
    ``seed`` makes the run reproducible.  Returns the uniform driver dict (see
    ``driver.py``): one history entry per population member (``popsize`` per
    generation), ``diagnostics["sigma"]`` tracking the step size per
    generation (the CMA-ES convergence signal).
    """
    options = {
        "bounds": [lo.tolist(), hi.tolist()],
        "verbose": -1,  # suppress the package's own console logging
        "seed": seed,
    }
    if pop_size is not None:
        options["popsize"] = pop_size
    sigma0 = float(np.max(0.3 * (hi - lo) / 2.0))

    es = cma.CMAEvolutionStrategy(np.array(x0, dtype=float), sigma0, options)
    history = []
    sigmas = []
    best_loss = float("inf")
    best_x = x0.copy()

    for gen in range(max_iters):
        X = [np.clip(np.asarray(x, dtype=float), lo, hi) for x in es.ask()]
        losses = np.empty(len(X))
        for k, xk in enumerate(X):
            out_k = fn(xk)
            if isinstance(out_k, tuple):
                loss_k = float(out_k[0])
                entry = {
                    "iteration": len(history),
                    "loss": loss_k,
                    "x": np.asarray(xk, dtype=float).tolist(),
                    "inductance": float(out_k[1]) if not np.isnan(out_k[1])
                                   else float("nan"),
                }
            else:
                loss_k = float(out_k)
                entry = {
                    "iteration": len(history),
                    "loss": loss_k,
                    "x": np.asarray(xk, dtype=float).tolist(),
                }
            losses[k] = loss_k
            history.append(entry)
            if loss_k < best_loss:
                best_loss = loss_k
                best_x = xk.copy()
        es.tell(X, losses.tolist())
        sigmas.append(float(es.sigma))

        if verbose:
            print(f"  CMA-ES gen {gen + 1}/{max_iters}: loss={best_loss:.4e}, "
                  f"sigma={es.sigma:.4e}")

    return {
        "x_best": best_x,
        "best_loss": best_loss,
        "history": history,
        "nit": len(history),
        "diagnostics": {"sigma": sigmas, "gens": max_iters},
    }


def run_cmaes(core_spec, x0=None, max_iters=40, w_vol=0.0, pop_size=None,
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

    x0 = objective.default_x0(core_spec) if x0 is None else np.asarray(x0, dtype=float)
    x0 = np.clip(x0, lo, hi)

    def fn(x):
        loss_k, L_k, _ = eval_fn(jnp.asarray(x))
        return float(loss_k), float(L_k)

    t_start = time.time()
    out = _cmaes_driver(fn, x0, lo, hi, max_iters=max_iters, pop_size=pop_size,
                        seed=seed, verbose=verbose)
    opt_time = time.time() - t_start

    best_x = np.clip(out["x_best"], lo, hi)
    best_loss = out["best_loss"]
    params_dict = objective.make_params_dict(best_x, core_spec)
    L_opt, _ = objective.solve_forward(params_dict, mesh_size, backend=backend)
    L_opt = float(L_opt)
    history = driver.enrich_history(out["history"])

    if verbose:
        print(f"\n  CMA-ES completed in {max_iters} gens, {opt_time:.2f}s")
        print(f"  Final L = {L_opt * 1e9:.1f} nH, Target = {target_L * 1e9:.1f} nH")

    return {
        "optimizer": "CMA-ES",
        "core": core_spec["name"],
        "x_opt": best_x,
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
    """Run CMA-ES on one core; report + plots under ``output/cmaes/CMA-ES/``.

    Usage::

        python optimizers/cmaes.py [core] [mesh_size] [max_iters]
    """
    from new_fem_engine.report import report_design, OUTPUT_DIR

    core_name = sys.argv[1] if len(sys.argv) > 1 else "pq_40x40"
    mesh_size = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
    max_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    spec = objective.load_core_spec(core_name)
    target_nH = spec["target_L"] * 1e9
    print(f"=== CMA-ES | core {core_name} | target {target_nH:.0f} nH | "
          f"mesh {mesh_size} m | {max_iters} gens ===")

    r = run_cmaes(spec, max_iters=max_iters, mesh_size=mesh_size, verbose=True)

    out_dir = os.path.join(OUTPUT_DIR, "cmaes", "CMA-ES")
    os.makedirs(out_dir, exist_ok=True)
    report_design(spec, mesh_size, r["x_opt"], out_dir, label="CMA-ES")
    driver.save_convergence_plot(
        r["history"], os.path.join(out_dir, "convergence.png"),
        title=f"CMA-ES (mesh {mesh_size * 1e3:.3f} mm)", target_nH=target_nH)
    print(f"\nReport + plots saved under {out_dir}/")


if __name__ == "__main__":
    main()
