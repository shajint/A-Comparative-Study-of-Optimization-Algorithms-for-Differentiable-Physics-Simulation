"""main_run.py -- one file to run ALL optimizers on ONE core.

Edit the CONFIG block below and run ``python comparative_study/main_run.py``.
The file runs the configured optimizers on the configured core, overrides the
target inductance in memory (the core YAML is a read-only source file and is
never modified), then produces a run_forward-style report for each optimised
design (material load, assembly/solve timing, L, energy split, Geometry &
Source Verification, Simulated/Analytical/Datasheet AL) plus a per-design
backend comparison and the cross-optimizer table.

Mesh policy: L-BFGS-B always runs at ``LBFGS_MESH_SIZE`` (coarse sweep mesh, to
avoid the large one-time XLA compile at the production mesh); the other
optimizers run at ``MESH_SIZE`` (the production 0.001 mesh).  To keep the
comparison fair, every optimised design is re-solved at ``MESH_SIZE`` and
reported in the table as ``L@1mm``.

Outputs land under ``output/comparative_study/<CORE>/`` (one subfolder per
optimizer with the run_forward-style plots, plus ``benchmark.png``).
"""

import os
import sys
import time
import copy

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from optimizers import objective, backends
from optimizers.benchmark import (
    _runner, budget_iters, solves_of, analytical_L_at, _print_table, _plot,
)
from new_fem_engine import run_forward
from new_fem_engine.report import OUTPUT_DIR

# =============================================================================
# CONFIG -- edit these before running
# =============================================================================

CORE = "pq_40x40"        # core spec under cores/<CORE>/<CORE>.yaml (one core)
TARGET_L = 250e-9          # None -> use the YAML's target; a number overrides it
                         # in memory, e.g. TARGET_L = 300e-9.  The YAML (the
                         # datasheet AL) is read-only and never modified.
SOLVER = "lineax"        # linear solver: spsolve | lineax | feax
MESH_SIZE = 0.001        # production mesh for Adam/Bayesian/CMA-ES/NSGA3
LBFGS_MESH_SIZE = 0.002  # L-BFGS-B runs coarse (sweep mesh, avoids the big compile)
BUDGET = 120             # common FEM solve-equivalents per optimizer
OPTIMIZERS = ["lbfgs", "adam", "bayesian", "cmaes", "nsga3"]

# =============================================================================
# Helpers
# =============================================================================

def mesh_for(name):
    """L-BFGS-B always runs on the coarse mesh; the rest on the converged one."""
    return LBFGS_MESH_SIZE if name.partition(":")[0] == "lbfgs" else MESH_SIZE


def _design_spec(core_spec, x, mesh_size):
    """Full params dict + spec for the optimised design ``x`` at ``mesh_size``."""
    s = dict(core_spec)
    s["params"] = objective.make_params_dict(x, core_spec)
    s["mesh_size"] = mesh_size
    return s


def report_design(core_spec, x, mesh_size, label, out_dir):
    """run_forward-style report + plots for one optimised design.

    Mirrors ``run_forward.main``: reference (spsolve) solve + verification,
    then the backend comparison across all three solvers, plots saved under
    ``out_dir``.  Returns the reference metrics dict.
    """
    t0 = time.time()
    s = _design_spec(core_spec, x, mesh_size)
    prob, Nr, Nz, _ = run_forward.build_problem(s)
    fill, (f_core, _, _) = run_forward.geometry_fill(prob, Nr, Nz, mesh_size)

    cfg = run_forward.ExperimentConfig()
    cfg.core_name = core_spec["name"]

    ref = run_forward.run_backend(
        run_forward.spsolve_options(None), os.path.join(out_dir, "reference"),
        prob, fill, f_core, Nr, Nz, mesh_size, s["params"]["mur"],
        save_plots=True,
    )
    run_forward.print_reference(ref, prob, f_core, Nr, Nz, mesh_size, s)

    results = {}
    for name in cfg.backends:
        results[name] = run_forward.run_backend(
            run_forward.build_solver_options(name, cfg),
            os.path.join(out_dir, name), prob, fill, f_core, Nr, Nz, mesh_size,
            s["params"]["mur"], save_plots=True,
        )
    run_forward.print_backend_comparison(results, cfg)
    print(f"[{label}] design report took {time.time() - t0:.2f} s "
          f"(reference + {len(results)} backend solves + plots)")
    print()
    return ref


def converged_L(core_spec, x, mesh_size, backend):
    """Re-solve the design at the production mesh for the fair comparison."""
    params = objective.make_params_dict(x, core_spec)
    L, _ = objective.solve_forward(params, mesh_size, backend=backend)
    return float(L)


# =============================================================================
# Main
# =============================================================================

def main():
    all_start = time.time()

    if SOLVER not in backends.BACKENDS:
        raise ValueError(f"SOLVER must be one of {backends.BACKENDS}, "
                         f"got {SOLVER!r}")

    spec = objective.load_core_spec(CORE)
    target_nH = spec["target_L"] * 1e9
    if TARGET_L is not None:
        spec = copy.deepcopy(spec)
        spec["target_L"] = float(TARGET_L)
        target_nH = spec["target_L"] * 1e9
        print(f"Target inductance OVERRIDDEN to {TARGET_L * 1e9:.1f} nH "
              f"(in memory only; YAML untouched)")
    print(f"=== main_run | core {CORE} | target {target_nH:.1f} nH | "
          f"solver {SOLVER} | mesh {MESH_SIZE * 1000:.3f} mm "
          f"(lbfgs {LBFGS_MESH_SIZE * 1000:.3f} mm) | budget {BUDGET} solves ===")
    print()

    results = []
    for name in OPTIMIZERS:
        run_fn, label = _runner(name)
        ms = mesh_for(name)
        iters = budget_iters(name, BUDGET)
        kw = dict(mesh_size=ms, max_iters=iters, backend=SOLVER, verbose=True)
        if name.partition(":")[0] == "nsga3":
            kw["pop_size"] = 40
        if name.partition(":")[0] == "bayesian":
            kw["n_init"] = 5
        if name.partition(":")[0] == "adam":
            kw["traceable"] = SOLVER in backends.TRACEABLE_BACKENDS

        print(f"  --- {label} (mesh {ms * 1000:.3f} mm, budget {BUDGET} "
              f"solves -> {iters} iters) ---")
        t0 = time.time()
        r = run_fn(spec, **kw)
        r["wall_time"] = time.time() - t0
        r["label"] = label
        r["mesh_size"] = ms
        r["n_solves"] = solves_of(r)
        r["L_analytical"] = float(analytical_L_at(r))
        results.append(r)
        print(f"  -> L_fem = {r['L_opt'] * 1e9:.1f} nH | "
              f"L_analytical = {r['L_analytical'] * 1e9:.1f} nH | "
              f"target = {target_nH:.1f} nH | {r['n_solves']} solves | "
              f"{r['time']:.1f}s (+ wall {r['wall_time']:.1f}s)")
        print()

        # run_forward-style report of the optimised design
        out_dir = os.path.join(OUTPUT_DIR, "comparative_study", CORE, label)
        report_design(spec, r["x_opt"], ms, label, out_dir)

    # Fair comparison: every design re-solved at the converged mesh.
    t0 = time.time()
    for r in results:
        r["L_converged"] = converged_L(spec, r["x_opt"], MESH_SIZE, SOLVER)
    print(f"[fair re-check] all designs re-solved at {MESH_SIZE * 1000:.3f} mm "
          f"({time.time() - t0:.2f}s)")

    print(f"\n=== Comparison table (L_fem at each optimizer's own mesh; "
          f"L@1mm at the production mesh) ===")
    print(f"{'optimizer':<16}{'mesh mm':>8}{'L_fem nH':>10}"
          f"{'L@1mm nH':>10}{'L_an nH':>10}{'target':>9}{'loss':>10}"
          f"{'solves':>8}{'time s':>9}")
    for r in results:
        print(f"{r['label']:<16}{r['mesh_size'] * 1000:>8.3f}"
              f"{r['L_opt'] * 1e9:>10.2f}{r['L_converged'] * 1e9:>11.2f}"
              f"{r['L_analytical'] * 1e9:>10.2f}{target_nH:>9.1f}"
              f"{r['loss']:>10.1e}{r['n_solves']:>8}{r['time']:>9.1f}")
    Ls = [r["L_converged"] * 1e9 for r in results]
    print(f"\n  cross-optimizer spread of L@1mm: "
          f"{max(Ls) - min(Ls):.2f} nH (range {min(Ls):.1f}..{max(Ls):.1f})")
    print()
    _print_table(results, target_nH)

    try:
        _plot(results, spec, MESH_SIZE)
    except Exception as exc:
        print(f"  (skipping cross-optimizer plot: {exc})")

    print("=== Timing ===")
    print(f"Total main_run execution time: {time.time() - all_start:.2f} s")
    print()


if __name__ == "__main__":
    main()
