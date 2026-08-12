"""report.py -- run_forward-style report + plots for an optimizer's design.

``report_design(core_spec, mesh_size, x, out_dir, label)`` builds the forward
problem at the geometry vector ``x`` (the optimized design), solves it with the
scipy spsolve reference backend, prints the same report blocks as
``run_forward.py`` (material load, assembly/solve timing, base L, core/air
energy split, Geometry & Source Verification, the Simulated/Analytical/
Datasheet AL triple) and saves the same plots (``geometry.png``,
``geometry_mesh.png``, ``psi.png``, ``bfield.png``) under ``out_dir``.

Reuses ``run_forward.py``'s own functions so an optimizer report is byte-for-
byte the same style as the forward demo; nothing is duplicated.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers import objective
from new_fem_engine.run_forward import (
    build_problem,
    geometry_fill,
    run_backend,
    print_material_load,
    print_reference,
    print_timing,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# scipy spsolve reference options (same as run_forward's spsolve builder).
SP_OPTIONS = {"newton": {"linear": {"spsolve_solver": {}}}}


def report_design(core_spec, mesh_size, x, out_dir, label=None):
    """Report + plots for the design ``x``, saved under ``out_dir``.

    Prints a run_forward-style report for the geometry vector ``x`` (length-5
    over :data:`optimizers.objective.VARY_GEOM_KEYS`; the frozen keys stay at
    their datasheet values) and saves the standard plots
    (geometry, geometry_mesh, psi, bfield) under ``out_dir``.  Returns the
    reference solve metrics dict from :func:`run_forward.run_backend`.

    The problem is built from :func:`optimizers.objective.base_params` so the
    reported L is the *same* objective the optimizer minimized (the loader
    spec lacks gap/yoke defaults which run_forward would otherwise default to
    a no-gap, flat-yoke geometry).
    """
    p = objective.make_params_dict(np.asarray(x), core_spec)
    spec = dict(core_spec)
    spec["params"] = p
    spec["mesh_size"] = mesh_size

    target_nH = core_spec["target_L"] * 1e9
    gap = spec["params"].get("gap_size", 0.0)
    gap_number = spec["params"].get("gap_number", 0)

    print("=== Experiment configuration ===")
    print(f"Optimizer:   {label or 'forward'}")
    print(f"Core:        {core_spec['name']}")
    print(f"Mesh size:   {mesh_size * 1000:.3f} mm")
    print(f"Gap size:    {gap * 1000:.3f} mm (x{gap_number})")
    print(f"Target L:    {target_nH:.1f} nH")
    print()

    material = spec["params"]["material"]
    print_material_load(material, spec)

    t_start = time.time()
    prob, Nr, Nz, mesh_size = build_problem(spec)
    fill, (f_core, _, _) = geometry_fill(prob, Nr, Nz, mesh_size)
    ref = run_backend(
        SP_OPTIONS, out_dir, prob, fill, f_core, Nr, Nz, mesh_size,
        spec["params"]["mur"], save_plots=True,
    )
    print_reference(ref, prob, f_core, Nr, Nz, mesh_size, spec)
    print_timing(t_start, ref, label="Report-phase")

    return ref


def output_root():
    """Repository ``output/`` folder shared with run_forward.py."""
    return OUTPUT_DIR


if __name__ == "__main__":
    # Standalone: report on a core's default design.
    from optimizers import objective

    core_name = sys.argv[1] if len(sys.argv) > 1 else "pq_40x40"
    mesh_size = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
    spec = objective.load_core_spec(core_name)
    x = objective.default_x0(spec)
    out_dir = os.path.join(OUTPUT_DIR, "report", core_name)
    os.makedirs(out_dir, exist_ok=True)
    report_design(spec, mesh_size, x, out_dir, label=f"default ({core_name})")
    print(f"Report + plots saved under {out_dir}/")
