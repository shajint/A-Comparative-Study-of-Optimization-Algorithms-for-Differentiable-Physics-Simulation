"""solver_speed_comparison.py -- solver scaling comparison (spsolve vs lineax vs feax).

Runs the ``new_fem_engine`` magnetostatics forward solve on the bundled PQ
cores (``pq_32x30`` and ``pq_40x40`` by default) over a mesh-size sweep (1.0
mm down to 0.2 mm) and records the steady-state linear-solve and assembly
times for the three linear-solver backends: scipy spsolve (direct), lineax
(cg) and feax (Krylov).  Every timed solve also reports the simulated 1-turn
inductance (energy method), which is tabulated against the analytical
reluctance-model L and the core target (260 nH for the bundled cores), so
solver speed is always presented together with a correctness /
mesh-convergence check.  Prints both tables per core and saves a single
solver-speed-vs-mesh PNG with one subplot per core (left = first core in
``CORES``, right = last) under ``fully_differentiable_simulation/output/``.

Timing protocol
---------------
Every (mesh, backend) pair is warmed up once (the first solve triggers the
JAX/XLA compile for the lineax/feax iterative solvers) and the reported number
is the median of ``REPEATS`` subsequent steady-state solves.  Only the
non-traceable ``solver()`` Newton path is used: the traceable BCOO path
(``solver_jax``) reports zero timing, and spsolve can only run there anyway.
All three backends therefore share the same assembly + Newton code and differ
only in the linear solve.

Depends only on ``new_fem_engine`` (``run_forward`` + ``solver``).  It does not
import ``comparative_study/main_run.py`` or the ``optimizers`` package.
"""

import os
import sys
import time
import copy
import argparse
import statistics

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from new_fem_engine import run_forward
import new_fem_engine.solver as slv

# =============================================================================
# CONFIG -- edit these before running
# =============================================================================

CORE = "pq_40x40"
CORES = ["pq_32x30", "pq_40x40"]
"""Cores benchmarked in one run, in display order (left-to-right in the
output figure: left = pq_32x30, right = pq_40x40).  Override on the command
line with ``--cores``."""
MESH_SIZES = [0.001, 0.0008, 0.0006, 0.0004, 0.0002]
REPEATS = 3
TARGET_L_NH = 260.0
"""Mesh sweep (m), steady-state repeats, and target inductance (nH).  The
sweep runs 1.0 mm down to 0.2 mm.  If the core spec carries a ``target_L``
(``pq_40x40`` does: 2.6e-7 H = 260 nH) that value is used instead of this
constant.  Per mesh the simulated L is checked against the analytical
reluctance-model L and the target, so solver speed is never reported without a
correctness check."""

BACKENDS = [
    {"name": "spsolve", "label": "scipy spsolve", "overrides": {}},
    {"name": "lineax", "label": "lineax cg+jacobi",
     "overrides": {"lineax": {"solver": "cg", "rtol": 1e-10, "atol": 1e-10,
                              "max_steps": 10000, "jacobi": True}}},
    {"name": "feax", "label": "feax bicgstab+jacobi",
     "overrides": {"feax": {"kind": "krylov", "solver": "bicgstab",
                            "rtol": 1e-10, "atol": 1e-10, "maxiter": 10000,
                            "jacobi": True}}},
]
"""``overrides`` are applied to a fresh ``run_forward.ExperimentConfig`` before
``build_solver_options`` maps them to the backend's solver dict."""

OUT_FILE = os.path.join(run_forward.OUTPUT_DIR, "solver_speed_comparison.png")

# =============================================================================
# Measurement
# =============================================================================


def measure_one(prob, fill, options, repeats):
    """Median steady-state times + inductance for one (mesh, backend).

    Returns ``(solve_s, assemble_s, L_nh)``.  The first call is a warmup
    (JIT/XLA compile + cached-solve priming) and is never counted.  L is the
    simulated 1-turn inductance (energy method) in nH, so every timed solve
    also verifies the physics.
    """
    prob.set_params(fill)
    slv.solver(prob, options, return_timing=True)
    solves, assembles, Ls = [], [], []
    for _ in range(repeats):
        sol, timing = slv.solver(prob, options, return_timing=True)
        solves.append(timing["linear"])
        assembles.append(timing["local_assembly"] + timing["global_matrix"])
        Ls.append(float(prob.compute_inductance(sol)) * 1e9)
    return statistics.median(solves), statistics.median(assembles), statistics.median(Ls)


def run_sweep(core, mesh_sizes, repeats):
    """Benchmark every backend at every mesh size.

    Returns ``(rows, target_nh)`` where ``rows`` is a list of dicts with
    ``mesh_size``, ``dofs``, ``label``, ``solve_s``, ``assemble_s``, ``L_nh``,
    ``L_an_nh``, sorted by (mesh_size ascending, backend order), and
    ``target_nh`` is the target inductance read from the core spec (or the
    ``TARGET_L_NH`` constant).
    """
    rows = []
    cfg = run_forward.ExperimentConfig()
    cfg.core_name = core
    spec0 = copy.deepcopy(run_forward.load_core_spec(core))
    target_nh = float(spec0.get("target_L", TARGET_L_NH * 1e-9)) * 1e9

    for ms in mesh_sizes:
        cfg.mesh_size = ms
        spec = copy.deepcopy(spec0)
        run_forward.apply_config(spec, cfg)
        prob, Nr, Nz, mesh_size = run_forward.build_problem(spec)
        fill, _ = run_forward.geometry_fill(prob, Nr, Nz, mesh_size)
        dofs = int(prob.num_total_dofs_all_vars)
        L_an_nh = float(run_forward.analytical_inductance(spec)) * 1e9

        print(f"=== mesh {ms * 1000:.3f} mm | {dofs} DOFs | "
              f"L_analytical = {L_an_nh:.1f} nH | target = {target_nh:.1f} nH ===")
        for b in BACKENDS:
            for key, value in b["overrides"].items():
                setattr(cfg, key, value)
            options = run_forward.build_solver_options(b["name"], cfg)
            solve_s, assemble_s, L_nh = measure_one(prob, fill, options, repeats)
            rows.append({"mesh_size": ms, "dofs": dofs, "label": b["label"],
                         "solve_s": solve_s, "assemble_s": assemble_s,
                         "L_nh": L_nh, "L_an_nh": L_an_nh})
            print(f"   {b['label']}"
                  f"\n    FEM matrix assembly time: {assemble_s:.3f} s"
                  f"\n    FEM solve time: {solve_s:.3f} s"
                  f"\n    L_sim = {L_nh:7.1f} nH  (L_analytical = {L_an_nh:.1f} nH, "
                  f"target = {target_nh:.1f} nH)")
        print()
    return rows, target_nh

# =============================================================================
# Output
# =============================================================================


def print_table(rows):
    """Console table of solve + assembly time per (mesh, backend)."""
    print("=== Solve-time comparison (median over steady-state solves) ===")
    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
    col = 20
    print("* FEM solve time [s], median over steady-state solves\n")
    header = (f"{'mesh mm':>9} {'dofs':>9}"
              + "".join(f" {l[:col]:>{col}}" for l in labels))
    print(header)
    for ms in sorted({r["mesh_size"] for r in rows}):
        line = [f"{ms * 1000:>9.3f}"]
        sub = [r for r in rows if r["mesh_size"] == ms]
        line.append(f"{sub[0]['dofs']:>9}")
        for l in labels:
            for r in sub:
                if r["label"] == l:
                    line.append(f" {r['solve_s']:>{col - 3}.4f} s")
        print("".join(line))
    print()


def print_inductance(rows):
    """Console table of L per mesh: simulated per backend, analytical, target.

    The simulated L should be identical across backends (same physics, same
    tolerance); a large spread flags a solver accuracy problem, while the
    error against ``L_an`` and the target flags mesh-convergence issues.
    """
    print("=== Inductance check (L in nH, 1-turn base) ===")
    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
    col = 13
    header = (f"{'mesh mm':>9} {'dofs':>8}"
              + "".join(f" {l[:col]:>{col}}" for l in labels)
              + f" {'L_an':>{col}} {'target':>{col}} {'vs_tgt%':>9}")
    print(header)
    for ms in sorted({r["mesh_size"] for r in rows}):
        sub = [r for r in rows if r["mesh_size"] == ms]
        line = [f"{ms * 1000:>9.3f}", f"{sub[0]['dofs']:>8}"]
        for l in labels:
            for r in sub:
                if r["label"] == l:
                    line.append(f" {r['L_nh']:>{col - 1}.1f}")
        L_an = sub[0]["L_an_nh"]
        target = sub[0].get("target_nh", TARGET_L_NH)
        L_sim = statistics.median(r["L_nh"] for r in sub)
        line.append(f" {L_an:>{col - 1}.1f}")
        line.append(f" {target:>{col - 1}.1f}")
        line.append(f" {100 * (L_sim - target) / target:>8.2f}")
        print("".join(line))
    worst = 0.0
    for ms in {r["mesh_size"] for r in rows}:
        vals = [r["L_nh"] for r in rows if r["mesh_size"] == ms]
        worst = max(worst, max(vals) - min(vals))
    print(f"   max |dL| across backends (same mesh): {worst:.3f} nH")
    print()


def plot_results(results, out_file):
    """Solver speed vs mesh, one subplot per core (left-to-right as given)."""
    cores = list(results.keys())
    fig, axes = plt.subplots(1, len(cores), figsize=(9 * len(cores), 6),
                             squeeze=False)
    for ax, core in zip(axes[0], cores):
        data = {}
        for r in results[core]:
            d = data.setdefault(r["label"], {"dofs": [], "solve": [],
                                             "ms_mm": []})
            d["dofs"].append(r["dofs"])
            d["solve"].append(r["solve_s"])
            d["ms_mm"].append(r["mesh_size"] * 1000)

        for label, d in data.items():
            order = np.argsort(d["ms_mm"])
            ax.plot(np.asarray(d["ms_mm"])[order], np.asarray(d["solve"])[order],
                    marker="o", linewidth=2, label=label)
            for i in order:
                ax.annotate(f"{d['dofs'][i]:,}", (d["ms_mm"][i], d["solve"][i]),
                            textcoords="offset points", xytext=(0, 6),
                            fontsize=8)

        ax.set_yscale("log")
        ax.set_xlabel("mesh size [mm] (fine -> coarse)")
        ax.set_ylabel("FEM solve time [s] (log scale)")
        ax.set_title(core)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.suptitle("Solver speed vs mesh")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=150)
    print(f"Plot saved: {out_file}")


def main():
    all_start = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", nargs="+", default=CORES,
                        help="core names to benchmark (default: all of CORES)")
    parser.add_argument("--meshes", type=float, nargs="+",
                        default=MESH_SIZES, help="mesh sizes in meters")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--out", default=OUT_FILE)
    args = parser.parse_args()

    print(f"=== Solver speed comparison | cores: {', '.join(args.cores)} ===")
    print(f"mesh sizes: {', '.join(f'{m * 1000:.3f} mm' for m in args.meshes)}")
    print(f"repeats per (mesh, backend): {args.repeats}")
    print(f"backends: {', '.join(b['label'] for b in BACKENDS)}")
    print("first solve per (mesh, backend) is a warmup and is discarded\n")

    results = {}
    for core in args.cores:
        print(f"########## Core: {core} ##########")
        rows, target_nh = run_sweep(core, args.meshes, args.repeats)
        for r in rows:
            r["target_nh"] = target_nh
        results[core] = rows
        print_table(rows)
        print_inductance(rows)

    plot_results(results, args.out)

    print("=== Timing ===")
    print(f"Total execution time: {time.time() - all_start:.2f} s")
    print()


if __name__ == "__main__":
    main()
