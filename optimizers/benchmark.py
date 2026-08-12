"""Fair cross-optimizer benchmark on the ``new_fem_engine`` objective.

Every optimizer runs on the same core and mesh under one common ``budget``
measured in FEM solve-equivalents (one forward solve = 1 unit; an Adam /
L-BFGS iteration costs 2 because the implicit-adjoint gradient solve is a
second forward-equivalent).  The common budget converts to a per-optimizer
iteration count, so a comparison is about *quality per solve*, not per
iteration.

Each optimised design is then cross-checked three ways:

- ``L_fem``          -- forward FEM inductance at ``x_opt``;
- ``L_analytical``   -- reluctance-model inductance (McLyman fringing) at the
                        SAME ``x_opt`` params;
- ``target``         -- the datasheet-derived design requirement.

A result is trusted only when FEM and analytical agree (within the ~10 % model
tolerance) AND the optimizers agree with each other (small cross-optimizer
spread of ``L_fem``).  Cross-backend agreement (~1e-8) is pinned separately by
``tests/test_optimizer_backends.py``.

Entry point (console table + ``benchmark.png`` under ``output/benchmark/<core>/``)::

    python optimizers/benchmark.py [core] [mesh_size] [budget]

The five families are benchmarked; ``bayesian`` defaults to the TPE member
(``bayesian:tpe_optuna``); ``bayesian:gp_skopt`` selects the GP member.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from optimizers import objective
from optimizers.adam import build_vg
from new_fem_engine import run_forward
from new_fem_engine.report import OUTPUT_DIR

# -- one representative per family --------------------------------------------
# name:member syntax: "bayesian" -> TPE member, "bayesian:gp_skopt" -> GP member.
DEFAULT_OPTIMIZERS = ["lbfgs", "adam", "bayesian", "cmaes", "nsga3"]

# Seed statistics measure optimizer stability (not physics accuracy), so they
# are meaningful only for the stochastic families; L-BFGS-B and Adam are
# deterministic and would waste the budget re-running.
DETERMINISTIC = {"lbfgs", "adam"}

# per-member: (run callable, display label)
def _runner(name):
    family, _, member = name.partition(":")
    if family == "bayesian":
        member = member or "tpe_optuna"
        from optimizers import bayesian
        labels = {"tpe_optuna": "Bayesian (TPE)", "gp_skopt": "Bayesian (GP)"}
        return (lambda spec, **kw: bayesian.run_bayesian(spec, name=member, **kw),
                labels[member])
    if family == "lbfgs":
        from optimizers import lbfgs
        return (lbfgs.run_lbfgs, "L-BFGS-B")
    if family == "adam":
        from optimizers import adam
        return (lambda spec, **kw: adam.run_optax(spec, "adam", **kw), "Adam")
    if family == "cmaes":
        from optimizers import cmaes
        return (cmaes.run_cmaes, "CMA-ES")
    if family == "nsga3":
        from optimizers import nsga3
        return (nsga3.run_nsga3, "NSGA3")
    raise ValueError(f"Unknown benchmark optimizer {name!r}; "
                     f"choices: {DEFAULT_OPTIMIZERS} + bayesian:gp_skopt")


def budget_iters(name, budget, pop_size=40, n_init=5):
    """Convert a FEM solve-equivalent budget into per-optimizer iterations."""
    family, _, member = name.partition(":")
    if family == "adam":
        return max(budget // 2, 1)          # 1 fwd + 1 adjoint per iteration
    if family == "lbfgs":
        return max(budget // 2, 1)
    if family == "bayesian":
        return max(budget - n_init, 1)      # n_init initial + guided evals
    if family == "cmaes":
        return max(budget // 9, 1)          # population 9 at 5 free dims
    if family == "nsga3":
        return max(budget // pop_size, 1)   # population 40
    raise ValueError(name)


def solves_of(result):
    """Actual FEM solve-equivalents spent by the optimizer."""
    o = result["optimizer"].lower()
    if o == "lbfgs":
        return int(result.get("nfev", 0)) + int(result.get("njev", 0))
    if o == "adam":
        return 2 * int(result["nit"])
    return int(result["nit"])               # bayesian / cmaes / nsga3: 1 per eval


def solves_axis(result):
    """Cumulative solve-equivalents at each history entry (plot x-axis)."""
    hist = result["history"]
    nit = max(len(hist), 1)
    o = result["optimizer"].lower()
    if o == "adam":
        return [2.0 * (i + 1) for i in range(len(hist))]
    if o == "lbfgs":
        n_solves = solves_of(result)
        return [(i + 1) * n_solves / nit for i in range(len(hist))]
    return [float(i + 1) for i in range(len(hist))]


def analytical_L_at(result):
    """Reluctance-model inductance (Henry) at the optimised params."""
    return run_forward.analytical_inductance({"params": result["params"]})


def run_benchmark(core_spec, mesh_size=None, budget=360, optimizers=None,
                  backend="lineax", n_init=5, pop_size_nsga3=40, verbose=True,
                  tol=0.25, patience=40, improve_tol=1e-3):
    """Run the requested optimizers under one common solve-equivalent budget.

    Returns a list of result dicts (each augmented with ``label``, ``n_solves``
    and ``L_analytical``) and prints the validation table.

    The gradient families (Adam, L-BFGS-B) share ONE compiled traceable
    ``value_and_grad`` (:func:`build_vg`) so the ~10 s XLA compile and problem
    build happen a single time, not once per optimizer.  Adam additionally uses
    early termination (``tol``/``patience``, see ``adam._adam_driver``) — loss
    ``<= tol`` (i.e. |L-target| < 0.5 nH here) stops the run, cutting the
    post-convergence tail iters.  L-BFGS-B has its own optimality criterion
    (projected-gradient ``tol=1e-8``) inside the solver.
    """
    mesh_size = mesh_size or core_spec["mesh_size"]
    optimizers = optimizers or list(DEFAULT_OPTIMIZERS)
    target_nH = core_spec["target_L"] * 1e9

    # One shared compiled objective for both gradient families.
    if any(n.partition(":")[0] in ("adam", "lbfgs") for n in optimizers):
        t_shared = time.time()
        loss_s, vg_s = build_vg(core_spec, mesh_size=mesh_size, backend=backend)
        print(f"  [shared gradient objective] #{1000 * (time.time() - t_shared):.0f} "
              f"ms one-time XLA compile + problem build")
    else:
        loss_s = vg_s = None

    results = []
    for name in optimizers:
        run_fn, label = _runner(name)
        family = name.partition(":")[0]
        iters = budget_iters(name, budget, pop_size=pop_size_nsga3, n_init=n_init)
        kw = dict(mesh_size=mesh_size, max_iters=iters, backend=backend,
                  verbose=verbose)
        if family == "nsga3":
            kw["pop_size"] = pop_size_nsga3
        if family == "bayesian":
            kw["n_init"] = n_init
        if family == "adam" and vg_s is not None:
            kw["vg"], kw["loss_fn"] = vg_s, loss_s
            kw["tol"], kw["patience"], kw["improve_tol"] = tol, patience, improve_tol
            kw["max_grad_norm"] = 1.0   # clip the ~1e7 raw grads (RAdam/Fromage safety)
            kw.pop("backend", None)          # run_optax reuses the shared build
        if family == "lbfgs" and vg_s is not None:
            kw["vg"], kw["loss_fn"] = vg_s, loss_s
            kw["tol_loss"] = 0.25   # |ΔL| < 0.5 nH
            kw["maxls"] = 10        # cap the hidden line-search FEM evals
        print(f"\n  --- {label} (budget {budget} solves -> {iters} iters) ---")
        t0 = time.time()
        r = run_fn(core_spec, **kw)
        r["wall_time"] = time.time() - t0
        r["label"] = label
        r["n_solves"] = solves_of(r)
        r["L_analytical"] = float(analytical_L_at(r))
        results.append(r)
        print(f"  -> L_fem = {r['L_opt'] * 1e9:.1f} nH | "
              f"L_analytical = {r['L_analytical'] * 1e9:.1f} nH | "
              f"target = {target_nH:.1f} nH | {r['n_solves']} solves | "
              f"{r['time']:.1f}s | {r['message']}")

    _print_table(results, target_nH)
    _plot(results, core_spec, mesh_size)
    return results


def run_multi_seed(core_spec, mesh_size=None, budget=360, n_seeds=5, seeds=None,
                   optimizers=None, backend="lineax", n_init=5, pop_size_nsga3=40):
    """Seed-robustness sweep for the stochastic families (CMA-ES, BO, NSGA3).

    Runs each stochastic optimizer across ``n_seeds`` seeds at the same budget
    and reports ``success_rate`` (best_loss < 1.0), ``mean +/- std`` of
    ``best_loss`` and ``L_fem``, plus the worst seed — averages alone cannot
    expose an optimizer that *sometimes* fails outright.  Deterministic
    optimizers (L-BFGS-B, Adam) are stated as deterministic and skipped.  A
    coarse mesh is the right setting: seed statistics need many samples,
    physics accuracy needs a fine mesh, never both at once.
    """
    mesh_size = mesh_size or core_spec["mesh_size"]
    optimizers = optimizers or [o for o in DEFAULT_OPTIMIZERS
                                if o.partition(":")[0] not in DETERMINISTIC]
    target_nH = core_spec["target_L"] * 1e9
    seeds = seeds or list(range(n_seeds))

    print(f"\n=== Multi-seed robustness | core {core_spec['name']} | "
          f"target {target_nH:.0f} nH | mesh {mesh_size} m | "
          f"budget {budget} solves | {len(seeds)} seeds ===")

    for name in optimizers:
        run_fn, label = _runner(name)
        iters = budget_iters(name, budget, pop_size=pop_size_nsga3, n_init=n_init)
        kw = dict(mesh_size=mesh_size, max_iters=iters, backend=backend,
                  verbose=False)
        if name.partition(":")[0] == "nsga3":
            kw["pop_size"] = pop_size_nsga3
        if name.partition(":")[0] == "bayesian":
            kw["n_init"] = n_init

        losses, Ls = [], []
        print(f"\n  --- {label} ({iters} iters x {len(seeds)} seeds) ---")
        for s in seeds:
            r = run_fn(core_spec, seed=s, **kw)
            losses.append(float(r["loss"]))
            Ls.append(float(r["L_opt"]) * 1e9)
            print(f"    seed {s}: loss = {r['loss']:.3e} | "
                  f"L_fem = {r['L_opt'] * 1e9:.2f} nH")

        losses = np.asarray(losses)
        Ls = np.asarray(Ls)
        success_rate = float(np.mean(losses < 1.0))   # objective contract
        k_worst = int(np.argmax(losses))
        print(f"  -> success_rate = {success_rate:.0%} ({int(np.sum(losses < 1.0))}"
              f"/{len(seeds)})")
        print(f"     best_loss   mean +/- std = {losses.mean():.2e} +/- "
              f"{losses.std():.2e} | worst = {losses[k_worst]:.2e}")
        print(f"     L_fem [nH]  mean +/- std = {Ls.mean():.2f} +/- "
              f"{Ls.std():.2f} | range {Ls.min():.1f}..{Ls.max():.1f}")
        if success_rate < 1.0:
            print(f"     WARNING: failed on {int(np.sum(losses >= 1.0))} seed(s) — "
                  f"optimizer is unstable at this budget")


def _print_table(results, target_nH):
    print("\n=== Benchmark table ===")
    print(f"{'optimizer':<16}{'L_fem nH':>11}{'L_an nH':>11}"
          f"{'target':>9}{'|dL| nH':>10}{'|FEM-AN|':>10}{'loss':>10}"
          f"{'solves':>8}{'time s':>9}")
    for r in results:
        dL = abs(r["L_opt"] * 1e9 - target_nH)
        ana = abs(r["L_opt"] - r["L_analytical"]) * 1e9
        print(f"{r['label']:<16}{r['L_opt'] * 1e9:>11.2f}"
              f"{r['L_analytical'] * 1e9:>11.2f}{target_nH:>9.1f}"
              f"{dL:>10.2f}{ana:>10.2f}{r['loss']:>10.1e}"
              f"{r['n_solves']:>8}{r['time']:>9.1f}")
    Ls = [r["L_opt"] * 1e9 for r in results]
    print(f"\n  cross-optimizer spread of L_fem: "
          f"{max(Ls) - min(Ls):.2f} nH (range {min(Ls):.1f}..{max(Ls):.1f})")
    for r in results:
        rel_an = abs(r["L_opt"] - r["L_analytical"]) / r["L_analytical"]
        trusted = (r["success"] and abs(r["L_opt"] * 1e9 - target_nH) < 1.0
                   and rel_an < 0.10)
        flag = "TRUSTED" if trusted else "check"
        print(f"  {r['label']:<16} rel|FEM-analytical| = {rel_an * 100:5.1f}%  "
              f"{flag}")


def _plot(results, core_spec, mesh_size):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  (skipping plot: {exc})")
        return
    out_dir = os.path.join(OUTPUT_DIR, "benchmark", core_spec["name"])
    os.makedirs(out_dir, exist_ok=True)
    target_nH = core_spec["target_L"] * 1e9

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 13))
    for r in results:
        xs = solves_axis(r)
        loss = [h["loss"] for h in r["history"]]
        ax1.plot(xs, loss, label=r["label"])
        Ls = [h.get("inductance") for h in r["history"]]
        if all(v is not None for v in Ls):
            ax2.plot(xs, np.asarray(Ls) * 1e9, label=r["label"])
    ax1.set_ylabel("loss"); ax1.set_yscale("log"); ax1.grid(alpha=0.3)
    ax1.legend(); ax1.set_title(f"{core_spec['name']} benchmark, "
                                f"mesh {mesh_size * 1000:.3f} mm, "
                                f"target {target_nH:.0f} nH")
    ax2.axhline(target_nH, color="gray", ls="--", lw=1, label="target")
    ax2.set_ylabel("L [nH]"); ax2.set_xlabel("FEM solves"); ax2.grid(alpha=0.3)
    ax2.legend()

    labels = [r["label"] for r in results]
    xpos = np.arange(len(results))
    w = 0.28
    ax3.bar(xpos - w, [r["L_opt"] * 1e9 for r in results], w, label="L_fem")
    ax3.bar(xpos, [r["L_analytical"] * 1e9 for r in results], w,
            label="L_analytical")
    ax3.bar(xpos + w, [target_nH] * len(results), w, label="target")
    ax3.axhline(target_nH, color="gray", ls="--", lw=1)
    ax3.set_xticks(xpos); ax3.set_xticklabels(labels, rotation=20, ha="right")
    ax3.set_ylabel("L [nH]"); ax3.legend(); ax3.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = os.path.join(out_dir, "benchmark.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  benchmark plots saved under {out_dir}/")


def main():
    core_name = sys.argv[1] if len(sys.argv) > 1 else "pq_40x40"
    mesh_size = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 360
    n_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    spec = objective.load_core_spec(core_name)
    if n_seeds:
        # seed-robustness sweep on the coarse mesh (stochastic families only)
        run_multi_seed(spec, mesh_size=max(mesh_size, 0.002),
                       budget=budget, n_seeds=n_seeds)
    else:
        print(f"=== Benchmark | core {core_name} | target {spec['target_L'] * 1e9:.0f} nH "
              f"| mesh {mesh_size} m | budget {budget} FEM solves ===")
        run_benchmark(spec, mesh_size=mesh_size, budget=budget)


if __name__ == "__main__":
    main()
