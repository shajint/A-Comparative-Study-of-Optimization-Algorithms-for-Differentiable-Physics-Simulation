"""Shared generic-driver contract for the optimizer adapters.

Layering
--------
Each adapter is split into two pieces:

1. A *generic driver* ``_<family>_driver(fn, x0, lo, hi, ...)`` — the real
   optimization loop, written against raw flat parameter vectors only.  It
   never imports the FEM objective and never sees ``GEOM_KEYS``: the objective
   ``fn`` is any callable on a raw ``x`` vector, and the returned history
   carries raw ``x`` vectors.  This is the testable core that the known-answer
   (toy) tests in ``tests/test_optimizer_correctness.py`` drive in
   milliseconds — the fast CI safety net for the optimization machinery.

2. A thin FEM wrapper ``run_*(core_spec, ...)`` — builds the compiled FEM
   objective (``build_vg`` / ``make_eval``), calls the driver, *enriches* the
   driver history with ``GEOM_KEYS``-keyed ``params`` (and inductance), and
   assembles the FEM result dict consumed by ``benchmark.py`` /
   ``report_design`` / ``run_forward``.  The wrapper's return dict is
   regression-pinned by the existing test suite.

Driver output (uniform across all families)::

    {
        "x_best":      np.ndarray,   # argmin over the recorded history
        "best_loss":   float,
        "history":     [{"iteration": int, "loss": float, "x": [float, ...]}, ...],
        "nit":         int,          # len(history)
        "diagnostics": dict,         # per-family convergence signals
    }

Objective contracts
-------------------
- gradient-based (L-BFGS-B, Adam): ``fn(x) -> (loss, grad)``
- gradient-free (BO, CMA-ES, NSGA3, later): ``fn(x) -> scalar loss``
  (the wrapper supplies a scalar extractor; ``x`` is still a raw vector).
"""

import os

import numpy as np

from optimizers import objective


def enrich_history(driver_history):
    """Attach ``VARY_GEOM_KEYS``-keyed ``params`` to each driver history entry.

    The single place the FEM parameter keys enter the driver output.  The x
    vector runs over the free design keys (:data:`objective.VARY_GEOM_KEYS`);
    frozen keys (``coil_clearance``) are pinned by the FEM layer and never
    appear in the history x.  Returns a new list (the driver's generic history
    is never mutated), so the toy tests and the FEM consumers share one code
    path safely.
    """
    return [
        {**entry, "params": {k: float(entry["x"][j])
                             for j, k in enumerate(objective.VARY_GEOM_KEYS)}}
        for entry in driver_history
    ]


def track_best(history):
    """``(best_loss, x_best)`` = argmin over the recorded history entries."""
    best = min(history, key=lambda e: e["loss"])
    return float(best["loss"]), np.asarray(best["x"])


def save_convergence_plot(history, out_path, title="", target_nH=None,
                          standard=False):
    """Loss-vs-iteration convergence plot (log-y), PNG saved to ``out_path``.

    Every driver history carries ``iteration`` + ``loss``, so this works for
    all five families.  CMA-ES / NSGA3 entries also carry ``inductance``; when
    present, a second panel plots L-vs-iteration against ``target_nH`` (the
    datasheet target, in nH).  matplotlib is optional — silently no-ops if it
    is unavailable, so the toy tests and headless CI never depend on it.

    The default renders the standard single loss-vs-iteration line.

    ``standard=True`` (Adam family only) plots the best-so-far running minimum
    instead: Adam's raw loss sawtooths around the optimum on a log axis, so
    the monotone best-so-far line is its clean convergence signal.  Other
    families use the default single-line rendering.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    import numpy as np

    it = [h["iteration"] for h in history]
    loss = [h["loss"] for h in history]
    has_L = all("inductance" in h for h in history)
    n = 2 if has_L else 1
    fig, axes = plt.subplots(n, 1, figsize=(8, 4.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    if standard:
        best = np.minimum.accumulate(np.asarray(loss))
        axes[0].plot(it, best, ls="-", lw=1.5, color="C0")
    else:
        axes[0].plot(it, loss, marker=".", ls="-", ms=3, color="C0")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(title + ("  (loss)" if has_L else ""))

    if has_L:
        axes[1].plot(it, [h["inductance"] * 1e9 for h in history],
                     marker=".", ls="-", ms=3)
        if target_nH:
            axes[1].axhline(target_nH, color="gray", ls="--", lw=1,
                            label=f"target {target_nH:.0f} nH")
            axes[1].legend()
        axes[1].set_ylabel("L [nH]")
        axes[1].grid(alpha=0.3)

    axes[-1].set_xlabel("iteration")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
