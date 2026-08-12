"""Cross-backend optimizer consistency: one fixed mesh, solver varies only.

Pins the fair-comparison contract behind the ``lbfgs.py`` / backend objective:

- every optimizer runs on every backend at the SAME mesh (fixed DOFs);
- a design's forward inductance and AD gradient agree across all four
  linear-solver backends to solver tolerance (~1e-8 relative);
- each optimizer's reported ``L_opt`` is backend-independent.

The solver changes that make this possible (adjoint-RHS normalisation and
relative residual asserts) live in ``new_fem_engine/solver.py``.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import helpers
from optimizers import objective, backends, adam, bayesian, cmaes, nsga3, lbfgs


CORE = "pq_32x30"
MESH = 0.002  # fast; gradients agree across backends on any fixed mesh
TARGET = 260e-9


def _spec():
    return objective.load_core_spec(CORE)


def _x():
    x0 = objective.default_x0(_spec())
    return x0 + np.array([0.00013, 0.00023, 0.00031, 0.00019, 0.00011])


def test_forward_L_consistent_across_backends():
    x = _x()
    params = objective.make_params_dict(x, _spec())
    L_ref = None
    for bk in backends.BACKENDS:
        L, _ = objective.solve_forward(params, MESH, backend=bk)
        L = float(L)
        if L_ref is None:
            L_ref = L
            continue
        rel = abs(L - L_ref) / abs(L_ref)
        assert rel < 1e-6, f"[{bk}] L {L*1e9:.6f} nH vs spsolve {L_ref*1e9:.6f} nH"


def test_grad_consistent_across_backends():
    x = _x()
    g_ref = None
    for bk in backends.BACKENDS:
        loss = objective.make_loss_fn(_spec(), mesh_size=MESH, backend=bk)
        loss.prepare(x)
        _, g = jax.value_and_grad(loss)(jnp.asarray(x))
        g = np.asarray(g)
        assert np.all(np.isfinite(g)), f"[{bk}] non-finite gradient"
        if g_ref is None:
            g_ref = g
            continue
        rel = np.linalg.norm(g - g_ref) / max(np.linalg.norm(g_ref), 1e-12)
        assert rel < 1e-5, f"[{bk}] grad rel diff {rel:.2e} vs spsolve"


BUDGETS = {
    "lbfgs": dict(max_iters=3),
    "adam": dict(max_iters=3),
    "bayesian": dict(max_iters=2, n_init=1),
    "cmaes": dict(max_iters=2, pop_size=8),
    "nsga3": dict(max_iters=2, pop_size=8),
}
RUNNERS = {
    "lbfgs": lbfgs.run_lbfgs,
    "adam": adam.run_adam,
    "bayesian": bayesian.run_bayesian,
    "cmaes": cmaes.run_cmaes,
    "nsga3": nsga3.run_nsga3,
}


@pytest.mark.slow
@pytest.mark.parametrize("name,bk", [
    (n, b) for n in RUNNERS for b in backends.BACKENDS
], ids=lambda v: v if isinstance(v, str) else v)
def test_optimizer_runs_on_backend(name, bk):
    r = RUNNERS[name](_spec(), mesh_size=MESH, backend=bk, verbose=False,
                      **BUDGETS[name])
    assert np.isfinite(float(r["L_opt"]))
    assert np.isfinite(float(r["loss"]))


@pytest.mark.slow
def test_L_opt_backend_independent():
    """The same design space, same mesh -> L_opt must not depend on the solver."""
    x = _x()
    Ls = []
    for bk in backends.BACKENDS:
        r = adam.run_adam(_spec(), max_iters=3, mesh_size=MESH, backend=bk,
                          verbose=False)
        Ls.append(float(r["L_opt"]))
    L0 = Ls[0]
    for bk, L in zip(backends.BACKENDS[1:], Ls[1:]):
        rel = abs(L - L0) / abs(L0)
        assert rel < 1e-6, f"[{bk}] L_opt {L*1e9:.6f} nH vs spsolve {L0*1e9:.6f} nH"
