"""Optimizer objective: the certified differentiable loss + optimizer adapters.

Pins the contract that the 5 optimizer adapters in ``optimizers/`` rely on:

- ``make_eval(x) -> (loss, L, V)`` returns all three from ONE forward solve;
- a concrete ``eval(x)`` equals the plain ``solve_forward`` inductance
  (coils fresh — regression for the coil-state baking bug);
- the ``jax.grad`` of the loss (coils frozen at the prepared design, only the
  fill differentiated) matches central differences of the same frozen
  objective to ~1e-4;
- the optimizer registry exposes all five ``run_*`` adapters.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import helpers
from optimizers import objective


CORE = "pq_32x30"
MESH = 0.002
TARGET = 260e-9


def _spec():
    return objective.load_core_spec(CORE)


def _dealigned_x():
    """x0 nudged so no rect edge sits on a fine-grid cell boundary (mesh 0.002
    cell boundaries are multiples of 0.002; the kink would otherwise break the
    FD comparison — see the leg_height edge-alignment note in the objective)."""
    x0 = objective.default_x0(_spec())
    return x0 + np.array([0.00013, 0.00023, 0.00031, 0.00019, 0.00011])


def test_eval_matches_solve_forward():
    spec = _spec()
    ev = objective.make_eval(spec, mesh_size=MESH)
    for x in (objective.default_x0(spec), _dealigned_x()):
        ev.prepare(x)
        _, L, _ = ev(x)
        L_ref, _ = objective.solve_forward(objective.make_params_dict(x, spec), MESH)
        assert helpers.rel_err(L, L_ref) < 1e-6, (L, L_ref)


def test_eval_returns_loss_L_volume_in_one_solve():
    spec = _spec()
    ev = objective.make_eval(spec, mesh_size=MESH)
    x = _dealigned_x()
    ev.prepare(x)
    loss, L, V = ev(x)
    assert helpers.rel_err(loss, ((L - TARGET) * 1e9) ** 2) < 1e-6
    assert V > 0.0 and L > 0.0


def test_ad_grad_shape_and_finiteness():
    loss_fn = objective.make_loss_fn(_spec(), mesh_size=MESH)
    x = _dealigned_x()
    loss_fn.prepare(x)
    _, g = jax.value_and_grad(loss_fn)(jnp.asarray(x))
    g = np.asarray(g)
    assert g.shape == (5,)
    assert np.all(np.isfinite(g))


def test_ad_matches_frozen_fd():
    """Central differences of the frozen-coil objective == AD gradient.

    ``ev.frozen`` evaluates the fill path with coils fixed at the prepared
    design — the same function jax.grad differentiates.  A de-aligned design
    keeps every rect edge off a cell boundary so the fill is C1 there.
    """
    spec = _spec()
    loss_fn = objective.make_loss_fn(spec, mesh_size=MESH)
    ev = objective.make_eval(spec, mesh_size=MESH)
    x = _dealigned_x()
    loss_fn.prepare(x)
    ev.prepare(x)
    _, g = jax.value_and_grad(loss_fn)(jnp.asarray(x))
    g = np.asarray(g)
    h = 1e-5
    for idx in range(5):
        xp, xm = x.copy(), x.copy()
        xp[idx] += h
        xm[idx] -= h
        lp = ev.frozen(jnp.asarray(xp))[0]
        lm = ev.frozen(jnp.asarray(xm))[0]
        fd = (lp - lm) / (2 * h)
        denom = max(abs(fd), abs(g[idx]), 1e-30)
        assert abs(fd - g[idx]) / denom < 5e-3, (idx, fd, g[idx])


def test_optimizer_registry():
    from optimizers import OPTIMIZERS
    assert set(OPTIMIZERS) == {"lbfgs", "adam", "bayesian", "cmaes", "nsga3"}


@pytest.mark.slow
def test_lbfgs_converges_to_target():
    """End-to-end convergence: L-BFGS-B drives the loss to ~0.

    Uses the PQ 40x40 core at the coarse 0.002 mesh.  This validates the full
    coil/fill pipeline end-to-end (the coil-state baking bug previously made
    the optimizer converge to a stale L).  jaxopt's native-JAX L-BFGS-B is
    used (Level A: jitted step + Python loop).
    """
    from optimizers import run_lbfgs
    spec = objective.load_core_spec("pq_40x40")
    r = run_lbfgs(spec, max_iters=40, mesh_size=0.002, verbose=False)
    assert r["success"]
    assert helpers.rel_err(r["L_opt"], TARGET) < 1e-3


@pytest.mark.slow
def test_lbfgs_converges_at_production_mesh():
    """Pure-JAX L-BFGS-B converges at the production 0.001 mesh.

    The old scipy-Fortran L-BFGS-B line search stalled at the production mesh
    (ABNORMAL_TERMINATION_IN_LNSRCH, ~261 nH — the paper's documented
    'struggles during discretization').  jaxopt's native JAX L-BFGS-B (zoom
    line search in JAX) removes that stall: it reaches the 260 nH target to
    <0.1% at the same mesh.  This pins the new behaviour.
    """
    from optimizers import run_lbfgs
    spec = objective.load_core_spec("pq_40x40")
    r = run_lbfgs(spec, max_iters=40, mesh_size=0.001, verbose=False)
    assert r["success"]
    assert helpers.rel_err(r["L_opt"], TARGET) < 1e-3
