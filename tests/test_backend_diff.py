"""Differentiating the forward solve THROUGH lineax/feax backends.

``ad_wrapper`` escapes tracing (custom_vjp), so the forward Newton solve and
the adjoint solve can each run on any linear-solver backend.  This pins that
combination: for every (forward, adjoint) backend pair, ``jax.grad(L)(geom)``
must match central finite differences of the same forward L to the same bar
as the spsolve reference (test_backward).

Pairs deliberately mix lineax/feax/spsolve on both the forward and the
adjoint side (the gap the old suite left open).
"""

import numpy as np
import jax
import feax
import pytest

import helpers

FEAX_DIRECT = {"feax_solver": {"options": feax.DirectSolverOptions(solver="auto")}}

# (label, forward solver options, adjoint solver options)
PAIRS = [
    ("spsolve/spsolve", helpers.SP, helpers.ADJ),
    ("lineax-cg/lineax-cg",
     {"newton": {"linear": {"lineax_solver": {"solver": "cg"}}}},
     {"lineax_solver": {"solver": "cg"}}),
    ("lineax-bicgstab/lineax-bicgstab",
     {"newton": {"linear": {"lineax_solver": {"solver": "bicgstab"}}}},
     {"lineax_solver": {"solver": "bicgstab"}}),
    ("feax-direct/feax-direct",
     {"newton": {"linear": dict(FEAX_DIRECT)}}, dict(FEAX_DIRECT)),
    ("spsolve/feax-direct", helpers.SP, dict(FEAX_DIRECT)),
    ("lineax-cg/feax-direct",
     {"newton": {"linear": {"lineax_solver": {"solver": "cg"}}}}, dict(FEAX_DIRECT)),
]

GEOM = {"center_post_diameter": 0.0149, "gap_size": 0.0005}
EPS = 1e-6


@pytest.mark.parametrize("name,fwd,adj", PAIRS, ids=[p[0] for p in PAIRS])
def test_grad_through_forward_backend_matches_fd(name, fwd, adj):
    _, _, L = helpers.make_param_problem(mesh_size=0.002, fwd_opts=fwd, adj_opts=adj)

    grad = {k: float(v) for k, v in jax.grad(L)(GEOM).items()}
    assert all(np.isfinite(v) for v in grad.values())

    for key in GEOM:
        e = {k: 0.0 for k in GEOM}
        e[key] = EPS
        fp = float(L({k: v + e[k] for k, v in GEOM.items()}))
        fm = float(L({k: v - e[k] for k, v in GEOM.items()}))
        fd = (fp - fm) / (2 * EPS)
        rel = abs(fd - grad[key]) / max(abs(fd), 1e-12)
        assert rel < 1e-3, (
            f"[{name}] param {key}: grad {grad[key]:.6e} vs FD {fd:.6e} "
            f"(rel err {rel:.3e})")


def test_backends_give_consistent_gradients():
    """All backend pairs must land on the same gradient (solver tol ~1e-8)."""
    _, _, ref = helpers.make_param_problem(mesh_size=0.002)
    g_ref = {k: float(v) for k, v in jax.grad(ref)(GEOM).items()}

    for name, fwd, adj in PAIRS[1:]:
        _, _, L = helpers.make_param_problem(mesh_size=0.002, fwd_opts=fwd, adj_opts=adj)
        g = {k: float(v) for k, v in jax.grad(L)(GEOM).items()}
        for key in GEOM:
            rel = abs(g[key] - g_ref[key]) / max(abs(g_ref[key]), 1e-12)
            assert rel < 1e-3, (
                f"[{name}] param {key}: {g[key]:.6e} vs spsolve {g_ref[key]:.6e} "
                f"(rel err {rel:.3e})")
