"""Second-order: Hessian of L w.r.t. geometry params.

Uses finite differences of the adjoint gradient (H_adj) vs finite
differences of L directly (H_fd).  The two must agree, and H_adj must be
symmetric.  (Full jax.hessian through the custom_vjp is not exercised here
because the adjoint solve mixes in a scipy/numpy linear solve.)
"""

import numpy as np
import jax
import pytest

import helpers


@pytest.mark.slow
def test_hessian_from_adjoint_gradient_matches_fd():
    _, _, L = helpers.make_param_problem()
    p0 = np.array([0.0149, 0.0005])  # center_post_diameter, gap_size

    def f(p):
        return L({"center_post_diameter": p[0], "gap_size": p[1]})

    def g(p):
        return np.asarray(jax.grad(f)(p))

    h = 1e-4
    H_adj = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        H_adj[:, j] = (g(p0 + e) - g(p0 - e)) / (2 * h)

    H_fd = np.array([
        [helpers.central_hessian_fd(f, p0, i, j, h) for j in range(2)]
        for i in range(2)
    ])

    assert np.all(np.isfinite(H_adj))
    assert np.allclose(H_adj, H_adj.T, rtol=0.05), "adjoint Hessian asymmetric"
    assert np.allclose(H_adj, H_fd, rtol=0.05, atol=1e-6), (
        f"adjoint Hessian {H_adj} != FD Hessian {H_fd}")
