"""End-to-end differentiability: the optimizer smoke test.

``loss = (L(geom) - L_target)^2`` through the full chain
geom -> fill -> implicit solve -> L.  jax.grad must give a finite, non-zero
gradient and one small step along it must reduce the loss (local descent,
i.e. the plumbing the real optimizer relies on).
"""

import numpy as np
import jax
import pytest

import helpers

TARGET = 260e-9


@pytest.mark.slow
def test_loss_gradient_finite_and_nonzero():
    _, _, L = helpers.make_param_problem()
    geom = {
        "center_post_diameter": 0.0149,
        "leg_inner_diameter": 0.037,
        "leg_height": 0.0398,
        "window_height": 0.0295,
        "gap_size": 0.0009,
    }
    loss = lambda g: (L(g) - TARGET) ** 2
    g = jax.grad(loss)(geom)
    for v in g.values():
        assert np.isfinite(float(v)), "non-finite loss gradient"
    assert any(abs(float(v)) > 0 for v in g.values()), "zero loss gradient"


@pytest.mark.slow
def test_one_gradient_step_decreases_loss():
    _, _, L = helpers.make_param_problem()
    geom = {
        "center_post_diameter": 0.0149,
        "leg_inner_diameter": 0.037,
        "leg_height": 0.0398,
        "window_height": 0.0295,
        "gap_size": 0.0009,
    }
    loss = lambda g: (L(g) - TARGET) ** 2
    loss0 = float(loss(geom))
    g = jax.grad(loss)(geom)

    geom1 = {k: v - np.sign(float(g[k])) * 1e-4 * abs(v) for k, v in geom.items()}
    loss1 = float(loss(geom1))
    assert loss1 < loss0, f"gradient step raised loss {loss0:.3e} -> {loss1:.3e}"
