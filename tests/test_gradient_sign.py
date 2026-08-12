"""Gradient signs: the physics of dL/dfill and dL/d(geometry).

Signs, not magnitudes — robust to mesh.  Geometry gradients go through the
full chain param -> fill -> implicit solve -> L (the optimizer's path).
mur and the gap enter differently (mur via the material law at solve time,
gap via the fill), so mur is checked by monotonicity instead.
"""

import numpy as np
import jax
import jax.numpy as jnp

import helpers


def test_core_cells_positive_air_zero():
    prob, fill, f_core = helpers.build_system()
    fwd = helpers.make_fwd(prob)

    def L(f):
        return prob.compute_inductance(fwd(f))

    grad = np.asarray(jax.grad(L)(fill))
    fc = f_core.T.reshape(-1)  # r-major, matching cell order

    core_cells = np.flatnonzero(fc > 0.5)
    air_cells = np.flatnonzero(fc < 0.05)
    assert core_cells.size > 0 and air_cells.size > 0
    assert grad[core_cells, :].min() > 0, "some core cells have dL/df < 0"
    # far air cells carry essentially no gradient
    assert np.all(np.abs(grad[air_cells, :]) < 1e-9), "air cells not ~zero"


def test_geometry_param_gradient_signs():
    _, _, L = helpers.make_param_problem()
    geom = {
        "center_post_diameter": 0.0149,
        "leg_inner_diameter": 0.037,
        "leg_height": 0.0398,
        "window_height": 0.0295,
        "gap_size": 0.0003,
    }
    g = jax.grad(L)(geom)
    assert float(g["center_post_diameter"]) > 0, "wider post must raise L"
    assert float(g["leg_inner_diameter"]) < 0, "thinner legs must lower L"
    assert float(g["leg_height"]) > 0, "taller core must raise L"
    assert float(g["window_height"]) < 0, "taller window must lower L"
    assert float(g["gap_size"]) < 0, "more gap must lower L"
