"""Backward-path verification for ``new_fem_engine``.

The engine's reason to exist is differentiability: ``dL/d(fill)`` through the
full implicit solve.  These tests check that the adjoint (custom_vjp) gradient
of the physically meaningful scalar ``L = compute_inductance(sol)`` matches a
central finite difference, on a realistic PQ40 core/air fill field.

Covers the checklist item "Gradient check (adjoint/VJP vs finite difference)".
"""

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from new_fem_engine.generate_mesh import rectangle_mesh, create_mesh
from new_fem_engine.problem import MagnetostaticProblem
import new_fem_engine.solver as slv
from new_fem_engine.geometry import compute_fill_fractions, define_core_rectangles
from new_fem_engine.run_forward import geometry_fill


CORE_SPEC = {
    "params": {
        "center_post_diameter": 0.0149,
        "leg_inner_diameter": 0.037,
        "leg_height": 0.0398,
        "window_height": 0.0295,
        "coil_clearance": 0.0005,
        "gap_size": 0.0003,
        "gap_number": 1,
        "mur": 1680.0,
        "A_e": 189e-6,
        "l_e": 93e-3,
        "sigma_l_over_A": 492.0,
        "AL": 4300.0,
        "material": "N87",
        "yoke_taper": True,
    },
    "mesh_size": 0.002,
}


def make_problem():
    mesh_size = CORE_SPEC["mesh_size"]
    Nr = int(round(0.035 / mesh_size))
    Nz = int(round(0.06 / mesh_size))
    msh = rectangle_mesh(Nr, Nz, 0.0, 0.035, -0.03, 0.03)
    mesh = create_mesh(msh, "quad")
    locs = [
        lambda pt: jnp.isclose(pt[0], 0.035),
        lambda pt: jnp.isclose(pt[0], 0.0),
        lambda pt: jnp.isclose(pt[1], 0.03),
        lambda pt: jnp.isclose(pt[1], -0.03),
    ]
    prob = MagnetostaticProblem(
        mesh=mesh, vec=1, dim=2,
        dirichlet_bc_info=[locs, [0, 0, 0, 0], [lambda pt: 0.0] * 4],
        additional_info=(CORE_SPEC, mesh_size),
    )
    fill, (f_core, _, _) = geometry_fill(prob, Nr, Nz, mesh_size)
    prob.set_params(fill)
    return prob, fill, f_core


@pytest.fixture(scope="module")
def setup():
    return make_problem()


def test_L_gradient_wrt_fill_matches_finite_difference(setup):
    """dL/d(fill) from the adjoint must equal central finite differences.

    L is the physically meaningful inductance ``compute_inductance``.  We
    perturb individual cells (core, boundary, air) and compare against the
    adjoint gradient returned by ``jax.grad`` through the custom_vjp solve.
    """
    prob, fill, f_core = setup
    fwd = slv.ad_wrapper(
        prob,
        {"newton": {"linear": {"spsolve_solver": {}}}},
        {"spsolve_solver": {}},
    )

    def L(f):
        return prob.compute_inductance(fwd(f))

    grad = np.asarray(jax.grad(L)(fill))
    assert np.all(np.isfinite(grad)), "adjoint gradient contains NaN/Inf"

    # pick representative cells: a core cell, a core/air boundary cell, an
    # air cell, and two arbitrary interior cells (f_core is (Nz, Nr), so
    # transpose to the r-major cell index).
    f_core_rmaj = np.asarray(f_core).T.reshape(-1)
    num_cells = fill.shape[0]
    candidates = []
    for i in range(num_cells):
        if f_core_rmaj[i] > 0.5:
            candidates.append(i)
            break
    for i in range(num_cells):
        if 0.05 < f_core_rmaj[i] < 0.95:
            candidates.append(i)
            break
    for i in range(num_cells):
        if f_core_rmaj[i] < 0.05:
            candidates.append(i)
            break
    candidates += [num_cells // 3, 2 * num_cells // 3]
    cells = [(c, 0) for c in dict.fromkeys(candidates)]

    eps = 1e-5
    core_cell = candidates[0]  # first core cell found above
    for idx in cells:
        e = np.zeros_like(np.asarray(fill))
        e[idx] = eps
        fp = float(L(jnp.asarray(np.asarray(fill) + e)))
        fm = float(L(jnp.asarray(np.asarray(fill) - e)))
        fd = (fp - fm) / (2 * eps)
        rel = abs(fd - grad[idx]) / max(abs(fd), 1e-12)
        assert rel < 1e-3, (
            f"cell {idx}: adjoint {grad[idx]:.6e} vs FD {fd:.6e} "
            f"(rel err {rel:.3e})")

    # physical sanity: increasing fill of a core cell raises L
    assert grad[core_cell, 0] > 0, "core-cell gradient should be positive"


def test_L_gradient_wrt_geometry_params_matches_finite_difference(setup):
    """dL/d(geometry params) through the full chain must match finite diff.

    Chain: geometry param -> define_core_rectangles -> compute_fill_fractions
    -> fill -> implicit solve -> L.  This is the differentiation path the
    topology/shape optimizer relies on.
    """
    prob, fill, f_core = setup
    fwd = slv.ad_wrapper(
        prob,
        {"newton": {"linear": {"spsolve_solver": {}}}},
        {"spsolve_solver": {}},
    )

    Nr, Nz = f_core.shape[1], f_core.shape[0]
    ms = CORE_SPEC["mesh_size"]
    xs = 0.0 + (np.arange(Nr) + 0.5) * ms
    ys = -0.03 + (np.arange(Nz) + 0.5) * ms

    def fill_from_params(geom):
        p = dict(CORE_SPEC["params"])
        p.update(geom)
        rects, mask, prim, sec = define_core_rectangles(p)
        _, _, _, f_core = compute_fill_fractions(
            jnp.asarray(xs), jnp.asarray(ys), rects, mask, prim, sec, p["mur"])
        return jnp.repeat(f_core.T.reshape(-1)[:, None], 4, axis=1)

    def L_geom(geom):
        return prob.compute_inductance(fwd(fill_from_params(geom)))

    geom = {
        "center_post_diameter": CORE_SPEC["params"]["center_post_diameter"],
        "leg_inner_diameter": CORE_SPEC["params"]["leg_inner_diameter"],
        "leg_height": CORE_SPEC["params"]["leg_height"],
        "window_height": CORE_SPEC["params"]["window_height"],
    }
    grad = jax.grad(L_geom)(geom)
    assert np.all(np.isfinite(jnp.stack(list(grad.values()))))
    grad = {k: float(v) for k, v in grad.items()}

    eps = 1e-6
    for name in geom:
        e = {k: 0.0 for k in geom}
        e[name] = eps
        fp = float(L_geom({k: v + e[k] for k, v in geom.items()}))
        fm = float(L_geom({k: v - e[k] for k, v in geom.items()}))
        fd = (fp - fm) / (2 * eps)
        rel = abs(fd - grad[name]) / max(abs(fd), 1e-12)
        assert rel < 1e-3, (
            f"param {name}: adjoint {grad[name]:.6e} vs FD {fd:.6e} "
            f"(rel err {rel:.3e})")

    # physical sanity: a wider center post raises L (more core cross-section)
    assert grad["center_post_diameter"] > 0, "wider post should raise L"


def test_ad_wrapper_gradient_nonzero_and_finite(setup):
    """The adjoint must produce a finite, non-trivial gradient everywhere."""
    prob, fill, f_core = setup
    fwd = slv.ad_wrapper(
        prob,
        {"newton": {"linear": {"spsolve_solver": {}}}},
        {"spsolve_solver": {}},
    )

    def L(f):
        return prob.compute_inductance(fwd(f))

    grad = np.asarray(jax.grad(L)(fill))
    assert np.all(np.isfinite(grad))
    assert np.max(np.abs(grad)) > 0
    # only cells whose fill can affect the solve carry gradient: air cells far
    # from the geometry should be ~zero, core cells strictly positive.
    f_core_rmaj = np.asarray(f_core).T.reshape(-1)
    core_mask = f_core_rmaj > 0.5
    assert grad[core_mask].min() > 0, "some core cells have non-positive dL/df"
