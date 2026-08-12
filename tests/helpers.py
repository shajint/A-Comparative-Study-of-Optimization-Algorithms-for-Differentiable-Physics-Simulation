"""Shared test-support helpers for ``new_fem_engine`` physics tests.

Pure functions + a couple of closures; no pytest magic (fixtures live in
``conftest.py``).  Every test builds systems through :func:`build_system` so
all files share one code path.
"""

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from new_fem_engine.generate_mesh import rectangle_mesh, create_mesh
from new_fem_engine.problem import MagnetostaticProblem, COIL_GRID_RES
from new_fem_engine.geometry import (
    FIXED_BOUNDS, define_core_rectangles, compute_fill_fractions,
    compute_coil_areas,
)
from new_fem_engine.run_forward import geometry_fill, linkage_inductance
import new_fem_engine.solver as slv

SP = {"newton": {"linear": {"spsolve_solver": {}}}}
ADJ = {"spsolve_solver": {}}

RTOL = 1e-3  # default relative tolerance for physics equalities


def base_params(**overrides):
    """PQ40/40 params dict (tapered yoke is the project default)."""
    p = {
        "center_post_diameter": 0.0149,
        "leg_inner_diameter": 0.037,
        "leg_height": 0.0398,
        "window_height": 0.0295,
        "coil_clearance": 0.0005,
        "gap_size": 0.0,
        "gap_number": 1,
        "mur": 1680.0,
        "A_e": 189e-6,
        "l_e": 93e-3,
        "sigma_l_over_A": 492.0,
        "AL": 4300.0,
        "material": "N87",
        "yoke_taper": True,
    }
    p.update(overrides)
    return p


def build_system(mesh_size=0.002, params=None, nr=None, nz=None):
    """Build a solved-ready (prob, fill, f_core) on the fixed structured mesh.

    ``params`` may be any dict accepted by ``define_core_rectangles``
    (defaults to :func:`base_params`).  ``fill`` is the differentiable
    material field; call ``prob.set_params(fill)`` before solving.
    """
    params = params if params is not None else base_params()
    spec = {"params": params, "mesh_size": mesh_size}
    rmin, rmax, zmin, zmax = FIXED_BOUNDS
    Nr = nr or max(int(round((rmax - rmin) / mesh_size)), 1)
    Nz = nz or max(int(round((zmax - zmin) / mesh_size)), 1)
    msh = rectangle_mesh(Nr, Nz, rmin, rmax, zmin, zmax)
    mesh = create_mesh(msh, "quad")
    locs = [
        lambda pt: jnp.isclose(pt[0], rmax),
        lambda pt: jnp.isclose(pt[0], rmin),
        lambda pt: jnp.isclose(pt[1], zmax),
        lambda pt: jnp.isclose(pt[1], zmin),
    ]
    prob = MagnetostaticProblem(
        mesh=mesh, vec=1, dim=2,
        dirichlet_bc_info=[locs, [0, 0, 0, 0], [lambda pt: 0.0] * 4],
        additional_info=(spec, mesh_size),
    )
    fill, (f_core, _, _) = geometry_fill(prob, Nr, Nz, mesh_size)
    return prob, fill, np.asarray(f_core)


def make_fwd(prob, fwd_opts=SP, adj_opts=ADJ):
    """Differentiable forward wrapper bound to ``prob``.

    ``fwd_opts`` are the Newton+linear options for the forward solve,
    ``adj_opts`` the flat linear options for the adjoint solve.
    Defaults solve both stages with scipy spsolve."""
    return slv.ad_wrapper(prob, fwd_opts, adj_opts)


def solve_L(prob, fill):
    """Plain forward solve -> inductance (Henry)."""
    prob.set_params(fill)
    sol = slv.solver(prob, SP)
    return float(prob.compute_inductance(sol))


def swap_windings(prob):
    """Swap primary/secondary coil rects and recompute their areas."""
    prob.primary_rect, prob.secondary_rect = prob.secondary_rect, prob.primary_rect
    rmin, rmax, zmin, zmax = FIXED_BOUNDS
    xs = np.linspace(rmin, rmax, COIL_GRID_RES)
    ys = np.linspace(zmin, zmax, COIL_GRID_RES)
    _, f_prim, f_sec, _ = compute_fill_fractions(
        xs, ys, prob.core_rects_padded, prob.rect_mask,
        prob.primary_rect, prob.secondary_rect, prob.mur, prob.mu0,
    )
    cell_area = (xs[1] - xs[0]) * (ys[1] - ys[0])
    prob.area_prim, prob.area_sec = compute_coil_areas(f_prim, f_sec, cell_area)


def z_mirror_fill(f_core):
    """Mirror a (Nz, Nr) core grid about the z=0 plane."""
    return f_core[::-1]


def make_param_problem(mesh_size=0.002, fwd_opts=SP, adj_opts=ADJ):
    """Differentiable L(geom) closure for gradient w.r.t. geometry params.

    Returns ``(prob, fwd, L)`` where ``L(geom)`` recomputes the core fill
    from a geometry override dict (subset of :func:`base_params`) and solves
    through the implicit-differentiation wrapper.  ``fwd_opts``/``adj_opts``
    select the forward/adjoint linear-solver backends.
    """
    prob, _, f_core = build_system(mesh_size=mesh_size)
    fwd = make_fwd(prob, fwd_opts, adj_opts)
    Nr, Nz = f_core.shape[1], f_core.shape[0]
    rmin, rmax, zmin, zmax = FIXED_BOUNDS
    xs = rmin + (np.arange(Nr) + 0.5) * mesh_size
    ys = zmin + (np.arange(Nz) + 0.5) * mesh_size

    def fill_from_params(geom):
        p = base_params()
        p.update(geom)
        rects, mask, prim, sec = define_core_rectangles(p)
        _, _, _, fc = compute_fill_fractions(
            jnp.asarray(xs), jnp.asarray(ys), rects, mask, prim, sec, p["mur"])
        return jnp.repeat(fc.T.reshape(-1)[:, None], 4, axis=1)

    def L(geom):
        return prob.compute_inductance(fwd(fill_from_params(geom)))

    return prob, fwd, L


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def rel_err(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def richardson2(coarse, fine):
    """2nd-order Richardson extrapolation from L(h) and L(h/2)."""
    return (4.0 * fine - coarse) / 3.0


def central_grad_fd(f, x, i, h):
    e = np.zeros_like(x)
    e[i] = h
    return float((f(x + e) - f(x - e)) / (2 * h))


def central_hessian_fd(f, x, i, j, h):
    ei = np.zeros_like(x)
    ei[i] = h
    ej = np.zeros_like(x)
    ej[j] = h
    if i == j:
        e2 = np.zeros_like(x)
        e2[i] = 2 * h
        return float((f(x + e2) - 2 * f(x) + f(x - e2)) / (4 * h * h))
    return float((f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej))
                 / (4 * h * h))
