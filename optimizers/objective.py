"""Shared objective: geometry-parametrised inductance for ``new_fem_engine``.

This is the production adaptation of the certified differentiable closure in
``tests/helpers.make_param_problem`` (``test_backend_diff.py`` pins
``jax.grad(L)(geom)`` against finite differences for spsolve/lineax/feax
forward+adjoint pairs).  It wires the 5 optimizers to the current engine.

Free variables (``VARY_GEOM_KEYS``, in this fixed order, 5 dims)::

    center_post_diameter, leg_inner_diameter, leg_height,
    window_height, gap_size

``coil_clearance`` is frozen to its datasheet value (see
:data:`FROZEN_GEOM_KEYS`): it is a bobbin/winding packaging constraint, not an
inductance-design freedom.

``make_loss_fn(core_spec)`` returns a *differentiable* scalar loss
``loss(x) = (L(x) - target_L)^2`` on a flat length-5 vector ``x``; the
gradient ``jax.grad(loss)(x)`` is certified against central differences.

Physics caveats (documented, not hidden):
  - The geometry is baked into ``prob`` at construction; each evaluation
    recomputes the *core fill* from the geometry overrides and solves
    through ``solver.ad_wrapper`` (implicit differentiation, custom_vjp).
    Coil areas/rects come from the base params and are NOT re-derived per
    geometry — the certified, tested path.
  - ``gap_size = 0`` is a degenerate kink (the gap geometry activates for
    gap > 0), where the AD gradient is 0 but finite differences see a jump.
    The loader bounds are ``[5e-5, 0.005]`` (never 0), and the default
    ``x0`` starts at 1 mm, so the optimizer never sits on the kink.
  - L is mesh-limited at coarse meshes (0.5 mm -> ~224 nH, 1 mm -> ~68 nH
    for PQ32/30, gap 1 mm).  Use the fine mesh for production values.
"""

import logging
import os

import numpy as np
import jax
import jax.numpy as jnp

logging.getLogger("new_fem_engine").setLevel(logging.WARNING)

from new_fem_engine.generate_mesh import rectangle_mesh, create_mesh
from new_fem_engine.problem import MagnetostaticProblem, COIL_GRID_RES, MU0
from new_fem_engine.geometry import (
    FIXED_BOUNDS, define_core_rectangles, compute_fill_fractions,
    compute_coil_areas,
)
import new_fem_engine.solver as slv

from optimizers import backends

GEOM_KEYS = [
    "center_post_diameter",
    "leg_inner_diameter",
    "leg_height",
    "window_height",
    "coil_clearance",
    "gap_size",
]

# Design keys frozen to their datasheet (base) values.  ``coil_clearance`` is a
# bobbin/winding packaging constraint, not an inductance-design freedom, so the
# optimizer must never move it (doing so "cheats" L via leakage/coil position).
# The optimizer's x-vector runs over ``VARY_GEOM_KEYS`` (5 dims); the frozen
# keys stay pinned to ``base_params`` in every params dict (FEM + analytical +
# reports still see all 6).
FROZEN_GEOM_KEYS = ("coil_clearance",)

VARY_GEOM_KEYS = [k for k in GEOM_KEYS if k not in FROZEN_GEOM_KEYS]

# Backward-compatible aliases for the spsolve defaults.
SP = backends.SP
ADJ = backends.ADJ


_CORES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cores")


def load_core_spec(core_name):
    """Load a core spec directly from ``cores/<name>/<name>.yaml``.

    The YAML file IS the spec: flat and 1:1 with the returned dict, so there
    is no loader/translation layer.  ``<name>`` is the core type
    (``pq_40x40``, ``pq_32x30``, ...).
    """
    import yaml

    path = os.path.join(_CORES_DIR, core_name, f"{core_name}.yaml")
    if not os.path.exists(path):
        available = sorted(d for d in os.listdir(_CORES_DIR)
                           if os.path.isdir(os.path.join(_CORES_DIR, d)))
        raise ValueError(f"Unknown core: {core_name}. Available: {available}")
    with open(path) as f:
        return yaml.safe_load(f)


def base_params(core_spec):
    """Full params dict for ``define_core_rectangles`` from a core spec.

    The loader spec does not carry ``gap_size``/``gap_number``/``yoke_taper``,
    so they default here (1 mm gap, 1 gap, tapered yoke — the project default).
    """
    p = dict(core_spec["params"])
    p.setdefault("gap_size", 0.001)
    p.setdefault("gap_number", 1)
    p.setdefault("yoke_taper", True)
    return p


def default_x0(core_spec):
    """Starting point inside the loader bounds: datasheet geometry + 1 mm gap.

    Over the free keys (:data:`VARY_GEOM_KEYS`) only; frozen keys are pinned
    to their datasheet values and never enter the design vector.
    """
    idx = [i for i, k in enumerate(GEOM_KEYS) if k in VARY_GEOM_KEYS]
    lo = np.array([core_spec["bounds"][i][0] for i in idx])
    hi = np.array([core_spec["bounds"][i][1] for i in idx])
    base = base_params(core_spec)
    x0 = np.array([float(base[k]) for k in VARY_GEOM_KEYS])
    return np.clip(x0, lo + 1e-12, hi - 1e-12)


def bounds_arrays(core_spec):
    """``(lo, hi)`` over the free keys (:data:`VARY_GEOM_KEYS`) only."""
    idx = [i for i, k in enumerate(GEOM_KEYS) if k in VARY_GEOM_KEYS]
    lo = np.array([core_spec["bounds"][i][0] for i in idx], dtype=float)
    hi = np.array([core_spec["bounds"][i][1] for i in idx], dtype=float)
    return lo, hi


def _mesh_centers(mesh_size):
    rmin, rmax, zmin, zmax = FIXED_BOUNDS
    Nr = max(int(round((rmax - rmin) / mesh_size)), 1)
    Nz = max(int(round((zmax - zmin) / mesh_size)), 1)
    xs = rmin + (np.arange(Nr) + 0.5) * mesh_size
    ys = zmin + (np.arange(Nz) + 0.5) * mesh_size
    return xs, ys


def make_eval(core_spec, w_vol=0.0, mesh_size=None, fwd_opts=None, adj_opts=None,
              backend=None, traceable=False):
    """Single-solve ``eval(x) -> (loss, L, V)`` for the core's geometry vector.

    ``x`` is a flat length-5 vector over :data:`VARY_GEOM_KEYS` (the frozen
    keys stay at their datasheet base values).  The loss, the inductance and
    the axisymmetric core volume all come from the SAME forward solve, so
    gradient-free optimizers can record L (and V) without extra evaluations.

    Coil handling (non-traceable path): the engine's assembly kernels bake the
    coil geometry (rects/areas/mur) at jit time, so changing coils on a shared
    ``Problem`` does not take effect unless the kernels are rebuilt.  ``eval``
    therefore first calls :meth:`prepare` (eager, concrete geometry -> coils +
    rebuilt kernels) whenever ``x`` is not a JAX tracer.  Under ``jax.grad``
    the forward is traced with a tracer ``x``: ``prepare`` is then skipped and
    the kernels from the last concrete ``prepare`` are used, keeping the
    gradient path clean (coils frozen at the current design, only the fill is
    differentiated — the certified implicit-differentiation contract).

    Coil handling (``traceable=True``): the coil geometry is computed from
    ``x`` *inside* the traced body (``geometry_from_x``) and threaded into the
    solve and inductance as traced arguments, so the whole
    ``jax.jit(jax.value_and_grad(loss))`` is one compiled program and the coils
    move correctly without ``prepare`` or per-iteration kernel rebuilds.  This
    is the fast optimizer-loop path (measured ~3 ms/iter at 0.002 mesh).
    """
    mesh_size = mesh_size or core_spec["mesh_size"]
    if backend is not None:
        fwd_opts, adj_opts = backends.solver_backend(backend)
    elif fwd_opts is None:
        fwd_opts, adj_opts = (backends.solver_backend('lineax') if traceable
                              else (SP, ADJ))
    if traceable and ('spsolve_solver' in str(fwd_opts)
                      or 'spsolve_solver' in str(adj_opts)):
        raise ValueError(
            "traceable=True requires the lineax backend, not scipy spsolve.")
    base = base_params(core_spec)
    target = float(core_spec["target_L"])
    spec = {"params": base, "mesh_size": mesh_size}
    rmin, rmax, zmin, zmax = FIXED_BOUNDS
    Nr = max(int(round((rmax - rmin) / mesh_size)), 1)
    Nz = max(int(round((zmax - zmin) / mesh_size)), 1)
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
    if traceable:
        fwd = slv.ad_wrapper(prob, fwd_opts, adj_opts, traceable=True)
    else:
        fwd = slv.ad_wrapper(prob, fwd_opts, adj_opts)

    xs = rmin + (np.arange(Nr) + 0.5) * mesh_size
    ys = zmin + (np.arange(Nz) + 0.5) * mesh_size
    r_cell = np.asarray(xs)
    dv_cell = 2.0 * np.pi * r_cell * mesh_size * mesh_size

    @jax.jit
    def fill_core(x):
        """Core fill field on the mesh grid, traced/JIT'd once.

        ``jax.jit`` here is decisive: under ``jax.value_and_grad`` the
        ``eval_frozen`` body is re-traced on every call, and the raw
        ``define_core_rectangles`` + ``compute_fill_fractions`` python loops
        (a ~124-iteration ``rect_fill`` trace) cost ~1.3 s per evaluation.
        JIT-ing this function builds the jaxpr once and reuses it, cutting that
        to ~0.05 s.  ``base``/``xs``/``ys`` are baked in as constants.
        """
        p = dict(base)
        for i, k in enumerate(VARY_GEOM_KEYS):
            p[k] = x[i]
        p_c = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
               for k, v in p.items()}
        rects, mask, prim, sec = define_core_rectangles(p_c)
        _, _, _, fc = compute_fill_fractions(
            jnp.asarray(xs), jnp.asarray(ys), rects, mask, prim, sec, p["mur"])
        return fc

    @jax.jit
    def geometry_from_x(x):
        """Traced/JIT'd core fill + coil state from the geometry vector.

        Returns ``(fc, prim, sec, a_prim, a_sec)`` where ``fc`` is the core
        fill field on the mesh grid, the rects are the coil window bounds, and
        the areas are the coil cross-section areas on the COIL_GRID_RES grid
        (matching the eager ``prepare`` computation).  Everything is traced so
        the fused ``value_and_grad`` re-derives the coils per ``x`` — no stale
        kernel baking, no per-iteration ``prepare``.
        """
        p = dict(base)
        for i, k in enumerate(VARY_GEOM_KEYS):
            p[k] = x[i]
        p_c = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
               for k, v in p.items()}
        rects, mask, prim, sec = define_core_rectangles(p_c)
        _, _, _, fc = compute_fill_fractions(
            jnp.asarray(xs), jnp.asarray(ys), rects, mask, prim, sec, p_c["mur"])
        xs_c = jnp.linspace(rmin, rmax, COIL_GRID_RES)
        ys_c = jnp.linspace(zmin, zmax, COIL_GRID_RES)
        _, f_prim, f_sec, _ = compute_fill_fractions(
            xs_c, ys_c, rects, mask, prim, sec, p_c["mur"], MU0)
        cell_area = (xs_c[1] - xs_c[0]) * (ys_c[1] - ys_c[0])
        a_prim, a_sec = compute_coil_areas(f_prim, f_sec, cell_area)
        return fc, prim, sec, a_prim, a_sec

    def prepare(x):
        """Eagerly set the problem's coil state + rebuild assembly kernels."""
        geom = {k: float(x[i]) for i, k in enumerate(VARY_GEOM_KEYS)}
        p = dict(base)
        p.update(geom)
        p_c = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
               for k, v in p.items()}
        rects, mask, prim, sec = define_core_rectangles(p_c)
        prob.mur = p_c["mur"]
        prob.primary_rect = prim
        prob.secondary_rect = sec
        xs_c = np.linspace(rmin, rmax, COIL_GRID_RES)
        ys_c = np.linspace(zmin, zmax, COIL_GRID_RES)
        _, f_prim, f_sec, _ = compute_fill_fractions(
            xs_c, ys_c, rects, mask, prim, sec, p_c["mur"], MU0)
        cell_area = (xs_c[1] - xs_c[0]) * (ys_c[1] - ys_c[0])
        prob.area_prim, prob.area_sec = compute_coil_areas(f_prim, f_sec, cell_area)
        prob.pre_jit_fns()

    def eval_frozen(x):
        """Traced body: geometry + solve in one jitted program."""
        if traceable:
            fc, prim, sec, a_prim, a_sec = geometry_from_x(x)
            fill = jnp.repeat(fc.T.reshape(-1)[:, None], 4, axis=1)
            sol = fwd((fill, prim, sec, a_prim, a_sec))
            l = prob.compute_inductance(sol, prim, sec, a_prim, a_sec)
        else:
            fc = fill_core(x)
            fill = jnp.repeat(fc.T.reshape(-1)[:, None], 4, axis=1)
            l = prob.compute_inductance(fwd(fill))
        val = ((l - target) * 1e9) ** 2
        vol = jnp.sum(fc * dv_cell[None, :])
        if w_vol:
            val = val + w_vol * (vol * 1e6) ** 2
        return val, l, vol

    def eval(x):
        if traceable:
            return eval_frozen(x)
        if not isinstance(x, jax.core.Tracer):
            prepare(x)
        return eval_frozen(x)

    eval.prepare = prepare
    eval.frozen = eval_frozen
    return eval


def make_loss_fn(core_spec, w_vol=0.0, mesh_size=None, fwd_opts=None, adj_opts=None,
                 backend=None, traceable=False):
    """Differentiable ``loss(x) = (L(x) - target)^2 + w_vol * V(x)``.

    ``x`` is a flat length-5 vector over :data:`VARY_GEOM_KEYS`.  Use
    ``jax.value_and_grad(loss)`` (optionally ``jax.jit``-wrapped, required for
    the ``traceable`` path) for gradient-based optimizers.
    """
    eval = make_eval(core_spec, w_vol=w_vol, mesh_size=mesh_size,
                     fwd_opts=fwd_opts, adj_opts=adj_opts, backend=backend,
                     traceable=traceable)
    loss = lambda x: eval(x)[0]
    loss.prepare = eval.prepare
    loss.frozen = eval.frozen
    return loss


def make_params_dict(x, core_spec):
    """Full params dict from a design vector over the free keys.

    ``x`` maps onto :data:`VARY_GEOM_KEYS` in order; the frozen keys
    (:data:`FROZEN_GEOM_KEYS`, e.g. ``coil_clearance``) keep their datasheet
    base values, so the returned dict always carries all 6 params.
    """
    params = base_params(core_spec)
    for i, k in enumerate(VARY_GEOM_KEYS):
        params[k] = float(x[i])
    return params


def solve_forward(params_dict, mesh_size, fwd_opts=SP, backend=None):
    """Plain (non-differentiated) forward solve -> (L, None).

    Keeps the old ``(L, ctx)`` return contract of the legacy ``fem`` helpers;
    ``ctx`` is always ``None`` in the new engine (plotting now lives in
    ``run_forward``).
    """
    if backend is not None:
        fwd_opts, _ = backends.solver_backend(backend)
    base = dict(params_dict)
    spec = {"params": base, "mesh_size": mesh_size}
    rmin, rmax, zmin, zmax = FIXED_BOUNDS
    Nr = max(int(round((rmax - rmin) / mesh_size)), 1)
    Nz = max(int(round((zmax - zmin) / mesh_size)), 1)
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
    xs = rmin + (np.arange(Nr) + 0.5) * mesh_size
    ys = zmin + (np.arange(Nz) + 0.5) * mesh_size
    rects, mask, prim, sec = define_core_rectangles(base)
    _, _, _, fc = compute_fill_fractions(
        jnp.asarray(xs), jnp.asarray(ys), rects, mask, prim, sec, base["mur"])
    fill = jnp.repeat(fc.T.reshape(-1)[:, None], 4, axis=1)
    prob.set_params(fill)
    sol = slv.solver(prob, fwd_opts)
    L = float(prob.compute_inductance(sol))
    return L, None
