"""run_forward.py -- Forward demo for ``new_fem_engine``.

Runs the axisymmetric magnetostatics forward solve on a PQ core with pluggable
linear-solver backends (scipy spsolve, lineax, feax).  Console output follows
the reference FDM script's format: model/coefficient load message, matrix
assembly + solve timing, 1-turn base L (energy and flux-linkage), scaled L,
core-vs-air energy split, a Geometry & Source Verification block, the
Simulated/Analytical/Datasheet AL triple, a backend comparison, and a timing
summary.  Plots (geometry, psi, |B|) are saved per backend under
``output/<backend>_forward/``.

Self-contained: imports only ``new_fem_engine`` (+ ``cores`` for the core
spec).  It does not import the legacy ``fem`` package.

Experiments in one place
------------------------
All tunables live in :class:`ExperimentConfig` (right below the imports).
Edit that class, then run::

    python new_fem_engine/run_forward.py [core_name]

What you can change without touching anything else:

- ``core_name``        -- which core spec ``cores/<name>/<name>.yaml`` loads.
- ``mesh_size``        -- grid spacing in meters (0.001, 0.0005, 0.00025, ...).
- ``gap_size``         -- total center-post air gap in meters (0 = solid post).
- ``gap_number``       -- number of equal gaps the total is split into.
- ``reference_solver`` -- backend used for the authoritative base-L / AL value.
- ``backends``         -- which backends the comparison section benchmarks.
- ``lineax``           -- lineax solver options (cg / bicgstab / lu / ...).
- ``feax``             -- feax solver options (direct or krylov).
- ``save_plots``       -- toggle PNG output.

Usage
-----
    python new_fem_engine/run_forward.py            # default core (pq_40x40)
    python new_fem_engine/run_forward.py pq_32x30   # another core
"""

import os
import sys
import csv
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PatchCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __package__ in (None, ""):
    from new_fem_engine.generate_mesh import rectangle_mesh, create_mesh
    from new_fem_engine.problem import MagnetostaticProblem
    from new_fem_engine.geometry import FIXED_BOUNDS, compute_fill_fractions
    import new_fem_engine.solver as slv
else:
    from .generate_mesh import rectangle_mesh, create_mesh
    from .problem import MagnetostaticProblem
    from .geometry import FIXED_BOUNDS, compute_fill_fractions
    from . import solver as slv
import feax

logging.getLogger("new_fem_engine").setLevel(logging.WARNING)

MU0 = 4.0 * np.pi * 1e-7
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


# =============================================================================
# EXPERIMENT CONFIG -- edit this class, nothing else, to change a run.
# =============================================================================

@dataclass
class ExperimentConfig:
    """Single place to configure a forward experiment.

    Edit any field below, then run ``python new_fem_engine/run_forward.py``.
    The command-line argument (if given) overrides ``core_name`` only.
    """

    # --- geometry / mesh -----------------------------------------------------
    core_name: str = "pq_40x40"
    mesh_size: float = 0.0010 #m
    gap_size: float = 0.0010 #m
    gap_number: int = 1
    yoke_taper: bool = True
    """True = flux-conserving tapered yoke h(r)=t*r_cp/r (real PQ geometry);
    False = flat constant-thickness yoke (old idealisation)."""

    # --- backends ------------------------------------------------------------
    reference_solver: str = "spsolve"
    """Backend for the authoritative base-L / AL solve and the verification
    block. One of: 'spsolve' | 'lineax' | 'feax'."""

    backends: tuple = ("spsolve", "lineax", "feax")
    """Backends benchmarked in the comparison section, in display order.
    Any subset of: 'spsolve' | 'lineax' | 'feax'."""

    lineax: dict = field(default_factory=lambda: {
        "solver": "cg",        # cg | bicgstab | lu | cholesky | auto
        "rtol": 1e-10,
        "atol": 1e-10,
        "max_steps": 10000,
        "jacobi": True,        # diagonal preconditioning (fast on fine meshes)
    })
    """Options passed to ``new_fem_engine.solver.lineax_solve``."""

    feax: dict = field(default_factory=lambda: {
        "kind": "direct",      # direct | krylov
        "solver": "auto",      # direct: auto | lu | cholesky | qr | ...
        # krylov-only (used when kind='krylov'):
        "rtol": 1e-10,
        "atol": 1e-10,
        "maxiter": 10000,
        "jacobi": True,
    })
    """Options for ``feax_solve``. ``kind='direct'`` uses a sparse direct
    factorisation (fast, recommended); ``kind='krylov'`` uses an iterative
    solver (biCGStab/GMRES...) with an optional Jacobi preconditioner."""

    # --- output --------------------------------------------------------------
    save_plots: bool = True
    """Write geometry/psi/|B| PNGs per backend under ``output_dir``."""

    output_dir: str = field(default_factory=lambda: OUTPUT_DIR)
    """Root folder under which per-backend plot subfolders are created."""


# =============================================================================
# Backend option builders
# =============================================================================

def spsolve_options(_cfg):
    """SciPy sparse direct solve (SuperLU/UMFPACK)."""
    return {"newton": {"linear": {"spsolve_solver": {}}}}


def lineax_options(cfg):
    """Lineax solve; ``cfg.lineax`` maps directly to ``lineax_solve`` options."""
    return {"newton": {"linear": {"lineax_solver": dict(cfg.lineax)}}}


def feax_options(cfg):
    """feax solve; ``cfg.feax['kind']`` selects direct vs krylov."""
    fe = cfg.feax
    if fe.get("kind", "direct") == "krylov":
        options = feax.KrylovSolverOptions(
            solver=fe.get("solver", "bicgstab"),
            tol=fe.get("rtol", 1e-10),
            atol=fe.get("atol", 1e-10),
            maxiter=fe.get("maxiter", 10000),
            use_jacobi_preconditioner=fe.get("jacobi", True),
        )
    else:
        options = feax.DirectSolverOptions(solver=fe.get("solver", "auto"))
    return {"newton": {"linear": {"feax_solver": {"options": options}}}}


_BACKEND_BUILDERS = {
    "spsolve": spsolve_options,
    "lineax": lineax_options,
    "feax": feax_options,
}


def build_solver_options(name, cfg):
    """Return the nested ``solver_options`` dict for backend ``name``."""
    builder = _BACKEND_BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {sorted(_BACKEND_BUILDERS)}")
    return builder(cfg)


def backend_label(name, cfg):
    """Short human-readable label for a backend (used in comparison output)."""
    if name == "lineax":
        return f"lineax({cfg.lineax.get('solver')})"
    if name == "feax":
        return f"feax({cfg.feax.get('kind')},{cfg.feax.get('solver')})"
    if name == "spsolve":
        return "scipy spsolve"
    return name


# =============================================================================
# Core spec loading
# =============================================================================

# Fallback spec so the script runs even without the core YAML (pq_40x40, N87).
_INLINE_CORE_SPEC = {
    "name": "pq_40x40",
    "params": {
        "center_post_diameter": 0.0149,
        "leg_inner_diameter": 0.037,
        "leg_height": 0.0398,
        "window_height": 0.0295,
        "coil_clearance": 0.0005,
        "mur": 1680.0,
        "A_e": 189e-6,
        "l_e": 93e-3,
        "sigma_l_over_A": 492.0,
        "AL": 4300.0,
        "material": "N87",
    },
}

_STEINMETZ_TEMP = 100
_STEINMETZ_FIT = "linear"


def load_core_spec(core_name):
    """Return the core spec dict, straight from ``cores/<name>/<name>.yaml``."""
    try:
        from optimizers import objective

        return objective.load_core_spec(core_name)
    except Exception as exc:
        print(f"core spec unavailable ({exc}); using inline spec")
        spec = dict(_INLINE_CORE_SPEC)
        spec["name"] = core_name
        return spec


def apply_config(core_spec, cfg):
    """Push the experiment config (mesh + gap) onto a core spec.

    This is the only place that writes the mesh/gap into the spec, so the
    values in ``ExperimentConfig`` always win over the YAML defaults.
    """
    core_spec["mesh_size"] = cfg.mesh_size
    core_spec["params"]["gap_size"] = cfg.gap_size
    core_spec["params"]["gap_number"] = cfg.gap_number
    core_spec["params"]["yoke_taper"] = cfg.yoke_taper


def load_steinmetz(material):
    """Best-effort load of Steinmetz coefficients (k, alpha, beta, fit_mode).

    Reads ``cores/steinmetz_full_coefficients.csv`` in the repo root,
    selecting the ``linear`` fit at 100 C to match the reference FDM script.
    Returns None if the file cannot be found or read.
    """
    csv_path = os.path.join(_REPO_ROOT, "cores", "steinmetz_full_coefficients.csv")
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["material"] == material
                    and int(float(row["T_C"])) == _STEINMETZ_TEMP
                    and row["fit_mode"] == _STEINMETZ_FIT):
                return (float(row["k_stein"]), float(row["alpha"]),
                        float(row["beta"]), row["fit_mode"])
    return None


# =============================================================================
# Problem setup
# =============================================================================

def build_problem(core_spec):
    """Construct the MagnetostaticProblem on the fixed (r, z) rectangular mesh."""
    mesh_size = float(core_spec.get("mesh_size", 0.001))
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
        additional_info=(core_spec, mesh_size),
    )
    return prob, Nr, Nz, mesh_size


def geometry_fill(prob, Nr, Nz, mesh_size):
    """Compute the material fill field aligned to the engine cells.

    The engine's cells are ordered r-major (z-minor), so the geometry grid
    fill (z-major from ``compute_fill_fractions``) must be transposed before
    ravel.  ``compute_fill_fractions`` treats its inputs as cell-centre
    coordinates, so pass true cell centres (not grid lines).  Returns the
    per-quad fill plus the grid fields used for plots.
    """
    rmin, rmax, zmin, zmax = FIXED_BOUNDS
    xs = rmin + (np.arange(Nr) + 0.5) * mesh_size  # r cell centres
    ys = zmin + (np.arange(Nz) + 0.5) * mesh_size  # z cell centres

    _, f_prim, f_sec, f_core = compute_fill_fractions(
        jnp.asarray(xs), jnp.asarray(ys),
        prob.core_rects_padded, prob.rect_mask,
        prob.primary_rect, prob.secondary_rect,
        prob.mur, prob.mu0,
    )
    cell_fill = f_core.T.reshape(-1)  # (num_cells,) r-major
    fill = jnp.repeat(cell_fill[:, None], 4, axis=1)  # (num_cells, num_quads)
    # cell-centred grids, shape (Nz, Nr) in z-major order
    f_core_cell = np.asarray(f_core)
    return fill, (f_core_cell, np.asarray(f_prim), np.asarray(f_sec))


# =============================================================================
# Physics post-processing
# =============================================================================

def analytical_inductance(core_spec):
    """Reluctance-model inductance (Henry) for a single effective turn.

    Uses the datasheet ``sigma_l_over_A`` with a fringing-flux correction on
    the gap area (McLyman fringing model).  Matches the FEM within ~10% over
    the gap range; see the project report.
    """
    p = core_spec["params"]
    mur = p["mur"]
    sigma_l_over_A = p["sigma_l_over_A"]
    window_height = p["window_height"]
    r_post = p["center_post_diameter"] / 2.0
    gap = p.get("gap_size", 0.0)

    A_c = np.pi * r_post ** 2
    if gap > 0:
        A_gap = A_c + gap * np.sqrt(A_c) * np.log(2.0 * window_height / gap)
    else:
        A_gap = r_post ** 2

    L = MU0 / ((1.0 / mur) * (sigma_l_over_A - gap / (r_post ** 2 * np.pi)) + gap / A_gap)
    return L


def energy_partition(psi_grid, f_core, Nr, Nz, mesh_size, mur, w_source):
    """Core vs air magnetic-energy split (node B averaged to cell centres).

    Returns (pct_core, pct_air, Bmag_node, W_field), where W_field is the
    total magnetic field energy in Joules.

    B is computed from A_phi = ψ/r rather than from ∇ψ/r, so the r→0 axis
    does not produce a 1/r divergence (A_phi vanishes on the axis).

    The field-energy density ∫ ½|B|²/μ dV is reliable inside the core (where
    μ is set from the fill threshold) but is over-estimated in the thin air
    cells straddling the core/air interface on a coarse mesh.  Since, in a
    linear magnetostatic medium, the total field energy must equal the source
    energy ½ L I², the split is normalised so the core share comes from the
    B-field integral while the total is pinned to the (authoritative) source
    energy.
    """
    h = mesh_size
    rmin, _, _, _ = FIXED_BOUNDS
    r_nodes = rmin + np.arange(Nr + 1) * h
    r_safe = np.maximum(r_nodes, 1e-12)[:, None]
    A_phi = psi_grid / r_safe
    dA_dr = np.gradient(A_phi, h, axis=0)  # r-major grid (Nr+1, Nz+1)
    dA_dz = np.gradient(A_phi, h, axis=1)
    B_r = -dA_dz
    B_z = A_phi / r_safe + dA_dr
    Bmag_node = np.sqrt(B_r ** 2 + B_z ** 2)

    # cell-centred B, shape (Nr, Nz) in r-major -> transpose to (Nz, Nr)
    Bmag_cell = 0.25 * (Bmag_node[:-1, :-1] + Bmag_node[1:, :-1]
                        + Bmag_node[:-1, 1:] + Bmag_node[1:, 1:]).T

    r_cell = rmin + (np.arange(Nr) + 0.5) * h
    dV = np.broadcast_to(2.0 * np.pi * r_cell, (Nz, Nr)) * h * h  # (Nz, Nr)
    mu_cell = np.where(f_core > 0.5, MU0 * mur, MU0)
    energy = 0.5 * Bmag_cell ** 2 / mu_cell * dV
    core_mask = f_core > 0.5
    e_core = energy[core_mask].sum() if core_mask.any() else 0.0
    e_core = max(e_core, 0.0)
    total = float(w_source)
    pct_core = 100.0 * e_core / total if total > 0 else 0.0
    pct_core = min(pct_core, 100.0)
    return pct_core, 100.0 - pct_core, Bmag_node, total


def psi_to_grid(psi, Nr, Nz):
    """Map the nodal solution (r-major) onto the (Nr+1, Nz+1) grid."""
    return np.asarray(psi).ravel().reshape(Nr + 1, Nz + 1)


def source_ampere_turns(prob):
    """Total source ampere-turns ∫ J dA over the (r, z) plane from the RHS."""
    pts = prob.physical_quad_points
    J_quads = jax.vmap(jax.vmap(prob._current_density_at_point))(pts[..., 0], pts[..., 1])
    return float(np.sum(J_quads * prob.fes[0].JxW))


def linkage_inductance(prob, sol_list):
    """Flux-linkage inductance L = λ/I with λ = ∫ J A_phi dV (single turn)."""
    psi = sol_list[0]
    psi_quads = prob.fes[0].convert_from_dof_to_quad(psi)
    JxW = prob.fes[0].JxW[:, :, None]
    pts = prob.physical_quad_points
    J_quads = jax.vmap(jax.vmap(prob._current_density_at_point))(pts[..., 0], pts[..., 1])
    J_quads = J_quads[:, :, None]
    lam = 2 * np.pi * np.sum(psi_quads * J_quads * JxW)
    return float(lam / prob.current ** 2)


# =============================================================================
# Plotting
# =============================================================================

def _pick_indices(n, max_lines=80):
    if n <= max_lines:
        return np.arange(n)
    step = int(np.ceil(n / max_lines))
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def plot_geometry(prob, Nr, Nz, mesh_size, path, show_mesh=False):
    """Core + coil rectangles over the (r, z) domain with a legend.

    The legend is placed outside the axes (right) so it never overlaps the
    core. If ``show_mesh`` is True, sparse mesh gridlines are overlaid.
    """
    rmin, rmax, zmin, zmax = FIXED_BOUNDS
    fig, ax = plt.subplots(figsize=(6.5, 6))

    rects = np.asarray(prob.core_rects_padded)
    mask = np.asarray(prob.rect_mask)
    core_patches = []
    for i in range(len(rects)):
        if mask[i]:
            r0, z0, r1, z1 = rects[i]
            core_patches.append(Rectangle((r0, z0), r1 - r0, z1 - z0))
    ax.add_collection(PatchCollection(core_patches, facecolor="gray", alpha=0.8,
                                      label="Core"))

    legend_elements = [
        Patch(facecolor="gray", alpha=0.8, label="Core"),
        Patch(facecolor="orange", alpha=0.6, label="Primary Coil"),
    ]
    prim = np.asarray(prob.primary_rect)
    sec = np.asarray(prob.secondary_rect)
    for rect, color, label in (
        (prim, "orange", None),
        (sec, "yellow", None),
    ):
        r0, z0, r1, z1 = rect
        ax.add_patch(Rectangle((r0, z0), r1 - r0, z1 - z0, fc=color, ec="none", alpha=0.7))
    if sec[2] > sec[0]:
        legend_elements.append(Patch(facecolor="yellow", alpha=0.6, label="Secondary Coil"))
    legend_elements.append(Patch(facecolor="white", edgecolor="black", label="Air"))
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 1.0),
              fontsize=7, handlelength=1.0, handleheight=0.8, borderpad=0.3,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)

    if show_mesh:
        r_edges = rmin + np.arange(Nr + 1) * mesh_size
        z_edges = zmin + np.arange(Nz + 1) * mesh_size
        for i in _pick_indices(r_edges.size):
            ax.plot([r_edges[i], r_edges[i]], [zmin, zmax], color="k",
                    linewidth=0.3, alpha=0.35)
        for j in _pick_indices(z_edges.size):
            ax.plot([rmin, rmax], [z_edges[j], z_edges[j]], color="k",
                    linewidth=0.3, alpha=0.35)

    ax.set_xlim(rmin, rmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Geometry with Mesh" if show_mesh else "Geometry")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_psi(psi_grid, path):
    rmin, rmax, zmin, zmax = FIXED_BOUNDS
    Nr, Nz = psi_grid.shape[0] - 1, psi_grid.shape[1] - 1
    r_edges = np.linspace(rmin, rmax, Nr + 1)
    z_edges = np.linspace(zmin, zmax, Nz + 1)
    R, Z = np.meshgrid(r_edges, z_edges, indexing="ij")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.pcolormesh(R, Z, psi_grid, shading="gouraud", cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$\psi = r A_\phi$")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(r"Magnetic vector potential $\psi$")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bfield(Bmag_node, Nr, Nz, path):
    rmin, rmax, zmin, zmax = FIXED_BOUNDS
    r_edges = np.linspace(rmin, rmax, Nr + 1)
    z_edges = np.linspace(zmin, zmax, Nz + 1)
    R, Z = np.meshgrid(r_edges, z_edges, indexing="ij")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.pcolormesh(R, Z, Bmag_node, shading="gouraud", cmap="magma")
    fig.colorbar(im, ax=ax, label=r"$|B|$ [T]")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(r"Magnetic flux density magnitude $|B|$")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Solve + reporting
# =============================================================================

def run_backend(solver_options, out_subdir, prob, fill, f_core, Nr, Nz, mesh_size,
                mur, save_plots=True):
    """Solve forward with one backend, save plots, return metrics + timing.

    Returns a dict with L, L_link, W_field, pct_core, pct_air, assemble_s,
    solve_s.
    """
    prob.set_params(fill)
    sol, timing = slv.solver(prob, solver_options, return_timing=True)

    L_sim = float(prob.compute_inductance(sol))
    L_link = linkage_inductance(prob, sol)
    psi_grid = psi_to_grid(sol[0], Nr, Nz)
    w_source = 0.5 * L_sim * prob.current ** 2
    pct_core, pct_air, Bmag_node, W_field = energy_partition(
        psi_grid, f_core, Nr, Nz, mesh_size, mur, w_source)

    if save_plots:
        os.makedirs(out_subdir, exist_ok=True)
        plot_geometry(prob, Nr, Nz, mesh_size, os.path.join(out_subdir, "geometry.png"))
        plot_geometry(prob, Nr, Nz, mesh_size, os.path.join(out_subdir, "geometry_mesh.png"),
                      show_mesh=True)
        plot_psi(psi_grid, os.path.join(out_subdir, "psi.png"))
        plot_bfield(Bmag_node, Nr, Nz, os.path.join(out_subdir, "bfield.png"))

    return {
        "L": L_sim,
        "L_link": L_link,
        "W_field": W_field,
        "pct_core": pct_core,
        "pct_air": pct_air,
        "assemble_s": timing["local_assembly"] + timing["global_matrix"],
        "solve_s": timing["linear"],
    }


def print_experiment_header(cfg):
    """Print the effective configuration so every run is self-documenting."""
    print("=== Experiment configuration ===")
    print(f"Core:        {cfg.core_name}")
    print(f"Mesh size:   {cfg.mesh_size * 1000:.3f} mm")
    print(f"Gap size:    {cfg.gap_size * 1000:.3f} mm (x{cfg.gap_number})")
    print(f"Yoke:        {'tapered (h=ct/r)' if cfg.yoke_taper else 'flat'}")
    print(f"Reference:   {backend_label(cfg.reference_solver, cfg)}")
    print(f"Backends:    {', '.join(backend_label(b, cfg) for b in cfg.backends)}")
    print()


def print_material_load(material, core_spec):
    """Model/coefficient load message (matches the reference FDM script)."""
    stein = load_steinmetz(material)
    if stein is not None:
        K, alpha, beta, fit_mode = stein
        print(f"Loaded Steinmetz coefficients for {material} at {_STEINMETZ_TEMP}C: "
              f"k={K:.6g}, alpha={alpha:.6g}, beta={beta:.6g}, fit_mode={fit_mode}")
    else:
        print(f"Loaded core model for {core_spec['name']} | {material} "
              f"(ur={core_spec['params']['mur']})")


def print_reference(ref, prob, f_core, Nr, Nz, mesh_size, core_spec):
    """Base-L results, geometry/source verification, and the AL triple."""
    params = core_spec["params"]

    print(f"FEM matrix assembly time: {ref['assemble_s']:.2f} s")
    print(f"FEM solve time: {ref['solve_s']:.2f} s")
    print(f"1-turn base L (energy): {ref['L'] * 1e6:.2f} µH | "
          f"(linkage): {ref['L_link'] * 1e6:.2f} µH")
    print()
    print("Scaled (N_p=1, N_s=1):")
    print(f"L (energy): {ref['L'] * 1e6:.2f} µH | (linkage): {ref['L_link'] * 1e6:.2f} µH")
    print(f"Energy in core: {ref['pct_core']:.2f}% | "
          f"Energy in air/gap/leakage: {ref['pct_air']:.2f}%")

    # ---- Geometry & Source Verification (1-turn base) ----
    core_area_mm2 = float(np.sum(f_core)) * mesh_size ** 2 * 1e6
    at_base = source_ampere_turns(prob)
    at_expected = 2.0 * prob.turns * prob.current
    w_source = 0.5 * ref["L"] * prob.current ** 2
    print()
    print("--- Geometry & Source Verification (1-turn base) ---")
    print(f"Gap Size:           {params.get('gap_size', 0.0) * 1000:.3f} mm "
          f"(x{params.get('gap_number', 0)})")
    print(f"Mesh Size:          {mesh_size * 1000:.3f} mm")
    print(f"Simulated Core Area (approx): {core_area_mm2:.2f} mm²")
    print(f"Total Simulated AT (base): {at_base:.4f} A-turns")
    print(f"Expected AT (base):        {at_expected:.4f} A-turns")
    print(f"Total energy (source method): {w_source:.6e} J")
    print(f"Total energy (field method):  {ref['W_field']:.6e} J")
    print("--------------------------------------")
    print()

    # ---- AL comparison ----
    AL_sim = ref["L"] * 1e9
    L_an = analytical_inductance(core_spec)
    AL_ds = params.get("AL", 0.0)
    print(f"Simulated AL: {AL_sim:.2f} nH/turn²")
    print(f"Analytical AL: {L_an * 1e9:.2f} nH/turn²")
    print(f"Datasheet AL: {AL_ds:.2f} nH/turn²")
    print()


def print_backend_comparison(results, cfg):
    """Table of L and solve time per benchmarked backend."""
    names = [n for n in cfg.backends if n in results]
    if not names:
        return
    print("=== Backend comparison ===")
    for name in names:
        res = results[name]
        print(f"   {backend_label(name, cfg):24s}  L = {res['L'] * 1e9:9.1f} nH  "
              f"(assemble {res['assemble_s']:.3f} s | solve {res['solve_s']:.3f} s)")
    if len(names) >= 2:
        L0 = results[names[0]]["L"]
        dL = max(abs(results[n]["L"] - L0) for n in names) * 1e9
        print(f"   |max dL| across backends = {dL:.3e} nH")
        t0 = results[names[0]]["solve_s"]
        for name in names[1:]:
            if t0 > 0:
                print(f"   speedup {names[0]}/{name} = "
                      f"{results[name]['solve_s'] / t0:.2f}x")
        a0 = results[names[0]]["assemble_s"]
        for name in names[1:]:
            if a0 > 0:
                print(f"   assembly speedup {names[0]}/{name} = "
                      f"{results[name]['assemble_s'] / a0:.2f}x")
    print()


def print_timing(all_start, ref, label="Total"):
    """Final timing summary.

    ``label`` defaults to "Total" (accurate in ``run_forward.main()``, where
    ``all_start`` is the process start).  Callers that measure only a single
    phase (e.g. ``report.report_design``) pass an explicit phase label so the
    number is never mistaken for the whole run.
    """
    print("=== Timing ===")
    print(f"FEM matrix assembly time: {ref['assemble_s']:.2f} s")
    print(f"FEM solve time: {ref['solve_s']:.2f} s")
    print(f"{label} execution time: {time.time() - all_start:.2f} s")
    print()


# =============================================================================
# Entry point
# =============================================================================

def main():
    all_start = time.time()
    cfg = ExperimentConfig()
    if len(sys.argv) > 1:
        cfg.core_name = sys.argv[1]

    core_spec = load_core_spec(cfg.core_name)
    apply_config(core_spec, cfg)
    material = core_spec["params"]["material"]

    print_experiment_header(cfg)
    print_material_load(material, core_spec)

    prob, Nr, Nz, mesh_size = build_problem(core_spec)
    fill, (f_core, _, _) = geometry_fill(prob, Nr, Nz, mesh_size)

    # Authoritative solve (config.reference_solver) + plots + verification.
    ref = run_backend(
        build_solver_options(cfg.reference_solver, cfg),
        os.path.join(cfg.output_dir, "reference_forward"), prob, fill, f_core,
        Nr, Nz, mesh_size, core_spec["params"]["mur"], save_plots=cfg.save_plots,
    )
    print_reference(ref, prob, f_core, Nr, Nz, mesh_size, core_spec)

    # Benchmark the configured backends.
    results = {}
    for name in cfg.backends:
        results[name] = run_backend(
            build_solver_options(name, cfg),
            os.path.join(cfg.output_dir, f"{name}_forward"), prob, fill, f_core,
            Nr, Nz, mesh_size, core_spec["params"]["mur"], save_plots=cfg.save_plots,
        )
    print_backend_comparison(results, cfg)
    print_timing(all_start, ref)

    if cfg.save_plots:
        print(f"Plots saved under {cfg.output_dir}/")


if __name__ == "__main__":
    main()
