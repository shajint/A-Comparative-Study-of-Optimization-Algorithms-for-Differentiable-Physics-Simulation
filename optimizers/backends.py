"""Backend registry: friendly names -> (forward_opts, adjoint_opts).

The single production mapping for every optimizer.  ``solver_backend(name)``
returns the Newton-wrapped options for the forward solve and the flat options
for the adjoint solve (``implicit_vjp`` uses a flat dict — no ``newton``
layer).  All backends are interchangeable on one fixed mesh: they agree to
~1e-11 nH at the same geometry, so a fair comparison varies only the linear
solver.

Names: ``spsolve`` | ``lineax`` | ``feax``.
"""

import copy

SP = {"newton": {"linear": {"spsolve_solver": {}}}}
ADJ = {"spsolve_solver": {}}

LINEAX_DEFAULTS = {
    "solver": "cg",        # cg | bicgstab | lu | cholesky | auto
    "rtol": 1e-10,
    "atol": 1e-10,         # capped to 1e-13 internally under Jacobi
    "max_steps": 20000,
    "jacobi": True,
}


def _feax_options(kind="direct"):
    """feax solver options; ``kind`` is 'direct' (recommended) or 'krylov'."""
    import feax

    if kind == "krylov":
        return feax.KrylovSolverOptions(
            solver="bicgstab", tol=1e-10, atol=1e-10, maxiter=10000,
            use_jacobi_preconditioner=True,
        )
    return feax.DirectSolverOptions(solver="auto")


def solver_backend(name, lineax=None, feax_kind="direct"):
    """Return ``(fwd_opts, adj_opts)`` for backend ``name``."""
    name = str(name).lower()
    la = dict(LINEAX_DEFAULTS)
    la.update(lineax or {})

    if name == "spsolve":
        return copy.deepcopy(SP), copy.deepcopy(ADJ)
    if name == "lineax":
        fwd = {"newton": {"linear": {"lineax_solver": la}}}
        return fwd, {"lineax_solver": la}
    if name == "feax":
        opts = _feax_options(feax_kind)
        fwd = {"newton": {"linear": {"feax_solver": {"options": opts}}}}
        return fwd, {"feax_solver": {"options": opts}}
    raise ValueError(f"Unknown backend {name!r}; choices: spsolve | lineax | feax")


BACKENDS = ("spsolve", "lineax", "feax")

# Backends whose linear solver can run under the traced objective (``build_vg``,
# the jaxopt/optax fast loops).  lineax is a Krylov solver; feax is defaulted
# to a direct solver which the traceable BCOO path rejects (see
# ``new_fem_engine/solver.py``), so it is excluded here.
TRACEABLE_BACKENDS = ("lineax",)


def backend_label(name):
    return {"spsolve": "scipy spsolve", "lineax": "lineax",
            "feax": "feax"}.get(str(name), str(name))
