import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
import scipy
import time
from jax.experimental.sparse import BCOO

import lineax
import feax
import equinox as eqx

from . import logger
from jax import config
config.update("jax_enable_x64", True)

def _timing_record(timing, name, dt):
    timing[name] += dt


def _log_newton_iter_start(iter_num):
    logger.info("  iter %d", iter_num)


def _log_newton_iter_summary(iter_num, local_s, global_s, res_val, rel_res_val, linear_s=None):
    logger.info("           nonlinear residual: L2 norm = %.3g (relative to initial = %.3g)",
                res_val, rel_res_val)
    if linear_s is None:
        logger.info("           timing: local assembly %6.3f s, global matrix %6.3f s",
                    local_s, global_s)
    else:
        logger.info("           timing: linear solve %6.3f s, local assembly %6.3f s, global matrix %6.3f s",
                    linear_s, local_s, global_s)
        
def _log_timing_table(n_iters, parts, wall_s):
    """Log the per-phase timing breakdown of the Newton loop.

    Parameters
    ----------
    n_iters : int
        Number of Newton iterations taken.
    parts : dict
        Accumulated wall times with keys 'local_assembly', 'global_matrix',
        'linear' (filled by ``_timing_record``).
    wall_s : float
        Total wall time of the solve (``time.perf_counter()`` delta).
    """
    rows = (
        ('local_assembly', 'local'),
        ('global_matrix', 'global'),
        ('linear', 'linear'),
    )

    logger.info("Timing summary — %d Newton iter, %.3f s wall", n_iters, wall_s)
    for key, label in rows:
        dt = parts[key]
        pct = 100. * dt / wall_s if wall_s > 0 else 0.
        logger.info("  %-8s %7.3f s  %5.1f%%", label, dt, pct)
    other = wall_s - sum(parts.values())
    if other >= 0.01:
        pct = 100. * other / wall_s if wall_s > 0 else 0.
        logger.info("  %-8s %7.3f s  %5.1f%%", "other", other, pct)
        
def scipy_spsolve(A, b):
    """SciPy direct sparse solve (SuperLU, or UMFPACK if scikits.umfpack is
    installed and applicable).

    Parameters
    ----------
    A : scipy.sparse.csr_array
        Global tangent assembled by ``get_A`` (Dirichlet rows eliminated).
    b : JaxArray
        RHS with Dirichlet BC applied.
    """
    logger.debug("Scipy Solver - Solving linear system with scipy.sparse.linalg.spsolve")
    x = scipy.sparse.linalg.spsolve(A, onp.array(b))
    logger.debug("Scipy Solver - Finished solving, linear solve res = %.3g",
                 np.linalg.norm(A @ x - b))
    return x

def _csr_to_bcoo(A):
    """Convert a scipy CSR matrix to a JAX BCOO in ~1 ms.

    ``BCOO.from_scipy_sparse`` + ``sort_indices`` costs ~0.4 s for the 0.5 mm
    meshes (~8.6k DOFs / 73k nnz) — measured overhead that dominated the
    iterative solves.  CSR is already row-major ordered, so ``A.tocoo()``
    yields sorted (row, col) pairs and no ``sort_indices`` is required.
    """
    coo = A.tocoo()
    indices = np.stack([np.asarray(coo.row, dtype=onp.int32),
                        np.asarray(coo.col, dtype=onp.int32)], axis=1)
    return BCOO((np.asarray(coo.data), indices), shape=A.shape)


class _SparseMatvec(eqx.Module):
    """BCOO-backed matvec stored as a pytree leaf.

    ``lineax.FunctionLinearOperator`` defaults to ``closure_convert=True``,
    which bakes the matrix into a freshly-created lambda *static*.  JAX's jit
    cache hashes that static by identity, so every solve rebuilt a different
    lambda and recompiled the whole CG machinery (~0.4 s per call at 0.5 mm).
    Keeping the BCOO as a traced leaf (``closure_convert=False``) makes the
    cache key on the matrix *structure* only: any matrix with the same
    sparsity pattern reuses the compiled solve and runs at ~0.08 s.
    """
    bcoo: BCOO

    def __call__(self, v):
        return self.bcoo @ v


################################################################################
# Traceable BCOO assembly + linear solves
#
# The jitted optimizer path must not touch scipy objects: ``get_A_bcoo`` keeps
# only ``V`` (the flat per-cell Jacobian entries) traced, with the sparsity
# pattern ``(I, J)`` and the Dirichlet-DOF set baked in as mesh constants, so
# the assembly and every linear solve below run inside ``jax.jit``.

def _bcoo_structure(problem):
    """Static ``(n, I_f, J_f, keep, n_bc)`` assembly structure for a problem.

    ``keep`` is a bool mask over the flat ``V`` entries selecting the
    unconstrained rows/columns; ``I_f``/``J_f`` are the filtered indices plus
    the unit diagonal entries for the constrained DOFs.  Computed once and
    cached on the problem (the mesh never changes across a sweep).
    """
    struct = getattr(problem, "_bcoo_struct", None)
    if struct is not None:
        return struct
    n = problem.num_total_dofs_all_vars
    I = onp.asarray(problem.I)
    J = onp.asarray(problem.J)
    bc = _bc_dofs(problem)
    keep = ~(onp.isin(I, bc) | onp.isin(J, bc))
    I_f = onp.concatenate([I[keep], bc])
    J_f = onp.concatenate([J[keep], bc])
    struct = (n, I_f, J_f, keep, bc.size)
    problem._bcoo_struct = struct
    return struct


def get_A_bcoo(problem, V=None):
    """Assemble the global tangent as a traceable JAX ``BCOO`` matrix.

    Same Dirichlet row/column elimination as :func:`get_A`, but only ``V`` is
    traced — ``I``, ``J`` and the constrained-DOF set are mesh constants — so
    the function can live inside ``jax.jit`` (unlike ``get_A``, which forces
    ``onp.asarray(V)``).
    """
    n, I_f, J_f, keep, n_bc = _bcoo_structure(problem)
    if V is None:
        V = problem.V
    V_keep = V[keep]
    data = np.concatenate([V_keep, np.ones(n_bc, dtype=V.dtype)])
    indices = np.stack([np.asarray(I_f), np.asarray(J_f)], axis=1)
    return BCOO((data, indices), shape=(n, n))


def _bcoo_diag(A_bcoo):
    """Dense diagonal of a BCOO matrix (duplicate diagonal entries summed)."""
    row, col = A_bcoo.indices[:, 0], A_bcoo.indices[:, 1]
    diag_entries = np.where(row == col, A_bcoo.data, np.zeros_like(A_bcoo.data))
    d = np.zeros(A_bcoo.shape[0], dtype=A_bcoo.dtype)
    return d.at[row].add(diag_entries)


def _scale_bcoo(A_bcoo, s):
    """Return BCOO ``D A D`` with ``D = diag(s)``: ``data[i] *= s[r] s[c]``."""
    row, col = A_bcoo.indices[:, 0], A_bcoo.indices[:, 1]
    return BCOO((A_bcoo.data * s[row] * s[col], A_bcoo.indices),
                shape=A_bcoo.shape)


def bcoo_solve_lineax(A_bcoo, b, x0, linear_options):
    """Traceable lineax linear solve on a BCOO matrix.

    Mirrors :func:`lineax_solve` but consumes the BCOO directly (no scipy CSR
    round-trip).  Jacobi preconditioning scales the operator as
    ``D^-1/2 A D^-1/2`` via :func:`_scale_bcoo`.
    """
    solver_name = linear_options.get('solver', 'auto')
    rtol = linear_options.get('rtol', 1e-10)
    atol = linear_options.get('atol', 1e-10)
    max_steps = linear_options.get('max_steps', 10000)
    jacobi = linear_options.get('jacobi', True)
    if jacobi and solver_name in ('cg', 'bicgstab'):
        rtol = min(rtol, 1e-13)
        atol = min(atol, 1e-13)

    if solver_name == 'cg':
        solver = lineax.CG(rtol=rtol, atol=atol, max_steps=max_steps)
    elif solver_name == 'bicgstab':
        solver = lineax.BiCGStab(rtol=rtol, atol=atol, max_steps=max_steps)
    elif solver_name == 'lu':
        solver = lineax.LU()
    elif solver_name == 'cholesky':
        solver = lineax.Cholesky()
    else:
        solver = lineax.AutoLinearSolver(well_posed=True)

    x0 = np.zeros_like(b) if x0 is None else x0

    if solver_name in ('cg', 'cholesky'):
        tags = frozenset({lineax.symmetric_tag, lineax.positive_semidefinite_tag})
    else:
        tags = frozenset()

    direct = solver_name in ('lu', 'cholesky', 'auto')
    if direct:
        operator = lineax.MatrixLinearOperator(A_bcoo.todense(), tags=tags)
        solution = lineax.linear_solve(operator, b - operator.mv(x0), solver=solver)
        x = x0 + solution.value
    elif jacobi:
        d = _bcoo_diag(A_bcoo)
        dinv_sqrt = 1.0 / np.sqrt(d)
        As = _scale_bcoo(A_bcoo, dinv_sqrt)
        operator = lineax.FunctionLinearOperator(
            _SparseMatvec(As),
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            tags=tags,
            closure_convert=False,
        )
        r0 = b - A_bcoo @ x0
        solution = lineax.linear_solve(operator, np.asarray(dinv_sqrt) * r0, solver=solver)
        x = x0 + np.asarray(dinv_sqrt) * solution.value
    else:
        operator = lineax.FunctionLinearOperator(
            _SparseMatvec(A_bcoo),
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            tags=tags,
            closure_convert=False,
        )
        solution = lineax.linear_solve(operator, b - operator.mv(x0), solver=solver)
        x = x0 + solution.value
    return x


def bcoo_solve_feax(A_bcoo, b, x0, linear_options):
    """Traceable feax linear solve on a BCOO matrix (Krylov only).

    feax direct solvers consume a CSR triple, which cannot be built from a
    traced BCOO cheaply; the traceable path therefore supports the Krylov
    backends (default biCGStab + Jacobi).
    """
    x0 = np.zeros_like(b) if x0 is None else x0

    solver_options = linear_options.get(
        'options',
        feax.KrylovSolverOptions(solver='bicgstab', tol=1e-10, atol=1e-10,
                                 maxiter=10000, use_jacobi_preconditioner=True),
    )

    if isinstance(solver_options, feax.KrylovSolverOptions) and solver_options.solver == 'auto':
        solver_options = feax.resolve_iterative_solver(
            solver_options, feax.MatrixProperty.SPD)
    if isinstance(solver_options, feax.DirectSolverOptions):
        raise NotImplementedError(
            "feax direct solvers are not supported on the traceable BCOO path; "
            "use a KrylovSolverOptions (e.g. bicgstab) instead.")

    solve_fn = feax.create_linear_solve_fn(solver_options)
    return solve_fn(A_bcoo, b, x0)


def linear_solver_bcoo(A_bcoo, b, x0, linear_options):
    """Traceable dispatch: linear solve on a BCOO matrix (jax-native backends).

    Same key layout as :func:`linear_solver`, but every backend consumes the
    BCOO directly so the call is ``jax.jit``-compatible.  scipy spsolve is
    rejected (it would break the trace).  With no solver key, defaults to
    lineax cg.
    """
    options = dict(linear_options)

    if len(options.keys() & _LINEAR_SOLVER_KEYS) == 0:
        options['lineax_solver'] = {'solver': 'cg'}

    if 'lineax_solver' in options:
        return bcoo_solve_lineax(A_bcoo, b, x0, options['lineax_solver'])
    if 'feax_solver' in options:
        return bcoo_solve_feax(A_bcoo, b, x0, options['feax_solver'])
    if 'spsolve_solver' in options:
        raise NotImplementedError(
            "scipy spsolve is not traceable; use a jax-native backend "
            "('lineax_solver' or 'feax_solver') in the jitted path.")
    raise NotImplementedError(f"Unknown linear solver.")


def lineax_solve(A, b, x0, linear_options):
    """Lineax linear solve on the global tangent.

    lineax 0.1.1 ships no sparse linear operator.  Direct solvers
    ('lu', 'cholesky') must densify into a ``MatrixLinearOperator`` — fine for
    coarse meshes (~2k scalar DOFs).  Iterative solvers ('cg', 'bicgstab')
    use a sparse matvec operator built with ``FunctionLinearOperator``
    wrapping a JAX ``BCOO``, keeping O(n) memory so they scale to fine meshes.
    ``gmres`` is deliberately unsupported: lineax 0.1.1's GMRES calls
    ``jax.lax.linalg.ormqr``, removed in jax 0.9.2.
    ``lineax.linear_solve`` takes no initial guess, so the BC-corrected ``x0``
    is folded in via the shifted solve ``A (x0 + d) = b``.

    Parameters
    ----------
    A : scipy.sparse.csr_array
        Global tangent assembled by ``get_A`` (Dirichlet rows eliminated).
    b : JaxArray
        RHS with Dirichlet BC applied.
    x0 : JaxArray
        Initial guess (BC-corrected).
    linear_options : dict
        ``solver``: 'cg' | 'bicgstab' | 'lu' | 'cholesky' | 'auto' (default 'auto').
        Iterative keys: ``rtol`` (1e-10), ``atol`` (1e-10), ``max_steps`` (10000),
        ``jacobi`` (True): diagonal preconditioning for cg/bicgstab.
        The sparse matvec uses ``_csr_to_bcoo`` (~1 ms vs ~0.4 s for the
        ``from_scipy_sparse``+``sort_indices`` path) and ``_SparseMatvec`` with
        ``closure_convert=False``, so the compiled solve is reused across every
        matrix of the same sparsity structure (see those docstrings).
    """
    solver_name = linear_options.get('solver', 'auto')
    rtol = linear_options.get('rtol', 1e-10)
    atol = linear_options.get('atol', 1e-10)
    max_steps = linear_options.get('max_steps', 10000)
    # Jacobi (diagonal) preconditioning for the iterative solvers: solves
    # (D^-1/2 A D^-1/2) dy = D^-1/2 r0 with D = diag(A).  It collapses the
    # mu_r=1680 core/air contrast, dropping CG from ~10k to a few hundred
    # iterations on fine meshes (measured: 5.8s -> 0.06s at 0.5mm).  The
    # D^-1/2 scaling also inflates the unscaled residual by ~1e5, so the
    # effective tolerance is capped tight enough that the Newton loop (tol
    # ~1e-6) still terminates; at rtol=1e-10 the Newton loop grinds
    # (~50s at 0.5mm), at 1e-13 it converges in one step (~0.8s).
    jacobi = linear_options.get('jacobi', True)
    if jacobi and solver_name in ('cg', 'bicgstab'):
        rtol = min(rtol, 1e-13)
        atol = min(atol, 1e-13)

    if solver_name == 'gmres':
        raise ValueError(
            "lineax GMRES is unsupported: lineax 0.1.1's GMRES calls "
            "jax.lax.linalg.ormqr, which was removed in jax 0.9.2. "
            "Use 'cg', 'bicgstab', 'lu', 'cholesky' or 'auto' instead.")
    if solver_name == 'cg':
        solver = lineax.CG(rtol=rtol, atol=atol, max_steps=max_steps)
    elif solver_name == 'bicgstab':
        solver = lineax.BiCGStab(rtol=rtol, atol=atol, max_steps=max_steps)
    elif solver_name == 'lu':
        solver = lineax.LU()
    elif solver_name == 'cholesky':
        solver = lineax.Cholesky()
    else:
        solver = lineax.AutoLinearSolver(well_posed=True)

    logger.debug("Lineax Solver - Solving linear system with %s", type(solver).__name__)

    x0 = np.zeros_like(b) if x0 is None else x0

    # The magnetostatic tangent is symmetric positive definite (after the
    # Dirichlet row/column elimination in ``get_A``). ``MatrixLinearOperator``
    # defaults to the "Any" tag, which makes CG/Cholesky reject the operator, so
    # advertise the SPD tag when one of those solvers is selected.
    if solver_name in ('cg', 'cholesky'):
        tags = frozenset({lineax.symmetric_tag, lineax.positive_semidefinite_tag})
    else:
        tags = frozenset()

    direct = solver_name in ('lu', 'cholesky', 'auto')
    if direct:
        operator = lineax.MatrixLinearOperator(np.asarray(A.toarray()), tags=tags)
        solution = lineax.linear_solve(operator, b - operator.mv(x0), solver=solver)
        x = x0 + solution.value
        err = np.linalg.norm(operator.mv(x) - b)
    else:
        A_bcoo = _csr_to_bcoo(A)
        if jacobi:
            # diagonal preconditioning: x0-folded shifted system becomes
            # (D^-1/2 A D^-1/2) dy = D^-1/2 (b - A x0),  x = x0 + D^-1/2 dy.
            d = A.diagonal()
            dinv_sqrt = 1.0 / onp.sqrt(d)
            As = A.multiply(dinv_sqrt[None, :]).multiply(dinv_sqrt[:, None]).tocsr()
            operator = lineax.FunctionLinearOperator(
                _SparseMatvec(_csr_to_bcoo(As)),
                jax.ShapeDtypeStruct(b.shape, b.dtype),
                tags=tags,
                closure_convert=False,
            )
            r0 = b - A_bcoo @ x0
            solution = lineax.linear_solve(operator, np.asarray(dinv_sqrt) * r0, solver=solver)
            x = x0 + np.asarray(dinv_sqrt) * solution.value
        else:
            operator = lineax.FunctionLinearOperator(
                _SparseMatvec(A_bcoo),
                jax.ShapeDtypeStruct(b.shape, b.dtype),
                tags=tags,
                closure_convert=False,
            )
            solution = lineax.linear_solve(operator, b - operator.mv(x0), solver=solver)
            x = x0 + solution.value
        err = np.linalg.norm(A_bcoo @ x - b)

    logger.debug("Lineax Solver - Finished solving, linear solve res = %.3g", err)
    rel_err = err / (np.linalg.norm(b) + 1e-30)
    assert rel_err < 1e-6, (
        f"Lineax linear solver failed to converge with rel_err = {rel_err}")
    return x
def feax_solve(A, b, x0, linear_options):
    """feax linear solve on the global tangent.

    Minimal-port path: our ``get_A`` (scipy CSR, Dirichlet rows already
    identity) is handed to feax unchanged — direct solvers consume it as a
    ``feax.csr.CSRMatrix``, iterative solvers as a ``BCOO``. feax only
    performs the matrix solve; BC row-elimination stays in ``get_A``.

    Parameters
    ----------
    A : scipy.sparse.csr_array
        Global tangent assembled by ``get_A`` (Dirichlet rows eliminated).
    b : JaxArray
        RHS with Dirichlet BC applied.
    x0 : JaxArray
        Initial guess (BC-corrected).
    linear_options : dict
    ``options``: a ``feax.KrylovSolverOptions`` or ``feax.DirectSolverOptions``
    instance. Default is biCGStab + Jacobi.
    """
    # A may be CSC (e.g. A^T from ``A.transpose()``); feax.csr.CSRMatrix expects
    # a true CSR triple, so coerce first. Otherwise the matrix is misread and
    # the solve silently returns garbage.
    A = A.tocsr()
    x0 = np.zeros_like(b) if x0 is None else x0

    solver_options = linear_options.get(
        'options',
        feax.KrylovSolverOptions(solver='bicgstab', tol=1e-10, atol=1e-10,
                                 maxiter=10000, use_jacobi_preconditioner=True),
    )

    if isinstance(solver_options, feax.DirectSolverOptions) and solver_options.solver == 'auto':
        solver_options = feax.resolve_direct_solver(
            solver_options, feax.MatrixProperty.SPD, feax.MatrixView.FULL)
    elif isinstance(solver_options, feax.KrylovSolverOptions) and solver_options.solver == 'auto':
        solver_options = feax.resolve_iterative_solver(
            solver_options, feax.MatrixProperty.SPD)

    solve_fn = feax.create_linear_solve_fn(solver_options)
    logger.debug("Feax Solver - Solving linear system with %s", type(solver_options).__name__)

    if isinstance(solver_options, feax.DirectSolverOptions):
        A_f = feax.csr.CSRMatrix(
            onp.asarray(A.data),
            onp.asarray(A.indptr),
            onp.asarray(A.indices),
            A.shape,
        )
    else:
        A_f = _csr_to_bcoo(A)

    x = solve_fn(A_f, b, x0)

    err = np.linalg.norm(A_f @ x - b)
    logger.debug("Feax Solver - Finished solving, linear solve res = %.3g", err)
    assert err < 0.1, f"Feax linear solver failed to converge with err = {err}"
    return x

_LINEAR_SOLVER_KEYS = frozenset({'spsolve_solver', 'lineax_solver', 'feax_solver', 'custom_solver'})


def linear_solver(A, b, x0, linear_options):
    # Copy so the caller's dict is never mutated by the default injection below.
    options = dict(linear_options)

    # If user does not specify any solver, set spsolve as the default one.
    if len(options.keys() & _LINEAR_SOLVER_KEYS) == 0:
        options['spsolve_solver'] = {}

    if 'spsolve_solver' in options:
        x = scipy_spsolve(A, b)
    elif 'lineax_solver' in options:
        x = lineax_solve(A, b, x0, options['lineax_solver'])
    elif 'feax_solver' in options:
        x = feax_solve(A, b, x0, options['feax_solver'])
    elif 'custom_solver' in options:
        x = options['custom_solver'](A, b, x0, options)
    else:
        raise NotImplementedError(f"Unknown linear solver.")

    return x

################################################################################
# Dirichlet boundary conditions ("row elimination")

def apply_bc_vec(res_vec, dofs, problem, scale=1.):
    res_list = problem.unflatten_fn_sol_list(res_vec)
    sol_list = problem.unflatten_fn_sol_list(dofs)

    for ind, fe in enumerate(problem.fes):
        res = res_list[ind]
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            res = (res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]], unique_indices=True))
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].add(-fe.vals_list[i]*scale)

        res_list[ind] = res

    return jax.flatten_util.ravel_pytree(res_list)[0]


def apply_bc(res_fn, problem, scale=1.):
    def res_fn_bc(dofs):
        """Apply Dirichlet boundary conditions
        """
        res_vec = res_fn(dofs)
        return apply_bc_vec(res_vec, dofs, problem, scale)
    return res_fn_bc


def assign_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(fe.vals_list[i])
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def assign_ones_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(1.)
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def assign_zeros_bc(dofs, problem):
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i],
                         fe.vec_inds_list[i]].set(0.)
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def copy_bc(dofs, problem):
    new_dofs = np.zeros_like(dofs)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    new_sol_list = problem.unflatten_fn_sol_list(new_dofs)
  
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        new_sol = new_sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            new_sol = (new_sol.at[fe.node_inds_list[i],
                                  fe.vec_inds_list[i]].set(sol[fe.node_inds_list[i],
                                          fe.vec_inds_list[i]]))
        new_sol_list[ind] = new_sol

    return jax.flatten_util.ravel_pytree(new_sol_list)[0]


################################################################################
# Newton helpers: flattening and tangent probe

def get_flatten_fn(fn_sol_list, problem):

    def fn_dofs(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        val_list = fn_sol_list(sol_list)
        return jax.flatten_util.ravel_pytree(val_list)[0]

    return fn_dofs


def operator_to_matrix(operator_fn, problem):
    """Only used for when debugging.
    Can be used to print the matrix, check the conditional number, etc.
    """
    J = jax.jacfwd(operator_fn)(np.zeros(problem.num_total_dofs_all_vars))
    return J


def _bc_dofs(problem):
    """Global DOF indices constrained by Dirichlet B.C.

    'unique' so a corner node shared by two boundary sets is not added twice
    (which would sum to a 2 on the diagonal under coo->csr duplicate handling).

    The node/vec index lists are JAX arrays; they are converted with
    ``onp.asarray`` *before* any arithmetic so this helper stays pure-numpy and
    can run inside a ``jax.jit`` trace (JAX ops on concrete arrays otherwise
    create tracers that numpy rejects).
    """
    return onp.unique(onp.concatenate([
        onp.asarray(fe.node_inds_list[i]) * fe.vec + onp.asarray(fe.vec_inds_list[i])
        for fe in problem.fes for i in range(len(fe.node_inds_list))
    ]))


def get_A(problem, V=None):
    """Assemble the global tangent stiffness as a scipy CSR matrix.

    ``Problem.newton_update`` now *returns* the flat per-cell Jacobian entries
    ``V`` (row/column indices are ``problem.I``/``problem.J``); pass it here.
    If omitted, falls back to ``problem.V`` for backward compatibility with
    tests and legacy callers.

    Dirichlet DOFs are eliminated on the raw ``(I, J, V)`` triple *before* the
    sparse matrix is built: constrained rows and columns are dropped (equivalent
    to zeroing) and a unit diagonal is appended.  This avoids the LIL
    fancy-indexed assignment ``A[bc, :] = 0`` etc., which measured ~0.38 s of
    the 0.42 s 'global_matrix' phase at the 0.5 mm mesh.  The result is
    symmetric positive definite, which the CG/Cholesky backends require.
    """
    n = problem.num_total_dofs_all_vars
    if V is None:
        V = problem.V
    I = onp.asarray(problem.I)
    J = onp.asarray(problem.J)
    V = onp.asarray(V)

    bc_dofs = _bc_dofs(problem)
    keep = ~(onp.isin(I, bc_dofs) | onp.isin(J, bc_dofs))
    I = onp.concatenate([I[keep], bc_dofs])
    J = onp.concatenate([J[keep], bc_dofs])
    V = onp.concatenate([V[keep], onp.ones_like(bc_dofs, dtype=V.dtype)])

    return scipy.sparse.csr_matrix((V, (I, J)), shape=(n, n))


def newton_step(problem, res_vec, A, dofs, newton_cfg, timing):
    """One Newton correction: solve A \\Delta u = -R, then update ``dofs``.

    Returns
    -------
    dofs : ndarray
    linear_s : float
        Linear solve wall time (also accumulated in ``timing``).
    """
    logger.debug(f"Solving linear system...")
    b = -res_vec

    # x0 will always be correct at boundary locations
    x0_1 = assign_bc(np.zeros(problem.num_total_dofs_all_vars), problem)
    x0_2 = copy_bc(dofs, problem)
    x0 = x0_1 - x0_2

    t0 = time.perf_counter()
    inc = linear_solver(A, b, x0, newton_cfg.get('linear', {}))
    linear_s = time.perf_counter() - t0
    _timing_record(timing, 'linear', linear_s)

    if newton_cfg.get('line_search_flag', False):
        dofs = line_search(problem, dofs, inc)
    else:
        dofs = dofs + inc

    return dofs, linear_s

def line_search(problem, dofs, inc):
    """Backtracking line search on the L2 residual norm.

    Only needed for strongly nonlinear problems, dormant for the linear magnetostatics solve.
    """
    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem)

    def res_norm_fn(alpha):
        res_vec = res_fn(dofs + alpha*inc)
        return np.linalg.norm(res_vec)

    alpha = 1.
    res_norm = res_norm_fn(alpha)
    for i in range(3):
        alpha *= 0.5
        res_norm_half = res_norm_fn(alpha)
        logger.debug(f"i = {i}, res_norm = {res_norm}, res_norm_half = {res_norm_half}")
        if res_norm_half > res_norm:
            alpha *= 2.
            break
        res_norm = res_norm_half

    return dofs + alpha*inc


################################################################################
# solver_options registry and dispatch
#
# Top level: 'newton' block, or a legacy flat dict (auto-wrapped as Newton).
#
#   {'newton': {
#       'tol': 1e-6, 'rel_tol': 1e-8, 'line_search_flag': False,
#       'initial_guess': sol_list,
#       'linear': {'lineax_solver': {'solver': 'bicgstab'}},
#   }}

_METHOD_KEYS = frozenset({'newton'})
_LINEAR_OPTION_KEYS = frozenset({
    'spsolve_solver', 'lineax_solver', 'feax_solver', 'custom_solver',
})
_NEWTON_OPTION_KEYS = frozenset({'tol', 'rel_tol', 'line_search_flag', 'initial_guess'})


def _resolve_solver_options(solver_options):
    """Return (nonlinear_method, method_cfg). Legacy flat dicts become Newton."""
    opts = solver_options or {}
    methods = [m for m in _METHOD_KEYS if m in opts]

    if not methods:
        linear = {k: opts[k] for k in _LINEAR_OPTION_KEYS if k in opts}
        cfg = {k: opts[k] for k in _NEWTON_OPTION_KEYS if k in opts}
        if linear:
            cfg['linear'] = linear
        return 'newton', cfg

    if len(methods) > 1:
        raise ValueError(f"Pick one nonlinear method, got {methods}.")

    method = methods[0]
    if not isinstance(opts[method], dict):
        raise ValueError(f"solver_options['{method}'] must be a dict.")
    return method, opts[method]
def _with_traceable(solver_options):
    """Deep-copy ``solver_options`` with ``newton.traceable`` set to True."""
    import copy
    opts = copy.deepcopy(solver_options) if solver_options else {}
    method, _ = _resolve_solver_options(opts)
    opts.setdefault(method, {})['traceable'] = True
    return opts


def solver(problem, solver_options={}, return_timing=False):
    r"""Solve the (possibly nonlinear) problem with Newton's method.

    Dirichlet B.C. are imposed by "row elimination". Conceptually,

    .. math::
        r(u) = D \, r_{\text{unc}}(u) + (I - D)u - u_b \\
        A = \frac{\text{d}r}{\text{d}u} = D \frac{\text{d}r}{\text{d}u} + (I - D)

    where :math:`D` masks the constrained DOFs, :math:`u_b` holds the
    Dirichlet values, and :math:`A` is the tangent stiffness (global
    Jacobian). For a linear problem a single Newton step suffices.

    Parameters
    ----------
    problem : Problem
        The problem to solve.
    solver_options : dict
        A ``{'newton': {...}}`` block, or a legacy flat dict. Newton keys:

        - ``tol`` → ``1e-6`` (absolute residual :math:`\ell_2` norm)
        - ``rel_tol`` → ``1e-8`` (relative to the initial residual)
        - ``line_search_flag`` → ``False``
        - ``initial_guess`` → zero vector
        - ``linear``: exactly one linear backend:

          - ``{'spsolve_solver': {}}`` — SciPy direct (SuperLU/UMFPACK)
          - ``{'lineax_solver': {'solver': 'bicgstab'}}`` — lineax
          - ``{'feax_solver': {'options': <KrylovSolverOptions|DirectSolverOptions>}}`` — feax
          - ``{'custom_solver': fn}``

        All of the following are equivalent::

            solver_options = {}
            solver_options = {'newton': {}}
            solver_options = {'newton': {'linear': {}}}
            solver_options = {'newton': {'linear': {'spsolve_solver': {}}}}

    Returns
    -------
    sol_list : list
        If ``return_timing`` is True, returns ``(sol_list, timing)`` where
        ``timing`` is a dict with the ``local_assembly``, ``global_matrix``
        and ``linear`` wall times in seconds.
    """
    method, cfg = _resolve_solver_options(solver_options)
    assert method == 'newton'

    if cfg.get('traceable'):
        # JAX-native path: fixed-iteration linear Newton on BCOO, safe under
        # ``jax.jit`` (no scipy objects, no Python control flow, no logging).
        sol_list = solver_jax(problem, cfg)
        if return_timing:
            return sol_list, {'local_assembly': 0., 'global_matrix': 0., 'linear': 0.}
        return sol_list

    logger.info("Solving the nonlinear problem...")
    timing = {'local_assembly': 0., 'global_matrix': 0., 'linear': 0.}
    wall_start = time.perf_counter()

    if 'initial_guess' in cfg:
        # Don't let the initial guess enter the differentiation chain.
        initial_guess = jax.lax.stop_gradient(cfg['initial_guess'])
        dofs = jax.flatten_util.ravel_pytree(initial_guess)[0]
    else:
        dofs = np.zeros(problem.num_total_dofs_all_vars)

    rel_tol = cfg.get('rel_tol', 1e-8)
    tol = cfg.get('tol', 1e-6)

    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        t0 = time.perf_counter()
        res_list, V = problem.newton_update(sol_list)
        local_s = time.perf_counter() - t0
        _timing_record(timing, 'local_assembly', local_s)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)

        t0 = time.perf_counter()
        A = get_A(problem, V)
        global_s = time.perf_counter() - t0
        _timing_record(timing, 'global_matrix', global_s)
        return res_vec, A, local_s, global_s

    _log_newton_iter_start(0)
    res_vec, A, local_s, global_s = newton_update_helper(dofs)
    res_val = np.linalg.norm(res_vec)
    res_val_initial = res_val
    rel_res_val = res_val/res_val_initial
    _log_newton_iter_summary(0, local_s, global_s, res_val, rel_res_val)
    n_iters = 0
    while (rel_res_val > rel_tol) and (res_val > tol):
        n_iters += 1
        _log_newton_iter_start(n_iters)
        dofs, linear_s = newton_step(problem, res_vec, A, dofs, cfg, timing)
        res_vec, A, local_s, global_s = newton_update_helper(dofs)
        res_val = np.linalg.norm(res_vec)
        rel_res_val = res_val/res_val_initial
        _log_newton_iter_summary(n_iters, local_s, global_s, res_val, rel_res_val, linear_s)

    assert np.all(np.isfinite(res_val)), f"res_val contains NaN, stop the program!"
    assert np.all(np.isfinite(dofs)), f"dofs contains NaN, stop the program!"

    sol_list = problem.unflatten_fn_sol_list(dofs)

    _log_timing_table(n_iters, timing, time.perf_counter() - wall_start)

    logger.info(f"max of dofs = {np.max(dofs)}")
    logger.info(f"min of dofs = {np.min(dofs)}")

    if return_timing:
        return sol_list, timing
    return sol_list


def solver_jax(problem, cfg):
    """Traceable forward solve for linear problems.

    The magnetostatic tangent does not depend on the state (``kernel_jac``
    builds it from the material field alone), so Newton collapses to a single
    solve: assemble ``A`` once, form ``b`` from the BC-applied residual at
    ``u0 = 0``, and solve ``A u = b`` on BCOO with a jax-native backend.
    No scipy objects, Python control flow, or wall-clock logging, so the whole
    function may be traced inside ``jax.jit``.

    """
    dofs = np.zeros(problem.num_total_dofs_all_vars)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    res_list, V = problem.newton_update(sol_list)
    res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
    res_vec = apply_bc_vec(res_vec, dofs, problem)

    A = get_A_bcoo(problem, V)
    b = -res_vec
    linear = cfg.get('linear', {})
    inc = linear_solver_bcoo(A, b, np.zeros_like(b), linear)
    dofs = dofs + inc
    return problem.unflatten_fn_sol_list(dofs)


def _constraint_fn(problem, dofs, params):
    """BC-applied residual ``c(u, p)`` at given state and params."""
    problem.set_params(params)
    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem)
    return res_fn(dofs)


def _vjp_contraint_fn_params(problem, params, sol_list):
    """``v -> v * (partial dc/dp)`` for the adjoint seed ``v``."""
    def constraint_fn_sol_to_sol(sol_list, params):
        dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
        con_vec = _constraint_fn(problem, dofs, params)
        return problem.unflatten_fn_sol_list(con_vec)

    def partial_params_c_fn(params):
        return constraint_fn_sol_to_sol(sol_list, params)

    def vjp_linear_fn(v_list):
        primals_output, f_vjp = jax.vjp(partial_params_c_fn, params)
        val, = f_vjp(v_list)
        return val

    return vjp_linear_fn


def _solve_adjoint(A, v_vec, adjoint_solver_options):
    """Solve ``A^T z = v`` (adjoint), normalizing the seed.

    The loss seed is O(1e11) for nH^2 losses, which makes absolute-tolerance
    stopping in iterative solvers run to max_steps (or fail absolute-residual
    asserts).  The adjoint system is linear, so solving ``A^T z = v/||v||`` and
    scaling back is exact.  ``A`` is a scipy CSR (eager) or BCOO (traceable);
    the scipy path uses ``linear_solver`` on the CSR transpose, the BCOO path
    uses ``linear_solver_bcoo`` on ``A.T``.
    """
    v_vec = jax.flatten_util.ravel_pytree(v_vec)[0]
    v_norm = np.linalg.norm(v_vec)
    v_norm_safe = np.where(v_norm > 0.0, v_norm, 1.0)
    v_vec_s = v_vec / v_norm_safe

    if isinstance(A, BCOO):
        A_T = A.T
        adjoint_vec = linear_solver_bcoo(A_T, v_vec_s, np.zeros_like(v_vec_s),
                                         adjoint_solver_options)
    else:
        # scipy .transpose() returns a new matrix (no in-place mutation) — A intact
        A_T = A.transpose()
        adjoint_vec = linear_solver(A_T, v_vec_s, np.zeros_like(v_vec_s),
                                    adjoint_solver_options)
    return adjoint_vec * v_norm_safe


def implicit_vjp(problem, sol_list, params, v_list, adjoint_solver_options):
    problem.set_params(params)
    _, V = problem.newton_update(sol_list)

    A = get_A(problem, V)
    adjoint_vec = _solve_adjoint(A, v_list, adjoint_solver_options)

    vjp_linear_fn = _vjp_contraint_fn_params(problem, params, sol_list)
    vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
    vjp_result = jax.tree_util.tree_map(lambda x: -x, vjp_result)

    return vjp_result


def implicit_vjp_bcoo(problem, sol_list, params, v_list, adjoint_solver_options):
    """Traceable adjoint: same as :func:`implicit_vjp` but fully jax-native.

    The forward solve is linear, so ``A`` does not depend on the state and the
    re-assembly here matches the one in :func:`solver_jax`.  The adjoint solve
    runs on ``A.T`` as a BCOO with a jax-native backend (``linear_solver_bcoo``),
    keeping the whole backward pass free of scipy objects for ``jax.jit``.
    """
    problem.set_params(params)
    _, V = problem.newton_update(sol_list)

    A = get_A_bcoo(problem, V)
    adjoint_vec = _solve_adjoint(A, v_list, adjoint_solver_options)

    vjp_linear_fn = _vjp_contraint_fn_params(problem, params, sol_list)
    vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
    vjp_result = jax.tree_util.tree_map(lambda x: -x, vjp_result)

    return vjp_result
def ad_wrapper(problem, solver_options={}, adjoint_solver_options={}, traceable=False):
    """Automatic differentiation wrapper for the forward problem.

    Parameters
    ----------
    problem : Problem
    solver_options : dict
        Same layout as :func:`solver` (Newton + nested ``linear``).  Set
        ``solver_options['newton']['traceable'] = True`` (or pass
        ``traceable=True``) to select the jax-native BCOO forward.
    adjoint_solver_options : dict
        Linear solver options for the adjoint solve only (flat dict, e.g.
        ``{'lineax_solver': {'solver': 'bicgstab'}}``).
    traceable : bool
        When True, both the forward (``solver_jax``) and the adjoint
        (``implicit_vjp_bcoo``) run entirely on BCOO with jax-native backends,
        so ``fwd_pred`` can be traced inside ``jax.jit``.  Requires
        jax-native linear backends (lineax/feax), never scipy.

    Returns
    -------
    fwd_pred : callable
    """
    if traceable:
        solver_options = _with_traceable(solver_options)

    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        sol_list = solver(problem, solver_options)
        return sol_list

    def f_fwd(params):
        sol_list = fwd_pred(params)
        return sol_list, (params, sol_list)

    def f_bwd(res, v):
        logger.info("Running backward and solving the adjoint problem...")
        params, sol_list = res
        if traceable:
            vjp_result = implicit_vjp_bcoo(
                problem, sol_list, params, v, adjoint_solver_options)
        else:
            vjp_result = implicit_vjp(
                problem, sol_list, params, v, adjoint_solver_options)
        return (vjp_result, )

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred