import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import scipy.sparse as sp

import feax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from new_fem_engine.generate_mesh import rectangle_mesh, create_mesh
from new_fem_engine.problem import MagnetostaticProblem
import new_fem_engine.solver as slv


CORE_SPEC = {
    "params": {
        "center_post_diameter": 0.015,
        "leg_inner_diameter": 0.0275,
        "leg_height": 0.044,
        "window_height": 0.0294,
        "coil_clearance": 0.0005,
        "mur": 1680.0,
        "A_e": 122.6e-6,
        "l_e": 44.4e-3,
        "sigma_l_over_A": 362.0,
        "AL": 4300.0,
        "material": "N87",
    }
}


def make_problem(Nr=6, Nz=8):
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
        additional_info=(CORE_SPEC, 0.001),
    )
    prob.set_params(jnp.ones((prob.num_cells, 4)))
    return prob


@pytest.fixture(scope="module")
def prob():
    return make_problem()


def reset_params(prob):
    prob.set_params(jnp.ones((prob.num_cells, 4)))


def dirichlet_dofs(prob):
    fe = prob.fes[0]
    return np.concatenate(
        [np.asarray(fe.node_inds_list[i]) * fe.vec + np.asarray(fe.vec_inds_list[i])
         for i in range(len(fe.node_inds_list))]
    )


FORWARD_BACKENDS = [
    ("spsolve", {"newton": {"linear": {"spsolve_solver": {}}}}),
    ("lineax-bicgstab", {"newton": {"linear": {"lineax_solver": {"solver": "bicgstab"}}}}),
    ("lineax-cg", {"newton": {"linear": {"lineax_solver": {"solver": "cg"}}}}),
    ("lineax-lu", {"newton": {"linear": {"lineax_solver": {"solver": "lu"}}}}),
    ("lineax-auto", {"newton": {"linear": {"lineax_solver": {}}}}),
    ("feax-direct", {"newton": {"linear": {"feax_solver": {"options": feax.DirectSolverOptions(solver="auto")}}}}),
    ("feax-krylov", {"newton": {"linear": {"feax_solver": {}}}}),
]

ADJOINT_BACKENDS = [
    ("spsolve", {"spsolve_solver": {}}),
    ("lineax-cg", {"lineax_solver": {"solver": "cg"}}),
    ("feax-direct", {"feax_solver": {"options": feax.DirectSolverOptions(solver="auto")}}),
    ("feax-krylov", {"feax_solver": {}}),
]


def test_get_A_symmetric_spd(prob):
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    A = slv.get_A(prob)
    assert A.shape == (prob.num_total_dofs_all_vars, prob.num_total_dofs_all_vars)
    assert sp.issparse(A)
    Ad = A.toarray()
    assert np.allclose(Ad, Ad.T, atol=1e-6)
    assert np.linalg.eigvalsh(Ad).min() > 0


def test_get_A_dirichlet_rows_unit(prob):
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    A = slv.get_A(prob).toarray()
    identity = np.eye(A.shape[0])
    for d in dirichlet_dofs(prob):
        assert np.allclose(A[d], identity[d], atol=1e-8)


def test_newton_jacobian_matches_jacfwd(prob):
    """problem.V/I/J (the get_A data source) is the true residual Jacobian."""
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    n = prob.num_total_dofs_all_vars
    M = sp.coo_matrix((np.asarray(prob.V), (np.asarray(prob.I), np.asarray(prob.J))),
                      shape=(n, n)).toarray()

    def raw_res(x):
        return jax.flatten_util.ravel_pytree(prob.compute_residual([jnp.asarray(x)[:, None]]))[0]

    J = np.asarray(jax.jacfwd(raw_res)(np.zeros(n)))
    np.testing.assert_allclose(M, J, atol=1e-6)


@pytest.mark.parametrize("name,opts", FORWARD_BACKENDS)
def test_solver_backend_forward(prob, name, opts):
    reset_params(prob)
    sol = slv.solver(prob, opts)
    psi = np.asarray(sol[0]).ravel()
    assert psi.shape == (prob.fes[0].num_total_nodes,)
    assert np.all(np.isfinite(psi))
    assert np.max(np.abs(psi)) > 0
    assert np.allclose(psi[dirichlet_dofs(prob)], 0.0, atol=1e-10)


def test_solver_backends_agree(prob):
    ref = None
    for name, opts in FORWARD_BACKENDS:
        reset_params(prob)
        psi = np.asarray(slv.solver(prob, opts)[0]).ravel()
        if ref is None:
            ref = psi
        else:
            np.testing.assert_allclose(psi, ref, atol=1e-8)


def test_solver_empty_options_uses_spsolve_default(prob):
    reset_params(prob)
    psi = np.asarray(slv.solver(prob, {})[0]).ravel()
    assert np.all(np.isfinite(psi))
    assert np.max(np.abs(psi)) > 0


def test_solver_legacy_flat_options(prob):
    reset_params(prob)
    psi = np.asarray(slv.solver(prob, {"spsolve_solver": {}})[0]).ravel()
    assert np.all(np.isfinite(psi))
    assert np.max(np.abs(psi)) > 0


def test_linear_solver_does_not_mutate_options(prob):
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    A = slv.get_A(prob)
    b = np.ones(prob.num_total_dofs_all_vars)
    opts = {}
    slv.linear_solver(A, b, None, opts)
    assert opts == {}


def test_lineax_solve_matches_scipy(prob):
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    A = slv.get_A(prob)
    b = np.ones(prob.num_total_dofs_all_vars)
    xl = np.asarray(slv.lineax_solve(A, b, None, {"solver": "lu"}))
    xs = slv.scipy_spsolve(A, b)
    np.testing.assert_allclose(xl, xs, atol=1e-6)


def test_feax_solve_matches_scipy(prob):
    reset_params(prob)
    prob.newton_update([jnp.zeros((prob.fes[0].num_total_nodes, 1))])
    A = slv.get_A(prob)
    b = np.ones(prob.num_total_dofs_all_vars)
    xf = np.asarray(slv.feax_solve(A, b, None, {"options": feax.DirectSolverOptions(solver="auto")}))
    xs = slv.scipy_spsolve(A, b)
    np.testing.assert_allclose(xf, xs, atol=1e-6)


@pytest.mark.parametrize("name,adjopts", ADJOINT_BACKENDS)
def test_adjoint_backend_agrees_with_reference(prob, name, adjopts):
    reset_params(prob)
    ref = slv.ad_wrapper(prob, {"newton": {"linear": {"spsolve_solver": {}}}}, {"spsolve_solver": {}})
    fwd = slv.ad_wrapper(prob, {"newton": {"linear": {"spsolve_solver": {}}}}, adjopts)
    fill = jnp.ones((prob.num_cells, 4))

    def total(f):
        return jnp.sum(fwd(f)[0])

    grad = np.asarray(jax.grad(total)(fill))
    assert np.all(np.isfinite(grad))
    assert np.max(np.abs(grad)) > 0

    def total_ref(f):
        return jnp.sum(ref(f)[0])

    grad_ref = np.asarray(jax.grad(total_ref)(fill))
    np.testing.assert_allclose(grad, grad_ref, rtol=1e-6, atol=1e-10)


def test_ad_wrapper_gradient_matches_finite_difference(prob):
    reset_params(prob)
    fwd = slv.ad_wrapper(prob, {"newton": {"linear": {"spsolve_solver": {}}}}, {"spsolve_solver": {}})

    def total(fill):
        return jnp.sum(fwd(fill)[0])

    fill = jnp.ones((prob.num_cells, 4))
    grad = np.asarray(jax.grad(total)(fill))
    assert np.all(np.isfinite(grad))

    eps = 1e-4
    for idx in [(0, 0), (5, 2), (7, 3)]:
        e = np.zeros_like(np.asarray(fill))
        e[idx] = eps
        fp = float(total(jnp.asarray(np.asarray(fill) + e)))
        fm = float(total(jnp.asarray(np.asarray(fill) - e)))
        assert np.isclose(fp - fm, 2 * eps * grad[idx], rtol=1e-2)
