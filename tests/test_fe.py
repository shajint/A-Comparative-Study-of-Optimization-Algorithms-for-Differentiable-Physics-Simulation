import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from new_fem_engine.generate_mesh import rectangle_mesh, create_mesh
from new_fem_engine.fe import FiniteElement


def make_fe(Nr=3, Nz=4, vec=1):
    msh = rectangle_mesh(Nr, Nz, 0.0, 0.01, 0.0, 0.02)
    mesh = create_mesh(msh, "quad")
    return FiniteElement(mesh, vec=vec, dim=2, ele_type="QUAD4", dirichlet_bc_info=None)


def test_finite_element_attributes():
    fe = make_fe(3, 4)
    assert fe.num_cells == 12
    assert fe.num_total_nodes == 20
    assert fe.num_total_dofs == 20
    assert fe.num_quads == 4
    assert fe.num_nodes == 4
    assert fe.shape_vals.shape == (4, 4)
    assert fe.shape_grads_ref.shape == (4, 4, 2)
    assert fe.quad_weights.shape == (4,)


def test_shape_grads_shapes():
    fe = make_fe(3, 4)
    shape_grads_physical, JxW = fe.get_shape_grads()
    assert shape_grads_physical.shape == (12, 4, 4, 2)
    assert JxW.shape == (12, 4)


def test_jxw_sum_is_area():
    """THE anchor: sum of JxW over all cells/quads = physical domain area."""
    fe = make_fe(3, 4)
    _, JxW = fe.get_shape_grads()
    assert np.isclose(np.asarray(JxW).sum(), 0.01 * 0.02)


def test_shape_grads_partition_of_unity():
    """Sum of shape gradients over nodes = 0 (derivative of constant = 0)."""
    fe = make_fe(3, 4)
    shape_grads_physical, _ = fe.get_shape_grads()
    s = np.asarray(shape_grads_physical).sum(axis=2)  # (num_cells, num_quads, dim)
    assert np.allclose(s, 0.0, atol=1e-4)


def test_identity_mapping_unit_square():
    """Unit-square cell: physical grads == ref grads, JxW == weights."""
    msh = rectangle_mesh(1, 1, 0.0, 1.0, 0.0, 1.0)
    mesh = create_mesh(msh, "quad")
    fe = FiniteElement(mesh, vec=1, dim=2, ele_type="QUAD4")
    shape_grads_physical, JxW = fe.get_shape_grads()
    assert np.allclose(np.asarray(shape_grads_physical), np.asarray(fe.shape_grads_ref)[None, :, :, :])
    assert np.allclose(np.asarray(JxW), np.asarray(fe.quad_weights)[None, :])


def test_physical_quad_points():
    fe = make_fe(3, 4)
    pqp = fe.get_physical_quad_points()
    assert pqp.shape == (12, 4, 2)
    pts = np.asarray(pqp)
    assert pts[:, :, 0].min() >= 0.0 and pts[:, :, 0].max() <= 0.01
    assert pts[:, :, 1].min() >= 0.0 and pts[:, :, 1].max() <= 0.02


def test_dirichlet_boundary_conditions():
    msh = rectangle_mesh(2, 2, 0.0, 0.01, 0.0, 0.02)
    mesh = create_mesh(msh, "quad")
    fe = FiniteElement(
        mesh, vec=1, dim=2, ele_type="QUAD4",
        dirichlet_bc_info=[[lambda pt: jnp.isclose(pt[0], 0.0)], [0], [lambda pt: 0.0]],
    )
    node_inds = np.asarray(fe.node_inds_list[0])
    assert len(node_inds) == 3          # 3 nodes on r=0 for a 2x2 grid
    assert np.all(np.asarray(mesh.points)[node_inds, 0] == 0.0)


def test_get_boundary_conditions_inds():
    fe = make_fe(2, 2)
    b_inds = fe.get_boundary_conditions_inds([lambda pt: jnp.isclose(pt[0], 0.0)])
    assert len(b_inds) == 1
    assert b_inds[0].shape[0] == 2      # 2 cells touch the r=0 face


def test_convert_from_dof_to_quad_constant():
    fe = make_fe(3, 4)
    sol = jnp.ones((fe.num_total_nodes, 1))
    u = fe.convert_from_dof_to_quad(sol)
    assert u.shape == (12, 4, 1)
    assert np.allclose(np.asarray(u), 1.0)   # partition of unity: 1 stays 1


def test_sol_to_grad_linear():
    """A = r  ->  grad = (1, 0) everywhere (exercises stored self.shape_grads)."""
    msh = rectangle_mesh(1, 1, 0.0, 1.0, 0.0, 1.0)
    mesh = create_mesh(msh, "quad")
    fe = FiniteElement(mesh, vec=1, dim=2, ele_type="QUAD4")
    r = np.asarray(mesh.points)[:, 0]
    sol = jnp.stack([jnp.asarray(r)], axis=1)
    u_grads = fe.sol_to_grad(sol)
    assert u_grads.shape == (1, 4, 1, 2)
    assert np.allclose(np.asarray(u_grads)[0, :, 0, :], np.array([1.0, 0.0]), atol=1e-10)


def test_sol_to_grad_linear_multicell():
    """A = r  ->  grad = (1, 0) on a multi-cell mesh."""
    fe = make_fe(3, 4)
    r = np.asarray(fe.mesh.points)[:, 0]
    sol = jnp.stack([jnp.asarray(r)], axis=1)
    u_grads = np.asarray(fe.sol_to_grad(sol))
    assert u_grads.shape == (12, 4, 1, 2)
    assert np.allclose(u_grads[:, :, 0, :], np.array([1.0, 0.0]), atol=1e-5)


def test_convert_from_dof_to_quad_interpolation():
    """f = z at nodes -> quad values == z of the physical quad points."""
    fe = make_fe(3, 4)
    z = np.asarray(fe.mesh.points)[:, 1]
    sol = jnp.stack([jnp.asarray(z)], axis=1)
    u = np.asarray(fe.convert_from_dof_to_quad(sol))
    pqp = np.asarray(fe.get_physical_quad_points())
    assert u.shape == (12, 4, 1)
    assert np.allclose(u[:, :, 0], pqp[:, :, 1], atol=1e-5)


def test_vec_multiple_components():
    fe = make_fe(3, 4, vec=2)
    assert fe.num_total_dofs == fe.num_total_nodes * 2 == 40


def test_dirichlet_bc_values():
    msh = rectangle_mesh(2, 2, 0.0, 0.01, 0.0, 0.02)
    mesh = create_mesh(msh, "quad")
    fe = FiniteElement(
        mesh, vec=1, dim=2, ele_type="QUAD4",
        dirichlet_bc_info=[[lambda pt: jnp.isclose(pt[0], 0.0)], [0], [lambda pt: 7.5]],
    )
    assert np.allclose(np.asarray(fe.vals_list[0]), 7.5)


def test_dirichlet_vec_component():
    msh = rectangle_mesh(2, 2, 0.0, 0.01, 0.0, 0.02)
    mesh = create_mesh(msh, "quad")
    fe = FiniteElement(
        mesh, vec=2, dim=2, ele_type="QUAD4",
        dirichlet_bc_info=[[lambda pt: jnp.isclose(pt[0], 0.0)], [1], [lambda pt: 2.0]],
    )
    assert np.all(np.asarray(fe.vec_inds_list[0]) == 1)
    assert np.allclose(np.asarray(fe.vals_list[0]), 2.0)


def test_dirichlet_2arg_location_fn():
    msh = rectangle_mesh(2, 2, 0.0, 0.01, 0.0, 0.02)
    mesh = create_mesh(msh, "quad")
    fe = FiniteElement(
        mesh, vec=1, dim=2, ele_type="QUAD4",
        dirichlet_bc_info=[[lambda pt, ind: jnp.isclose(pt[0], 0.0) & (ind < 5)], [0], [lambda pt: 0.0]],
    )
    node_inds = np.asarray(fe.node_inds_list[0])
    assert len(node_inds) == 3
    assert np.all(np.asarray(mesh.points)[node_inds, 0] == 0.0)


def test_update_dirichlet_boundary_conditions():
    fe = make_fe(2, 2)
    fe.update_Dirichlet_boundary_conditions(
        [[lambda pt: jnp.isclose(pt[1], 0.0)], [0], [lambda pt: 3.0]]
    )
    node_inds = np.asarray(fe.node_inds_list[0])
    assert len(node_inds) == 3
    assert np.all(np.asarray(fe.mesh.points)[node_inds, 1] == 0.0)
    assert np.allclose(np.asarray(fe.vals_list[0]), 3.0)


def test_get_physical_surface_quad_points_on_boundary():
    fe = make_fe(3, 4)
    b_inds = fe.get_boundary_conditions_inds([lambda pt: jnp.isclose(pt[0], 0.0)])[0]
    sqp = np.asarray(fe.get_physical_surface_quad_points(b_inds))
    assert sqp.shape == (4, 2, 2)
    assert np.allclose(sqp[:, :, 0], 0.0, atol=1e-10)
    assert sqp[:, :, 1].min() >= 0.0 and sqp[:, :, 1].max() <= 0.02


def test_face_shape_grads_nanson_length():
    """Anchor: integral over the r=0 boundary == its physical length (0.02)."""
    fe = make_fe(3, 4)
    b_inds = fe.get_boundary_conditions_inds([lambda pt: jnp.isclose(pt[0], 0.0)])[0]
    _, nanson = fe.get_face_shape_grads(b_inds)
    assert nanson.shape == (4, 2)
    assert np.isclose(np.asarray(nanson).sum(), 0.02)


def test_convert_from_dof_to_face_quad_linear():
    """f = z interpolated on the r=0 face == z of the face quad points."""
    fe = make_fe(3, 4)
    b_inds = fe.get_boundary_conditions_inds([lambda pt: jnp.isclose(pt[0], 0.0)])[0]
    z = np.asarray(fe.mesh.points)[:, 1]
    sol = jnp.stack([jnp.asarray(z)], axis=1)
    u = fe.convert_from_dof_to_face_quad(sol, b_inds)
    sqp = np.asarray(fe.get_physical_surface_quad_points(b_inds))
    assert u.shape == (4, 2, 1)
    assert np.allclose(np.asarray(u)[:, :, 0], sqp[:, :, 1], atol=1e-5)