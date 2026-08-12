import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from new_fem_engine.generate_mesh import (
    Mesh,
    rectangle_mesh,
    create_mesh,
    get_meshio_cell_type,
    get_face_elems,
    get_face_edges,
    get_face_nodes,
    get_node_indices,
)
import jax.numpy as jnp


def test_rectangle_mesh_shapes():
    mesh = rectangle_mesh(3, 4, 0.0, 0.01, 0.0, 0.02)
    assert mesh.points.shape == (20, 2)      # (3+1)*(4+1)
    assert mesh.cells_dict["quad"].shape == (12, 4)  # 3*4 cells


def test_rectangle_mesh_single_cell():
    mesh = rectangle_mesh(1, 1, 0.0, 0.01, 0.0, 0.02)
    assert mesh.points.shape == (4, 2)
    assert mesh.cells_dict["quad"].shape == (1, 4)


def test_rectangle_mesh_bounds():
    mesh = rectangle_mesh(3, 4, 0.0, 0.01, -0.03, 0.03)
    pts = mesh.points
    assert np.isclose(pts[:, 0].min(), 0.0)
    assert np.isclose(pts[:, 0].max(), 0.01)
    assert np.isclose(pts[:, 1].min(), -0.03)
    assert np.isclose(pts[:, 1].max(), 0.03)


def test_rectangle_mesh_ccw():
    mesh = rectangle_mesh(3, 4, 0.0, 0.01, 0.0, 0.02)
    pts = mesh.points
    cells = mesh.cells_dict["quad"]
    v0 = pts[cells[:, 0]]
    v1 = pts[cells[:, 1]]
    v3 = pts[cells[:, 3]]
    cross = (v1 - v0)[:, 0] * (v3 - v0)[:, 1] - (v1 - v0)[:, 1] * (v3 - v0)[:, 0]
    assert np.all(cross > 0)


def test_create_mesh_basix_order():
    """Single cell: verify reorder gives basix vertex order [BL, BR, TL, TR]."""
    msh = rectangle_mesh(1, 1, 0.0, 0.01, 0.0, 0.02)
    mesh = create_mesh(msh, "quad")

    assert mesh.ele_type == "QUAD4"
    assert mesh.points.shape == (4, 2)
    assert mesh.cells.shape == (1, 4)

    # points: BL=(0,0), TL=(0,0.02), BR=(0.01,0), TR=(0.01,0.02)
    pts = np.asarray(mesh.points)
    assert np.allclose(pts[0], [0.0, 0.0])
    assert np.allclose(pts[1], [0.0, 0.02])     # TL
    assert np.allclose(pts[2], [0.01, 0.0])     # BR
    assert np.allclose(pts[3], [0.01, 0.02])    # TR

    # basix vertex order [BL, BR, TL, TR]
    assert np.array_equal(np.asarray(mesh.cells), np.array([[0, 2, 1, 3]]))


def test_create_mesh_dtype():
    msh = rectangle_mesh(2, 2, 0.0, 0.01, 0.0, 0.02)
    mesh = create_mesh(msh, "quad", data_type=np.float32)
    assert mesh.points.dtype == np.float32


def test_create_mesh_unsupported_cell_type():
    msh = rectangle_mesh(2, 2, 0.0, 0.01, 0.0, 0.02)
    with pytest.raises(NotImplementedError):
        create_mesh(msh, "tet")


def test_mesh_orientation_guard():
    msh = rectangle_mesh(3, 4, 0.0, 0.01, 0.0, 0.02)
    mesh = create_mesh(msh, "quad")   # must not raise
    assert mesh.points.shape[0] == 20


def test_mesh_orientation_negative():
    """Clockwise cell ordering must raise AssertionError."""
    pts = np.array([[0.0, 0.0], [0.0, 0.01], [0.01, 0.0], [0.01, 0.01]])
    cw_cells = np.array([[0, 1, 3, 2]])  # BL, TL, TR, BR -> clockwise
    with pytest.raises(AssertionError):
        Mesh(pts, cw_cells, ele_type="QUAD4")


def test_get_meshio_cell_type():
    assert get_meshio_cell_type("QUAD4") == "quad"
    try:
        get_meshio_cell_type("TET4")
        assert False, "should have raised"
    except NotImplementedError:
        pass


def test_get_face_elems():
    mesh = rectangle_mesh(2, 2, 0.0, 0.01, 0.0, 0.02)
    cells = mesh.cells_dict["quad"]
    faces = get_face_elems(cells, 4)
    assert faces.shape == (4, 4, 2)
    # every face must be a valid edge: consecutive corner pair
    for k in range(4):
        assert np.array_equal(faces[k, 0], cells[0, [k, (k + 1) % 4]])


def test_get_face_edges():
    """Each face edge is a pair of consecutive edge-elements, closing the loop."""
    faces = np.array([[[0, 1], [1, 2], [2, 3], [3, 0]]])
    edges = get_face_edges(faces)
    assert edges.shape == (1, 4, 2, 2)
    expected = np.array([
        [[0, 1], [1, 2]],
        [[1, 2], [2, 3]],
        [[2, 3], [3, 0]],
        [[3, 0], [0, 1]],
    ])
    assert np.array_equal(edges[0], expected)


def test_get_face_nodes():
    pts = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01]])
    faces = np.array([[[0, 1], [1, 3], [3, 2], [2, 0]]])
    fn = get_face_nodes(pts, faces)
    assert fn.shape == (1, 4, 2, 2)
    assert np.allclose(fn[0, 0], [[0.0, 0.0], [0.01, 0.0]])
    assert np.allclose(fn[0, 1], [[0.01, 0.0], [0.01, 0.01]])


def test_count_selected_faces_left():
    msh = rectangle_mesh(2, 2, 0.0, 0.01, 0.0, 0.02)
    mesh = create_mesh(msh, "quad")
    count = mesh.count_selected_faces(lambda pt: jnp.isclose(pt[0], 0.0))
    assert count == 2


def test_count_selected_faces_outer_boundary():
    msh = rectangle_mesh(2, 2, 0.0, 0.01, 0.0, 0.02)
    mesh = create_mesh(msh, "quad")

    def on_outer(pt):
        return (jnp.isclose(pt[0], 0.0) | jnp.isclose(pt[0], 0.01) |
                jnp.isclose(pt[1], 0.0) | jnp.isclose(pt[1], 0.02))

    assert mesh.count_selected_faces(on_outer) == 8


def test_count_selected_faces_none():
    msh = rectangle_mesh(2, 2, 0.0, 0.01, 0.0, 0.02)
    mesh = create_mesh(msh, "quad")
    assert mesh.count_selected_faces(lambda pt: jnp.array(False)) == 0


def test_get_node_indices():
    pts = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01],
                    [0.0, 0.02], [0.01, 0.02]])
    edge_nodes = np.array([[0.0, 0.0], [0.0, 0.01], [0.0, 0.02]])

    inds = get_node_indices(edge_nodes, pts)
    assert np.array_equal(np.sort(inds), np.array([0, 2, 4]))

    inds_partial = get_node_indices(edge_nodes, pts, method="partial", interval=(0.0, 0.01))
    assert np.array_equal(np.sort(inds_partial), np.array([0, 2]))


def test_get_node_indices_partial_requires_interval():
    pts = np.array([[0.0, 0.0], [0.01, 0.0]])
    with pytest.raises(ValueError):
        get_node_indices(np.array([[0.0, 0.0]]), pts, method="partial")
