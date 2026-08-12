import os
import gmsh
import numpy as onp
import meshio

from .basis import get_elements
from .basis import get_face_shape_vals_and_grads

import jax
import jax.numpy as np


class Mesh():
    """Mesh manager.

    Attributes
    ----------
    points : NumpyArray
        Shape is (num_total_nodes, dim).
    cells : NumpyArray
        Shape is (num_cells, num_nodes).
    """

    def __init__(self, points, cells, ele_type=None):
        self.points = np.asarray(points)
        self.cells = np.asarray(cells)
        self.ele_type = ele_type
        assert self._check_cell_orientation(), "Cells must be in CCW order"

    def _check_cell_orientation(self):
        v0 = self.points[self.cells[:, 0]]
        v1 = self.points[self.cells[:, 1]]
        v3 = self.points[self.cells[:, 3]]
        cross = (v1 - v0)[:, 0] * (v3 - v0)[:, 1] - (v1 - v0)[:, 1] * (v3 - v0)[:, 0]
        return bool(onp.all(cross > 0))

    def count_selected_faces(self, location_fn):
        """Compute the number of faces that satisfy the location function.
        Useful for setting up distributed load conditions.

        Parameters
        ----------
        location_fns : list
            :attr:`~new_fem_engine.problem.Problem.location_fns`

        Returns
        -------
        face_count : int
        """
        _, _, _, _, face_inds = get_face_shape_vals_and_grads(self.ele_type)
        cell_points = self.points[self.cells]
        cell_face_points = cell_points[:, face_inds]

        vmap_location_fn = jax.vmap(location_fn)

        def on_boundary(cell_points):
            boundary_flag = vmap_location_fn(cell_points)
            return np.all(boundary_flag)

        vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
        boundary_flags = vvmap_on_boundary(cell_face_points)
        boundary_inds = onp.argwhere(np.asarray(boundary_flags))
        return boundary_inds.shape[0]


def get_meshio_cell_type(ele_type):
    """Convert element type to the meshio-compatible string."""
    if ele_type == "QUAD4":
        return "quad"
    raise NotImplementedError(f"Unsupported element type: {ele_type}")


def rectangle_mesh(Nr, Nz, r_min, r_max, z_min, z_max):
    """Generate structured QUAD4 mesh for the 2D axisymmetric (r, z) domain.

    Nodes ordered CCW per cell: BL, BR, TR, TL.
    Returns a meshio.Mesh (in memory — no .msh file needed).
    """
    r = onp.linspace(r_min, r_max, Nr + 1)
    z = onp.linspace(z_min, z_max, Nz + 1)
    rv, zv = onp.meshgrid(r, z, indexing="ij")
    points = onp.stack((rv, zv), axis=2).reshape(-1, 2)

    inds = onp.arange(len(points)).reshape(Nr + 1, Nz + 1)
    i1, i2 = inds[:-1, :-1], inds[1:, :-1]
    i3, i4 = inds[1:, 1:], inds[:-1, 1:]
    cells = onp.stack((i1, i2, i3, i4), axis=2).reshape(-1, 4)

    return meshio.Mesh(points=points, cells={"quad": cells})


def create_mesh(meshio_mesh, cell_type, data_type=onp.float64):
    """Convert a meshio.Mesh into a jax-compatible Mesh object.

    Applies reorder_inds so the file's vertex order matches basix's
    shape-function ordering.
    """
    if cell_type == "quad":
        _, _, _, _, _, re_order = get_elements("QUAD4")
    else:
        raise NotImplementedError(f"Unsupported cell type: {cell_type}")

    points = meshio_mesh.points.astype(data_type)
    cells = meshio_mesh.cells_dict[cell_type].astype(onp.int32)

    cells = cells[:, re_order]

    return Mesh(points, cells, ele_type="QUAD4")


def get_face_elems(elems, num_face_quads):
    """Extract the face (edge) elements of each cell.

    Returns shape (num_faces, num_cells, 2). For QUAD4: 4 faces, 2 nodes each.
    """
    num_cells, num_nodes = elems.shape
    face_elems = onp.zeros((num_face_quads, num_cells, 2), dtype=onp.int32)
    for i in range(num_face_quads):
        face_elems[i] = elems[:, [i, (i + 1) % num_nodes]]
    return face_elems


def get_face_nodes(points, faces):
    """Given points and face elements, compute coordinates of the face nodes."""
    return points[faces]


def get_face_edges(faces):
    """Given face elements, compute the edges of each face."""
    face_edges = []
    for face in faces:
        num_nodes = len(face)
        edges = [[face[i], face[(i + 1) % num_nodes]] for i in range(num_nodes)]
        face_edges.append(edges)
    return onp.array(face_edges)


def get_node_indices(edge_nodes, all_nodes, method="all", interval=None):
    """Find indices of boundary nodes within all_nodes.

    Parameters
    ----------
    edge_nodes : NumpyArray
        Shape is (num_boundary_nodes, dim).
    all_nodes : NumpyArray
        Shape is (num_total_nodes, dim).
    method : str
        "all" → every edge node; "partial" → only those within interval.
    interval : tuple
        (low, high) along the last coordinate axis, for "partial".

    Returns
    -------
    node_inds : NumpyArray
    """
    match = (all_nodes[None, :, :] == edge_nodes[:, None, :]).all(axis=-1)
    inds = onp.argwhere(match.any(axis=0))[:, 0]

    if method == "partial":
        if interval is None:
            raise ValueError("interval is required for method='partial'")
        low, high = interval
        mask = (all_nodes[inds, -1] >= low) & (all_nodes[inds, -1] <= high)
        inds = inds[mask]
    return inds