import basix
import numpy as onp
from . import logger

def get_elements(ele_type):
    """Obtain element information for basix to handle.

    Parameters
    ----------
    ele_type : str
        Element type string, e.g. 'QUAD4'.

    Returns
    -------
    element_family, basix_ele, basix_face_ele, quadrature_order, degree, re_order
    """
    
    element_family = basix.ElementFamily.P
    if ele_type == 'QUAD4':
        re_order = [0, 1, 3, 2]
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        quadrature_order = 2
        degree = 1
    else:
        raise NotImplementedError(f"Only QUAD4 is supported, got {ele_type}")

    return element_family, basix_ele, basix_face_ele, quadrature_order, degree, re_order


def reorder_inds(inds, re_order):
    """Apply re-ordering transformation for node indices."""
    new_inds = []
    for ind in inds.reshape(-1):
        new_inds.append(onp.argwhere(re_order == ind))
    new_inds = onp.array(new_inds).reshape(inds.shape)
    return new_inds


def _normalize_quadrature_rule(rule):
    """Convert input to a basix QuadratureType or None (uses basix string_to_type for strings)."""
    if rule is None:
        return basix.QuadratureType.default
    if isinstance(rule, basix.QuadratureType):
        return rule
    if isinstance(rule, str):
        return basix.quadrature.string_to_type(rule)
    raise TypeError(
        "quadrature_rule must be None, a basix.QuadratureType, or a string accepted by "
        "basix.quadrature.string_to_type"
    )


def get_shape_vals_and_grads(ele_type, quadrature_rule=None, quadrature_order=None):
    """Use basix to get shape function values and gradients for an element.

    Parameters
    ----------
    ele_type : str
        Element type, e.g. 'QUAD4'.
    quadrature_rule : optional
        basix quadrature rule specification.
    quadrature_order : int, optional
        Order of Gaussian quadrature.

    Returns
    -------
    shape_values : NumpyArray
        Shape (num_quads, num_nodes), e.g. (4, 4) for QUAD4.
    shape_grads_ref : NumpyArray
        Shape (num_quads, num_nodes, dim), e.g. (4, 4, 2) for QUAD4.
    weights : NumpyArray
        Shape (num_quads,), e.g. (4,) for QUAD4.
    """
    element_family, basix_ele, basix_face_ele, quadrature_order_default, degree, re_order = get_elements(ele_type)

    quadrature_rule = _normalize_quadrature_rule(quadrature_rule)
    if quadrature_order is None:
        quadrature_order = quadrature_order_default

    quad_points, weights = basix.make_quadrature(basix_ele, quadrature_order, rule=quadrature_rule)

    element = basix.create_element(element_family, basix_ele, degree)
    vals_and_grads = element.tabulate(1, quad_points)
    shape_values = vals_and_grads[0, :, :, 0]
    shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0], axes=(1, 2, 0))
    logger.info(f"ele_type = {ele_type}, quad_points.shape = {quad_points.shape}")
    return shape_values, shape_grads_ref, weights

def get_face_shape_vals_and_grads(ele_type, quadrature_rule=None, quadrature_order=None):
    """Use basix to get shape function values and gradients for element faces.

    Parameters
    ----------
    ele_type : str
        Element type, e.g. 'QUAD4'.
    quadrature_rule : optional
        basix quadrature rule specification.
    quadrature_order : int, optional
        Order of Gaussian quadrature.

    Returns
    -------
    face_shape_vals : NumpyArray
        Shape (num_faces, num_face_quads, num_nodes), e.g. (4, 1, 4) for QUAD4.
    face_shape_grads_ref : NumpyArray
        Shape (num_faces, num_face_quads, num_nodes, dim), e.g. (4, 1, 4, 2) for QUAD4.
    face_weights : NumpyArray
        Shape (num_faces, num_face_quads), e.g. (4, 1) for QUAD4.
    face_normals : NumpyArray
        Shape (num_faces, dim), e.g. (4, 2) for QUAD4.
    face_inds : NumpyArray
        Shape (num_faces, num_face_vertices), e.g. (4, 2) for QUAD4.
    """
    element_family, basix_ele, basix_face_ele, quadrature_order_default, degree, re_order = get_elements(ele_type)

    # TODO: Separate quadrature_order for volume integral and surface integral.
    # Currently they use the same quadrature_order.
    quadrature_rule = _normalize_quadrature_rule(quadrature_rule)
    if quadrature_order is None:
        quadrature_order = quadrature_order_default
    points, weights = basix.make_quadrature(basix_face_ele, quadrature_order, rule=quadrature_rule)

    map_degree = 1
    lagrange_map = basix.create_element(basix.ElementFamily.P, basix_face_ele, map_degree)
    values = lagrange_map.tabulate(0, points)[0, :, :, 0]
    vertices = basix.geometry(basix_ele)
    dim = len(vertices[0])
    facets = basix.cell.sub_entity_connectivity(basix_ele)[dim - 1]
    # Map face points
    face_quad_points = []
    face_inds = []
    face_weights = []
    for f, facet in enumerate(facets):
        mapped_points = []
        for i in range(len(points)):
            vals = values[i]
            mapped_point = onp.sum(vertices[facet[0]] * vals[:, None], axis=0)
            mapped_points.append(mapped_point)
        face_quad_points.append(mapped_points)
        face_inds.append(facet[0])
        jacobian = basix.cell.facet_jacobians(basix_ele)[f]
        if dim == 2:
            size_jacobian = onp.linalg.norm(jacobian)
        else:
            size_jacobian = onp.linalg.norm(onp.cross(jacobian[:, 0], jacobian[:, 1]))
        face_weights.append(weights * size_jacobian)
    face_quad_points = onp.stack(face_quad_points)
    face_weights = onp.stack(face_weights)

    face_normals = basix.cell.facet_outward_normals(basix_ele)
    face_inds = onp.array(face_inds)
    num_faces, num_face_quads, dim = face_quad_points.shape
    element = basix.create_element(element_family, basix_ele, degree)
    vals_and_grads = element.tabulate(1, face_quad_points.reshape(-1, dim))
    face_shape_vals = vals_and_grads[0, :, :, 0].reshape(num_faces, num_face_quads, -1)
    face_shape_grads_ref = vals_and_grads[1:, :, :, 0].reshape(dim, num_faces, num_face_quads, -1)
    face_shape_grads_ref = onp.transpose(face_shape_grads_ref, axes=(1, 2, 3, 0))
    logger.info(f"face_quad_points.shape = (num_faces, num_face_quads, dim) = {face_quad_points.shape}")
    return face_shape_vals, face_shape_grads_ref, face_weights, face_normals, face_inds