import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from new_fem_engine.basis import get_shape_vals_and_grads


def test_shape_vals_partition_of_unity():
    """Shape functions must sum to 1 at every Gauss point."""
    shape_vals, _, _ = get_shape_vals_and_grads("QUAD4")
    assert np.allclose(shape_vals.sum(axis=1), 1.0)


def test_shape_vals_shapes():
    """QUAD4: (num_quads, num_nodes), grads (quads, nodes, dim), weights (quads,)."""
    shape_vals, shape_grads_ref, weights = get_shape_vals_and_grads("QUAD4")
    assert shape_vals.shape == (4, 4)
    assert shape_grads_ref.shape == (4, 4, 2)
    assert weights.shape == (4,)


def test_quadrature_weights_sum_to_one():
    """Basix reference square is [0,1]^2 (area 1); 2x2 Gauss weights sum to 1."""
    _, _, weights = get_shape_vals_and_grads("QUAD4")
    assert np.isclose(weights.sum(), 1.0)