"""Mesh convergence at the gapped operating point (gap = 1 mm).

The gapped regime is NOT mesh-insensitive: the fringing field around the gap
corners keeps improving as the mesh resolves it, so L climbs with refinement
(h=0.5mm 266.8nH, 0.25mm 271.5nH, 0.125mm 277.0nH).  The sharp corners carry a
field singularity, so uniform refinement never reaches a flat plateau at
practical cost; the production mesh is therefore frozen at 0.125mm (8 cells
across the 1mm gap) where the finer member of the (0.25, 0.125) pair sits
within 2% of the 2nd-order Richardson extrapolate and the last halving moved
L by < 3%.  That residual (~1-2% low vs the converged asymptote) is a known,
documented bias of the production mesh.
"""

import pytest

import helpers


@pytest.mark.slow
def test_gapped_L_is_mesh_converged_at_operating_point():
    gap = 0.001
    params = helpers.base_params(gap_size=gap)
    L_coarse = helpers.solve_L(*helpers.build_system(mesh_size=0.00025, params=params)[:2])
    L_fine = helpers.solve_L(*helpers.build_system(mesh_size=0.000125, params=params)[:2])

    L_inf = helpers.richardson2(L_coarse, L_fine)
    assert L_inf > 0
    assert helpers.rel_err(L_fine, L_inf) < 0.02, (
        f"finest mesh {helpers.rel_err(L_fine, L_inf):.3%} off Richardson "
        f"extrapolate {L_inf:.6e}")


@pytest.mark.slow
def test_gapped_L_last_refinement_step_small():
    """Freeze rule: the last mesh halving must move L by < 3%."""
    gap = 0.001
    params = helpers.base_params(gap_size=gap)
    L_h = helpers.solve_L(*helpers.build_system(mesh_size=0.00025, params=params)[:2])
    L_h2 = helpers.solve_L(*helpers.build_system(mesh_size=0.000125, params=params)[:2])
    assert abs(L_h2 - L_h) / L_h2 < 0.03, (
        f"0.25->0.125 mm moved L by {abs(L_h2 - L_h) / L_h2:.3%}")
