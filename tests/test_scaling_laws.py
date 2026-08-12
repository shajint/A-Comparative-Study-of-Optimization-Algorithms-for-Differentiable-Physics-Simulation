"""Scaling laws of the magnetostatic model.

- L scales with the square of the turn count (L ~ N^2).
- L decreases monotonically with the air gap, and matches the fringing
  corrected reluctance model within 10%.
"""

import numpy as np

import helpers


def test_L_scales_with_turns_squared():
    # current/turns are baked into the jitted assembly at the first trace,
    # so each configuration gets a fresh system with the value set first.
    prob1, fill1, _ = helpers.build_system()
    L_half = helpers.solve_L(prob1, fill1)          # turns = 0.5

    prob2, fill2, _ = helpers.build_system()
    prob2.turns = 1.0
    L_one = helpers.solve_L(prob2, fill2)           # turns = 1.0

    assert helpers.rel_err(L_one, 4.0 * L_half) < 1e-3, (
        f"L(turn) not ~ N^2: {L_one:.6e} vs 4*{L_half:.6e}")


def test_L_decreases_monotonically_with_gap():
    gaps = (0.0005, 0.001, 0.002)
    ls = []
    for g in gaps:
        prob, fill, _ = helpers.build_system(
            mesh_size=0.0005, params=helpers.base_params(gap_size=g))
        ls.append(helpers.solve_L(prob, fill))
    assert all(b < a for a, b in zip(ls, ls[1:])), f"L not decreasing: {ls}"


def test_gapped_L_matches_fringing_analytical():
    gap = 0.001
    prob, fill, _ = helpers.build_system(
        mesh_size=0.0005, params=helpers.base_params(gap_size=gap))
    L_sim = helpers.solve_L(prob, fill)
    spec = {"params": helpers.base_params(gap_size=gap)}
    from new_fem_engine.run_forward import analytical_inductance
    L_ana = analytical_inductance(spec)
    assert helpers.rel_err(L_sim, L_ana) < 0.10, (
        f"sim {L_sim:.6e} vs analytical {L_ana:.6e}")
