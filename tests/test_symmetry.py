"""Symmetry invariants of the axisymmetric solve.

1. Swapping which winding is "primary" vs "secondary" must not change L
   (both carry the same magnetising current density).
2. Z-mirroring the material fill (the core is z-symmetric) must not change L.
"""

import jax.numpy as jnp

import helpers


def test_winding_swap_invariance():
    prob, fill, _ = helpers.build_system()
    L_norm = helpers.solve_L(prob, fill)

    prob2, fill2, _ = helpers.build_system()
    helpers.swap_windings(prob2)
    L_swap = helpers.solve_L(prob2, fill2)

    assert helpers.rel_err(L_swap, L_norm) < 1e-3, (
        f"winding swap changed L: {L_norm:.6e} -> {L_swap:.6e}")


def test_z_mirror_fill_invariance():
    prob, fill, f_core = helpers.build_system()
    L_orig = helpers.solve_L(prob, fill)

    f_mirror = helpers.z_mirror_fill(f_core)
    fill_mirror = jnp.repeat(
        jnp.asarray(f_mirror).T.reshape(-1)[:, None], 4, axis=1)
    L_mirror = helpers.solve_L(prob, fill_mirror)

    assert helpers.rel_err(L_mirror, L_orig) < 1e-3, (
        f"z-mirror changed L: {L_orig:.6e} -> {L_mirror:.6e}")
