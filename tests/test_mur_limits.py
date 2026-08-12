"""mu_r limits.

- mur = 1 makes every core cell as permeable as air -> the solve must equal
  the all-air (no core) solve: pure leakage inductance.
- Both limits must be far below the ferrite (mur = 1680) inductance.
"""

import numpy as np
import jax.numpy as jnp

import helpers


def test_mur_one_equals_no_core():
    params = helpers.base_params(mur=1.0)
    prob_core, fill_core, _ = helpers.build_system(params=params)
    L_mur1 = helpers.solve_L(prob_core, fill_core)

    prob_air, _, _ = helpers.build_system()
    fill_air = jnp.zeros_like(fill_core)
    L_air = helpers.solve_L(prob_air, fill_air)

    assert helpers.rel_err(L_air, L_mur1) < 1e-6, (
        f"mur=1 core ({L_mur1:.6e}) != no-core ({L_air:.6e})")


def test_mur_one_is_leakage_only_and_small():
    params = helpers.base_params(mur=1.0)
    prob_core, fill_core, _ = helpers.build_system(params=params)
    L_mur1 = helpers.solve_L(prob_core, fill_core)

    prob_full, fill_full, _ = helpers.build_system()
    L_full = helpers.solve_L(prob_full, fill_full)

    assert L_mur1 > 0
    assert L_mur1 < 0.5 * L_full, (
        f"mur=1 L ({L_mur1:.6e}) not far below ferrite L ({L_full:.6e})")


def test_L_monotone_increasing_in_mur():
    ls = []
    for mur in (100.0, 1680.0, 10000.0):
        prob, fill, _ = helpers.build_system(params=helpers.base_params(mur=mur))
        ls.append(helpers.solve_L(prob, fill))
    assert ls[0] < ls[1] < ls[2], f"L not increasing with mur: {ls}"
