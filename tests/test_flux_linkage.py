"""Flux-linkage vs stored-energy inductance.

Both definitions of L (lambda/I from flux linkage and 2W/I^2 from field
energy) must agree on the same solve.
"""

from new_fem_engine.run_forward import linkage_inductance
import new_fem_engine.solver as slv

import helpers


def test_energy_and_linkage_L_agree():
    prob, fill, _ = helpers.build_system()
    L_energy = helpers.solve_L(prob, fill)

    prob.set_params(fill)
    sol = slv.solver(prob, helpers.SP)
    L_link = linkage_inductance(prob, sol)

    assert helpers.rel_err(L_link, L_energy) < 1e-3, (
        f"linkage {L_link:.6e} vs energy {L_energy:.6e}")
