"""The optax Adam-family sweep infrastructure (paper-fidelity requirement)."""

import numpy as np
import pytest

from optimizers import objective
from optimizers.adam import ADAM_FAMILY, build_vg, run_optax

EXPECTED_MEMBERS = {
    "adam", "adamw", "adamax", "amsgrad", "nadam", "nadamw",
    "adabelief", "radam", "yogi", "fromage", "lion",
}

CONTRACT = {"optimizer", "core", "x_opt", "params", "L_opt", "loss",
            "history", "nit", "time", "success", "message"}


def test_adam_family_members():
    assert set(ADAM_FAMILY) == EXPECTED_MEMBERS
    for name, factory in ADAM_FAMILY.items():
        tx = factory(3e-4)
        assert hasattr(tx, "init") and hasattr(tx, "update")


@pytest.mark.slow
def test_run_optax_shares_compiled_vg():
    """Two members share one ``build_vg`` (the ~10 s compile happens once)."""
    spec = objective.load_core_spec("pq_40x40")
    loss_fn, vg = build_vg(spec, mesh_size=0.002)
    for name in ["adam", "lion"]:
        r = run_optax(spec, name, max_iters=20, mesh_size=0.002,
                      vg=vg, loss_fn=loss_fn, verbose=False)
        assert set(r) >= CONTRACT
        assert r["optimizer"] == name
        assert r["core"] == spec["name"]
        assert len(r["x_opt"]) == len(objective.VARY_GEOM_KEYS)
        assert len(r["history"]) == 20
        assert np.isfinite(r["loss"])
        assert r["L_opt"] > 0
        assert r["params"]["gap_size"] == float(r["x_opt"][-1])


@pytest.mark.slow
def test_run_optax_hits_target_on_production_mesh():
    """Adam-family must drive L to the target on the production (0.001) mesh."""
    spec = objective.load_core_spec("pq_40x40")
    target = spec["target_L"]
    r = run_optax(spec, "adam", max_iters=300, mesh_size=0.001, verbose=False)
    assert r["success"], f"loss {r['loss']:.3e} (target {target * 1e9:.0f} nH)"
    assert abs(r["L_opt"] - target) / target < 1e-3
