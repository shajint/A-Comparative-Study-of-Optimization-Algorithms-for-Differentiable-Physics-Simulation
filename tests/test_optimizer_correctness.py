"""Known-answer correctness of the optimizer *drivers* (the generic cores).

The drivers (``adam._adam_driver``, ``lbfgs._lbfgs_driver``) are the
GEOM_KEYS-free loops that the FEM ``run_*`` wrappers call.  They are run here
against toy objectives with **known** optima, in milliseconds — the fast CI
safety net that proves the optimization machinery (convergence, box
constraints, best-tracking, history accounting) is correct *in isolation*
from the FEM problem.

The FEM side (objective + gradient) is verified separately by
``test_backward.py`` / ``test_gradient_sign.py`` / ``test_optimizer_objective.py``
and by the analytical cross-check in the benchmark; the FEM result-dict
contract is pinned by ``test_optimizer_backends.py`` / ``test_optax_family.py``
and the slow pin at the bottom of this file.

Per-family assertion contract (drivers, not FEM):

- L-BFGS-B: tight tolerance (1e-8 on value, projected-gradient -> 0 for the
  active-bound case); ``len(history) == nit`` (line search lives inside the
  step, one history entry per iteration).
- Adam: interior 1e-2, boundary 1e-2; ``len(history) == max_iters`` (1:1).
- CMA-ES / NSGA3: medium/loosest tolerance; ``len(history) == gens x
  popsize`` (population-multiplied); seeded determinism.
- Bayesian (skopt/optuna): tight on best-loss; ``len(history) ==
  n_init + max_iters`` (initialization points are recorded too).
"""

import numpy as np
import pytest
import optax
import jax.numpy as jnp

from optimizers import adam, bayesian, cmaes, driver, objective, lbfgs, nsga3


# --- toy objectives (known minima) -------------------------------------------

def _styblinski_tang(x):
    """2D Styblinski-Tang; global min at (-2.903534, -2.903534), f* ~ -78.332."""
    x = jnp.asarray(x, dtype=jnp.float64)
    return jnp.sum(x ** 4 - 16.0 * x ** 2 + 5.0 * x) / 2.0


def _styblinski_tang_grad(x):
    x = jnp.asarray(x, dtype=jnp.float64)
    return 2.0 * x ** 3 - 16.0 * x + 2.5


def _boundary_quad(x):
    """min (x-5)^2 on x in [0,3]: true optimum at x=3 (active upper bound)."""
    x = jnp.asarray(x, dtype=jnp.float64)
    return jnp.sum((x - 5.0) ** 2)


def _boundary_quad_grad(x):
    return 2.0 * (jnp.asarray(x, dtype=jnp.float64) - 5.0)


ST_ARGMIN = jnp.array([-2.903534, -2.903534])   # known argmin (book value)
ST_BOUNDS = (jnp.array([-5.0, -5.0]), jnp.array([5.0, 5.0]))
BQ_BOUNDS = (jnp.array([0.0]), jnp.array([3.0]))


def _st_fn(x):
    return _styblinski_tang(x), _styblinski_tang_grad(x)


def _st_scalar(x):
    return float(_styblinski_tang(x))


def _bq_fn(x):
    return _boundary_quad(x), _boundary_quad_grad(x)


def _bq_scalar(x):
    return float(_boundary_quad(x))


def _st_two_obj(x):
    return float(_styblinski_tang(x)), float(_styblinski_tang(x))


# --- interior known minimum: Styblinski-Tang ---------------------------------

def test_interior_known_min():
    """Both drivers must hit the *known* global min of Styblinski-Tang."""
    f_star = float(_styblinski_tang(ST_ARGMIN))          # computed, no magic number
    lo, hi = ST_BOUNDS

    out_l = lbfgs._lbfgs_driver(_st_fn, np.array([0.0, 0.0]), np.asarray(lo),
                                np.asarray(hi), max_iters=200)
    out_a = adam._adam_driver(_st_fn, np.array([0.0, 0.0]), np.asarray(lo),
                              np.asarray(hi), tx=optax.adam(learning_rate=1e-2),
                              max_iters=3000)

    for out in (out_l, out_a):
        assert out["best_loss"] < f_star + 1e-2, (out["best_loss"], f_star)
        # best-so-far == argmin over history (best-tracking consistency)
        reported, xb = driver.track_best(out["history"])
        assert out["best_loss"] == pytest.approx(reported, rel=1e-12)
        assert np.all(out["x_best"] == xb)
        # interior optimum stays inside the box
        assert np.all(np.asarray(lo) <= out["x_best"]) and \
               np.all(out["x_best"] <= np.asarray(hi))

    # gradient norm must decay (Adam optimality signal)
    gn = out_a["diagnostics"]["grad_norm"]
    assert gn[-1] < gn[0] * 0.1, (gn[0], gn[-1])


# --- active-bound optimum: KKT (projected gradient -> 0) ---------------------

def test_boundary_active_bounds_lbfgs():
    """The only case that exercises the box-constraint path; optimum sits ON
    the upper bound, so correctness means x -> 3 AND projected grad -> 0."""
    lo, hi = BQ_BOUNDS
    out = lbfgs._lbfgs_driver(_bq_fn, np.array([1.0]), np.asarray(lo),
                              np.asarray(hi), max_iters=200)
    assert out["x_best"][0] == pytest.approx(3.0, abs=1e-6)
    assert out["best_loss"] == pytest.approx((3.0 - 5.0) ** 2, abs=1e-8)
    assert out["diagnostics"]["proj_grad_error"][-1] <= 1e-6   # KKT


def test_boundary_active_bounds_adam():
    lo, hi = BQ_BOUNDS
    out = adam._adam_driver(_bq_fn, np.array([1.0]), np.asarray(lo),
                            np.asarray(hi), tx=optax.adam(learning_rate=1e-2),
                            max_iters=3000)
    assert out["x_best"][0] == pytest.approx(3.0, abs=1e-2)
    assert out["best_loss"] == pytest.approx((3.0 - 5.0) ** 2, abs=1e-2)
    # clip keeps every iterate in-box (recorded history must respect bounds)
    for e in out["history"]:
        assert 0.0 <= e["x"][0] <= 3.0


# --- determinism --------------------------------------------------------------

def test_deterministic_drivers():
    """Same inputs -> identical results (no RNG in the gradient-based path)."""
    lo, hi = BQ_BOUNDS
    a1 = adam._adam_driver(_bq_fn, np.array([1.0]), np.asarray(lo),
                           np.asarray(hi), tx=optax.adam(learning_rate=1e-2),
                           max_iters=50)
    a2 = adam._adam_driver(_bq_fn, np.array([1.0]), np.asarray(lo),
                           np.asarray(hi), tx=optax.adam(learning_rate=1e-2),
                           max_iters=50)
    assert a1["best_loss"] == a2["best_loss"]
    assert a1["history"] == a2["history"]

    l1 = lbfgs._lbfgs_driver(_bq_fn, np.array([1.0]), np.asarray(lo),
                             np.asarray(hi), max_iters=50)
    l2 = lbfgs._lbfgs_driver(_bq_fn, np.array([1.0]), np.asarray(lo),
                             np.asarray(hi), max_iters=50)
    assert l1["history"] == l2["history"]


# --- driver contract: uniform shape, GEOM_KEYS-free ---------------------------

def test_driver_output_shape_and_geometries_free():
    """Uniform driver dict; history carries raw x, never GEOM_KEYS params."""
    lo, hi = BQ_BOUNDS
    for out in (
        lbfgs._lbfgs_driver(_bq_fn, np.array([1.0]), np.asarray(lo),
                            np.asarray(hi), max_iters=20),
        adam._adam_driver(_bq_fn, np.array([1.0]), np.asarray(lo),
                          np.asarray(hi), tx=optax.adam(learning_rate=1e-2),
                          max_iters=20),
    ):
        assert {"x_best", "best_loss", "history", "nit", "diagnostics"} <= set(out)
        assert out["nit"] == len(out["history"])
        for e in out["history"]:
            assert set(e) == {"iteration", "loss", "x"}
            assert isinstance(e["x"], list)
            assert "params" not in e          # GEOM_KEYS stay in the FEM layer
        assert isinstance(out["diagnostics"], dict)

    # enrichment is the ONE place GEOM_KEYS enter — and it works off raw x,
    # mapping the 5-vector onto VARY_GEOM_KEYS in order
    fake = [{"iteration": 0, "loss": 1.0, "x": [1.0, 2.0, 3.0, 4.0, 5.0]}]
    enriched = driver.enrich_history(fake)
    assert set(enriched[0]) == {"iteration", "loss", "x", "params"}
    assert set(enriched[0]["params"]) == set(objective.VARY_GEOM_KEYS)
    assert list(enriched[0]["params"].values()) == [1.0, 2.0, 3.0, 4.0, 5.0]


# --- per-family accounting -----------------------------------------------------

def test_history_accounting():
    """1:1 families -> len(history) == max_iters; L-BFGS history == nit."""
    lo, hi = BQ_BOUNDS
    out_a = adam._adam_driver(_bq_fn, np.array([1.0]), np.asarray(lo),
                              np.asarray(hi), tx=optax.adam(learning_rate=1e-2),
                              max_iters=100)
    assert len(out_a["history"]) == 100 == out_a["nit"]

    out_l = lbfgs._lbfgs_driver(_bq_fn, np.array([1.0]), np.asarray(lo),
                                np.asarray(hi), max_iters=200)
    assert len(out_l["history"]) == out_l["nit"] <= 200
    assert out_l["diagnostics"]["nfev"] >= out_l["nit"]   # line search inside


# --- gradient-free families: CMA-ES / NSGA3 / Bayesian -----------------------

def test_cmaes_reaches_known_min():
    lo, hi = ST_BOUNDS
    out = cmaes._cmaes_driver(_st_scalar, np.array([0.0, 0.0]), np.asarray(lo),
                              np.asarray(hi), max_iters=60, pop_size=8)
    f_star = float(_styblinski_tang(ST_ARGMIN))
    assert out["best_loss"] < f_star + 5e-2, (out["best_loss"], f_star)
    # accounting: popsize entries per generation
    assert len(out["history"]) == 60 * 8 and out["nit"] == len(out["history"])
    # step size shrinks as the population concentrates (convergence signal)
    sig = out["diagnostics"]["sigma"]
    assert sig[-1] < sig[0]
    assert np.all(np.asarray(lo) <= out["x_best"]) and \
           np.all(out["x_best"] <= np.asarray(hi))


def test_nsga3_reaches_known_min():
    lo, hi = ST_BOUNDS
    out = nsga3._nsga3_driver(_st_two_obj, np.array([0.0, 0.0]), np.asarray(lo),
                              np.asarray(hi), max_iters=30, pop_size=20)
    f_star = float(_styblinski_tang(ST_ARGMIN))
    assert out["best_loss"] < f_star + 5e-2, (out["best_loss"], f_star)
    # accounting: pop_size entries per generation
    assert len(out["history"]) == 30 * 20 and out["nit"] == len(out["history"])
    assert np.all(np.asarray(lo) <= out["x_best"]) and \
           np.all(out["x_best"] <= np.asarray(hi))


@pytest.mark.parametrize("name,max_iters,tol", [
    ("gp_skopt", 40, 1e-2),
    ("tpe_optuna", 60, 5e-2),
])
def test_bayesian_reaches_known_min(name, max_iters, tol):
    lo, hi = ST_BOUNDS
    out = bayesian._bayesian_driver(_st_scalar, np.array([0.0, 0.0]),
                                    np.asarray(lo), np.asarray(hi), name,
                                    n_init=5, max_iters=max_iters)
    f_star = float(_styblinski_tang(ST_ARGMIN))
    assert out["best_loss"] < f_star + tol, (out["best_loss"], f_star)
    # accounting: initialization points are recorded too (n_init + iters)
    assert len(out["history"]) == 5 + max_iters and out["nit"] == len(out["history"])
    assert np.all(np.asarray(lo) <= out["x_best"]) and \
           np.all(out["x_best"] <= np.asarray(hi))


def test_stochastic_drivers_seeded_deterministic():
    """Same seed -> identical history for the RNG-driven families."""
    lo, hi = ST_BOUNDS   # 2-D (CMA-ES does not support 1-D)

    c1 = cmaes._cmaes_driver(_st_scalar, np.zeros(2), np.asarray(lo),
                             np.asarray(hi), max_iters=8, pop_size=6)
    c2 = cmaes._cmaes_driver(_st_scalar, np.zeros(2), np.asarray(lo),
                             np.asarray(hi), max_iters=8, pop_size=6)
    assert c1["history"] == c2["history"]

    b1 = bayesian._bayesian_driver(_bq_scalar, np.array([1.0]),
                                   np.asarray([0.0]), np.asarray([3.0]),
                                   "tpe_optuna", n_init=3, max_iters=6)
    b2 = bayesian._bayesian_driver(_bq_scalar, np.array([1.0]),
                                   np.asarray([0.0]), np.asarray([3.0]),
                                   "tpe_optuna", n_init=3, max_iters=6)
    assert b1["history"] == b2["history"]

    n1 = nsga3._nsga3_driver(_st_two_obj, np.zeros(2), np.asarray(lo),
                             np.asarray(hi), max_iters=8, pop_size=6)
    n2 = nsga3._nsga3_driver(_st_two_obj, np.zeros(2), np.asarray(lo),
                             np.asarray(hi), max_iters=8, pop_size=6)
    assert n1["history"] == n2["history"]


# --- FEM result-dict contract pin (superset of today's keys) -------------------

@pytest.mark.slow
def test_fem_result_dict_contract_preserved():
    """The refactor must keep run_adam's FEM result dict shape intact."""
    spec = objective.load_core_spec("pq_32x30")
    r = adam.run_adam(spec, max_iters=2, mesh_size=0.002, verbose=False)
    expected = {"optimizer", "core", "x_opt", "params", "L_opt", "loss",
                "history", "nit", "time", "success", "message"}
    assert expected <= set(r)
    assert set(objective.GEOM_KEYS) <= set(r["params"])  # all 6 params present
    assert len(r["x_opt"]) == len(objective.VARY_GEOM_KEYS)  # 5 free dims
    assert len(r["history"]) == 2
    for h in r["history"]:
        assert set(h) >= {"iteration", "loss", "params"}
