"""Certification of the traceable jitted forward+backward path.

``make_loss_fn(..., traceable=True)`` fuses the whole geometry -> fill -> solve
-> inductance -> gradient into one ``jax.jit(jax.value_and_grad(loss))``
program on BCOO (no scipy objects, no per-iteration ``prepare``).  This pins:

- forward parity: jitted L matches the eager spsolve forward to ~1e-10;
- gradient parity: the jitted 5-D gradient matches central finite differences
  (including the *coil* parameters — window_height, center_post_diameter,
  leg_inner_diameter — which the legacy frozen-coil path reported with zero
  gradient; ``coil_clearance`` is now frozen to its datasheet value and is not
  a free variable);
- coil movement: one compiled program must re-derive the coils per ``x`` (no
  stale baked kernels).

Marked ``slow`` (one-time ~10 s XLA compile plus finite-difference solves).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import helpers
from optimizers import objective

EPS = 1e-6


def _jitted_vg(core_spec, mesh_size):
    loss = objective.make_loss_fn(core_spec, mesh_size=mesh_size, traceable=True)
    return jax.jit(jax.value_and_grad(loss)), loss


def _x0(core_spec):
    return jnp.asarray(objective.default_x0(core_spec))


@pytest.mark.slow
def test_traceable_forward_matches_spsolve():
    spec = objective.load_core_spec("pq_40x40")
    vg, loss = _jitted_vg(spec, 0.002)
    x = _x0(spec)
    vg(x)  # compile
    _, L_jit, _ = loss.frozen(x)  # (val, raw L, vol)

    params = objective.make_params_dict(x, spec)
    L_ref, _ = objective.solve_forward(params, 0.002)
    rel = abs(float(L_jit) - L_ref) / abs(L_ref)
    assert rel < 1e-9, f"jitted L {L_jit:.10e} vs spsolve {L_ref:.10e} (rel {rel:.2e})"


@pytest.mark.slow
def test_traceable_grad_matches_fd():
    spec = objective.load_core_spec("pq_40x40")
    vg, _ = _jitted_vg(spec, 0.002)
    x = _x0(spec)
    _, grad = vg(x)

    for i, key in enumerate(objective.VARY_GEOM_KEYS):
        e = jnp.zeros(len(objective.VARY_GEOM_KEYS))
        e = e.at[i].set(EPS)
        fp = float(vg(x + e)[0])
        fm = float(vg(x - e)[0])
        fd = (fp - fm) / (2 * EPS)
        ad = float(grad[i])
        rel = abs(ad - fd) / max(abs(fd), 1e-12)
        assert rel < 1e-3, (
            f"{key}: AD {ad:.6e} vs FD {fd:.6e} (rel {rel:.3e})")


@pytest.mark.slow
def test_traceable_coil_params_have_nonzero_gradient():
    """Coil-affecting free params must NOT be frozen (gradient-free) anymore."""
    spec = objective.load_core_spec("pq_40x40")
    vg, _ = _jitted_vg(spec, 0.002)
    x = _x0(spec)
    _, grad = vg(x)
    for i, key in enumerate(objective.VARY_GEOM_KEYS):
        assert abs(float(grad[i])) > 1e-10, (
            f"{key} gradient is ~0 — coils frozen under the traceable path?")


@pytest.mark.slow
def test_traceable_coils_move_without_recompile():
    """One compiled program must respond to coil-geometry changes in x."""
    spec = objective.load_core_spec("pq_40x40")
    vg, _ = _jitted_vg(spec, 0.002)
    x = _x0(spec)
    lo, hi = objective.bounds_arrays(spec)
    i = objective.VARY_GEOM_KEYS.index("window_height")

    def l_at(c):
        xc = x.at[i].set(c)
        return float(vg(xc)[0])

    l_base = l_at(float(x[i]))
    # pick a nearby window height that keeps the coil window non-degenerate
    span = hi[i] - lo[i]
    l_other = l_at(float(x[i]) + 0.25 * span)
    assert abs(l_base - l_other) / abs(l_base) > 1e-6, (
        "L did not change with window_height — stale baked coils under jit")
