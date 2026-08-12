"""Adam-family optimizers (optax) on the ``new_fem_engine`` objective.

Single home for every Adam-related optimizer in the project:

- :data:`ADAM_FAMILY` — the 11 optax Adam-family transforms (adam, adamw,
  adamax, amsgrad, nadam, nadamw, adabelief, radam, yogi, fromage, lion).  All
  algorithms come from optax (paper-fidelity requirement: the optimizer
  machinery is never written from scratch).
- :func:`run_optax` — generic runner for any family member on the jitted
  traceable objective (forward + implicit adjoint fused into one compiled
  ``jax.jit(jax.value_and_grad(loss))`` program; coils traced through, no
  per-iteration ``prepare``).
- :func:`build_vg` — builds one shared compiled ``value_and_grad`` so a whole
  sweep pays the ~10 s XLA compile and problem build only once.
- :func:`run_adam` — single-optimizer Adam adapter, back-compatible with the
  non-traceable (stateful ``Problem``) path via ``traceable=False``.

The traceable loop measures roughly 0.6 s/iteration at the production 0.001
mesh (each iteration is one jitted FEM forward + adjoint solve; ~0.1-0.2 s at
the 0.002 mesh); ``run_optax`` follows the objective result-dict contract
(``optimizer, core, x_opt, params, L_opt, loss, history, nit, time, success,
message``).
"""

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from optimizers import driver, objective

# Name -> optax transform factory (all accept ``learning_rate``).  The
# weight-decay members are called with weight_decay=0.0 so the comparison is
# purely about the adaptive-momentum mechanism, not regularisation strength.
ADAM_FAMILY = {
    "adam": lambda lr: optax.adam(learning_rate=lr),
    "adamw": lambda lr: optax.adamw(learning_rate=lr, weight_decay=0.0),
    "adamax": lambda lr: optax.adamax(learning_rate=lr),
    "amsgrad": lambda lr: optax.amsgrad(learning_rate=lr),
    "nadam": lambda lr: optax.nadam(learning_rate=lr),
    "nadamw": lambda lr: optax.nadamw(learning_rate=lr, weight_decay=0.0),
    "adabelief": lambda lr: optax.adabelief(learning_rate=lr),
    "radam": lambda lr: optax.radam(learning_rate=lr),
    "yogi": lambda lr: optax.yogi(learning_rate=lr),
    "fromage": lambda lr: optax.fromage(learning_rate=lr),
    "lion": lambda lr: optax.lion(learning_rate=lr, weight_decay=0.0),
}

DEFAULT_LR = {
    # ``radam`` used to be untunable on this objective (see _adam_driver) but
    # now converges with the production max_grad_norm=1.0 clip: the warm-up
    # phase emits the un-normalised moment m = lr*||g||, so without the clip
    # the first steps (~lr*1.2e7 m) slam the design into a bound corner and
    # pin it at a degenerate L=0 geometry.  Clip fixed; LR kept at the family
    # default for a fair mechanism comparison.
    "radam": 3e-4,
    "fromage": 3e-4,   # fromage normalizes each step to norm lr; 1e-2 spanned
                       # 50-100% of the whole bound box per step (permanent
                       # bound-bouncing).  Now same scale as adam (L=259.6 nH).
    "lion": 3e-4,
}


def build_vg(core_spec, w_vol=0.0, mesh_size=None, backend=None):
    """One compiled ``jax.jit(jax.value_and_grad(loss))`` shared across runs."""
    loss_fn = objective.make_loss_fn(
        core_spec, w_vol=w_vol, mesh_size=mesh_size, backend=backend,
        traceable=True)
    vg = jax.jit(jax.value_and_grad(loss_fn))
    vg(jnp.asarray(objective.default_x0(core_spec)))  # trigger the ~10 s compile
    return loss_fn, vg


def _adam_driver(fn, x0, lo, hi, tx, max_iters, prepare=None, pass_params=True,
                 tol=None, patience=0, improve_tol=1e-6, max_grad_norm=None):
    """Generic Adam-family loop — GEOM_KEYS-free, testable on toy objectives.

    ``fn(x) -> (loss, grad)`` on a raw flat parameter vector; ``tx`` is any
    optax transform (:data:`ADAM_FAMILY` member or plain ``optax.adam``).
    ``prepare`` is an optional per-iteration hook (the non-traceable FEM path
    bakes coil state before each solve); ``pass_params`` mirrors the
    ``optimizer.update(grad, state, params=x)`` call the traceable path uses.

    ``max_grad_norm`` (opt-in) clips the gradient norm before the optimizer
    update — the standard robustness measure for this objective, whose raw
    ``(L-target)`` gradient is ~1e7 at ``x0``.  Without clipping, any
    member whose update is *not* Adam-normalised (RAdam's warm-up phase emits
    the raw moment ``m = lr*||g||``; Fromage uses a trust-ratio step) takes a
    step thousands of times the box width and pins the design to a bound
    corner (loss plummets to the degenerate L=0 geometry).  Clipping caps the
    pre-``tx`` gradient at a unit-ish norm so every family's step is ``O(lr)``.

    Early termination (opt-in; defaults OFF so contract tests keep 1:1
    ``len(history) == max_iters``): stop as soon as the *best* loss is
    ``<= tol``, or when the best loss has not improved by more than
    ``improve_tol`` for ``patience`` consecutive iterations.  The loop still
    records the true best loss over the history regardless of ``improve_tol``.

    Returns the uniform driver dict (see ``driver.py``); ``diagnostics``
    record the per-iteration (clipped) gradient norm.
    """
    opt_state = tx.init(x0)
    x = jnp.asarray(x0)
    history = []
    best_x = np.asarray(x0, dtype=float)
    best_loss = float("inf")
    grad_norms = []
    stall = 0
    stopped = False
    stop_reason = None

    t_start = time.time()
    for i in range(max_iters):
        if prepare is not None:
            prepare(x)
        loss_val, grad_val = fn(x)
        loss_val = float(loss_val)
        grad_val = np.asarray(grad_val)

        if max_grad_norm is not None:
            gn = float(np.linalg.norm(grad_val))
            if gn > max_grad_norm:
                grad_val = grad_val * (max_grad_norm / gn)

        prev_best = best_loss
        if loss_val < best_loss:
            best_loss = loss_val
            best_x = np.asarray(x)
        stall = 0 if best_loss < prev_best - improve_tol else stall + 1

        history.append({
            "iteration": i,
            "loss": loss_val,
            "x": np.asarray(x).tolist(),
        })
        grad_norms.append(float(np.linalg.norm(grad_val)))

        if tol is not None and best_loss <= tol:
            stopped = True
            stop_reason = f"best loss {best_loss:.2e} <= tol {tol:.2e}"
            break
        if patience and stall >= patience:
            stopped = True
            stop_reason = (f"best loss stalled (no improvement > {improve_tol:.1e} "
                           f"for {patience} iters)")
            break

        kwargs = {"params": x} if pass_params else {}
        updates, opt_state = tx.update(grad_val, opt_state, **kwargs)
        x = optax.apply_updates(x, updates)
        x = jnp.clip(x, jnp.asarray(lo), jnp.asarray(hi))

    opt_time = time.time() - t_start

    return {
        "x_best": best_x,
        "best_loss": best_loss,
        "history": history,
        "nit": len(history),
        "diagnostics": {"grad_norm": grad_norms, "time": opt_time},
        "message": f"early-stop: {stop_reason}" if stopped else "completed",
    }


_BOX_U_BOUND = 12.0
"""Re-parametrization bound for the unconstrained ``u`` variable.

Adam has no native box support; hard-clipping the physical parameters while
the optax momentum state stays unaware causes boundary bouncing (momentum
keeps pushing outside the box, the clip snaps back, loss spikes).  The fix
used by ``run_optax`` / ``run_adam`` is to optimise an unconstrained ``u``
and decode it monotonically into the box ``x = lo + (hi-lo)*sigmoid(k*u)``
(with ``k = 4/(hi-lo)``), so every iterate is strictly interior, the momentum
state is always consistent with the parameters, and the decode saturates
smoothly at the box edge.  ``_BOX_U_BOUND`` only caps the ``u`` iterate; the
decode is already fully saturated at the bounds by then.
"""


def _box_decode(u, lo, hi):
    """Monotonic box decode: strictly-interior ``x = lo + (hi-lo)*sigmoid(k*u)``.

    The per-coordinate input scale ``k = 4/(hi-lo)`` makes the local slope
    ``dx/du = 4*sigmoid'(k u)`` O(1) in the box interior, so a ``u``-step of
    size ``lr`` moves the physical parameter by ~``lr`` metres — the same step
    the raw clip-based path produced, but without the clip.
    """
    lo = jnp.asarray(lo, dtype=u.dtype)
    hi = jnp.asarray(hi, dtype=u.dtype)
    k = 4.0 / (hi - lo)
    return lo + (hi - lo) * jax.nn.sigmoid(k * u)


def _box_encode(x, lo, hi):
    """Inverse of :func:`_box_decode` (logit) to map a physical start into ``u``."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    k = 4.0 / (hi - lo)
    p = np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p)) / k


def _box_u_adapter(fn, prepare, lo, hi):
    """Wrap a physical ``(loss, grad_x)`` objective into the unconstrained ``u``.

    Evaluates the physical objective at ``x = _box_decode(u)`` and returns the
    gradient w.r.t. ``u`` by the chain rule (the decode is element-wise, so
    ``grad_u = grad_x * d(x)/du``).  ``prepare`` (the eager coil-state hook of
    the non-traceable path) is folded in here, applied at the decoded physical
    ``x``.
    """
    def fn_u(u):
        x = _box_decode(u, lo, hi)
        if prepare is not None:
            prepare(x)
        loss, grad_x = fn(x)
        k = 4.0 / (jnp.asarray(hi) - jnp.asarray(lo))
        d = k * (jnp.asarray(hi) - jnp.asarray(lo)) * jax.nn.sigmoid(k * u) * (
            1.0 - jax.nn.sigmoid(k * u))
        return loss, grad_x * d
    return fn_u


def _decode_history(history, lo, hi):
    """History x (raw ``u``) -> physical box ``x``, then GEOM_KEYS enrichment."""
    decoded = []
    for e in history:
        entry = dict(e)
        entry["x"] = [float(v) for v in _box_decode(jnp.asarray(e["x"]), lo, hi)]
        decoded.append(entry)
    return driver.enrich_history(decoded)


def run_optax(core_spec, name, x0=None, max_iters=200, lr=None, w_vol=0.0,
              mesh_size=None, vg=None, loss_fn=None, verbose=True,
              tol=None, patience=0, improve_tol=1e-6, max_grad_norm=None):
    """Run an Adam-family member on the jitted traceable objective.

    ``vg``/``loss_fn`` come as a pair from :func:`build_vg` (shared across the
    family so the ~10 s XLA compile and the problem build happen once).  If
    omitted they are built here.  ``tol``/``patience``/``improve_tol`` and
    ``max_grad_norm`` opt in to early termination / gradient clipping (see
    :func:`_adam_driver`); defaults are OFF so the recorded history stays 1:1
    with ``max_iters`` for contract tests.  The production entry
    (:func:`main`, :func:`~optimizers.benchmark.run_benchmark`) enable them.
    """
    if name not in ADAM_FAMILY:
        raise ValueError(
            f"Unknown Adam-family optimizer {name!r}; choices: {sorted(ADAM_FAMILY)}")
    mesh_size = mesh_size or core_spec["mesh_size"]
    target_L = core_spec["target_L"]
    lo, hi = objective.bounds_arrays(core_spec)
    lr = lr if lr is not None else DEFAULT_LR.get(name, 3e-4)

    if vg is None:
        loss_fn, vg = build_vg(core_spec, w_vol=w_vol, mesh_size=mesh_size,
                               backend=None)

    x0 = objective.default_x0(core_spec) if x0 is None else np.asarray(x0, dtype=float)
    x0 = np.clip(x0, lo, hi)

    u0 = _box_encode(x0, lo, hi)
    u_lo = -_BOX_U_BOUND * np.ones_like(lo)
    u_hi = _BOX_U_BOUND * np.ones_like(hi)
    fn_u = _box_u_adapter(vg, None, lo, hi)
    out = _adam_driver(fn_u, u0, u_lo, u_hi, ADAM_FAMILY[name](lr), max_iters,
                       tol=tol, patience=patience, improve_tol=improve_tol,
                       max_grad_norm=max_grad_norm)

    best_x = np.clip(
        np.asarray(_box_decode(jnp.asarray(out["x_best"]), lo, hi), dtype=float),
        lo, hi)
    best_loss = out["best_loss"]
    params_dict = objective.make_params_dict(best_x, core_spec)
    _, L_opt, _ = loss_fn.frozen(best_x)
    L_opt = float(L_opt)
    history = _decode_history(out["history"], lo, hi)

    if verbose:
        print(f"  {name}: completed {out['nit']} iters "
              f"({out['message']}), {out['diagnostics']['time']:.2f}s")
        print(f"  Final L = {L_opt * 1e9:.1f} nH, Target = {target_L * 1e9:.1f} nH")
        print(f"  Best loss = {best_loss:.4e}")

    return {
        "optimizer": name,
        "core": core_spec["name"],
        "x_opt": best_x,
        "params": params_dict,
        "L_opt": L_opt,
        "loss": best_loss,
        "history": history,
        "nit": out["nit"],
        "time": out["diagnostics"]["time"],
        "lr": lr,
        "success": best_loss < 1.0,
        "message": out["message"],
    }


def run_adam(core_spec, x0=None, max_iters=200, lr=3e-4, w_vol=0.0, mesh_size=None,
             fwd_opts=None, adj_opts=None, backend=None, verbose=True, traceable=False,
             vg=None, loss_fn=None, tol=None, patience=0, improve_tol=1e-6,
             max_grad_norm=None):
    mesh_size = mesh_size or core_spec["mesh_size"]
    target_L = core_spec["target_L"]
    lo, hi = objective.bounds_arrays(core_spec)

    if traceable and vg is not None and loss_fn is not None:
        loss_and_grad = vg
    else:
        loss_fn = objective.make_loss_fn(
            core_spec, w_vol=w_vol, mesh_size=mesh_size,
            fwd_opts=fwd_opts, adj_opts=adj_opts,
            backend=backend, traceable=traceable,
        )
        if traceable:
            # One fused, compiled forward+backward; coils traced through (no
            # per-iteration prepare).  Each iteration is one FEM solve + adjoint.
            loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
        else:
            loss_and_grad = jax.value_and_grad(loss_fn)

    x0 = objective.default_x0(core_spec) if x0 is None else np.asarray(x0, dtype=float)
    x0 = np.clip(x0, lo, hi)

    u0 = _box_encode(x0, lo, hi)
    u_lo = -_BOX_U_BOUND * np.ones_like(lo)
    u_hi = _BOX_U_BOUND * np.ones_like(hi)
    fn_u = _box_u_adapter(
        loss_and_grad, loss_fn.prepare if not traceable else None, lo, hi)
    out = _adam_driver(fn_u, u0, u_lo, u_hi, optax.adam(learning_rate=lr),
                       max_iters, prepare=None, pass_params=False,
                       tol=tol, patience=patience, improve_tol=improve_tol,
                       max_grad_norm=max_grad_norm)

    best_x = np.clip(
        np.asarray(_box_decode(jnp.asarray(out["x_best"]), lo, hi), dtype=float),
        lo, hi)
    best_loss = out["best_loss"]
    params_dict = objective.make_params_dict(best_x, core_spec)
    if traceable:
        # Reuse the compiled forward: ``frozen(best_x) -> (loss, L, vol)`` — no
        # extra problem build, no scipy.
        _, L_opt, _ = loss_fn.frozen(best_x)
        L_opt = float(L_opt)
    else:
        L_opt, _ = objective.solve_forward(params_dict, mesh_size, backend=backend)
        L_opt = float(L_opt)
    history = _decode_history(out["history"], lo, hi)
    opt_time = out["diagnostics"]["time"]

    if verbose:
        print(f"  Adam completed in {out['nit']} iters ({out['message']}), "
              f"{opt_time:.2f}s")
        print(f"  Final L = {L_opt * 1e9:.1f} nH, Target = {target_L * 1e9:.1f} nH")
        print(f"  Best loss = {best_loss:.4e}")

    return {
        "optimizer": "Adam",
        "core": core_spec["name"],
        "x_opt": best_x,
        "params": params_dict,
        "L_opt": L_opt,
        "loss": best_loss,
        "history": history,
        "nit": out["nit"],
        "time": opt_time,
        "success": best_loss < 1.0,
        "message": out["message"],
    }


def main():
    """Run the whole Adam family on one core, one report folder per optimizer.

    A single shared ``build_vg`` compile is reused across the family; each
    optimizer's optimized design is then reported run_forward-style (console
    print blocks + geometry/psi/bfield PNGs) under
    ``output/adam/<optimizer_name>/``.

    Usage::

        python optimizers/adam.py [core] [mesh_size] [max_iters] [optimizer]

    The optional 4th argument runs ONLY that family member (e.g. ``adam`` or
    ``adabelief``).  Omit it to run all 11.
    """
    from new_fem_engine.report import report_design, OUTPUT_DIR

    core_name = sys.argv[1] if len(sys.argv) > 1 else "pq_40x40"
    mesh_size = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
    max_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    only = sys.argv[4] if len(sys.argv) > 4 else None

    if only is not None and only not in ADAM_FAMILY:
        raise ValueError(
            f"Unknown Adam-family optimizer {only!r}; choices: {sorted(ADAM_FAMILY)}")

    spec = objective.load_core_spec(core_name)
    target_nH = spec["target_L"] * 1e9

    family = [only] if only else sorted(ADAM_FAMILY)
    print(f"=== Adam family | core {core_name} | target {target_nH:.0f} nH | "
          f"mesh {mesh_size} m | {max_iters} iters/opt ===")
    print(f"family: {', '.join(family)}\n")

    t_build = time.time()
    loss_fn, vg = build_vg(spec, mesh_size=mesh_size)
    build_s = time.time() - t_build
    print(f"shared JIT compile + problem build: {build_s:.1f}s (one-time)\n")
    out_root = os.path.join(OUTPUT_DIR, "adam")

    for name in family:
        t_wall = time.time()
        print(f"--- {name} ---")
        r = run_optax(spec, name, max_iters=max_iters, mesh_size=mesh_size,
                      vg=vg, loss_fn=loss_fn, verbose=False,
                      tol=0.25, patience=40, improve_tol=1e-3,
                      max_grad_norm=1.0)   # |ΔL| < 0.5 nH; clip the ~1e7 raw grads
        ms_per_iter = r["time"] / max(r["nit"], 1) * 1000.0
        print(f"  L_opt = {r['L_opt'] * 1e9:.1f} nH | loss = {r['loss']:.3e}")
        print("  final design dimensions (mm):")
        for k in objective.GEOM_KEYS:
            print(f"    {k:24s} = {r['params'][k] * 1e3:9.3f} mm")
        print(f"  timing: {r['nit']} iters in {r['time']:.1f}s "
              f"({ms_per_iter:.0f} ms/iter; each iter = one jitted FEM "
              f"forward + adjoint solve at {mesh_size * 1e3:.1f} mm mesh)\n")
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        report_design(spec, mesh_size, r["x_opt"], out_dir, label=name)
        driver.save_convergence_plot(
            r["history"], os.path.join(out_dir, "convergence.png"),
            title=f"{name} (mesh {mesh_size * 1e3:.3f} mm)",
            target_nH=target_nH, standard=True)
        total = time.time() - t_wall
        report_s = total - r["time"]
        print(f"  report + plots saved under {out_dir}/")
        print(f"  RUN SUMMARY ({name}): optimizer {r['time']:.1f}s + "
              f"report/solve {report_s:.1f}s = total wall {total:.1f}s "
              f"(the report block above only measures the final single solve "
              f"+ plots, not the optimizer)\n")
    print(f"\nDone. Adam-family reports under {out_root}/")


if __name__ == "__main__":
    main()
