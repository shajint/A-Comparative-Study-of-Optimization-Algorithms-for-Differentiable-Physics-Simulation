"""Bounded L-BFGS optimizer adapter on the ``new_fem_engine`` objective.

Quasi-Newton with box constraints, fully JAX (Level A: jitted step + Python
loop, parity with the Adam family).  A bounded two-loop-recursion L-BFGS in
the L-BFGS-B family runs as a Python driver loop, calling our compiled,
jitted forward + implicit adjoint (``build_vg``) directly and computing the
search direction in-jax; the box bounds are enforced by projecting every
iterate onto ``(lo, hi)`` and the step size follows an Armijo backtracking
line search.  Nothing scipy is in the traced program.

jaxopt.LBFGSB was the original backend but its zoom line search runs the
objective inside a ``lax.while_loop`` trace, which re-traces the whole FEM
solve on every function evaluation (~7 s each) — so the driver was replaced
by the Python loop that calls the compiled gradient (sub-ms each).

The learning rate is set by the line search (hence no learning-rate /
meta-optimisation, exactly as in the paper).  The result follows the objective
contract (``optimizer, core, x_opt, params, L_opt, loss, history, nit, time,
success, message``) plus the solve-cost counters (``nfev``/``njev``) used by
the paper's comparison.
"""

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from optimizers import driver, objective, backends
from optimizers.adam import build_vg


def _lbfgs_driver(fn, x0, lo, hi, max_iters=80, tol=1e-8, jit=True, prepare=None,
                  tol_loss=None, maxls=10, memory=10, step_cap=0.2):
    """Generic bounded L-BFGS loop — GEOM_KEYS-free, testable on toy objectives.

    Level A parity with the Adam family: a Python driver loop calling ``fn(x)
    -> (value, grad)`` (the compiled ``build_vg`` gradient) directly, with the
    box-constrained two-loop-recursion quasi-Newton step computed in-jax.

    Why not jaxopt.LBFGSB?  Its zoom line search runs ``fun`` inside a
    ``lax.while_loop`` trace, which re-traces the whole FEM solve on every
    function evaluation (~6.7 s each here); this loop calls the compiled
    gradient with concrete arrays (sub-ms each) and does the Armijo
    backtracking in Python.  The method stays a bounded quasi-Newton in the
    L-BFGS-B family: two-loop recursion for the search direction, projection
    onto ``[lo, hi]``, Armijo sufficient-decrease line search, and a step-size
    cap (``step_cap`` × box width) so the raw quasi-Newton step cannot slam
    the box when the gradient is huge.

    ``maxls`` caps the line-search evals per step, ``tol_loss`` opt-in
    early-terminates once the best loss is ``<= tol_loss`` (production sweep
    uses it so it stops at |ΔL| < 0.5 nH instead of grinding to a 1e-8
    projected gradient).  ``tol`` is the projected-gradient optimality
    threshold (KKT).  ``jit`` is retained for API compatibility (the loop is
    always a Python loop; the per-eval gradient is the caller's choice).
    Returns the uniform driver dict (see ``driver.py``); ``diagnostics``
    carry the proj-grad error trajectory and the ``nfev``/``njev`` counters.
    """
    def fun(x):
        """(value, grad); NaN-guarded so a bad step can't kill the loop."""
        if prepare is not None:
            prepare(x)
        value, grad = fn(x)
        grad = jnp.where(jnp.isnan(grad), 0.0, grad)
        return value, grad

    x = jnp.asarray(x0, dtype=jnp.float64)
    lo_a = jnp.asarray(lo, dtype=jnp.float64)
    hi_a = jnp.asarray(hi, dtype=jnp.float64)
    max_step = step_cap * float(jnp.max(hi_a - lo_a))

    value, grad = fun(x)
    nfev = njev = 1

    history = []
    errors = []
    S, Y = [], []          # L-BFGS memory: (s, y) curvature pairs
    early = False

    for i in range(max_iters):
        history.append({
            "iteration": i,
            "loss": float(value),
            "x": np.asarray(x).tolist(),
        })
        # projected-gradient error (KKT residual) — stops at a bound optimum
        errors.append(float(jnp.max(jnp.abs(jnp.clip(x - grad, lo_a, hi_a) - x))))
        if tol_loss is not None and float(value) <= tol_loss:
            early = True
            break
        if errors[-1] <= tol:
            break

        # --- two-loop recursion: d = -H^-1 grad (in-jax, 5-dim, cheap) -------
        q = grad
        n_pairs = len(S)
        alphas = []
        for k in range(n_pairs - 1, -1, -1):
            s, y = S[k], Y[k]
            rho = 1.0 / jnp.maximum(jnp.dot(y, s), 1e-12)
            a = rho * jnp.dot(s, q)
            alphas.append(a)
            q = q - a * y
        gamma = 1.0
        if n_pairs:
            s, y = S[-1], Y[-1]
            gamma = jnp.dot(s, y) / jnp.maximum(jnp.dot(y, y), 1e-12)
        r = gamma * q
        for k in range(n_pairs):
            s, y = S[k], Y[k]
            rho = 1.0 / jnp.maximum(jnp.dot(y, s), 1e-12)
            b = rho * jnp.dot(y, r)
            r = r + s * (alphas[n_pairs - 1 - k] - b)
        d = -r

        d_norm = float(jnp.max(jnp.abs(d)))
        if d_norm > max_step:
            d = d * (max_step / d_norm)

        # --- projected Armijo backtracking line search ------------------------
        alpha = 1.0
        accepted = False
        for _ in range(maxls):
            xt = jnp.clip(x + alpha * d, lo_a, hi_a)
            vt, gt = fun(xt)
            nfev += 1
            njev += 1
            if float(vt) <= float(value) + 1e-4 * alpha * float(jnp.dot(grad, xt - x)):
                s, y = xt - x, gt - grad
                # keep the pair only if the curvature condition s^T y > 0 holds
                # (L-BFGS memory validity; skip on nonconvex/overshooting steps)
                if float(jnp.dot(s, y)) > 1e-12:
                    S.append(s)
                    Y.append(y)
                    if len(S) > memory:
                        S.pop(0)
                        Y.pop(0)
                x, value, grad = xt, vt, gt
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break                    # line search failed; keep best-so-far

    # Best design = argmin loss over the recorded history (mirrors run_optax).
    if history:
        best = min(history, key=lambda e: e["loss"])
        best_loss = float(best["loss"])
        x_best = np.clip(np.asarray(best["x"]), lo, hi)
    else:
        best_loss = float("nan")
        x_best = np.clip(np.asarray(x0), lo, hi)

    return {
        "x_best": x_best,
        "best_loss": best_loss,
        "history": history,
        "nit": len(history),
        "early": early,
        "diagnostics": {
            "proj_grad_error": errors,
            "nfev": int(nfev),
            "njev": int(njev),
        },
    }


def run_lbfgs(core_spec, x0=None, max_iters=80, w_vol=0.0, mesh_size=None,
              backend=None, vg=None, loss_fn=None, verbose=True, traceable=None,
              tol_loss=None, maxls=30):
    """Run bounded L-BFGS (Python loop, compiled gradient) on the FEM objective.

    ``vg``/``loss_fn`` come as a pair from :func:`build_vg` (shared with the
    Adam-family sweep so the ~10 s XLA compile and problem build happen once).
    If omitted they are built here.

    ``traceable=True`` (default when ``backend`` is None or a lineax) uses the
    one compiled ``jax.jit(value_and_grad)`` program — sub-ms function evals,
    so the L-BFGS line search is cheap too.  On the non-traceable
    spsolve/feax backends (``traceable=False``) the gradient is computed by
    the eager custom_vjp implicit-adjoint (the path pinned by
    ``test_backend_diff.py``), so every backend can drive the loop.

    ``tol_loss``/``maxls`` default OFF/conservative to keep contract tests
    unchanged; the production sweep (:func:`main`, ``run_benchmark``) enables
    them — early-stop at a loss that already means |ΔL| < 0.5 nH and a
    line-search cap of 10.
    """
    mesh_size = mesh_size or core_spec["mesh_size"]
    target_L = core_spec["target_L"]
    lo, hi = objective.bounds_arrays(core_spec)
    if traceable is None:
        traceable = backend is None or backend in backends.TRACEABLE_BACKENDS

    if vg is None or loss_fn is None:
        if traceable:
            loss_fn, vg = build_vg(core_spec, w_vol=w_vol, mesh_size=mesh_size,
                                   backend=backend)
        else:
            loss_fn = objective.make_loss_fn(
                core_spec, w_vol=w_vol, mesh_size=mesh_size, backend=backend,
                traceable=False)
            vg = jax.value_and_grad(loss_fn)

    x0 = objective.default_x0(core_spec) if x0 is None else np.asarray(x0, dtype=float)
    x0 = np.clip(x0, lo, hi)

    tol = 1e-8  # projected-gradient optimality threshold (KKT residual)
    t_start = time.time()
    out = _lbfgs_driver(vg, x0, lo, hi, max_iters=max_iters, tol=tol,
                        prepare=loss_fn.prepare if not traceable else None,
                        tol_loss=tol_loss, maxls=maxls)
    opt_time = time.time() - t_start

    x_opt = np.clip(out["x_best"], lo, hi)
    best_loss = out["best_loss"]
    params_dict = objective.make_params_dict(x_opt, core_spec)
    if traceable:
        # Reuse the compiled forward: ``frozen(x_opt) -> (loss, L, vol)`` — no
        # extra problem build, no scipy.
        _, L_opt, _ = loss_fn.frozen(x_opt)
        L_opt = float(L_opt)
    else:
        L_opt, _ = objective.solve_forward(params_dict, mesh_size, backend=backend)
        L_opt = float(L_opt)
    history = driver.enrich_history(out["history"])
    nit = len(history)
    proj_err = out["diagnostics"]["proj_grad_error"][-1]

    converged = proj_err <= tol
    success = best_loss < 1.0   # objective contract (loss = ((L-target)*1e9)^2 < 1)
    if verbose:
        stop = ("early-stop" if out["early"] else
                "converged" if converged else "max_iters")
        print(f"  L-BFGS-B {stop} in {nit} iters, {opt_time:.2f}s")
        print(f"  Final L = {L_opt * 1e9:.1f} nH, Target = {target_L * 1e9:.1f} nH")
        print(f"  Best loss = {best_loss:.4e} | proj-grad error = {proj_err:.2e}")

    return {
        "optimizer": "L-BFGS-B",
        "core": core_spec["name"],
        "x_opt": x_opt,
        "params": params_dict,
        "L_opt": L_opt,
        "loss": best_loss,
        "history": history,
        "nit": nit,
        "time": opt_time,
        "nfev": out["diagnostics"]["nfev"],
        "njev": out["diagnostics"]["njev"],
        "success": success,
        "converged": bool(converged),
        "message": (f"L-BFGS-B {('early-stop ' if out['early'] else '')}"
                    f"proj-grad error {proj_err:.2e}"),
    }


def main():
    """Run L-BFGS-B on one core; report + plots under ``output/lbfgs/L-BFGS-B/``.

    Usage::

        python optimizers/lbfgs.py [core] [mesh_size] [max_iters]
    """
    from new_fem_engine.report import report_design, OUTPUT_DIR

    core_name = sys.argv[1] if len(sys.argv) > 1 else "pq_40x40"
    mesh_size = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
    max_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 80

    spec = objective.load_core_spec(core_name)
    target_nH = spec["target_L"] * 1e9
    print(f"=== L-BFGS-B | core {core_name} | target {target_nH:.0f} nH | "
          f"mesh {mesh_size} m | {max_iters} iters ===")

    r = run_lbfgs(spec, max_iters=max_iters, mesh_size=mesh_size, verbose=True,
                  tol_loss=0.25, maxls=10)   # |ΔL| < 0.5 nH; cap line-search evals

    out_dir = os.path.join(OUTPUT_DIR, "lbfgs", "L-BFGS-B")
    os.makedirs(out_dir, exist_ok=True)
    report_design(spec, mesh_size, r["x_opt"], out_dir, label="L-BFGS-B")
    driver.save_convergence_plot(
        r["history"], os.path.join(out_dir, "convergence.png"),
        title=f"L-BFGS-B (mesh {mesh_size * 1e3:.3f} mm)", target_nH=target_nH)
    print(f"\nReport + plots saved under {out_dir}/")


if __name__ == "__main__":
    main()
