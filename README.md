# A Comparative Study of Optimization Algorithms for Differentiable Physics Simulation

A fully differentiable, JAX-based finite-element (FEM) framework for the inverse
design of magnetic components, used to systematically benchmark **gradient-based
and gradient-free** optimization algorithms — including evolutionary and
surrogate-based methods — on a common solver.

Built with JAX / Equinox / Lineax, the framework propagates exact gradients
through the entire magnetostatics solve — geometry construction, sparse matrix
assembly, and the linear solve — enabling physics-based AI optimization
(gradient-based design optimization) of PQ-core transformers without any
discretization of the physics by hand.

## Highlights

- **Fully differentiable 2D axisymmetric FEM solver** for magnetic components,
  with sparse stiffness assembly and exact `jax.grad` support through the solve.
- **Pluggable linear-solver backends** — `scipy.sparse` spsolve, Lineax (CG/BiCGSTAB/LU),
  and FEAX (direct/krylov) — interchangeable on a fixed mesh with agreement to
  ~1e-11 nH, enabling fair backend-vs-backend timing studies.
- **Five optimizer families under two paradigms** — a uniform contract
  (`run_<name>(core_spec)` -> result dict) that directly compares
  **gradient-based optimizers** (L-BFGS-B; Adam and its family — AdamW,
  AdaBelief, AMSGrad, RAdam, NAdam, Yogi, Lion, Fromage, ...) against
  **gradient-free optimizers** (CMA-ES, NSGA-III, and Bayesian optimization via
  Optuna/Scikit-Optimize).
- **Multi-objective objective** — inductance targeting with optional volume
  regularization and core-vs-air energy reporting.
- **Reproducible comparison protocol** — every optimized design is re-solved at a
  fixed production mesh for a fair cross-optimizer table, with wall-clock and
  solve-equivalent budgets.
- **Test suite** — 25+ tests covering solver correctness, gradient accuracy
  (finite-difference checks), optimizer correctness, mesh convergence, symmetry,
  and linearity.

## Project Structure

```
├── cores/              — PQ core geometry specs (32×30, 40×40) with YAML + drawings
├── new_fem_engine/     — Differentiable FEM engine (mesh, basis, assembly, solver)
│   ├── run_forward.py  — forward demo: single-core solve + backend comparison
│   ├── problem.py      — magnetostatic problem setup (BCs, material fill)
│   ├── solver.py       — differentiable linear solvers (spsolve / lineax / feax)
│   └── report.py       — plot + report generation
├── optimizers/         — optimizer adapters + comparison framework
│   ├── benchmark.py    — the uniform comparison harness
│   ├── objective.py    — differentiable multi-objective loss
│   ├── adam.py         — Adam family via optax
│   ├── lbfgs.py        — L-BFGS-B via SciPy
│   ├── cmaes.py        — CMA-ES
│   ├── nsga3.py        — NSGA-III (multi-objective, pymoo)
│   └── bayesian.py     — Bayesian optimization (Optuna / skopt)
├── comparative_study/  — end-to-end optimizer benchmark entry points
│   ├── main_run.py             — run ALL optimizers on ONE core
│   └── solver_speed_comparison.py — solver scaling study across mesh sizes
└── tests/              — 25+ unit tests (pytest)
```

## Installation

```bash
# Python 3.10+
pip install -r requirements.txt
pip install -e .          # installs the new_fem_engine package
```

The three linear-solver backends have different optional dependencies:

| Backend | Dependencies |
|---------|--------------|
| `spsolve` | `scipy` (always available) |
| `lineax` | `lineax`, `equinox` |
| `feax` | `feax` |

Optimizers load lazily, so optional frameworks (`optax`, `pymoo`, `cma`,
`optuna`, `scikit-optimize`) are only required when that optimizer is used.

## Quick Start

### 1. Forward solve with backend comparison

```bash
python new_fem_engine/run_forward.py            # default core (pq_40x40)
python new_fem_engine/run_forward.py pq_32x30   # another core
```

### 2. Full optimizer comparison on one core

```bash
python comparative_study/main_run.py
```

Edit the `CONFIG` block at the top of `main_run.py` to choose the core, target
inductance, solver backend, mesh size, budget, and optimizer list. Results and
plots land under `output/comparative_study/<core>/`.

### 3. Solver scaling study

```bash
python comparative_study/solver_speed_comparison.py
```

## Running the Tests

```bash
pytest                     # fast tests
pytest -m slow             # long-running physics/verification tests
```

## Methodology

All optimizers — gradient-based (L-BFGS-B, Adam family) and gradient-free
(CMA-ES, NSGA-III, Bayesian) — optimize the same differentiable objective on the
same core at a fixed budget (measured in FEM solve-equivalents), giving a
controlled comparison of the two paradigms for physics-based AI optimization.
L-BFGS-B runs on a coarse
sweep mesh to avoid a large one-time XLA compile; every optimized design is
then re-solved at the production mesh (1 mm) so the reported `L@1mm` comparison
is unbiased across optimizers. The linear-solver backends are validated to agree
to ~1e-11 nH on identical geometry, so backend choice does not confound the
optimizer comparison.

## Author

**Shajin Thankaswamy** — Project Student, Fraunhofer IISB

## License

This project is released for academic and research use.
