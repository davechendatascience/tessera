<p align="center">
  <img src="https://raw.githubusercontent.com/davechendatascience/tessera/main/resources/logo.png" alt="tessera" width="480">
</p>

# tessera

> *"Each piece of a mosaic is small and simple. The whole is composed."*

**Tessera** is a Python library for **symbolic regression with
measure-theoretic operators**. It searches for short, interpretable
mathematical formulas that fit your data — like PySR or Eureqa, but
with a richer operator vocabulary built around signed measures,
linear/bilinear functionals, and 2-D fields.

Distinct value vs other SR libraries:

- **First-class temporal/spatial structure.** Convolutions with
  arbitrary measures are primitive operators, not synthesised from
  pointwise compositions. Natural for time series, PDEs, images.
- **Axis-semantic type system.** Variables declare their dimensions
  *and* the invariance group acting on each (translation, causal
  translation, permutation, cyclic, ...). Constrain the search to
  respect data symmetry.
- **Branch-and-bound search.** Cheap interval-arithmetic bounds let
  the search prune provably-suboptimal candidates without evaluating
  them on the full dataset.
- **Honest about its scope.** Linear functionals are first-class via
  measure theory. Nonlinear (Volterra) extensions are exposed and
  clearly marked. GPU support is a tracked milestone (CPU works today).

## ⭐ Gradient-free symbolic regression — `tessera.search.csp`

Alongside the GP searcher, tessera ships a **gradient-free, enumerative** SR
engine: a CSP-generated *const-free expression dictionary* (symmetry-broken,
deduped) fit by a sparse **linear** combination — constants enter as
closed-form least-squares coefficients, **no gradient descent** — plus a
**top-down decomposition** driver (outer-op peel → polynomial-STLSQ leaf →
validation-gated separability) that recovers deep compositional laws a flat
search can't reach. It's the FFX / SINDy family with an AI-Feynman-style
decomposition prepass, in tessera's operator vocabulary.

```python
import numpy as np
from tessera.search.csp import discover_decompose

env = {"v": np.random.uniform(0, 0.9, 2000)}
y   = np.sqrt(1 - env["v"]**2)                      # relativistic factor
res = discover_decompose(env, y)
# -> sqrt(1 - v^2): recovered exactly (rel ~ 1e-30), gradient-free
```

**Feynman result (held-out, machine-precision symbolic match).** The
decomposition prepass lifts *genuine exact* recovery from **8/30 → 24/30** on a
30-equation Feynman subset, breaking the size-6 relativistic-form wall
(`1/√(1−v²/c²)`) that a single enumeration cannot reach — gradient-free, seconds
per equation on CPU. (`benchmarks/run_feynman_decompose.py`.)

### How it compares

A different *design point* from the heavyweight SR systems: it trades
exhaustive constant optimisation for being **gradient-free, deterministic,
fast, and machine-precision-exact** on the forms it reaches. The numbers below
are **not** all apples-to-apples — recovery metric and benchmark subset differ
(see caveats). The gplearn row is a true head-to-head on our exact subset and
metric (`benchmarks/run_feynman_vs_baselines.py`).

| Method | Family | Gradient-free | Feynman recovery | Cost |
|---|---|---|---|---|
| **`tessera.search.csp`** | enumerative dictionary + sparse-linear + decomposition | ✅ | **24/30 exact** + 4 approx (machine-precision, held-out) | gradient-free, CPU |
| gplearn | genetic programming (no const-opt) | ✅ | **4/30 exact** + 2 approx (same subset/metric) | ~0.5s/eq (fast cfg) |
| PySR [1] | GP + constant optimisation | ✅ | strong on full Feynman | Julia backend, slower |
| AI-Feynman [2] | NN + dimensional analysis | ❌ (trains a NN) | ~90/100 full set, looser tol | NN/eq, heavy |
| FFX [3] | basis enumeration + elastic-net | ✅ | linear-in-basis | fast, deterministic |
| SINDy [4] | fixed library + sparse regression | ✅ | dynamics (≈ our poly-leaf) | fast |

On this runnable head-to-head ([`run_feynman_vs_baselines.py`](benchmarks/run_feynman_vs_baselines.py)),
tessera recovers **24/30 exact vs gplearn's 4/30** — gplearn (basic GP, no
constant optimisation) reaches only the const-free product forms (`m·g·z`,
`q/C`, …). The heavier PySR (GP **+** constant optimisation) and AI-Feynman
(NN + dimensional analysis) recover more than us on the full set, but are far
costlier — that's the design-point trade.

> **Caveats (read these).** Our **24/30** uses a *strict* machine-precision
> metric (`rel < 1e-8` — the true closed form, not just a good fit) on a
> **30-equation subset**; AI-Feynman's ~90/100 is the **full 100** with a
> looser tolerance. The table positions *design points*, not a single
> leaderboard, and we do **not** claim to beat PySR/AI-Feynman on total
> recovery — they're heavier and stronger there. tessera's edge is being
> gradient-free, fast, exact-on-a-subset, and adding the decomposition prepass.

<sub>[1] Cranmer 2023, PySR. [2] Udrescu & Tegmark 2020, AI-Feynman. [3]
McConaghy 2011, FFX. [4] Brunton et al. 2016, SINDy. Decomposition follows
AI-Feynman's separability/dimensional-analysis idea, gradient-free.</sub>

## Quickstart

```bash
git clone https://github.com/davechendatascience/tessera
cd tessera
pip install -e .
pytest tests/                  # 392 tests should pass
```

(Tessera is installed from this git repo. PyPI publishing is deferred;
`pytessera` is reserved as the future distribution name when we ship.)

Then run any of the example benchmarks:

```bash
python benchmarks/run_lorenz63.py                # ODE rediscovery
python benchmarks/run_heat_equation_discovery.py # 2-D PDE discovery
python benchmarks/run_mnist_feature_discovery.py # CV (small subset)
python benchmarks/run_const_opt_demo.py          # how const polish helps
```

Each writes a result markdown under `benchmarks/results/`.

## Hello-world: symbolic regression in 10 lines

```python
import numpy as np
from tessera.search import GP, GPConfig

# Synthetic: y = sin(2*pi*x) + 0.1*noise
rng = np.random.default_rng(0)
x = rng.standard_normal(500)
y = np.sin(2 * np.pi * x) + 0.1 * rng.standard_normal(500)

gp = GP(GPConfig(pop_size=80, n_gens=30, optimize_constants_every=3))
front = gp.run({"x": x}, y, feature_names=["x"])

best = min(front, key=lambda c: c.train_loss)
print(f"loss={best.train_loss:.4g}  cx={best.complexity}  tree={best.tree}")
```

## Modules

| Module | Status | Purpose |
|---|---|---|
| [`tessera.expression`](src/tessera/expression/README.md) | shipping | Tree types, measure-theoretic operators, simplification, algebraic identities |
| [`tessera.search`](src/tessera/search/README.md) | shipping | GP, Simulated Annealing, Random Search, Hall of Fame, branch-and-bound bounds |
| [`tessera.search.csp`](docs/shipped/csp_symbolic_regression.md) | shipping | **Gradient-free enumerative SR**: CSP dictionary + sparse-linear fit + top-down decomposition (Feynman 8→24 exact). See [headline](#-gradient-free-symbolic-regression--tesserasearchcsp) |
| [`tessera.koopman`](src/tessera/koopman/README.md) | shipping | Closed-form latent Koopman with time-delay embedding |
| `tessera.expression.simplify` | shipping | Rule-based + AC-normalisation simplifier (canonical-form pass) |
| `tessera.expression.axes` | shipping | Axis-semantic type system: Translation, CausalTranslation, Permutation, Cyclic, ... |
| `tessera.expression.interval` | shipping | Sound interval arithmetic for B&B lower bounds (L1 norms on measures) |
| `tessera.backend` | scaffold | CPU/GPU switchable backend API; full JAX backend tracked in [`docs/shipped/gpu_backend.md`](docs/shipped/gpu_backend.md) |
| `tessera.ssm` | planned | Kalman / state-space filtering |
| `tessera.mts` | planned | Multi-timescale analysis |

## Examples by modality

### Time series — rediscover a known signal

```python
import numpy as np
from tessera.search import GP, GPConfig

# y = ema(x, halflife=24) - x.shift(7)
rng = np.random.default_rng(0)
x = rng.standard_normal(2000)
# (true target computed externally; SR finds a measure-theoretic form)
```

See `benchmarks/run_lorenz63.py` and `benchmarks/run_diff_eml_fhn.py`
for full examples on dynamical-systems data.

### 2-D PDE discovery

Discover the heat equation `dT/dt = α·∇²T` from data:

```bash
python benchmarks/run_heat_equation_discovery.py
```

Tessera's `FunctionalOp2D` with `Measure2D` represents the Laplacian
as a discoverable parameter rather than a hand-coded operator.

### Image feature discovery (CV)

`benchmarks/run_mnist_feature_discovery.py`: SR discovers a 2D
feature kernel for MNIST 0-vs-rest classification. With ~200 training
samples and 25 GP generations, the search converges on the classical
horizontal Laplacian `[+1, -2, +1]` (a recognisable edge detector)
in ~25 seconds and reaches 0.80-0.82 test accuracy. Kernel
visualisation saved to `benchmarks/results/mnist_discovered_kernel.png`.

## Axis-semantic declarations

For structured sensor data, declare what KIND of axis each dimension
is. The compatibility checker validates that operators respect the
declared symmetries:

```python
from tessera.expression.axes import Axis, Invariance, TypedVar

# Time series: causal translation invariance
ts = TypedVar("returns", axes=(
    Axis("time", 10000, Invariance.CAUSAL_TRANSLATION),
))

# Image: 2-D translation invariance
img = TypedVar("image", axes=(
    Axis("height", 28, Invariance.TRANSLATION),
    Axis("width", 28, Invariance.TRANSLATION),
))

# Multi-asset basket: time × asset (permutation-invariant)
basket = TypedVar("prices", axes=(
    Axis("time", 10000, Invariance.CAUSAL_TRANSLATION),
    Axis("asset", 8, Invariance.PERMUTATION),
))
```

The compatibility checker (`check_compatibility(tree, typed_env)`)
catches violations like applying a translation-equivariant
convolution to a permutation axis.

## Theory / research notes

The library is built around a Knuth-grounded formal framework:
SR-for-fit as a single-agent perfect-information game. The research
notes below develop the theory and connect it to specific
implementation pieces:

- [`docs/PROJECT_GOALS.md`](docs/PROJECT_GOALS.md) — what tessera is for
- [`docs/shipped/framework_synthesis.md`](docs/shipped/framework_synthesis.md) — every implementation mapped to its framework role
- [`docs/research/fit_as_perfect_info_game.md`](docs/research/fit_as_perfect_info_game.md) — Knuth-grounded framework
- [`docs/research/measure_theory_and_perfect_info.md`](docs/research/measure_theory_and_perfect_info.md) — measure algebra layer
- [`docs/research/search_as_energy_min.md`](docs/research/search_as_energy_min.md) — algebraic equivalence as free budget
- [`docs/research/invariance_in_sr.md`](docs/research/invariance_in_sr.md) — axis semantics direction
- [`docs/research/gpu_and_cv_via_sr.md`](docs/research/gpu_and_cv_via_sr.md) — GPU + CV scoping
- [`docs/shipped/gpu_backend.md`](docs/shipped/gpu_backend.md) — JAX backend port roadmap

## Status

- **392 tests passing** across `tessera.expression`, `tessera.search`,
  `tessera.koopman`, `tessera.backend`, and `tessera.expression.axes`
- CPU is the default; GPU/JAX backend has a public API (`set_backend`)
  with internal porting tracked in the milestone doc
- Multiple working benchmarks covering time series, ODEs, 2-D PDEs,
  weather (NCEP), and MNIST image classification

## Install

Currently installed from this git repo only. (PyPI publishing is
deferred; the bare `tessera` name is taken, `pytessera` is available
and reserved for future use.)

```bash
# Editable install for development
git clone https://github.com/davechendatascience/tessera
cd tessera
pip install -e .

# Or directly from a consuming project
pip install "git+https://github.com/davechendatascience/tessera.git"
```

Optional extras:

```bash
pip install -e .[dev]        # + pytest, ruff, mypy
pip install -e .[all]        # + sympy, pysr (optional)
```

JAX support (for the future GPU backend):

```bash
pip install jax jaxlib       # CPU JAX
# pip install jax[cuda12_pip] # GPU JAX
```

## License

MIT.
