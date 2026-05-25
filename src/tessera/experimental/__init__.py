"""tessera.experimental — implementations of research-note conjectures.

This subpackage mirrors `docs/research/` at the code level. A research
note proposes a conjecture; an experimental module implements it as
a testable artifact. If empirical results validate the conjecture,
the module graduates to its proper home (`tessera.search`,
`tessera.expression`, etc.). If results falsify it, the module is
removed and the falsification is documented in the research note.

The discipline
--------------

1. **Every module here implements at least one named conjecture** from
   a research note. The module docstring cites the note + the
   specific conjecture number (e.g., "C1 from process_discovery_sr.md").

2. **No production code imports from `tessera.experimental`.** Enforced
   by `tests/test_dependency_structure.py`. Experimental code can
   import from anywhere (it's downstream); production code must not
   depend on it (one-way relationship).

3. **Modules carry their own provenance + status.** Each module's
   docstring includes:
     - Provenance (which research note, which conjecture)
     - Status (untested / partially-validated / validated / falsified)
     - Graduation criterion (what evidence would promote it)
     - Removal criterion (what evidence would remove it)

4. **No silent promotions.** When a module graduates, the move is a
   commit that updates the research note (marking the conjecture
   validated), moves the code to its production home, and removes
   the experimental version. The CHANGELOG entry documents the
   graduation.

5. **Audit table maintained in this module's docstring.** Tracks
   the current contents and their status. Sec §Inventory below.

Lifecycle parallel
------------------

```
docs/research/X.md           tessera/experimental/X.py
  (conjecture)        ↔        (implementation)

       │ validated                       │ validated
       ▼                                 ▼
docs/planned/roadmap.md §N.M      tessera/<production-home>/X.py
  (committed)                       (production code)

       │ shipped                         │ shipped
       ▼                                 ▼
docs/shipped/X.md                 tessera/<home>/X.py + tests
  (validated artifact)              + CHANGELOG entry
```

Inventory (current)
-------------------

| Module | Conjecture | Status | Last evaluation |
|---|---|---|---|
| abc_scoring.py | C1-refined (process_discovery_sr.md §6.2 + §7.1) | **FALSIFIED** at β ∈ {0.1, 1.0} on heat eq | 2026-05-26 |
| causal_axes.py | C4 (process_discovery_sr.md §6.4) | **PARTIAL VALIDATION** on heat eq — eliminates Class A-temporal but doesn't boost Class C | 2026-05-26 |
| mdl_scoring.py | C3 (process_discovery_sr.md §6.3) | **FALSIFIED** — calibration math right, effect below empirical noise; ad-hoc effectively equivalent | 2026-05-26 |
| adaptive_search.py | C6 (process_discovery_sr.md §6.6) | **VALIDATED-AS-PREDICTED** — adaptive ≈ baseline; pre-analysis predicted no effect; experiment confirmed exactly | 2026-05-26 |

Reports:
- `benchmarks/results/heat_equation_abc_mvp71.md`
- `benchmarks/results/heat_equation_causal_axes_mvp_c4.md`
- `benchmarks/results/heat_equation_mdl_mvp_c3.md`
- `benchmarks/results/heat_equation_adaptive_mvp_c6.md`

To add a module: copy this checklist into the new module's docstring:

    Provenance: research note + conjecture (e.g., C1 from
                process_discovery_sr.md)
    Status: untested
    Graduation criterion: <empirical signal that validates>
    Removal criterion: <empirical signal that falsifies>
    Initial commit: <date>
    Last evaluation: <date or "never">

Future additions
----------------

Named-but-not-yet-implemented conjectures awaiting experimental work
(see `docs/research/process_discovery_sr.md` §6 + §7):

  - C1: ABC-style summary-statistics scoring suppresses Class B
        (target module: abc_scoring.py)
  - C2: Distributional-output trees capture stochastic dynamics
        (target module: distributional_trees.py)
  - C3: MDL with explicit log-likelihood beats ad-hoc parsimony
        (target module: mdl_scoring.py)
  - C4: Causal direction priors at tree-level reduce search space
        (target module: causal_axes.py)
  - C5: Counterfactual evaluation suppresses Class B
        (target module: counterfactual_eval.py)
  - C6: Iterative strategy refinement via residual diagnostics
        (target module: residual_diagnostics.py)

Each is currently a name only. Adding the module commits to
implementing the conjecture as a testable experiment.

Audit policy
------------

If a module has been in this subpackage for 6 months without an
evaluation update, the next maintainer should:

  - Re-read its provenance research note
  - Decide: extend the evaluation period (justify), graduate
    (with empirical evidence), or remove (with falsification note)

No module sits here indefinitely. The lifecycle is real.
"""
from __future__ import annotations

# The subpackage is intentionally empty at scaffold time.
# When experimental modules are added, they should NOT be re-exported
# here. Consumers should import explicitly:
#
#     from tessera.experimental.abc_scoring import abc_residual_loss
#
# This makes "I am using something experimental" explicit at the
# import site rather than hidden behind tessera.experimental.foo
# auto-reexport.

__all__: list[str] = []
