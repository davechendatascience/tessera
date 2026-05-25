"""MVP / Conjecture C4: causal direction priors via axis types.

Empirical test of conjecture C4 from
`docs/research/process_discovery_sr.md` §6.4 (second occupant of
tessera.experimental).

The conjecture: imposing the pure-spatial Measure2D constraint
(no temporal-derivative shortcuts) raises Class C discovery rate on
single-trajectory heat equation training, above the baseline rate
established in MVP 7.1 / C1.

Setup
-----
Two-mode comparison on single-trajectory training:
  - Baseline:    standard GP (no constraint)
  - With C4:     GP with penalty=1e6 on temporal-derivative M2Ds

Both modes use single trajectory (NOT multi-trajectory — that's a
separate intervention; we want to isolate the C4 effect). 5 seeds
per mode, pop=240, gens=100, β=0 (no held-out scoring; this isolates
the causal-axes effect).

Outcome interpretation
----------------------
- Class A (temporal-derivative tautology) should drop sharply with C4
- Class C should rise IF the GP redirects toward spatial mechanism
- If Class C doesn't rise and most seeds end up degenerate, the
  constraint is too strict — falsifies in that direction
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tessera.expression import (
    GP, GPConfig,
    FunctionalOp2D, iter_subtrees,
)
from tessera.expression.measure_2d import measure_2d_laplacian_5pt
from tessera.expression.tree import BinOp, Const, UnOp, evaluate as eval_tree
from tessera.experimental.causal_axes import (
    GPWithCausalAxes, count_violating_m2ds,
)

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_heat_equation_traintest_computescale import simulate_heat_with_ic  # noqa: E402


# Classification (same as C1 experiment)
def is_laplacian_atom(measure_2d) -> bool:
    if not hasattr(measure_2d, "atoms"):
        return False
    atoms = measure_2d.atoms
    if len(atoms) != 3:
        return False
    weights = sorted(round(float(a.weight), 4) for a in atoms)
    return weights == [-2.0, 1.0, 1.0]


def tree_has_reduce(node) -> bool:
    for sub in iter_subtrees(node):
        if isinstance(sub, UnOp) and sub.op.startswith("reduce_"):
            return True
    return False


def tree_has_laplacian(node) -> bool:
    for sub in iter_subtrees(node):
        if isinstance(sub, FunctionalOp2D):
            if is_laplacian_atom(sub.measure_2d):
                return True
    return False


def classify_tree(node, train_loss, test_loss, oracle_train, oracle_test):
    has_lap = tree_has_laplacian(node)
    has_red = tree_has_reduce(node)
    train_ratio = train_loss / max(oracle_train, 1e-30)
    test_ratio = test_loss / max(oracle_test, 1e-30) if oracle_test > 0 else float("inf")
    if not has_lap and train_ratio > 5.0:
        return "degenerate"
    if has_lap and not has_red:
        if train_ratio < 1.5 and test_ratio < 1.5:
            return "C"
        return "C-partial"
    if has_lap and has_red:
        return "B"
    return "A"


def oracle_loss_on(U, dt_U, alpha=0.05):
    lap = measure_2d_laplacian_5pt().apply(U, fill_warmup=0.0)
    interior = (slice(1, -1), slice(1, -1))
    return float(np.mean((alpha * lap[interior] - dt_U[interior]) ** 2))


def evaluate_tree_on(tree, U, dt_U):
    try:
        pred = eval_tree(tree, {"U": U}, fill_warmup=0.0)
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != dt_U.shape or not np.isfinite(pred).all():
            return float("inf")
        interior = (slice(1, -1), slice(1, -1))
        return float(np.mean((pred[interior] - dt_U[interior]) ** 2))
    except Exception:
        return float("inf")


def run_one_mode(mode_name, U_train, dt_U_train, U_test, dt_U_test,
                 seed, pop, gens):
    target_var = float(np.var(dt_U_train[1:-1, 1:-1]))
    parsimony = max(target_var * 0.001, 1e-9)
    cfg = GPConfig(
        pop_size=pop, n_gens=gens,
        init_max_depth=4, parsimony=parsimony,
        early_stop_patience=gens, seed=seed,
        enable_2d=True, fill_warmup=0.0, verbose=False,
    )
    if mode_name == "baseline":
        gp = GP(cfg)
    elif mode_name == "with_c4":
        gp = GPWithCausalAxes(cfg, penalty=1e6)
    else:
        raise ValueError(f"unknown mode {mode_name}")

    t0 = time.time()
    front = gp.run({"U": U_train}, dt_U_train, feature_names=["U"])
    rt = time.time() - t0
    best = min(front, key=lambda c: c.train_loss)

    test_loss = evaluate_tree_on(best.tree, U_test, dt_U_test)
    oracle_train = oracle_loss_on(U_train, dt_U_train)
    oracle_test = oracle_loss_on(U_test, dt_U_test)
    train_ratio = best.train_loss / oracle_train if oracle_train > 0 else float("inf")
    test_ratio = test_loss / oracle_test if oracle_test > 0 else float("inf")
    cls = classify_tree(best.tree, best.train_loss, test_loss,
                        oracle_train, oracle_test)
    violators = count_violating_m2ds(best.tree)
    return dict(
        mode=mode_name, seed=seed,
        train_loss=best.train_loss, test_loss=test_loss,
        train_ratio=train_ratio, test_ratio=test_ratio,
        best_cx=best.complexity,
        classification=cls,
        n_violating_m2ds=violators,
        front_size=len(front),
        runtime=rt,
        tree_str=str(best.tree),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--X", type=int, default=32)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--pop", type=int, default=240)
    p.add_argument("--gens", type=int, default=100)
    args = p.parse_args(argv)

    print("=== MVP / C4: causal direction priors ===")
    print(f"T={args.T} X={args.X} seeds={args.seeds} pop={args.pop} gens={args.gens}")
    print()

    print("Simulating TRAIN (ic_seed=100), TEST (ic_seed=999)...")
    U_train = simulate_heat_with_ic(T=args.T, X=args.X, alpha=0.05,
                                    noise_std=0.002, ic_seed=100, sim_seed=0)
    U_test = simulate_heat_with_ic(T=args.T, X=args.X, alpha=0.05,
                                   noise_std=0.002, ic_seed=999, sim_seed=2)

    def make_dt_U(U):
        dt = np.zeros_like(U)
        dt[:-1] = U[1:] - U[:-1]
        return dt

    dt_U_train = make_dt_U(U_train)
    dt_U_test = make_dt_U(U_test)
    print(f"Oracle TRAIN={oracle_loss_on(U_train, dt_U_train):.4g} "
          f"TEST={oracle_loss_on(U_test, dt_U_test):.4g}")
    print()

    results = []
    t_start = time.time()
    for mode_name in ["baseline", "with_c4"]:
        print(f"--- Mode {mode_name} ---")
        for seed_idx in range(args.seeds):
            seed = 2026 + seed_idx
            r = run_one_mode(
                mode_name=mode_name,
                U_train=U_train, dt_U_train=dt_U_train,
                U_test=U_test, dt_U_test=dt_U_test,
                seed=seed, pop=args.pop, gens=args.gens,
            )
            results.append(r)
            print(f"  seed={seed}  train/o={r['train_ratio']:.2f}  "
                  f"test/o={r['test_ratio']:.2f}  cx={r['best_cx']}  "
                  f"class={r['classification']}  violators={r['n_violating_m2ds']}  "
                  f"({r['runtime']:.1f}s)")
        print()

    total_rt = time.time() - t_start
    print(f"Total wall-clock: {total_rt:.1f}s")

    out_path = Path(__file__).resolve().parent / "results" / "heat_equation_causal_axes_mvp_c4.md"
    write_report(results, args, total_rt, out_path)
    return 0


def write_report(results, args, total_rt, out_path):
    L = ["# MVP / C4: causal direction priors — empirical test", ""]
    L.append("Empirical test of conjecture C4 from")
    L.append("`docs/research/process_discovery_sr.md` §6.4. Second occupant")
    L.append("of tessera.experimental.")
    L.append("")
    L.append("**The conjecture:** causal direction priors at the tree level")
    L.append("(restricting Measure2D to pure-spatial atoms when target is a")
    L.append("temporal derivative) raises Class C discovery above baseline")
    L.append("without losing the right answer.")
    L.append("")
    L.append(f"**Setup:** T={args.T}, X={args.X}, α=0.05. Single-trajectory")
    L.append("training (ic_seed=100); held-out evaluation on ic_seed=999.")
    L.append(f"pop={args.pop}, gens={args.gens}, {args.seeds} seeds/mode.")
    L.append(f"Wall-clock: {total_rt:.1f}s")
    L.append("")

    L.append("## Mode definitions")
    L.append("")
    L.append("- **Baseline**: standard tessera GP (no causal constraint)")
    L.append("- **With C4**: GPWithCausalAxes(penalty=1e6) — penalty per")
    L.append("  Measure2D whose atoms span multiple lag_t values")
    L.append("")

    # Class counts
    def count_classes(mode_name):
        runs = [r for r in results if r["mode"] == mode_name]
        counts = {"A": 0, "B": 0, "C": 0, "C-partial": 0, "degenerate": 0}
        for r in runs:
            counts[r["classification"]] = counts.get(r["classification"], 0) + 1
        return counts, len(runs)

    L.append("## Class distribution comparison")
    L.append("")
    L.append("| Mode | A (temporal tautology) | B (template/reduce_*) | C-partial | **C** (clean mechanism) | degenerate |")
    L.append("|---|---|---|---|---|---|")
    for mode_name in ["baseline", "with_c4"]:
        counts, n = count_classes(mode_name)
        L.append(f"| {mode_name} | {counts['A']}/{n} | {counts['B']}/{n} | "
                 f"{counts['C-partial']}/{n} | **{counts['C']}/{n}** | "
                 f"{counts['degenerate']}/{n} |")
    L.append("")

    # Causal violator counts
    L.append("## Causal-violator counts (number of temporal M2Ds in final best tree)")
    L.append("")
    L.append("| Mode | mean violators | seeds with violators |")
    L.append("|---|---|---|")
    for mode_name in ["baseline", "with_c4"]:
        runs = [r for r in results if r["mode"] == mode_name]
        violators = [r["n_violating_m2ds"] for r in runs]
        n_with = sum(1 for v in violators if v > 0)
        L.append(f"| {mode_name} | {np.mean(violators):.1f} | {n_with}/{len(runs)} |")
    L.append("")

    # Verdict
    counts_b, n_b = count_classes("baseline")
    counts_c4, n_c4 = count_classes("with_c4")
    c_baseline = counts_b["C"]
    c_with_c4 = counts_c4["C"]
    a_baseline = counts_b["A"]
    a_with_c4 = counts_c4["A"]
    deg_with_c4 = counts_c4["degenerate"]

    L.append("## Verdict")
    L.append("")
    L.append(f"- Class C (clean mechanism): baseline {c_baseline}/{n_b}, with C4 {c_with_c4}/{n_c4}")
    L.append(f"- Class A (temporal tautology): baseline {a_baseline}/{n_b}, with C4 {a_with_c4}/{n_c4}")
    L.append(f"- Degenerate: baseline {counts_b['degenerate']}/{n_b}, with C4 {deg_with_c4}/{n_c4}")
    L.append("")

    if c_with_c4 > c_baseline and a_with_c4 < a_baseline:
        L.append("**CONJECTURE SUPPORTED.** C4 increases Class C discovery while")
        L.append("eliminating Class A. The causal-axes constraint successfully")
        L.append("redirects the GP toward mechanism by forbidding the")
        L.append("temporal-derivative shortcut.")
    elif c_with_c4 == c_baseline and a_with_c4 < a_baseline:
        L.append("**C4 PARTIALLY VALIDATES.** Class A is eliminated (good) but")
        L.append("Class C rate didn't rise (the GP didn't redirect toward")
        L.append("mechanism). The constraint is informative but not transformative.")
    elif a_with_c4 < a_baseline and deg_with_c4 > counts_b["degenerate"]:
        L.append("**C4 TOO STRICT.** Class A eliminated but GP becomes degenerate")
        L.append("(predict-zero). The constraint removes too much of the search")
        L.append("space without providing alternative paths to mechanism.")
    elif c_with_c4 < c_baseline:
        L.append("**C4 HURTS Class C DISCOVERY.** Falsifies the conjecture —")
        L.append("the constraint interferes with finding the mechanism rather")
        L.append("than helping.")
    else:
        L.append(f"**MIXED RESULT.** Counts: baseline C={c_baseline}/A={a_baseline},")
        L.append(f"with_c4 C={c_with_c4}/A={a_with_c4}. May need more seeds.")
    L.append("")

    L.append("## Per-seed details")
    L.append("")
    L.append("| Mode | seed | train/oracle | test/oracle | cx | class | violators | tree (truncated) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        tree_str = r["tree_str"].encode("ascii", "replace").decode("ascii")
        if len(tree_str) > 60:
            tree_str = tree_str[:57] + "..."
        L.append(f"| {r['mode']} | {r['seed']} | {r['train_ratio']:.2f} | "
                 f"{r['test_ratio']:.2f} | {r['best_cx']} | "
                 f"{r['classification']} | {r['n_violating_m2ds']} | "
                 f"`{tree_str}` |")
    L.append("")

    L.append("## Reproducing")
    L.append("")
    L.append("```")
    L.append("python benchmarks/run_heat_equation_causal_axes_mvp_c4.py --seeds 5")
    L.append("```")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
