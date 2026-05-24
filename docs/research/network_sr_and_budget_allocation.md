# Research note: network-SR and budget allocation under perfect information

**Status:** ? RESEARCH. Thinking-aloud document, not yet a proposal. Captures user's evolving thought 2026-05-24 + the empirical result from the second IK benchmark run.

**Provenance:**

User (2026-05-24, first message): *"I definitely need to figure out the architectural design of unit SR, as our name tessera suggests. The assignment of special unit to special subproblems is probably a Knuth's category problem."*

User (2026-05-24, refinement): *"That was just an inspirational thought saying we can delegate the network of SR using specialised units that use Knuth's algorithm to figure out assignments. Basically, it's a bridge between discrete and continuous math. But my thought isn't complete on this. We should do more research."*

User (same message, on Gemini's MCTS-rollout suggestion): *"I don't think incremental rollout is very good. But using a game-theoretic approach, the measured loss value should be committed moves, and potential moves we can measure, but this then makes the game not perfect information. So we should think more on this."*

User (clarifying refinement): *"Or measured loss value is not committed moves, and the game evolving strategy needs to be based on perfect information."*

---

## 1. What the user's two thoughts actually say (and how they connect)

Two ideas, both unfinished, both pointing somewhere important:

**Thought A — "network-SR" architecture:**

> A complex SR problem decomposes into sub-problems. Each sub-problem belongs to a *category* (kinematics, image features, time-series convolution, …). For each category there's a *specialised SR unit* that does well on it. A meta-level **assignment algorithm** decides which unit handles which sub-problem. The assignment is *discrete* (which unit goes where); each unit's internal SR is *continuous* (numerical search). The whole system is a bridge between Knuth-style combinatorial assignment and continuous-math regression.

**Thought B — "strategy under budget, but still perfect info":**

> The landscape (all possible candidate trees with their deterministic losses) is perfect information — fully determined. Past evaluations don't "commit moves"; they just *reveal* parts of the landscape. The evolving search strategy decides which parts to reveal next under a finite eval budget. That strategy must be GROUNDED in perfect-information principles — not in bandit/MCTS-style stochastic exploration.

The two thoughts are connected: **the assignment algorithm in thought A is the search strategy in thought B.** Both ask the same question — *how do we allocate eval budget under deterministic-but-not-yet-known costs?* Thought A frames the question at the unit-network level (which unit gets which sub-problem); thought B frames it at the tree-level (which candidate gets evaluated next).

## 2. Why this matters now (the second IK benchmark result)

The IK benchmark run #2 (`benchmarks/results/ik_planar_3dof.md` 2026-05-24) showed something the user predicted obliquely. After shipping `atan2`/`acos`/`asin` — exactly the primitives the analytical IK uses — **the GP did not USE them**. Discovered trees stayed at Tier D with the *same* failure mode (algebraic approximations via `cos`/`pow`/`tanh`).

The diagnosis: **search-space explosion under uniform random sampling**. With ~30 ops in the alphabet, each new op has ~3% probability per tree slot. Composing the right IK formula needs *specific multi-op compositions* (`atan2(y_w, x_w) - atan2(...)`) at probabilities like `0.03² ≈ 0.1%`. The eval budget can't reach the right composition by random walk.

So vocabulary was necessary-but-not-sufficient. The next-level question is **search-strategy under finite budget** — exactly the user's thought B.

## 3. Rejecting the bandit/MCTS framing (and why)

Gemini suggested MCTS rollouts. The user pushed back on this, and the pushback is sharp:

| | Bandit / MCTS | SR-for-fit |
|---|---|---|
| Environment | Stochastic — same action gives different outcomes | Deterministic — same tree gives same loss |
| Exploration motivation | Reduce uncertainty about expected reward | Reduce uncertainty about unmapped landscape |
| Information principle | Bayesian update of belief distributions | Direct observation of deterministic values |
| Optimal strategy class | Probabilistic (Thompson sampling, UCB) | Deterministic (A*, B&B, IDA*) |

The bandit framework assumes you NEVER fully know the reward function — only expected-value estimates. SR-for-fit *does* know each candidate's loss exactly once you pay the eval cost. Treating SR as a bandit problem imports unnecessary stochastic-uncertainty machinery.

The right framework class: **deterministic search under budget with admissible heuristic** — Knuth's branch-and-bound and Korf's IDA* are the canonical instances. The "heuristic" is the loss-lower-bound from interval arithmetic (already shipped in tessera).

## 4. Restating the user's framing precisely

> *The static landscape of (tree, loss) pairs is perfect information. The eval-budget-constrained strategy navigates this landscape. Since the landscape is deterministic, the strategy must use deterministic admissible heuristics (B&B, A*), not stochastic exploration (bandits, MCTS).*

This converts the user's "thought B" into a concrete research direction: **what is the right deterministic admissible search strategy for SR**, given:

- The state space = trees of bounded complexity
- The cost function = O(N) eval per node
- The lower-bound oracle = interval arithmetic (already cheap, ~O(tree size))
- The incumbent = best-discovered-so-far Pareto candidate at each complexity

This is the **branch-and-bound game** from `fit_as_perfect_info_game.md` §5.2, but with the user's refinement: the *strategy* (not just the *bound check*) is the research question, and it lives outside bandit-class methods.

## 5. Where this leaves "network-SR" (thought A)

If thought B says "deterministic B&B over tree space," then thought A becomes: **decompose tree space into sub-spaces, each handled by a specialised unit; B&B at the assignment level + B&B within each unit**.

Concrete candidate decompositions (from `high_dim_symbolic_regression.md §6.3`):

| Sub-space | Suggested unit | Why it's a separate category |
|---|---|---|
| Time-series convolution | LinearFunctional + GP | Has its own structure (lag-translation invariance) |
| Pointwise physics-style | Trig + sqrt + GP | Smooth low-arity formulas |
| Image features | FunctionalOp2D + reduce | Spatial structure + invariance |
| Inverse kinematics | atan2/acos + GP | Quadrant-aware multi-trig |
| Constrained discrete | DLX-style enumeration | Exact-cover-like structure |

The *assignment* is: given a problem's metadata (input shape, output shape, structural hints), pick which unit handles it. Knuth's relevance: the assignment problem is itself a combinatorial structure (categorise the input → select a strategy). This is closer to Knuth's *algorithm-selection* spirit than his specific algorithms.

But: **before committing to network-SR, finish the simpler thing.** The IK benchmark's tier-D-due-to-search-explosion result tells us *even the universal GP needs better search strategy* before unit-architecture is the right lever. Search-strategy fixes (Knuth's B&B + admissible heuristics + smarter mutation distribution) may close enough of the gap that the unit-architecture becomes premature.

## 6. Open questions

| # | Question | How to make progress |
|---|---|---|
| 1 | What's the "right" deterministic budget-aware strategy for SR? | Read Korf's IDA*, Knuth's B&B paper, Russell & Norvig §3-4 |
| 2 | Does op-weight scheduling close the IK Tier-D gap? | One-day experiment: anneal OP_WEIGHTS biased toward new ops in early gens. Cheapest test of "the budget exists but the GP doesn't try the right ops." |
| 3 | Does template-based mutation close it? | Implement `template_atan2_composition` mutation; rerun. Two days. |
| 4 | If neither works, is the answer network-SR with a unit specifically for IK? | Cross-promotion to `high_dim_symbolic_regression.md` §6's experiment. |
| 5 | What's the bridge between B&B's heuristic search and the "perfect-info game" framing? | Theory work: the heuristic IS the partial knowledge of the landscape gained from prior evals. Formalise. |

## 7. What this note does NOT yet conclude

The user's framing is *inspirational*, not committed. This document captures it for future reference but **proposes no specific implementation work yet**. The right next moves are empirical: 

- Try op-weight scheduling on the IK benchmark (smallest test of "vocabulary present but unused" hypothesis)
- If that doesn't work, try template-mutation
- If THAT doesn't work, the network-SR architecture (thought A) is the lever, not search-strategy

The order matters because each successful step makes the next investment smaller. Knuth-style structural search (thought B's framing) is the cheaper hypothesis to test first.

## 8. Connection to existing research notes

- `fit_as_perfect_info_game.md` §5.2 + §12: B&B as the right paradigm for SR-for-fit. This note REINFORCES that conclusion: the user's "strategy must be based on perfect information" is the same claim, restated cleaner.
- `high_dim_symbolic_regression.md` §6: unit-architecture as Knuth's category-matching. This note: same architecture, but framed as *one possible response* to the budget-allocation question, not the only response.
- `benchmark_score_improvement.md` §4.2: template mutations. This note: empirically promotes that direction from "research-only" to "next candidate empirical experiment" given the IK rerun result.

## 9. The thread to come back to (when more thinking is done)

The user said: "we should think more on this." This doc is the **resting place** for the thoughts so far. Re-entering this thread requires reading:
- §1 (the two user thoughts)
- §3 (why bandit is wrong)
- §4 (the precise restatement)
- §6 (the five open questions)

Then deciding whether to:
(a) Pursue the deterministic-B&B-strategy direction (smallest commitment)
(b) Pursue the network-SR architecture (medium)
(c) Pursue both in parallel as different unit-architectures within tessera (largest)

## Changelog

- 2026-05-24: initial document. Provenance: user's evolving thoughts on architecture + game-theoretic framing + Gemini interaction + the IK rerun result. Status: thinking-aloud; no specific implementation proposed; five open questions left for further research.
