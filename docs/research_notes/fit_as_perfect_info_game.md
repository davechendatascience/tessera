# Research note: data-fitting as a single-agent perfect-information game

**Status:** open research direction. Written as a theoretical framework
for future tessera research, not immediate implementation. Independent
companion to `search_as_energy_min.md`.

**Provenance:** sparked by the chess-game analogy in the 2026-05-24
session. The user noted that the "counter-moves from the data" are
deterministic because the data is fixed — "we have full information."
This doc develops that intuition into a formal framework, grounded in
the algorithmic search literature.

## 1. Why the framing matters

Search-based symbolic regression is often described casually as "GP
searches the space of expressions." That metaphor is imprecise. To make
useful research claims we need a sharper formalism that distinguishes
SR-for-fit from:

- **Adversarial games** (chess, poker, GANs) — opponent has hidden plan;
  minimax / Nash search needed
- **Stochastic environments** (RL on noisy MDPs) — same state-action
  pair gives different rewards
- **Partial-information one-player games** (battleship, Mastermind) —
  some state is hidden from the player

SR-for-fit is **none of these**. The data is fixed and fully observable;
every candidate expression has a deterministically computable loss. The
correct framing is a *single-agent perfect-information game* — closer to
solitaire puzzles than to chess.

The framing change has operational consequences:

| Property | Adversarial search | SR-for-fit |
|---|---|---|
| Opponent's plan | hidden, minimax against worst case | none |
| Repeated states | rarely useful | identical state → cache hit |
| Algebraic equivalence | not exploitable | exploitable as free budget |
| Lower-bound pruning | requires alpha-beta (adversarial) | branch-and-bound (single-agent) |
| Convergence guarantee | none in general | exhaustive search → optimum |

The last row is the headline: **for finite-budget data-fitting, you can
in principle reach the global optimum by exhaustive enumeration**. SR
doesn't because the space is too large, but the *option* exists. That's
what differentiates it from chess.

## 2. Knuth's taxonomy for combinatorial search

Donald Knuth's TAOCP Volume 4 organises combinatorial search by the
*structure of the move set*:

- **Vol 4A — Combinatorial Algorithms, Part 1**: generating all binary
  strings, permutations, combinations, integer partitions. These are
  "stateless" enumerations — every legal state can be visited in turn.
- **Vol 4B — Combinatorial Algorithms, Part 2**: backtracking, dancing
  links (DLX), satisfiability, integer programming, constraint
  propagation. These are "stateful" searches — the move set depends on
  the current state.

SR sits squarely in Vol 4B's territory: the move set (mutation
operators) depends on the current candidate, and the constraint set
(don't produce ill-formed trees) is non-trivial.

Knuth's three foundational paradigms for stateful search are:

1. **Backtracking** (chapter 7.2.2 in TAOCP, 1968 paper *Backtrack
   programming*). Systematic depth-first enumeration of solutions, with
   explicit constraint propagation pruning failed branches early.
2. **Branch-and-bound** (TAOCP 7.2.2.2 and beyond). Maintain a global
   bound on the best solution seen; prune branches whose lower bound
   exceeds the global bound.
3. **Dancing links / Algorithm X** (Knuth, *Dancing Links*, 2000).
   Exact-cover problem solver via doubly-linked toroidal lists; the
   workhorse for Sudoku, polyomino tiling, and pentomino problems.
   Generalises to constraint-satisfaction with non-trivial backtrack
   ordering.

Of these, **branch-and-bound is most directly relevant to SR-for-fit**:
the loss function provides a natural global bound (= the incumbent
best), and dataset-derived lower bounds on subtree-derived losses
prune entire structural classes of candidates. Tessera's Exp 2
(interval-arithmetic lower-bound pruning) is the simplest instance.

## 3. The data-fitting game — formal definition

Let:
- $\mathcal{T}$ be the set of all well-formed Expr trees (under tessera's
  grammar).
- $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be the fixed dataset.
- $\mathcal{L} : \mathcal{T} \times \mathcal{D} \to \mathbb{R}$ be the
  loss function (MSE, PnL, etc.).
- $c : \mathcal{T} \to \mathbb{N}$ be the complexity function (= node
  count).
- $\lambda \in \mathbb{R}_{\geq 0}$ be the parsimony coefficient.

**The fit-game**: a player navigates $\mathcal{T}$ to minimise the
combined fitness $\mathcal{F}(t) = \mathcal{L}(t, \mathcal{D}) + \lambda
c(t)$ subject to a *move budget* $B$ (= number of full
loss-evaluations).

A *position* in this game is a candidate tree $t \in \mathcal{T}$.
A *move* is a tessera mutation operator (subtree_swap, term_insert,
constant_jitter, etc.) plus an optional cheap-bound check.

The game has three features distinguishing it from chess:

**(F1) Perfect information.** Given any $t$, $\mathcal{L}(t, \mathcal{D})$
is a deterministic function. No randomness, no opponent. Same $t$ →
same $\mathcal{L}(t, \mathcal{D})$ every time.

**(F2) Decomposable evaluation.** $\mathcal{L}$ usually decomposes as
$\mathcal{L}(t, \mathcal{D}) = \frac{1}{N}\sum_i \ell(t(x_i), y_i)$ for
some per-sample loss $\ell$ (e.g. squared error). This lets us compute
partial / interval bounds without full $N$-sample evaluation.

**(F3) Algebraic equivalence.** Many syntactically distinct trees
compute the same function on $\mathcal{D}$: $t_1 \sim t_2$ iff
$t_1(x_i) = t_2(x_i)$ for all $i$. Under MSE this equivalence is
finer than under any single dataset, since it can include points
$x \notin \mathcal{D}$ too — but on $\mathcal{D}$ the equivalence
classes are well-defined and can be enumerated.

(F1) and (F2) together say: **the game state has cheap, deterministic,
decomposable evaluation**. (F3) says: **many states are
equivalence-class-redundant**.

## 4. The cost structure of a move

In chess, evaluating a position costs ~1 unit of work (a simple
material/positional heuristic). In SR-for-fit, evaluating a candidate
costs **$O(N)$** — proportional to the dataset size. For $N \in
[10^3, 10^6]$ this is the dominant cost.

That asymmetry has design implications:

- **Mutation is cheap** (tree manipulation, O(tree size))
- **Evaluation is expensive** (O(N) on each candidate)
- **Per-sample lower bounds are O(N/k) for some $k > 1$** (interval
  arithmetic gives a scalar bound from one pass of constant-time-
  per-node operations — much cheaper than full per-sample evaluation)
- **Algebraic equivalence detection is O(tree size)** (term-rewriting,
  hash-consing)

So a budget-efficient search strategy spends most of its compute on the
*expensive evaluation step* and uses cheap mutation + cheap algebraic
work to AVOID redundant evaluations. This is exactly what Knuth's
branch-and-bound paradigm prescribes for the cost-asymmetric case.

## 5. Knuth-flavoured techniques applicable to SR-for-fit

### 5.1 Backtracking with constraint propagation

Tessera's tree-building / mutation step uses no constraint propagation
today — `random_tree` generates uniformly from the grammar with depth
limits. A backtracking variant would:

1. Start with an empty / minimal root
2. At each step, choose the next subtree to fill from a *constraint
   store* (variables that must appear, complexity budget remaining,
   forbidden subtrees from prior runs)
3. If constraints fail, backtrack to the last decision point

This is the standard CSP approach. In TAOCP 7.2.2 Knuth gives the
general recipe; for SR specifically there's prior work (PySR
constraints, Operon's tree-shape constraints) but no truly Knuthian
DLX-style backtracking.

**Open question**: can DLX represent the SR search space? DLX requires
the problem to be encoded as exact cover; SR trees are not naturally
exact cover instances. But individual *constraints* in SR (variable
must appear, max complexity, no FunctionalOp inside FunctionalOp) could
be DLX-encoded.

### 5.2 Branch-and-bound on tree extensions

The most directly applicable Knuth paradigm. For each partial / full
tree, compute:

- A loss **lower bound** (using interval arithmetic over the input
  range — tessera's Exp 2)
- A complexity **lower bound** (the partial tree's current node count)
- A combined **fitness lower bound** $\mathcal{F}_\downarrow(t) =
  \mathcal{L}_\downarrow(t) + \lambda c_\downarrow(t)$

If $\mathcal{F}_\downarrow(t) \geq \mathcal{F}^*$ (the incumbent best),
prune the entire branch under $t$.

Tessera's current Exp 2 implements this for **completed** trees only.
The extension is to apply it during *mutation construction*: if a
partially-mutated subtree already exceeds the bound, abort the
mutation. This requires the mutation operators to be incremental
(produce a sequence of partial trees) rather than atomic (jump from
one tree to another).

**Open question**: can the loss lower bound be tightened mid-mutation
without re-evaluating the whole tree? For sub-tree-swap, only the
swapped subtree's contribution to the loss changes; the rest can be
cached. This is a classic incremental-bound problem, well-studied in
the OR (operations research) branch-and-bound literature.

### 5.3 Dancing-links-style equivalence-class enumeration

In Knuth's *Dancing Links* (2000), the toroidal doubly-linked list
data structure allows exact cover solutions to be enumerated without
backtrack-overhead. The key insight: by carefully ordering the
"options" considered at each step, you can guarantee each equivalence
class is visited exactly once.

For SR, the analogous problem is: given the rewrite system $\sim$ on
trees, enumerate equivalence classes of trees up to complexity $K$,
visiting each class exactly once. This is a generalisation of the
"counting up to symmetry" problem (Burnside/Pólya enumeration).

**Open question**: is there a Knuth-style data structure for enumerating
SR equivalence classes under a confluent term-rewriting system? The
answer involves canonical-form normalisation (tessera's `simplify_ac` +
`simplify` is a partial implementation) plus a structural enumerator
that visits canonical forms in some order.

If solved, this would replace the GP's stochastic search with a
*deterministic systematic exploration of the equivalence-class
quotient space* — which is what perfect information truly enables.

### 5.4 BDDs / ZDDs for tree-shape constraints

Knuth's recent (Vol 4 fascicles) work on Binary Decision Diagrams
(BDDs) and Zero-suppressed Decision Diagrams (ZDDs) gives a compact
representation for sets of bitstrings, with poly-time set operations
(union, intersection, complementation).

For SR, a ZDD could represent "the set of all valid tree shapes of
complexity ≤ K" compactly, allowing:

- Counting (how many trees of each complexity are valid under
  constraints?)
- Sampling (uniform-random tree from the valid set)
- Constraint composition (add a constraint, intersect ZDDs)

This is orthogonal to the loss-evaluation problem but valuable for
the *search-space description* layer. Tessera currently generates
trees via stochastic recursion; a ZDD-backed generator would give
exact sampling without rejection.

**Open question**: how big is the BDD/ZDD for tessera's grammar at,
say, complexity ≤ 20? If polynomial, this is an immediate win.

### 5.5 Algorithm-analysis perspective

Knuth's algorithmic-analysis tradition prefers **rigorous asymptotic
guarantees** over empirical-only claims. The SR community has been
weak here: most claims are "this GP-variant beat that GP-variant on
SRBench" with no theoretical guarantee.

For SR-for-fit, asymptotic questions to ask:

- **Sample complexity**: as $N \to \infty$, does the best-found tree
  approach the Bayes-optimal expression for the data-generating
  process? Under what assumptions?
- **Computational complexity**: as the tree-budget $B \to \infty$ at
  fixed $N$, does the GP converge to the global optimum?
- **Approximation ratio**: at budget $B$, what's the best ratio
  $\mathcal{F}_{found} / \mathcal{F}_{opt}$ achievable?

Knuth's *Selected Papers on Analysis of Algorithms* (2000) is the
right model: it treats algorithm performance as a mathematical object,
not just an empirical phenomenon. For SR, the analogous treatise
doesn't exist.

**Open question**: under what assumptions on the data-generating
process is GP-for-SR provably consistent (best-tree → truth as
$N, B \to \infty$)?

## 6. The "perfect information" theorem we'd like to prove

A target theoretical result, stated informally:

> **Conjecture.** For the fit-game on a fixed dataset $\mathcal{D}$ of
> size $N$, with a fixed grammar $G$ of bounded complexity $K$, with
> the parsimony coefficient $\lambda > 0$, there exists an algorithm
> that finds the optimum $t^* = \arg\min_{t \in \mathcal{T}_K}
> \mathcal{F}(t, \mathcal{D})$ in time $O(N \cdot |E_K|)$, where
> $|E_K|$ is the number of equivalence classes of trees of complexity
> $\leq K$ under tessera's term-rewriting system.

The conjecture says: the *effective* search-space size is the number
of equivalence classes (not the number of syntactic trees). The
multiplicative $O(N)$ is the unavoidable cost of evaluating any single
candidate.

To prove the conjecture, one needs:

1. An enumerator for canonical forms (one per equivalence class), in
   complexity-ascending order
2. A bound-propagation argument that pruning preserves optimality
3. A counting argument that the number of equivalence classes is
   polynomial in $K$ (which would be surprising but worth checking)

If $|E_K|$ is exponential in $K$ (likely), the conjecture would need
to be stated in approximation form: "within $\epsilon$ of the optimum
in time $O(N \cdot \text{poly}(K, 1/\epsilon))$."

## 7. Open theoretical questions

A short list, with tessera-side experiments that would probe each:

1. **What is the equivalence-class count $|E_K|$ for tessera's
   grammar?** Tessera-side: enumerate canonical forms up to $K=10$ and
   count. This is empirical but bounds the theoretical question.

2. **What is the average-case tightness of the interval-arithmetic
   lower bound?** Tessera-side: for many candidates, compute both the
   actual loss and the interval bound; report the distribution of
   ratios. Tight bounds (ratio near 1) mean pruning works well; loose
   bounds (ratio near 0) mean pruning is rarely informative.

3. **Is there a canonical form for FunctionalOp subtrees?** Currently
   tessera's `simplify_canonical` doesn't normalise functionals; this
   is the dependency-problem analogue for measure-theoretic operators.

4. **How does the GP's stochastic exploration compare to a systematic
   equivalence-class enumeration on small problems?** Tessera-side:
   for grammar restricted to $K \leq 6$, enumerate all canonical forms
   exhaustively + compare to GP's discoveries.

5. **What is the BDD/ZDD size for tessera's grammar?** If small,
   immediate engineering win.

6. **Does adding a `WeightedIndicatorSum` primitive (option (b) from
   the previous roadmap) change the equivalence-class structure?**
   Specifically, does it shrink or grow $|E_K|$?

## 8. Connection to tessera's experiments

The fit-game framing gives a unified language for already-implemented
and planned tessera experiments:

| Tessera experiment | Knuth-paradigm role |
|---|---|
| `simplify` (rule-based) | Partial canonical-form normalisation |
| `simplify_ac` (AC norm) | Equivalence-class collapse (commutativity, associativity) |
| `simplify_canonical` | Composition of both → richer canonical form |
| `interval_evaluate` | Lower-bound oracle for branch-and-bound |
| `pareto_threshold` | Incumbent-best lookup for B&B pruning |
| `HallOfFame` | Per-complexity incumbent registry (Pareto-stratified B&B) |
| `optimize_constants` | Inner-loop "fix the tree shape, optimise the parameters" — the algorithm-analysis decomposition |
| `WeightedIndicatorSum` (planned) | Grammar-expansion experiment — does it change $|E_K|$? |
| Equality saturation (planned, Exp 4) | Full canonical-form algorithm via E-graphs |

The framework is *generative*: it suggests further experiments by
asking "what other Knuth paradigms haven't we tried?"

Three immediate suggestions:
- Constraint-propagation backtracking for tree construction (replaces
  rejection-sampling in `random_tree`)
- ZDD-based grammar enumeration (compact set-of-trees representation)
- Pareto-stratified branch-and-bound where the "incumbent" is the
  HoF entry at each complexity, not just the global best

## 9. Reading list (Knuth + adjacent)

In priority order for tessera-relevant material:

1. **Knuth, *The Art of Computer Programming Vol 4B* (Combinatorial
   Algorithms, Part 2)**, sections on backtracking and branch-and-bound.
   Most directly applicable.
2. **Knuth, *Dancing Links* (2000)**. arxiv:cs/0011047. Free PDF.
   Foundational for exact-cover problems; the structural ideas
   generalise.
3. **Knuth, *Selected Papers on Analysis of Algorithms* (2000)**, CSLI
   Publications. The right model for what rigorous SR analysis should
   look like.
4. **Knuth, *Backtrack programming* (1968 paper)**. Original paper; readable.
5. **Korf, *Iterative-Deepening A*: An Optimal Admissible Tree Search*
   (1985)**. For the single-agent perfect-information search.
6. **Lawler & Wood, *Branch-and-Bound Methods: A Survey* (1966)**. The
   foundational B&B paper.
7. **Coppersmith & Winograd, *Matrix multiplication via arithmetic
   progressions* (1990)**. Methodological inspiration — how to do
   rigorous algorithmic improvement on a deeply-studied problem.

## 10. Why this is worth pursuing

The SR literature is largely empirical: benchmark-driven, comparing
algorithms on standard datasets. The theoretical foundations are weak.
A rigorous Knuth-style treatment of SR-for-fit as a single-agent
perfect-information game would:

1. Provide convergence and approximation guarantees the field lacks
2. Identify *why* certain algorithms (PySR, AI Feynman) work better
   than others — currently mostly explained by empirical lore
3. Suggest new algorithms with provable properties
4. Give tessera a theoretical foundation distinguishing it from
   "yet another GP variant"

This is multi-year work. The immediate doc is a *framework*, not a
finished theory. Future tessera research can adopt the framework's
vocabulary and probe specific conjectures.

## Changelog
- 2026-05-25: initial document. Independent companion to
  search_as_energy_min.md.
