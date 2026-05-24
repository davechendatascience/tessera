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

## 11. Refined game-theoretic structure

The framing in §3 is correct but understates the game-theoretic content.
The single-agent perfect-information label is not the whole story —
there are five tighter game-theoretic readings of SR-for-fit that
either strengthen the doc's claims or open new ones.

### 11.1 Alpha-beta as the immediate ancestor (Knuth genealogy)

Knuth & Moore, *An Analysis of Alpha-Beta Pruning* (1975) is the
foundational paper on adversarial perfect-information game-tree search.
Alpha-beta is, structurally, **branch-and-bound *with an opponent***.
The doc's §2 mentions Knuth's branch-and-bound paradigm but does not
make the cleanest historical connection: **B&B for single-agent is what
remains when you drop the opponent from alpha-beta**. The min/max
recursion collapses to plain min over leaves, and the cutoff condition
simplifies from "exceeds opponent's bound" to "exceeds incumbent best."

So Knuth gave us both tools — alpha-beta for chess engines and B&B for
combinatorial optimization — and SR-for-fit is the single-agent
specialization of the same Knuth tradition. This is not just rhetorical;
it justifies why importing Knuth's algorithmic machinery (rather than,
say, Aumann's adversarial-game machinery) is the right move.

A useful way to phrase this in a paper introduction: *"We treat SR-for-fit
as the degenerate single-agent specialization of the game-tree search
problem Knuth & Moore (1975) studied. The simplification — no opponent
— turns alpha-beta into branch-and-bound and turns minimax into min."*

### 11.2 The eval-vs-rewrite move budget (algebraic equivalence as free moves)

The doc's §3, feature (F3), says many trees are equivalence-redundant.
In game-theoretic vocabulary this is sharper: **algebraic equivalence is
a class of *moves* that do not consume the move budget**. In chess you
cannot "rewrite your position" — what you have played is committed. In
the fit-game, `simplify(t)` is a real move that produces a different
position with provably equal value, and it costs only O(tree size), not
O(N).

The cleanest formal restatement:

> **The fit-game has *two* move budgets:**
> - **EVAL budget** $B_e$: bounded by O(N) per move; this is what
>   the player wants to minimize total spend on
> - **REWRITE budget** $B_r$: cheap moves (algebraic-equivalence
>   transformations); effectively unbounded compared to eval

In this two-budget game, **the optimal strategy spends the rewrite
budget maximally before each eval**. That is exactly what
equality-saturation (egg / snake-egg) does: it spends arbitrarily many
rewrites between evals to produce a canonical form, then evaluates the
canonical form.

This framing predicts (and `search_as_energy_min.md` quantifies) that
**the achievable speedup from equality saturation is bounded by
$|E_K|/|T_K|$, the orbit compression ratio**. The two-budget game
formalism makes that bound a *game-theoretic* statement: the maximum
information per eval move is determined by how aggressively rewrites
can collapse syntactically distinct positions before the eval is spent.

**Consequence for tessera roadmap:** an experiment that varies the
rewrite-budget per eval (current default ≈ 1: just simplify once) and
measures the resulting orbit compression and discovery rate would
directly validate this framing.

### 11.3 Burnside / Pólya as the formal bridge for $|E_K|$

The doc's §6 conjecture estimates search-space size as $|E_K|$, the
equivalence-class count. The doc does not explicitly state the algebraic
formula that lets you *compute* this bound without exhaustively
enumerating trees.

Burnside's lemma says: for a group $G$ acting on a set $X$, the number
of orbits is

$$|X/G| = \frac{1}{|G|} \sum_{g \in G} |\text{Fix}(g)|$$

where $\text{Fix}(g)$ is the set of $g$-fixed points.

For SR, $X = \mathcal{T}_K$ (trees of complexity $\leq K$) and $G$ is
the algebraic-equivalence group generated by:

- **Commutativity** of `add`, `mul`, `min`, `max` (involutions)
- **Associativity** of the same (more complex; gives infinite-cyclic structure on flat-list normalization)
- **Distributivity** `a*(b+c) = a*b + a*c` (relates trees of *different* shapes; the most powerful generator)
- **Identities**: `x + 0 = x`, `x * 1 = x`, etc. (relate trees to subtrees)
- **Inverses**: `x - x = 0`, `x / x = 1` (collapse to constants)

Tessera's `simplify_canonical` implements a *normalizer* for a subgroup
of $G$ — specifically the commutativity + identities part, plus AC
normalization for associativity. The full group includes distributivity,
which `simplify_canonical` does *not* normalize (it's much harder to do
soundly without an e-graph).

Burnside gives a path to *bound* $|E_K|$ without exhaustive enumeration:
just bound $|\text{Fix}(g)|$ for each generator. Pólya's theorem extends
this to weighted counting (e.g., counting orbits with a generating
function in complexity).

**Concrete tessera experiment** (worth a paper): compute
$|\text{Fix}(g)|$ for each commutativity / associativity / identity
generator at $K = 5, 6, 7, 8$, apply Burnside, and report the predicted
$|E_K|$. Compare to the empirical equivalence-class count from
exhaustive enumeration. This is the cleanest theoretical validation of
the perfect-info-as-orbit-counting framework.

### 11.4 Pareto front as 2-player cooperative game

The doc's framework treats fitness as a scalar $\mathcal{F}(t) =
\mathcal{L}(t) + \lambda c(t)$. The parsimony coefficient $\lambda$ is
ad-hoc. A cleaner game-theoretic reading: **the Pareto front is a
2-player cooperative game between Complexity and Accuracy**, two players
who must agree on a tree.

In cooperative-game language:
- Each Pareto-optimal tree is a *feasible bargaining outcome*
- The Pareto frontier is the *bargaining set*
- The chosen point on the front is the *bargaining solution*

Three canonical bargaining solutions from cooperative game theory:

- **Nash bargaining solution** (Nash 1950): maximize the product of
  utility gains over the disagreement point. For SR, this picks the
  point on the front where $\frac{\partial \mathcal{L}}{\partial c}$
  equals some reciprocal — a *principled* parsimony coefficient.
- **Kalai-Smorodinsky** (1975): match the ratio of gains to the ratio
  of utopia gains. Often gives a different point.
- **Egalitarian / Shapley** (Aumann-Shapley 1974): equal gain in each
  utility.

**Why this matters for tessera:**

Tessera's `HallOfFame` currently stores per-complexity best-ever trees
— a *stratified* representation of the front. The bargaining-theoretic
lens gives:

1. A principled scalarization: "pick the Nash bargaining solution on
   the current front" instead of a hardcoded $\lambda$
2. A way to compare two Pareto fronts: which front Pareto-dominates,
   and which gives a better bargaining solution
3. Convergence theory: the Nash bargaining solution is unique under
   convexity; non-convex Pareto fronts (typical in SR) have multiple
   admissible solutions, indicating *intrinsic ambiguity* in the
   parsimony tradeoff

**Tessera-side experiment**: implement a `nash_bargain(front)` function
that picks the Nash bargaining solution from the current Pareto front,
use it as the default selection criterion in `HallOfFame.report_best()`,
compare to current parsimony-based selection across Feynman + SRBench.

This connects SR directly to a 70-year-old cooperative-game-theory
literature with established convergence results.

### 11.5 The generalization game: Stackelberg with noise as adversary

The doc's §3 explicitly fixes $\mathcal{D}$ and dismisses noise as out
of scope. That is correct for the *train-time* problem but misleading
about the *deployment* problem.

The full SR lifecycle is:

1. Train data $\mathcal{D}_{train}$ shown to player
2. Player commits to tree $t^*$
3. Test data $\mathcal{D}_{test}$ revealed
4. Loss $\mathcal{L}(t^*, \mathcal{D}_{test})$ measured

This is a **Stackelberg game** where the SR searcher commits first and
nature reveals the test data after. **Overfitting is exactly the
SR-leader losing this game**: the player's best train-response over-fits
the leader's information set, then loses to a different test set.

Standard equilibrium concepts apply:

- **Subgame-perfect equilibrium**: player picks the tree that does best
  against the *worst-case* test set consistent with the train data.
  This is distributionally-robust SR.
- **Bayesian Nash equilibrium**: player picks the tree that minimizes
  expected test loss under a prior over test distributions. This is
  what cross-validation approximates.

Connection to existing tessera machinery:

- **Parsimony coefficient $\lambda$** is a regularizer; in Stackelberg
  language it is a *commitment device*: by penalizing complexity, the
  player commits not to over-fit, which limits the leader's
  vulnerability to the follower (nature/test set).
- **`pnl_loss_smooth` (Hamiltonian relaxation)** is exactly a
  distributionally-robust loss: it bounds the follower's response to a
  smooth class (no spike-thresholding).
- **Block bootstrap validation** is a Monte Carlo approximation of the
  follower's strategy space.

**Honest framing for the doc**: §3 should say "we study the train-time
*subgame* — single-agent, perfect-info, no opponent. The full SR
lifecycle is a Stackelberg game where this train-time subgame is the
leader's stage; deployment / generalization is the follower's stage. We
treat the follower as out of scope, but acknowledge it is where the
existing PAC-learning / DRO theory applies."

This is a small but important honesty correction: the doc's claim is
"the train-time fit-game has no opponent," not "SR-for-fit has no
opponent."

### 11.6 Synthesis: three actionable upgrades

The five angles above suggest three concrete tessera experiments, each
contained enough to ship and rigorous enough to publish:

| Item | Game-theoretic content | Expected effort |
|---|---|---|
| `nash_bargain(front)` in HallOfFame | Pareto = cooperative game; Nash bargaining as principled scalarization (replaces ad-hoc λ) | ~half day |
| Burnside-bound experiment | Empirically validate `\|E_K\|` predictions from group-orbit counting at $K \leq 8$; the core theoretical conjecture of §6 | 1-2 days |
| Two-budget eval-vs-rewrite ablation | Vary rewrite-budget-per-eval and measure orbit compression + discovery rate; validates the equality-saturation framing | 1 day |

All three serve goal 2 (theory) directly and goal 1 (workbench) by
producing publishable empirical validation. None block goal 3 (CV /
MNIST).

## 12. Open self-criticism: implementation hasn't cashed in on the framing

The framing in §3-§5 makes a sharp prediction: **a real perfect-info
engine doesn't evaluate every position; it prunes most.** Stockfish
evaluates a tiny fraction of legal positions; alpha-beta pruning
skips the rest. For SR-for-fit, the analogous claim is that tessera
should do *fewer* evals, not just *faster* evals.

This section is an honest audit of how the current implementation
lives up to that prediction. Short answer: **partially**. The recent
GPU work (Tiers 1-3, May 2026) made evaluations 25-35× faster on GPU
at MNIST scale, which is real progress on the *fast-eval* axis. But
the *fewer-evals* axis — which is what the perfect-info framing
actually predicts — is significantly underused.

### 12.1 The cost-structure inversion vs chess

In chess, evaluation is cheap (~100 ns per position via a 50-weight
heuristic) and search is the hard part. In SR-for-fit, evaluation
is expensive (O(N · tree_size), typically ms to seconds) and search
is the trivial part. The ratio is roughly N · tree_size : 1 in
favor of chess.

That asymmetry doesn't break the framing — it shifts where leverage
comes from. The doc's §4 acknowledges this. But the implementation
has not yet fully exploited the levers §4-§5 identify.

### 12.2 Lever utilisation audit

| Lever | Chess analog | Doc reference | Implementation |
|---|---|---|---|
| Subexpression caching | Transposition tables | §4 "cheap algebraic work" | `FunctionalCache` shipped (hit rate 30-80% in non-batched path); **bypassed in the new JAX-batched path** |
| Bound-based pruning | Alpha-beta | §5.2 | `prune_by_lower_bound=True` flag shipped; **off by default**, MSE-only, scalar-per-candidate not batched |
| Equivalence-class collapse | Rare in chess | §5.3, §6 conjecture | `simplify_canonical` shipped (rule-based + AC); **no e-graph yet** — only a few % of the theoretical `\|E_K\|/\|T_K\|` compression |
| Per-complexity incumbent | Bestline | §8 table | `HallOfFame` shipped |

Three out of four levers exist but are *underused*. The fourth
(equality saturation) is the biggest single unimplemented win and
remains in `search_as_energy_min.md` as research.

### 12.3 Where the May 2026 GPU work actually landed

Tiers 1-3 (`Measure.apply` JAX path, `compile_tree` jit, batched
`PopulationEvaluator` vmap) made **per-eval cost** drop sharply.
That's valuable: the eval-budget per generation goes from O(K · N
· tree_size) of CPU work to O(K · N · tree_size) of GPU work, with
~50× constant-factor improvement on Colab GPU at K=200, N=60K.

But that's solving the **wrong half** of the cost asymmetry. The
perfect-info-game framing predicts we should be doing **fewer evals**
via:

- Subexpression caching (don't recompute shared sub-trees across
  candidates)
- B&B pruning (skip evals provably worse than incumbent)
- Equivalence collapse (don't even consider syntactically distinct
  but semantically equivalent trees)

The GPU work made expensive evals cheaper. It did not make the *number*
of evals smaller. A Stockfish-shaped tessera would do both.

### 12.4 Concrete unrealized opportunities

The MNIST run we're currently shaping is illustrative. The numbers:
N=1000 balanced samples × K=60 trees × G=20 generations = 1.2M
tree-evaluations per feature. With Tier-3 vmap+jit this is tractable
(~30-90 s per feature). But a Knuth-style implementation would:

1. **Hash sub-expressions across the population.** Many candidate
   trees share `FunctionalOp2D(Laplacian, image)` or
   `reduce_mean(image)` — these should materialise once per
   generation, not K times. Current Tier-3 path bypasses
   `FunctionalCache` because topology-level vmap is incompatible with
   subexpression-level caching. They are complementary but not
   integrated.

2. **Batched B&B bound check.** The interval-arithmetic bound
   (`tessera.expression.interval.interval_evaluate`) is currently
   scalar-per-tree. Batching the bound check on GPU would let us
   skip the *full* eval for candidates whose bound exceeds the
   Pareto incumbent. The reported tightness ratio (median 0.47)
   says roughly half of candidates *could* be pruned if the check
   ran.

3. **Equality saturation.** Sketched in `search_as_energy_min.md`.
   Real e-graph saturation would collapse the population's K
   candidates into G_eq < K equivalence classes per generation,
   eliminating redundant evals before they happen. Conservative
   estimate from §5.3: `|E_K| / |T_K|` is well below 1, probably
   in the 0.1-0.5 range for tessera's grammar at typical K.

Combined effect, conservatively: another 5-10× fewer actual evals
on top of the 50× from Tier-3 GPU. Total leverage 250-500× over
the numpy baseline.

### 12.5 What this means for the doc

§3-§5 are correct as stated. They are not the *whole* story they
appeared to be when written, because the implementation has only
partially executed on them. The honest framing is:

> *The perfect-info game framing predicts three levers (caching,
> pruning, equivalence) and one performance dimension (per-eval
> speed). Current tessera has shipped partial implementations of all
> three levers plus a strong GPU port of the speed dimension. The
> levers are underused; pursuing them fully is the next theoretical-
> contribution sweep.*

The May 2026 work explicitly chose the fast-eval axis. That was
the right Goal-3 (MNIST) move because it unblocked the experiment.
But it's not a substitute for the §5 lever work, and this doc should
not be read as describing a *finished* system.

### 12.6 Action items (suggested, not committed)

In ascending effort:

| Item | Effort | Doc reference |
|---|---|---|
| Enable `prune_by_lower_bound=True` by default for MSE; report pruning rate alongside hit_rate | half day | §5.2 |
| Batched B&B bound check on GPU (one vmap call over the population's interval evaluations, compare to Pareto incumbent vector) | 1-2 days | §5.2 |
| Re-integrate `FunctionalCache` into the batched JAX path (subexpression hashing across the population) | 2-3 days | §5.1 / §4 |
| Equality saturation prototype (snake-egg or hand-rolled e-graph) for tessera's grammar at K≤8; measure `\|E_K\|` empirically | 1-2 weeks | §5.3, §6 |
| Burnside-bound experiment (§11.3): compute `\|E_K\|` analytically and compare to empirical | 1-2 days | §11.3 |

None of these block the current MNIST experiment. All of them
sharpen the doc's claims from "framework that predicts" to
"framework that the implementation actually realises."

## Changelog
- 2026-05-25: initial document. Independent companion to
  search_as_energy_min.md.
- 2026-05-24: added §11 (refined game-theoretic structure) with five
  under-articulated angles: alpha-beta genealogy, eval-vs-rewrite
  two-budget formalization, Burnside/Pólya bridge to |E_K|, Pareto as
  cooperative game with Nash bargaining solution, generalization as
  Stackelberg game with noise as follower. Three actionable
  experiments proposed.
- 2026-05-24: added §12 (open self-criticism). Honest audit of how the
  current implementation utilises the framing's predicted levers.
  Verdict: GPU port (Tiers 1-3) addressed fast-eval but not
  fewer-evals; three Knuth-style levers (caching, B&B, equality
  saturation) remain underused. Action items listed for sharpening
  the doc-implementation gap.
