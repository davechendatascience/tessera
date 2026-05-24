# Tessera doc lifecycle — research → planned → shipped

How ideas move from "loose exploration" to "in the library." Three locations, four transitions, one principle: **each transition is a small mechanical edit, not a rewrite.**

```
┌─────────────────────┐     promote     ┌─────────────────────┐
│  docs/research/X.md │ ──────────────▶ │  planned/roadmap.md │
│  status: ? RESEARCH │                 │  status: ○ PLANNED  │
└─────────────────────┘                 └──────────┬──────────┘
                                                   │ start work
                                                   ▼
                                        ┌─────────────────────┐
                                        │  planned/roadmap.md │
                                        │ status: ▷ IN PROGRESS│
                                        │  + TaskCreate       │
                                        │  + CHANGELOG entry  │
                                        └──────────┬──────────┘
                                                   │ land
                                                   ▼
                                        ┌─────────────────────┐
                                        │  docs/shipped/X.md  │
                                        │  status: ✓ DONE     │
                                        │  + CHANGELOG final  │
                                        │  + roadmap entry    │
                                        │    moves to         │
                                        │    "Recently shipped"│
                                        └─────────────────────┘
```

## Stage 1 — RESEARCH

**Lives in:** `docs/research/X.md`. Status header: **? RESEARCH** at the top.

**Purpose:** capture WHY we're exploring a direction, what's been tried elsewhere, what the open questions are, what would falsify the idea. May contain multiple sub-ideas in one doc.

**Contents:** problem statement, literature anchor, hypotheses, falsification criteria, sub-idea list with rough effort estimates. **Not:** detailed implementation specs; those wait for the "planned" stage.

**Example:** `docs/research/high_dim_symbolic_regression.md` discusses five scalable upgrade directions (§5.1-§5.5). None are committed; all are flagged ○ PLANNED or ? RESEARCH inside the doc.

**Done when:** the doc exists, has a falsification criterion, lists concrete sub-ideas. No code, no tests.

## Stage 2 — PROMOTION TO PLANNED

**The transition:** when we *decide to ship* a specific sub-idea from a research doc.

**Mechanical steps:**

1. Add a focused entry to `docs/planned/roadmap.md`. Format:
   ```markdown
   ### N.M [Title] ○ PLANNED
   
   **Origin:** [`docs/research/X.md`](../research/X.md) §[section].
   
   **What:** [1-2 sentence concrete deliverable]
   
   **Why now:** [why this lever, this moment, ahead of the others]
   
   **Effort:** [estimate]
   
   **Acceptance criterion:** [how we know it shipped + worked]
   ```

2. In the research doc, add a marker at the relevant section:
   ```markdown
   > **Promoted to PLANNED**: see [`docs/planned/roadmap.md`](../planned/roadmap.md) §N.M.
   ```

3. Status flag: leave research-doc internal status as ○ PLANNED (the table inside research/X.md). The roadmap entry is now the canonical-status source of truth.

**Done when:** roadmap.md has the new entry, research doc has the back-pointer.

**Do not yet:** create the task, touch CHANGELOG, write any code.

## Stage 3 — IN-PROGRESS

**The transition:** when we start writing the code.

**Mechanical steps:**

1. Roadmap entry status: ○ PLANNED → ▷ IN PROGRESS.
2. Create a TaskCreate task for tracking.
3. Add a [Unreleased] CHANGELOG entry at the time of the first commit (not when work starts — at-landing convention).
4. Commits use `[g2]` or `[g3]` tag matching the goal.

**Done when:** the task is created, work is happening.

## Stage 4 — SHIPPED

**The transition:** when the implementation lands AND has tests AND has design notes worth keeping.

**Mechanical steps:**

1. If the implementation has design notes worth preserving (architectural decisions, API contracts, perf characteristics): create `docs/shipped/X.md`. Otherwise the CHANGELOG entry and the test suite are enough.

   Heuristic: would a future maintainer need to read prose to understand WHY the code is shaped this way? If yes, write a shipped/ doc. If no (small enough that the code+tests tell the whole story), skip.

2. Roadmap entry moves from "open items" sections to the "Recently shipped (pointers, not detail)" table at the bottom of `roadmap.md`. Format: `| Item | ✓ DONE | Where (link to shipped/ doc or just CHANGELOG) |`.

3. CHANGELOG entry under [Unreleased] gets finalized.

4. Research doc back-pointer updates:
   ```markdown
   > **Shipped**: see [`docs/shipped/X.md`](../shipped/X.md) (or the CHANGELOG entry if no shipped/ doc was needed).
   ```

5. TaskUpdate to status: completed.

**Done when:** all five updates are in. The shipped/ doc (if it exists) and CHANGELOG together explain WHY + HOW the feature works.

## Why this matters

Without an explicit lifecycle, three things go wrong:

1. **Research docs grow forever.** A direction that was a one-paragraph note becomes a 500-line dissertation, then becomes the SHIPPING doc, then gets out of date.
2. **Planned doesn't equal committed.** "Maybe we'll do this" hangs around the roadmap for months. The promotion convention forces "are we actually doing this?" to be a discrete decision.
3. **Shipped features lack design notes.** Code lands without prose explaining WHY it's shaped a certain way. Future contributors reinvent the rationale.

The contract above is the smallest set of edits that prevents all three.

## Anti-patterns to avoid

- ❌ **Big design doc up front.** If you're writing a 1000-line shipped/ doc before any code, you're skipping the research stage. Stop and write a `docs/research/` doc first.
- ❌ **Status by prose only.** Inside a single doc, every sub-item should be status-flagged (○/▷/✓/?/×). Without flags, readers can't tell what's done.
- ❌ **Mixed-stage docs.** `roadmap.md` should never contain a "Recently shipped" *detail*; just a pointer. Done items belong in `shipped/`.
- ❌ **Skipping the research stage for "obvious" features.** Even "we should add X" benefits from a 2-paragraph research note explaining tradeoffs. It costs 30 minutes, saves rework.

## Quick reference

| Stage | Where it lives | Status flag | Required artifact |
|---|---|---|---|
| Research | `docs/research/X.md` | ? RESEARCH | Falsification criterion |
| Planned | `docs/planned/roadmap.md` §N.M | ○ PLANNED | Acceptance criterion |
| In progress | same + TaskCreate + CHANGELOG | ▷ IN PROGRESS | Task ID + draft CHANGELOG line |
| Shipped | `docs/shipped/X.md` (if design notes worth keeping) + CHANGELOG | ✓ DONE | Tests + (optionally) design doc |

## Changelog

- 2026-05-24: initial document. Codifies the four-stage lifecycle that's emerged from the May 2026 work but wasn't written down.
