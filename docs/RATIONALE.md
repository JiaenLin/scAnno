# Rationale

Why scAnno is shaped the way it is. Moved here from the README so that document
can describe the tool; the reasoning is unchanged.

See also [PRINCIPLES.md](PRINCIPLES.md) for the rules the code enforces.

---

## What motivated it

Three failures shaped this, all measured rather than imagined.

- **A label that depended on what else was in the sample.** Standardising a cluster against the
  other clusters in its own run makes its score a property of the run, not of the cluster.
  Deleting 2% of an object shifted every score by a median 19.4% and flipped a call. For any study
  measuring composition, that turns a technical property into the result.
- **A confidence gate that destroyed correct answers.** A softmax posterior over node scores was
  given veto power without anyone checking it predicted correctness. It didn't: it discarded three
  correct calls to catch one error, taking accuracy from 7/8 to 4/8.
- **A marker panel that scored a solid tissue on olfactory receptors.** A near-silent gene has a minute
  variance, so one stray count gives it an enormous z-score. Guarding against *exactly* zero
  variance does not catch it. The output table looked entirely plausible.

---

## Why the shape is this

- **The tree is rooted, and that is the whole design.** A forest of leaves has nowhere to truncate
  *to*, so its only failure action is `UNRESOLVED` and everything marginal gets thrown away. A
  root is what lets `Lymphoid` be an answer.
- **Only siblings are ever compared.** At each node the score is over that node's children alone,
  so a call never depends on how many distant cell types happen to be in the taxonomy.
- **One pass over cells, then milliseconds.** `cluster_profile` is a single sparse matmul against
  a one-hot cluster indicator. That construction is also what makes non-destructive exclusion
  exact: a cluster's profile depends on its own cells and nothing else.
- **Scores are standardised against a stored background, not against the run.** This is the
  property everything else rests on — see below.
- **Assignment is one code path.** Same reason scQC confines removal to step 7: a reader should be
  able to establish quickly that nothing else can put a label into `obs`.

---

## What is deliberately absent

Each was built, measured and removed. Listed so they do not come back without new evidence.

| removed | measured outcome |
|---|---|
| coverage term | identical failure with it on and off |
| softmax posterior + entropy gate | 7/8 → 4/8 |
| permutation null | every p pinned at the 1/(B+1) floor |
| correlation novelty gate | rejected 4 real populations on independent data |
| negative marker weights | 2 errors on independent data, **invisible on the self-test** |
| node-coherence gate | its statistic depended on which other nodes existed |
| design-differential gate | refused on a comparison where 2 libraries of 10 held 94% of the unresolved nuclei |
| cluster-share exclusion (`--exclude-mode cluster`) | withheld **783 of 2,680** nuclei (29.2%) that upstream QC had *passed*, while keeping 1,918 of 3,815 that it flagged; the size moved 42 → 4,080 with the caller's resolution from one unchanged flag |

**Eight capabilities were built here, measured, and deleted** — some of them only after they had
already shipped, and one of them (negative marker weights) causing errors that were invisible on
the self-test. Hence the standing rule:

> **No statistic gates an output until it has been shown to separate correct from incorrect calls
> on held-out data, reported as an AUC beside the gate it justifies.**

and, since 0.3.0, its companion:

> **scAnno annotates and does not decide what is technical.** Where an exclusion is applied it is
> exactly the per-cell flag it was given. A capability that is merely defaulted-off is one
> argument away from running — see `docs/PRINCIPLES.md` §5.
