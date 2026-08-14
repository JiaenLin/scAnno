# Principles

Five rules the code enforces. Each is here because it was got wrong first, and a principle that
lives only in a document is one that gets skipped.

---

## 1 · A score must not depend on what else is in the sample

Standardising a cluster against the other clusters in its own run makes its score a property of
the run. The same population scores differently in two samples, and **if composition shifts with
the study design — which is usually the hypothesis — the labels shift with it.**

Measured: deleting 2% of an object, none of it from the clusters being scored, moved every
winning score by a median of 19.4% and flipped one call.

`Z` is therefore standardised against a **gene background** — per-gene mean and sd across cell
types, held in the store. Nothing in the query's own composition enters. The residual after the
fix is under 1%.

**The background is not an atlas.** It is ~20k numbers per species, tissue-general and shippable.
The distinction matters: composition-independence does *not* require an annotated atlas for every
context, which is what lets the corpus-only path work at all.

## 2 · Truncate, do not abstain

A cluster that cannot be resolved to a leaf is not unknown. When T cell and NK cell cannot be
separated, the correct answer is `Lymphoid` — a true and useful statement.

A flat classifier with a reject option was implemented here and scored **4 of 8 where its own
argmax scored 7 of 8**. The rejected calls were correct. The tree must be **rooted** for
truncation to be expressible: a forest of leaves has nowhere to truncate to, so the only available
failure action is `UNRESOLVED`, and everything marginal gets thrown away.

`UNRESOLVED` is reserved for failure at the root.

## 3 · No statistic gates an output until it is shown to predict correctness

Four statistics in this design's history were given veto power without that check, and **all four
made results worse**:

| gate | outcome |
|---|---|
| softmax posterior + entropy | discarded 3 correct calls to catch 1 error |
| permutation null | every p-value pinned at the 1/(B+1) floor |
| correlation novelty test | rejected 4 real populations on independent data |
| negative marker weights | 2 errors on independent data |

Two of those were **invisible on a self-test** and only appeared on held-out data. So:

- every gating statistic reports its **AUC against correctness**, beside the gate it justifies;
- below ~0.6 it does not gate;
- accuracy is reported **with and without** every gate, and a gate that lowers it is a defect on
  the front page, not conservatism;
- **a self-test cannot certify a gate.** Proven twice.

`gate_auc()` exists for this and is called in the test suite.

## 4 · Unknown is not a value, and missing evidence is not weak evidence

Three shapes, all found here:

- **A near-zero scale in a denominator.** A gene detected in a handful of cells has a minute
  variance, so one stray count yields an enormous z-score — and every cluster in a heart scored on
  `Gm*` predicted genes and olfactory receptors, in a table that looked entirely plausible.
  Guarding against *exactly* zero does not catch it; the failing values are tiny, not zero.
  `safe_scale()` is the only route by which a scale becomes a denominator, and a test enforces it.
- **Redistributing missing evidence.** Normalising a node's weights by the mass that *survived*
  the query's gene coverage rescales a depleted node back to full strength. A panel that lost
  every real marker was scored on housekeeping leftovers and won. Normalisation is by **full**
  mass, so an unobservable node scores near zero.
- **A threshold that deletes what it cannot see.** A detection floor over all cells cannot be
  cleared by a population smaller than the floor. It is per **cluster**. The same bug then
  reappeared one layer up, in the store's own entry threshold, inside the fix for it — which is
  the argument for structural guards over careful intentions.

---

## 5 - scAnno annotates; it does not decide what is technical

An annotation tool that also removes cells is two tools, and the second one is invisible. Its
decisions arrive wearing the first one's name, they are made where nobody is auditing removals,
and downstream they are indistinguishable from biology: a cell type that is absent because it was
never there and one that is absent because the annotator withheld it produce the same table.

So scAnno computes no QC metric, applies no threshold to one, and has no code that turns a flag
into a different set of cells. `--exclude-flag` withholds **exactly** the nuclei named by the
column it is given. The excluded set is the flag - identical at every clustering resolution,
fingerprinted in the record so a reader can check which set actually ran.

**This was got wrong first, which is why it is a principle and not a preference.** Until 0.3.0 the
tool offered `--exclude-mode cluster`, which withheld a whole cluster once some share of it was
flagged. Measured on the cohort it was written for, at one resolution: 2,680 nuclei withheld, of
which **783 (29.2%) carried no flag at all** - cells upstream QC had passed, withheld for their
neighbours - while **1,918 of the 3,815 flagged nuclei were kept**, being in clusters under the
share. Neither a subset nor a superset of the decision it claimed to apply. And the size moved
with the caller's granularity: 42 nuclei at resolution 0.25, 4,080 at 2.0, from one flag that
never changed.

It was not the default and it was documented as discouraged. That is the part worth keeping:
**a capability that is merely defaulted-off is one argument away from running,** and it ran. The
mode was removed rather than re-defaulted, `tests/test_exclude.py` section 5 asserts its symbols
are gone, and the CLI refuses the retired options by name with the numbers above rather than with
argparse's "unrecognized arguments" - a generic error invites the reader to look for a typo when
the answer is that the behaviour no longer exists.

The corollary is a limit, stated because it is easy to read this as a safety property: withholding
exactly the flag is only as good as the flag. scAnno demands a `reason` with the exclusion and
reports what it cost, and it cannot tell you whether the nuclei deserved it.

---

## What this costs

More truncated labels and more refusals than a tool that always answers. That is the intended
trade. The failure these rules prevent — a confident, plausible, wrong label that every downstream
check passes — does not announce itself, and a cell-type name is the stratification variable for
everything computed after it.
