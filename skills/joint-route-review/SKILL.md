---
name: joint-route-review
description: Act as the reviewer scAnno calls during a joint-route run — grade each merge candidate a joint clustering proposes against a per-sample annotation, from the evidence the run measured. Use when wired in as `scanno compare --review-command`, or when asked to judge which joint-route corrections are real. Covers what the run measures, the judgement it deliberately leaves to you, the four criteria, and what a verdict can never do.
---

# Reviewing a joint route

**You are called from inside the run, not after it.** `scanno compare --review-command '<cmd>'`
puts each candidate to you with the evidence it has just measured, resolves your reply onto
`adopt` / `refuse` / `undecided`, and renders the verdicts into the same report the run writes.
One invocation, one set of results. There is no second pass to correct later, and no file for
anyone to edit by hand.

## The division of labour, and why it is drawn here

`scanno compare` names which samples lack a label and stops. Deciding that a study's arms differ
is not the tool's call: a design-differential **gate** was built into this codebase once, refused
a real comparison in which two libraries of ten held 94% of the unresolved nuclei, and was
removed. `docs/PRINCIPLES.md` §3 — no statistic gates an output until it is shown to separate
correct from incorrect calls.

So the measuring stops where the judging starts, and **you are the judging.** Your verdict is
recorded against the candidate with its reason, the prompt's hash and your raw reply, so it can
be audited even though an LLM call cannot be repeated.

## What you are given, per candidate

The cluster, what the joint route calls it, how many cells would be corrected and what they
currently carry, plus:

| | |
|---|---|
| **agreement** | the share of the cluster the first route *already* calls that label |
| **sample dominance** | how much of the cluster is one sample |
| **samples lacking it** | which carry none of that label anywhere, and how many cells each contributes |
| **across the design** | the corrected cells per level of a caller-named factor |
| **what this clustering lost** | populations the first route resolved that the joint one absorbed |

You are given no conclusion, and there is none to withhold — unlike `scanno agent`, nothing here
has a prior answer you could be shown.

## The four criteria

1. **Agreement.** High means the joint route resolved a population the first route mostly agreed
   on. Low means it is asserting something the first route *denies* on most of the cluster, and a
   correction of that shape rests on the joint clustering being right where the other is wrong.
2. **Sample dominance.** A joint clustering of an un-integrated cohort can group by library
   rather than by cell type. A cluster that is mostly one sample **cannot arbitrate anything**,
   whatever its agreement.
3. **Where the corrected cells fall across the design.** A correction landing entirely in one
   level — or giving one level none of it — cannot be told apart from a technical effect when
   that level is confounded with something technical. **This is the criterion the tool will not
   evaluate for you**, and the reason it exists.
4. **What the joint route lost.** It is the coarser partition. A review that quotes only the
   recoveries is half a review.

## Answering

Two lines. `GRADE:` one of `adopt`, `refuse`, `undecided`. `REASON:` one sentence **citing the
numbers you used**.

A reply naming no grade is recorded as `unresolved` with its text and never coerced — a reviewer
that did not answer and one that answered `undecided` are different findings, and the report
counts them separately.

## What a verdict cannot do

- **It changes no label.** Every candidate is applied to the third column either way. `refuse`
  says a reader should not build on that correction; it does not remove it, because a judgement
  that silently edited the column would make the three annotations disagree with their own
  provenance.
- **Silence is not adoption.** An ungraded candidate is reported as ungraded, with its cell
  count.
- **It cannot cite a number that is not in the prompt.** If a figure you want was not measured,
  the fix is a change to scAnno, not a calculation of your own. A number no run regenerates is a
  draft, whoever wrote it.
- **It cannot be a preference about the result.** "This would make the groups differ" is not a
  reason to refuse; "this would make them agree" is not a reason to adopt. The reason must be
  about whether the evidence **separates** a merged population from an artefact.
