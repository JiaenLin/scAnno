---
name: joint-route-review
description: Grade the merge candidates a scAnno joint route proposes, and record each verdict against the run. Use when a run directory contains report/review_request.md, or when asked which joint-route corrections are real. Needs no API key, no model and no network - the run writes down what a reviewer must read, you do the judging, and `scanno joint-review` records it and re-renders the document.
---

# Reviewing a joint route

**You are the reviewer.** Not a model the tool calls — there is no provider, no key and no
resident agent anywhere in this path. The run writes down what a reviewer has to read, you read
it, and one command records your verdict and re-renders the page. A tool that needed a provider
to be reviewable would be unreviewable on a compute node, which is where it runs.

## The three files

| | |
|---|---|
| `report/review_request.md` | **read this.** One block per candidate: what the joint route calls the cluster, how many cells would move, what they currently carry, the agreement, the sample dominance, which samples lack the label, where the corrected cells fall across the design, and what this clustering *lost* elsewhere |
| `report/joint_route.json` | everything the document was built from. `joint-review` reads it, so recording a verdict costs **no clustering, no annotation and no comparison** |
| `compare/verdicts.json` | what you write. Until you do, it records every candidate as ungraded — which is a result, not a gap |

## The division of labour, and why it is drawn here

`scanno compare` names which samples lack a label and stops. Deciding that a study's arms differ
is not the tool's call: a design-differential **gate** was built into this codebase once, refused
a real comparison in which two libraries of ten held 94% of the unresolved nuclei, and was
removed — `docs/PRINCIPLES.md` §3, no statistic gates an output until it is shown to separate
correct from incorrect calls.

The measuring stops where the judging starts, and **the judging is yours.** It is recorded
against the run rather than said in conversation, because a decision taken in a session cannot be
checked against the run it describes and is gone when the session ends.

## The four criteria

1. **Agreement** — the share of the cluster the first route *already* calls that label. High means
   the joint route resolved a population the first route mostly agreed on. Low means it asserts
   something the first route **denies** on most of the cluster.
2. **Sample dominance** — a joint clustering of an un-integrated cohort can group cells by library
   rather than by cell type. A cluster that is mostly one sample **cannot arbitrate anything**,
   whatever its agreement.
3. **Where the corrected cells fall across the design.** A correction landing entirely in one
   level — or giving one level none of it — cannot be told apart from a technical effect when that
   level is confounded with something technical. **This is the criterion the tool will not
   evaluate for you**, and the reason this step exists.
4. **What the joint route lost.** It is the coarser partition and absorbs populations as well as
   recovering them. A review quoting only the recoveries is half a review.

## Recording

```bash
scanno joint-review --payload <run>/report/joint_route.json \
    --reviewer "<who you are>" \
    --verdict "17=adopt:siblings under one parent, 85.6% agreement, spread across three levels" \
    --verdict "20=refuse:every corrected cell falls in one level of a confounded factor, and the first route disagrees on 82.5% of the cluster" \
    --out <run>/compare/verdicts.json \
    --out-report <run>/report/joint_route.html
```

`adopt` · `refuse` · `undecided`. **A reason is required.** It refuses a cluster that is not a
candidate, a grade outside the vocabulary, and an empty reason. Seconds, no analysis, and the
document comes back with the verdicts in it.

## What a verdict cannot do

- **It changes no label.** Every candidate is already applied to the joint column. `refuse` says a
  reader should not build on that correction; it does not remove it, because a judgement that
  silently edited the column would make the three annotations disagree with their own provenance.
- **Silence is not adoption.** An ungraded candidate is reported as ungraded with its cell count.
- **It cannot cite a number that is not in the request.** If a figure you want was not measured,
  the fix is a change to scAnno, not a calculation of your own. A number no run regenerates is a
  draft, whoever wrote it.
- **It cannot be a preference about the result.** "This would make the groups differ" is not a
  reason to refuse; "this would make them agree" is not a reason to adopt. The reason must be
  about whether the evidence **separates** a merged population from an artefact.
