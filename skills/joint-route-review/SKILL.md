---
name: joint-route-review
description: Grade the merge candidates a scAnno joint route proposes, and record each verdict against the run. Use after `scanno compare --out-h5ad --out-report`, when a joint clustering has proposed corrections to a per-sample annotation and somebody has to decide which of them are real. Covers what the tool measured, what it deliberately did not, the criteria a verdict rests on, and the things a reviewer must never do.
---

# Reviewing a joint route

**The tool measures and does not judge. You judge and do not measure.** Every number you use is
already on disk; producing one yourself — a script, a notebook cell, a figure — puts a number in
front of a reader that no run regenerates, and that is the one thing this step forbids.

## Why a verdict is recorded rather than said

`scanno compare` names which samples lack a label and stops. Deciding that a study's arms differ
is not the tool's call: a design-differential gate was built into this codebase once, refused a
real comparison in which two libraries of ten held 94% of the unresolved nuclei, and was removed.

But somebody still has to decide, and **a decision taken in conversation is not reproducible.**
It cannot be checked against the run it describes and it is gone with the session that made it.
So the decision is recorded against the candidate it is about, with its reason, and the report
renders it.

## What you are given

From the run, and nowhere else:

| file | what it holds |
|---|---|
| `<pair>_compare_<res>.json` | every candidate, the crosstab behind each, `lost_labels`, the netted `impact` |
| `<pair>_candidates_<res>.csv` | one row per candidate |
| `<pair>_impact_per_sample_<res>.csv` | every sample × label, before and after, in points |
| `report/joint_route.html` | the same, assembled |

## The four criteria, and what each is worth

1. **`pct_route_a_agrees` — how much of the cluster the first route already calls that label.**
   High means the joint route resolved a population the first route mostly agreed on. Low means
   the joint route is asserting something the first route *denies* on most of the cluster, and a
   correction of that shape rests on the joint clustering being right where the other is wrong.
2. **`top_share_pct` — how much of the cluster is one sample.** A joint clustering of an
   un-integrated cohort can group cells by library rather than by cell type. A cluster that is
   mostly one animal **cannot arbitrate anything**, whatever its agreement.
3. **Where the corrected cells fall across the design.** This is the one thing the tool will not
   read for you, deliberately. A correction landing entirely in one level — or giving one level
   none of it — cannot be distinguished from a batch effect when that level is confounded with
   something technical. **You must check the design yourself and say so in the reason.**
4. **What the joint route LOST.** `lost_labels` names every population the first route resolved
   that the joint clustering absorbed. A joint route that destroys a population while recovering
   another is not strictly better, and a review that quotes only the recoveries is half a review.

## Recording

```bash
scanno joint-review --compare <run>/compare/forced_compare_leiden_1p0.json \
    --verdict "17=adopt:siblings under one parent, 85.6% agreement, spread across three levels" \
    --verdict "20=refuse:every corrected cell falls in one level of a confounded factor, and the first route disagrees on 82.5% of the cluster" \
    --out <run>/compare/verdicts.json

scanno compare ... --verdicts <run>/compare/verdicts.json --out-report <run>/report/joint_route.html
```

`adopt` · `refuse` · `undecided`. **A reason is required** and is recorded verbatim. It refuses a
cluster that is not a candidate, a grade outside the vocabulary, and an empty reason.

## What a verdict does NOT do

**It changes no label.** The joint column is what `reconcile` wrote, and every candidate is
applied. `refuse` is a note saying a reader should not build on that correction; it does not
remove it, because a statistic — or a judgement — that silently edited the column would make the
three annotations disagree with their own provenance.

**An ungraded candidate is not adopted by silence.** It is in the column like every other and
simply has nobody's name against it. The report lists it as ungraded, with its cell count.

## Things this step must never do

- **Compute a number.** If a figure you want is not in the run, the fix is a change to scAnno, not
  a script beside it. A number that no run regenerates is a draft, whoever wrote it.
- **Edit an object, a table or a report by hand.** Everything is a run product.
- **Present co-membership as identity.** A candidate says these cells *group* with cells the joint
  route called that label. It does not say they were scored as one.
- **Grade on the design alone.** "This would make the arms differ" is not a reason to refuse, and
  "this would make them agree" is certainly not a reason to adopt. The reason must be about
  whether the *evidence separates* a merged population from an artefact.
- **Quote a number from a run with no `SEALED.txt`.**
