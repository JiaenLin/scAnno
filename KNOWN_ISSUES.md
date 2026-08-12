# Known issues

Measured, reproduced, and not yet fixed. This is where a defect lives between being found and
being repaired. Read it before quoting a number.

## Novelty detection is unsolved

A cluster whose cell type has no profile and no corpus panel may be assigned to a sibling rather
than withheld. CD34+ progenitors were called `Megakaryocyte` in one configuration and
`Myeloid/Dendritic cell` in another.

**Two formulations were built and both failed.** An absolute correlation threshold rejected four
real populations on an independent dataset, because the true-match correlation falls with gene
count while the unrelated-pair reference rises, and they cross. A self-normalised version scored
the genuinely novel population as *more* distinctive than six real ones.

Given that four unvalidated gates have now made results worse, this is reported rather than
patched with a fifth. The available mitigation is the refusal in `missing_nodes()`, which covers
the case where the user knows to expect the type.

## Training sharpens at the cost of novelty

Reliability weights raise confidence among the siblings they were trained on, so a cluster
belonging to none of them is pushed harder toward one. Measured: training fixed two sibling
confusions and converted one correctly-withheld novel type into a confident wrong call. The size
of that trade has not been bounded.

## Validated on one tissue and one species

Human blood, two datasets, 18 populations, all common types. No rare type, no disease state, no
single-nucleus data — and snRNA is the assay this was designed for. Every number in the README is
an existence proof, not a range.

## `gap_min` is declared, not derived

0.15 for the profile path and 0.30 for the corpus path. Both survived an independent test
unchanged and the sweep behind them is in `docs/CLASSIFIER.md`, which makes them defensible
defaults and not measurements. Neither has been calibrated against held-out labels.

## Context transfer is untested

The argument for training marker weights is that reliability learned in one tissue transfers to a
tissue with no atlas. That has never been tested — every result so far is within human blood.
`docs/TRAINING.md` §6 specifies the leave-one-context-out experiment; it needs atlases this
project does not have.

## Steps 0, 1 and 3 are specified and not built

Only step 2, the classifier, exists. There is no ingest, no clustering driver, no assign step, no
report and no task graph.
