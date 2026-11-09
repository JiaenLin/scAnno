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

## The design-differential gate was removed

A check refusing when the UNRESOLVED rate differed ≥3× across arms of an experimental design was
specified and never shipped in code. On its first real run it refused a diet comparison in which
two libraries of ten held 94% of the unresolved nuclei, and three of five samples in the
"affected" arm were at zero.

A per-arm rate is dominated by whichever libraries inside the arm happen to fail, and the
materiality floor does not help because the arm clears it on those same libraries. The gate was
never shown to predict correctness, which by the standing rule means it must not gate.

Per-arm and per-library rates remain worth computing — they are the diagnostic that found the two
libraries. They are not a refusal, and deciding that a study's arms differ is not the tool's call.

## Validated on one tissue and one species

Human blood, two datasets, 18 populations, all common types. No rare type, no disease state, no
single-nucleus data — and snRNA is the assay this was designed for. Every number in the README is
an existence proof, not a range.

## Rules chosen on whole cells are used on nuclei unvalidated

`--assay` decides the gene background a run accepts, and the unit the two reports and annotate's
summary name (USER_GUIDE, "The assay"). Three other rules are one rule for both assays, and each was set on human blood, which is
whole cells: `gap_min` (below), the corpus marker weights, and the store's size floors. The corpus
records no assay. Its assertions come mostly from whole-cell and tissue studies. A nucleus holds
little of the cytoplasm's mRNA, so on nuclei a marker that lives there can be under-detected.
Nothing corrects for that. The tool was nevertheless
designed for, and has mostly been run on, single-nucleus data. A per-assay value would be a number
the data has not yet decided, so none is invented here.

## Some prints still say "cells" whatever the assay

Only the two reports and annotate's own summary lines take their unit from `--assay`:
`annotate --report`, and `scanno report`'s pages, `report.json`, tables and figures. These still
say "cell(s)" under `snrna` too, so on nuclei they name the wrong unit:

- `background`: the per-object and total lines (`scanno/cli.py` `_background`);
- `calibrate`: the per-dataset line (`scanno/cli.py` `_calibrate`);
- `annotate`: the `--resolve` summary (`force.py` `format_resolved`), the `--scope` summary
  (`force.py` `format_force`), the `--l1-tree` summary (`emit.py` `format_independent_l1`), and
  the hint printed when nothing was written ("per CELL");
- `rescue`: "cell(s) renamed" and the reach line (`scanno/cli.py` `_rescue`);
- `compare`: "they are not the same cells";
- the README.md that `scanno report` writes beside the pages (`readme.py`);
- the documents and help of the commands that take no assay (`rescue`, `compare`,
  `joint-review`, `scope`, `resolution`, `embed`, `lab`), and help text in general.

No string in the package names the nucleus outside `scanno/assay.py`, and
tests/test_assay_rules.py fails if one does. This entry is the other direction: "cells" said of
nuclei. The fix is the same as the reports': the words from the assay where there is one, and a
neutral word where there is none.

## `scanno report` over objects annotated before the assay was recorded needs `--assay`

`annotate` records its assay in `uns['scanno_context']`, and `scanno report` reads it from there.
Objects annotated before that record existed carry none, and `scanno report` refuses them by name
until `--assay` is given, `--assay sn` for a nucleus cohort. It also refuses an `--assay` that
contradicts the record, and a cohort whose objects record different assays. This is the intended
behaviour, and the USER_GUIDE and `--help` say so. It is listed here because report jobs written
before it will stop at that refusal until they add the flag.

## Upstream QC's declaration does not say which assay it was

`annotate` checks `--assay` against the store and against an earlier annotation of the same object
(`uns['scanno_context']`). It cannot check it against the QC that produced the object, because
scQC's `uns['scqc']` declaration (`scqc/provenance@1`) records the flag and its digest and no
assay. An object QC'd as nuclei and annotated as cells is not caught here until that declaration
carries the assay.

## `gap_min` is declared, not derived

0.15 for the profile path and 0.30 for the corpus path. Both survived an independent test
unchanged and the sweep behind them is in `docs/CLASSIFIER.md`, which makes them defensible
defaults and not measurements. Neither has been calibrated against held-out labels.

## Context transfer is untested

The argument for training marker weights is that reliability learned in one tissue transfers to a
tissue with no atlas. That has never been tested — every result so far is within human blood.
`docs/TRAINING.md` §6 specifies the leave-one-context-out experiment; it needs atlases this
project does not have.

## Step 0, ingest, is specified and not built

Steps 1 to 3 exist: `scanno cluster`, the classifier, and the assignment that writes labels per
cell (`annotate --out-h5ad`), with `--report` and `scanno report` beside them. There is still no
ingest step that validates a samplesheet and the declared tree against the corpus before any work
starts, and no task graph: each command is run by its caller, in the order its caller chooses.

## Calibration is bounded by how many independent sources you have

With one source the partial pooling shrinks node-specific weights most of the way toward the
gene-level prior, which is correct but means a single-atlas calibration barely reorders anything.
`C1` additionally requires five sources of which three are label-clean, so a small calibration will
report `C1 0` and say so. This is a property of the evidence, not a defect, but it does mean the
command is worth running only when you have several independent atlases.
