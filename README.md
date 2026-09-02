# scAnno

**Hierarchical cell-type annotation that truncates rather than guesses.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-0.11.0-blue.svg)](#status)

scAnno returns a label at the deepest level the evidence supports and no deeper — `Lymphoid` when
it cannot separate T from NK, `Lymphoid/T cell` when it can.

📖 **[Quickstart](docs/QUICKSTART.md)** · **[User guide](docs/USER_GUIDE.md)** ·
**[Known issues](KNOWN_ISSUES.md)**

> **Validated on human blood only** — two PBMC datasets, 18 populations, zero errors. Not another
> tissue, not another species, and **not single-nucleus data**, on which it is nevertheless being
> used. See [Status](#status).

---

## Install

```bash
git clone https://github.com/JiaenLin/scAnno.git && cd scAnno
pip install -e '.[run]'      # '.[run]' adds anndata + scanpy for reading .h5ad
scanno selftest
```

The decision layer is numpy and scipy only. The marker corpus is **not** distributed: build a
SQLite database from a CellMarker release, or point `--db` at any database with an `assertion`
table carrying `species`, `tissue_class`, `cell_name`, `symbol_norm`, `evidence_tier` and
`n_pmids`.

Check what the corpus knows about your tissue first — if `panel` refuses, tuning will not help:

```bash
scanno panel --db corpus.db --species Human --tissue Blood --top 10
```

## Run

```bash
scanno cluster  --h5ad qc.h5ad --out clustered.h5ad --resolutions 0.5,1.0,2.0 --split-by sample

scanno annotate --h5ad clustered.h5ad --cluster-key leiden --tree tree.json \
                --db corpus.db --species Mouse --tissue Heart --store store.npz \
                --out-h5ad annotated.h5ad --report
```

`cluster` normalises, selects variable genes over every gene, runs PCA, neighbours and Leiden at
every resolution asked for, keeping all of them. It refuses rather than proceeding when it cannot
find raw counts to preserve. `--split-by` clusters each sample independently — no shared variable
genes, no joint embedding, no batch key.

`annotate` scores clusters against the tree and writes the annotation per **cell**.

## Weight sources

**Without an atlas** — the default. Node weights come from a curated marker corpus, each gene
charged its best competing sibling claim.

**With an atlas** — node weights come from expression profiles built from annotated datasets.
Better where available; the corpus covers far more tissues than atlases do.

Both share the same gene background (per-gene mean/sd, tissue-general) and the same rooted walk.
Measured on PBMC:

| | pbmc3k | pbmc68k *(independent, FACS labels, 765 genes)* |
|---|---|---|
| **corpus only, no atlas** | 8 correct · 0 coarser · **0 wrong** | 7 correct · 3 coarser · **0 wrong** |
| atlas profiles | 8 · 0 · **0** | 8 · 2 · **0** |

Nothing is fitted at runtime. A result is reproducible from the store digest plus the tree, and
every failure is attributable to one of three inputs rather than to hidden training state.

## Steps

| # | step | does | assigns | state |
|---|---|---|---|---|
| 0 | ingest | validate samplesheet and declared tree, bind it to the corpus | — | specified |
| 1 | cluster | normalise → HVG → PCA → neighbours → resolution sweep | — | built |
| 2 | score | one pass over cells, then walk the tree, scoring only each node's children | proposes | built |
| 3 | **assign** | the only code path that writes a label into `obs` | **yes — only here** | built |

Only siblings are ever compared, so a call never depends on how many distant cell types are in the
taxonomy. Scores are standardised against a stored background rather than against the run, so a
cluster's score does not change when something else is added to or removed from the object.

📄 **[Why the design is this shape](docs/RATIONALE.md)** ·
**[The classifier as built](docs/CLASSIFIER.md)**

## The annotations, and what separates them

scAnno can deliver several label columns for the same cells. They differ along **two independent
axes**, and confusing the two is the easiest way to misread the output.

### Axis one — which tree the walk ran against

| | |
|---|---|
| **L1** | an independent depth-1 walk over the complete declared compartment set, against the full corpus. No seal at any depth can move it; it is a real walk, not a truncation of the scope path |
| **scope** | the same walk run against the voted scope. At each node only children that survived the vote are scored, so labels terminate at whatever depth the cohort's evidence supports |

They can disagree, and the disagreement is information: the report carries the agreement between
them as a result. There is no "level 2" or "level 3" annotation — intermediate depths are
truncations of the scope path, and a share quoted at a depth nothing was annotated at has no call
behind it.

The scope restricts the **tree**, not the database. A subtype absent from the scope annotation
means the cohort could not agree to make that split.

### Axis two — how much the annotation is willing to assert

Three columns, each built from the one above it, all shipped side by side. **Each is a view over
its predecessor, never a replacement**, so moving back a row is a column drop and not a re-run.

| column | what it is | `UNRESOLVED` |
|---|---|---|
| `<prefix>_path` | **the walk, as it ran.** A cluster descends while the sibling contrast clears the gap bar and stops when it does not, so a cell carries the deepest label the evidence supported — `Lymphoid` where T and NK could not be separated. A cell whose walk failed at the root is `UNRESOLVED`, and that is a statement, not a gap | possible |
| `<prefix>_resolved_path` | **the forced assignment of the cells the walk declined.** `--resolve` pushes every `UNRESOLVED` cell from the root down to a leaf, and every cell stranded on an internal node down from there, using the argmax the walk ALREADY recorded — so nothing is scored that was not scored anyway and nothing is invented. **Their margin was below the gap bar by construction**: these are the least certain calls in the object. `<prefix>_resolved_origin` says per cell whether its leaf was reached or assigned | none |
| `<prefix>_path_joint` | **the joint correction of the forced label, in BOTH directions.** A second, JOINT clustering of the whole cohort is annotated against the same tree, scope, store and corpus, and the column follows what that partition says: a cell is **recovered** onto a joint cluster's label where the samples it came from carry that label nowhere, and **absorbed** off a label route B delivers nowhere at all, onto its own cluster's call. A coarser partition does both, and a column carrying only the first would make it look strictly better than the one it corrects. `<prefix>_path_joint_origin` says per cell `kept`, `joint_recovered` or `joint_absorbed` | none |

Use the plain column when the question is what the annotator was willing to assert; the forced
column where a column with no holes is needed — a composition table, a viewer colour-by, a
semi-supervised label; and the joint column as a **second opinion about the clustering**, not as
the answer.

### The joint route is not the authority

It is the **coarser** partition, and coarseness cuts both ways. It recovers populations the
per-sample clustering merged, and it merges populations the per-sample clustering recovered —
`scanno compare` reports both directions, and `lost_labels` names every label the first route
delivered that the joint clustering absorbed, what absorbed it, and which samples lost it.
Reporting only the direction where the joint route wins would present one route's losses as the
other's gains.

Nothing is gated. Every candidate is applied and each carries its cluster's **sample dominance**
and the share of the cluster the first route **already agreed** on — a cluster that is mostly one
animal cannot arbitrate anything, and a cluster the two routes disagree about is the joint route
asserting something the first one denies. Both are reported on every row and neither decides,
because a statistic does not gate an output here until it has been shown to separate correct from
incorrect calls (`docs/PRINCIPLES.md` §3).

```bash
# route B: ONE clustering over every sample together, annotated against the SAME scope,
# at EVERY resolution — one command, one store digest, one walk per granularity
scanno cluster  --h5ad cohort.h5ad --out joint.h5ad --resolutions 0.25:2.0:0.25
scanno annotate --h5ad joint.h5ad --tree tree.json \
                --cluster-key leiden_1p0 --cluster-key leiden_0p5 --cluster-key leiden_2p0 \
                --scope scope.json --store store.npz --species Mouse --tissue Heart --resolve

# the third column, the document, and the tables — route A is corrected, never replaced.
# --path-key-b reads the SWEEP CONSENSUS, so a correction rests on cells the sweep agrees about
scanno compare --a per_sample.h5ad --b joint_annotated.h5ad \
               --path-key cell_type_forced --path-key-b scanno_consensus \
               --agreement-key scanno_consensus_agreement \
               --sample-key sample --cluster-key leiden_1p0 --group-key group \
               --out-h5ad delivered.h5ad --out-key cell_type_joint_route \
               --out-report joint_route.html --out-table candidates.csv \
               --out-impact impact_per_sample.csv
```


## Output

`--out-h5ad` writes the input object with columns added and nothing else touched — `X`, `var` and
`obsm` come out as they went in.

| column | is |
|---|---|
| `scanno_cell_type` | the label, as a categorical |
| `scanno_path` | the full root-to-leaf path the walk took |
| `scanno_depth` | how deep it got before the evidence ran out |
| `scanno_gap` | the decision gap of the accepted step |
| `scanno_survival` | how much of the node's evidence survived the sibling contrast |
| `scanno_support` | curated tier-1/2 assertions behind the winning node |
| `scanno_resolved_path` | with `--resolve`: the forced label, no holes, plus `scanno_resolved_origin` |
| `scanno_resolved_path_r<tag>` | with more than one `--cluster-key`: one column per resolution, the label path and nothing else |
| `scanno_consensus` | the label a cell keeps **across the sweep**, with `scanno_consensus_agreement` — the share of the weight that *could* have voted for it. Written only when the sweep has at least two resolutions, because an agreement column that is 1.0 by construction reads exactly like one that was measured |
| `scanno_path_joint` | with `compare --out-h5ad`: the joint correction, plus `scanno_path_joint_origin` |

Three properties, asserted in `tests/test_emit.py`:

- a flagged cell is `EXCLUDED` whatever its cluster was called — the exclusion is per cell
- a statistic of a call that was not made is `NaN`, never `0`
- a cluster with no call raises rather than being labelled something plausible

## Upstream provenance

An object carrying scQC's `uns["scqc"]` declaration arms the exclusion automatically, after
verifying the flag against its digest. A declaration that does not check out refuses. An object
with a flag column and no declaration gets nothing — scAnno reads declarations and never infers
meaning from a column name.

`--exclude-flag` withholds exactly the flagged cells without deleting anything, and there is no
mode or threshold that widens that set.

## Report

`--report` writes one self-contained HTML file plus a `report.json` carrying every number in it:
composition, labels on the embedding, reliability by tree depth, the corpus markers behind the
calls, what was withheld and how unevenly, every cluster call, and provenance. Every section
states what it cannot show, and a missing limit is counted as a defect on the report's front page.

## Commands

`annotate` · `cluster` · `background` · `scope` · `compare` · `report` · `embed` · `lab` ·
`readme` · `resolution` · `calibrate` · `panel` · `store-info` · `agent` · `selftest`.
Exit code 2 is a refusal.

`resolution` picks a clustering resolution from the annotation rather than the geometry.
`compare` scores two annotated objects against each other, naming the confused pairs rather than
only a percentage. `agent` is an optional second opinion — bring your own key — and never replaces
`annotate`.

## Documentation

| | |
|---|---|
| [QUICKSTART](docs/QUICKSTART.md) | install, tree, annotate, read the output |
| [USER_GUIDE](docs/USER_GUIDE.md) | every command, every refusal, the gene background |
| [RATIONALE](docs/RATIONALE.md) | why the design is this shape, and what is deliberately absent |
| [PRINCIPLES](docs/PRINCIPLES.md) | the rules the code enforces |
| [CLASSIFIER](docs/CLASSIFIER.md) | the design as built: `Z`, the walk, both weight sources |
| [CALIBRATION](docs/CALIBRATION.md) | how the store learns from atlases |
| [TRAINING](docs/TRAINING.md) | whether trained weights help, measured |
| [READING_THE_OUTPUT](docs/READING_THE_OUTPUT.md) | a playbook for the annotated object, with a notebook |
| [KNOWN_ISSUES](KNOWN_ISSUES.md) | measured, reproduced, not yet fixed |

## Status

**0.11.0.**

| | |
|---|---|
| ✅ classifier | store, gene background, corpus weights, rooted walk with truncation |
| ✅ validated | human blood: PBMC, two datasets, 18 populations, zero errors on both paths. One dataset is independent with FACS labels |
| ⚠️ one tissue, one species | every number is an existence proof, not a range |
| ⚠️ no single-nucleus validation | nuclear and whole-cell transcriptomes differ systematically; the gene background would need building for the assay |
| ❌ novelty detection | a cluster whose type is absent from the store may be assigned to a sibling. Two formulations failed — see KNOWN_ISSUES |
| ✅ cluster, assign, report | `--out-h5ad` and `--report`, with the h5ad round trip asserted down to the categorical encoding |
| ✅ joint route | a third column correcting the forced one, with the document, the per-sample impact, and the populations the joint clustering ABSORBED reported beside what it recovered |
| ✅ resolution sweep | the joint route walked at every granularity against one store and one scope, voted per cell, with each candidate carrying how much of the sweep agrees — a population too rare to cluster at one resolution clusters at another, and one resolution cannot say which happened |
| ✅ the vote is not a plain majority | a coarse partition cannot separate a rare population **by construction**, so an equal count lets the resolutions blind to one outvote those that see it. A label's support is divided by the weight of the resolutions that deliver it *somewhere* — missing evidence is not weak evidence — and coarse partitions weigh less, by the number of clusters they actually produced |
| ✅ scope: SEAL / KEEP / FORCE | voted across a cohort. A forced call is never pooled with a gap-cleared one: `<prefix>_assignment` records how each cell was assigned and `uns` carries every step's margin |
| ✅ exclusion, provenance | equivalence to deletion asserted by digest, not only by count |
| ✅ calibration, resolution, kNN diagnostic, two-route check, agent | all built and tested |
| ❌ no ingest, no task graph | step 0 is specified, not built |
| ❌ one evidence stream | reference label transfer and de-novo marker lookup are designed, not built |

## Licence

MIT — see [LICENSE](LICENSE).
