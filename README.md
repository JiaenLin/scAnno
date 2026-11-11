# scAnno

**Hierarchical cell-type annotation that truncates rather than guesses.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-0.12.0-blue.svg)](#status)

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

To compare against a published number, build the locked environment instead:
`setup/install_env.sh --prefix <path>` creates it from `setup/environment.lock.yml`, which pins the
whole closure (51 packages), and refuses the result unless every version is the locked one and
nothing unpinned was added (pip, setuptools and wheel, the manager's own, aside; the check reads
what the new interpreter imports, so run it with PYTHONPATH unset and PYTHONNOUSERSITE=1);
`--no-editable` writes nothing into the checkout.

The decision layer is numpy and scipy only. The marker corpus is **not** distributed, and the
database built from it is never redistributed: fetch the CellMarker 3.0 release
(`all_cell_marker.zip`) yourself and build your own. `--reference-genes` takes your data's gene
axis (a text file of names, or an `.h5ad`, whose names are read as `annotate` reads them:
`--gene-key`, else `var['gene_symbol']` when present) and records which markers it can measure.

```bash
scanno build-markers --release all_cell_marker.zip --out corpus.db
```

Every delivered row is kept; each (species, tissue, cell name, gene) becomes one assertion keyed by
upper-cased symbol, graded E1-E5 by its evidence, and `corpus_info` records the release's sha256.
`--db` also takes any database with an `assertion` table carrying `species`, `tissue_class`,
`cell_name`, `symbol_norm`, `evidence_tier` and `n_pmids`.

Check what the corpus knows about your tissue first — if `panel` refuses, tuning will not help:

```bash
scanno panel --db corpus.db --species Human --tissue Blood --top 10
```

## Run

```bash
scanno cluster    --h5ad qc.h5ad --split-by sample --out-dir clustered --resolutions 0.5,1.0,2.0
scanno background --h5ad clustered/*_clustered.h5ad --cluster-key leiden_1p0 --out store.npz \
                  --species Human --tissue Blood --assay scrna
scanno annotate   --h5ad clustered/S1_clustered.h5ad --cluster-key leiden_1p0 --tree tree.json \
                  --db corpus.db --store store.npz --species Human --tissue Blood --assay scrna \
                  --out-h5ad annotated.h5ad --report report.html
```

`qc.h5ad` is your QC'd cohort with a `sample` column. `tree.json` is the tree you declare
([QUICKSTART](docs/QUICKSTART.md) §3) - `scanno tree propose` drafts one from the corpus for you
to approve or edit, from the tissue's own lineages, listing what it set aside and why - and `corpus.db` is your marker corpus.

`cluster` normalises, selects variable genes over every gene, runs PCA, neighbours and Leiden at
every resolution asked for, keeping all of them, one `leiden_<resolution>` column each (`leiden_1p0`
for 1.0). Raw counts are kept in `layers['counts']`. An object that arrives with none - `X`
log-normalised and no counts layer - is clustered as it is, with a REVIEW line and
`uns['scanno_cluster']['counts'] = 'absent'`; a scaled `X`, or one that is neither, is refused.
`--split-by` clusters each sample independently — no shared variable genes, no joint embedding, no
batch key — and writes `<sample>_clustered.h5ad` into `--out-dir`.

`background` builds ONE gene background from every sample's clusters, so a cluster's score does
not depend on which library it sat in. `annotate` scores clusters against the tree and writes the
annotation onto every cell (every nucleus, under `--assay snrna`). `--report` writes the HTML
document and a `report.json` beside it.

### The assay

`--assay` is required on `background`, `annotate`, `calibrate` and `agent`, and it has no default:
`scrna` for whole cells, `snrna` for nuclei (`sc` and `sn` are accepted as the same two). Single
cells are not single nuclei, and three things depend on which one you declared:

- **the gene background.** A store records the assay it was built from, and `annotate` refuses a
  store of the other assay, or one that records none, by name. `--background-from-clusters`
  builds the background from the object being annotated, so it is always that object's assay.
- **the words.** Every print and every document names cells for `scrna` and nuclei for `snrna`,
  or names no unit: the declared assay words `background`, `calibrate`, `annotate` and `report`;
  the assay the annotated objects recorded words `rescue`, `scope`, `compare`, `joint-review`,
  `embed` and `readme`; and `cluster`, `lab`, `resolution`, objects that record none and all help
  text say "cells or nuclei". `annotate` records the assay in `uns['scanno_context']`, and
  `scanno report`, `scope` and `rescue` read it from there and refuse a cohort annotated under
  both. Strings recorded in data products still say "cell(s)" whatever the assay, and are listed
  in [KNOWN_ISSUES](KNOWN_ISSUES.md).
- **what `agent` tells the model** it is annotating.

Every other rule is one rule for both assays: the corpus weights, the descent bar, the walk, the
normalisation. The [user guide](docs/USER_GUIDE.md#the-assay-what-depends-on-it-and-what-does-not)
lists each one with where its numbers came from. Most were chosen on whole cells.

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
# route A: every sample walked on its own, the walks voted into ONE scope, then every sample
# annotated against that scope - and the forced columns gathered into one object
for f in clustered/*_clustered.h5ad; do
  scanno annotate --h5ad "$f" --cluster-key leiden_1p0 --tree tree.json --db corpus.db \
                  --store store.npz --species Human --tissue Blood --assay scrna \
                  --out-h5ad "pass1/$(basename "$f" _clustered.h5ad).h5ad"
done
scanno scope --h5ad pass1/*.h5ad --tree tree.json --sample-key sample --out scope.json
for f in clustered/*_clustered.h5ad; do
  scanno annotate --h5ad "$f" --cluster-key leiden_1p0 --tree tree.json --scope scope.json \
                  --db corpus.db --store store.npz --species Human --tissue Blood --assay scrna \
                  --resolve --out-h5ad "annotated/$(basename "$f" _clustered.h5ad).h5ad"
done
scanno embed --h5ad annotated/*.h5ad --out per_sample.h5ad \
             --label-obs scanno_resolved_path=cell_type_forced

# route B: ONE clustering over every sample together, annotated against the SAME scope,
# at EVERY resolution — one command, one store digest, one walk per granularity
scanno cluster  --h5ad qc.h5ad --out joint.h5ad --resolutions 0.25:2.0:0.25
scanno annotate --h5ad joint.h5ad --tree tree.json --scope scope.json --db corpus.db \
                --store store.npz --species Human --tissue Blood --assay scrna --resolve \
                --cluster-key leiden_1p0 --cluster-key leiden_0p5 --cluster-key leiden_2p0 \
                --out-h5ad joint_annotated.h5ad

# the third column, the document, and the tables — route A is corrected, never replaced.
# route B is still ONE resolution's annotation; the sweep says how much of it agrees, per cell
scanno compare --a per_sample.h5ad --b joint_annotated.h5ad \
               --path-key cell_type_forced --path-key-b scanno_resolved_path \
               --agreement-key scanno_sweep_agreement \
               --sample-key sample --cluster-key leiden_1p0 --group-key group \
               --out-h5ad delivered.h5ad --out-key cell_type_joint_route \
               --out-report joint_route.html --out-table candidates.csv \
               --out-impact impact_per_sample.csv
```


### Rescuing a cell type a sample is missing

```bash
# annotate each sample at a LADDER of resolutions - the delivered one, then upward
scanno cluster  --h5ad qc.h5ad --split-by sample --out-dir laddered --resolutions 1.0:3.0:0.25
for f in laddered/*_clustered.h5ad; do
  scanno annotate --h5ad "$f" --tree tree.json --scope scope.json --db corpus.db \
                  --store store.npz --species Human --tissue Blood --assay scrna --resolve \
                  --cluster-key leiden_1p0 --cluster-key leiden_1p25 --cluster-key leiden_1p5 \
                  --cluster-key leiden_1p75 --cluster-key leiden_2p0 --cluster-key leiden_3p0 \
                  --out-h5ad "ladder/$(basename "$f" _clustered.h5ad).h5ad"
done

# the targeted search: only labels some units carry and others lack, only in the units that lack
scanno rescue --h5ad ladder/*.h5ad --label-key scanno_resolved_path --unit-key sample \
              --sweep-prefix scanno_resolved_path_r --cluster-prefix leiden_ \
              --from-resolution 1.0 --to-resolution 3.0 \
              --out-key cell_type_rescued --out-dir rescued \
              --out-table impact.csv --out-report rescue.html --out rescue.json
```

**The ceiling is part of the claim.** Past a granularity a study would report at, an appearance
stops being evidence that a population was merged and becomes evidence the partition was
shattered — so `--to-resolution` is a statement, not a budget, and a label not found below it is
reported as not found rather than chased.

## Output

`--out-h5ad` writes the input object with columns added; `X` and `obsm` come out as they went in.
`var` is re-keyed by `gene_symbol` by default (the accession is kept in `var['gene_id']`); pass
`--out-gene-key ""` to leave `var_names` alone. `uns['scanno_context']` records the species, tissue
and assay the run declared.

| column | is |
|---|---|
| `scanno_cell_type` | the label, as a categorical |
| `scanno_path` | the full root-to-leaf path the walk took |
| `scanno_depth` | how deep it got before the evidence ran out |
| `scanno_gap` | the gap of the **last** step: the accepted one when the walk reached a leaf or ran out of scorable children, the declining one (below the bar) when it truncated on the gap. NaN when no step was taken |
| `scanno_survival` | how much of the node's evidence survived the sibling contrast |
| `scanno_support` | curated tier-1/2 assertions behind the winning node |
| `scanno_resolved_path` | with `--resolve`: the forced label, no holes, plus `scanno_resolved_origin` |
| `scanno_resolved_path_r<tag>` | with more than one `--cluster-key`: one column per resolution, the label path and nothing else |
| `<label>_rescued` | `scanno rescue`: the delivered label with **only** the located clusters relabelled, plus `<label>_rescued_origin` — `kept` or `rescued` per cell |
| `scanno_sweep_agreement` | the share of the sweep whose label matches the delivered one, per cell. Written only when the sweep has at least two resolutions, because an agreement column that is 1.0 by construction reads exactly like one that was measured. **Evidence only — the sweep votes on nothing** |
| `scanno_path_joint` | with `compare --out-h5ad`: the joint correction, plus `scanno_path_joint_origin` |

Three properties, asserted in `tests/test_emit.py`:

- a flagged cell is `EXCLUDED` whatever its cluster was called — the exclusion is per cell
- a statistic of a call that was not made is `NaN`, never `0`
- a cluster with no call raises rather than being labelled something plausible

## What a command leaves behind

scAnno is several commands that compose into one run directory, so each seals itself beside its
own output. Four states are distinguishable from the filesystem alone:

| file | written | says |
|---|---|---|
| `STATUS.<cmd>.json` | first as `partial`, last with the outcome | `ok` · `partial` (it died) · `refused` with the `fix` the refusal named · `failed`; the commit read from `.git` by file, the job, the products on disk, every escape flag as an ask/decision pair with `--by`, and `timings`: the wall time, the observations read, and each step's own wall time |
| `RUNNING.<cmd>.txt` | at start | replaced at exit by one of the two below |
| `SEALED.<cmd>.txt` | exit 0 | the products listed |
| `FAILED.<cmd>.txt` | anything else | a crash is sealed before it is reported; a bad input is one line, not a traceback |

`--status-dir` chooses the directory; the default is beside the command's primary output.
`scanno describe` prints the declaration — needs, provides, `sees` (empty: an annotator is shown
no labels), the sentinels and their aliases, gates, escapes, `cannot_show`, `state_version` — as
JSON, for a host or a reader that will not read the code.

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

`annotate` · `cluster` · `background` · `scope` · `rescue` · `compare` · `joint-review` ·
`report` · `embed` · `lab` · `readme` · `resolution` · `calibrate` · `panel` · `store-info` ·
`agent` · `describe` · `selftest`. Exit code 2 is a refusal.

`resolution` picks a clustering resolution from the annotation rather than the geometry.
`compare` scores two annotated objects against each other, naming the confused pairs rather than
only a percentage. `agent` is an optional second opinion — bring your own key — and never replaces
`annotate`.

## Part of a family

Four tools share one contract. Each works on its own, and they fit together:

| tool | does |
|---|---|
| [scQC](https://github.com/JiaenLin/scQC) | quality control, and who set each threshold |
| [scAnno](https://github.com/JiaenLin/scAnno) | cell-type labels, only as deep as the evidence goes |
| [scIntegrate](https://github.com/JiaenLin/scIntegrate) | whether you need batch integration, and which method |
| [scProfile](https://github.com/JiaenLin/scProfile) | communication, velocity, pseudotime, differential expression |

[single-cell-harness](https://github.com/JiaenLin/single-cell-harness) runs them as one stack, so a
decision made in one can be taken back later and every tool downstream is told what that costs.

## Changing this tool

Using scAnno and changing it are different jobs, and the second one has its own suite.
`DEVPOINTS.yaml` in the root says what can be added here — mechanisms and sentinels — and what each one has to
declare. The suite lives in the harness:

```bash
git clone https://github.com/JiaenLin/single-cell-harness.git ../single-cell-harness
alias sch='PYTHONPATH=../single-cell-harness python3 -m sch'

sch dev map --root .                            # what plugs in here, and what it must declare
sch dev new mechanism my_mechanism       # a skeleton, a spec written before the code, and a test
sch dev check --point mechanism --name my_mechanism   # checks, cheapest first
```

Every check prints what it does **not** prove, so a green run is not read as more than it is. Two
of them run your addition against the same synthetic cohort written twice, with every column
renamed — code that asks for a role passes both, code that knows a column name passes one.

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

**0.12.0.**

| | |
|---|---|
| ✅ classifier | store, gene background, corpus weights, rooted walk with truncation |
| ✅ validated | human blood: PBMC, two datasets, 18 populations, zero errors on both paths. One dataset is independent with FACS labels |
| ⚠️ one tissue, one species | every number is an existence proof, not a range |
| ⚠️ no single-nucleus validation | nuclear and whole-cell transcriptomes differ systematically. The gene background is per assay, and a store of the other assay is refused. The corpus weights and the descent bar are one rule for both and were chosen on whole cells |
| ✅ assay rules | `--assay` decides the background a run accepts and the unit every print and document names, with no default; a command that takes no assay reads it from the objects or names both units. Every other rule is one for both, listed with its basis in the user guide. Strings recorded in data products that still say "cell(s)" are listed in KNOWN_ISSUES |
| ❌ novelty detection | a cluster whose type is absent from the store may be assigned to a sibling. Two formulations failed — see KNOWN_ISSUES |
| ✅ cluster, assign, report | `--out-h5ad` and `--report`, with the h5ad round trip asserted down to the categorical encoding |
| ✅ joint route | a third column correcting the forced one, with the document, the per-sample impact, and the populations the joint clustering ABSORBED reported beside what it recovered |
| ✅ targeted rescue | a rare cell type a unit lacks and another carries is looked for **in that unit alone**, by clustering it more finely step by step and annotating each step normally. On the first step where a cluster comes back as it, **that cluster's cells take the label and nothing else moves** — the finer clustering is discarded, never adopted. A search that finds nothing reports whether the finest clustering reached could even have held the population, so "absent" and "out of reach" are never printed the same way |
| ✅ resolution sweep | the joint route walked at every granularity against one store and one scope, voted per cell, with each candidate carrying how much of the sweep agrees — a population too rare to cluster at one resolution clusters at another, and one resolution cannot say which happened |
| ✅ scope: SEAL / KEEP / FORCE | voted across a cohort. A forced call is never pooled with a gap-cleared one: `<prefix>_assignment` records how each cell was assigned and `uns` carries every step's margin |
| ✅ exclusion, provenance | equivalence to deletion asserted by digest, not only by count |
| ✅ calibration, resolution, kNN diagnostic, two-route check, agent | all built and tested |
| ✅ the marker corpus | `scanno build-markers` builds it from a CellMarker 3.0 release: every row kept, E1-E5 tiers, the release's sha256 recorded. A rebuild of the first cohort's release reproduces its project-built corpus row for row |
| ❌ no ingest, no task graph | step 0 is specified, not built |
| ❌ one evidence stream | reference label transfer and de-novo marker lookup are designed, not built |

## Licence

MIT — see [LICENSE](LICENSE).
