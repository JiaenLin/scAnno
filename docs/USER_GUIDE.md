# User guide

Every command, what each refusal means, and the one input people get wrong.

Start at [QUICKSTART.md](QUICKSTART.md) if you have not run it yet.

---

## Exit codes

| | |
|---|---|
| **0** | pass, or review — the run completed and may have printed `REVIEW` lines |
| **1** | error — bad arguments, missing file, missing dependency |
| **2** | **refusal** — scAnno declined to produce a result, and says why |

A refusal is not a crash. It is the tool saying the answer it could give would not be
worth having. Every one of them can be satisfied by supplying something, and the message
says what.

---

## The gene background — the input people get wrong

A cluster's score is `(cluster mean − gene_mu) / gene_sd`, where `gene_mu` and `gene_sd`
are per-gene statistics **across cell types**. That is what makes a score a property of
the cluster rather than of the run.

Standardising against the run's own clusters instead makes the score depend on what else
was sequenced. Measured: deleting 2% of an object — none of it from the clusters being
scored — moved every winning score by a median of 19.4% and flipped one call. If your
composition varies across samples, and it usually does, so do your labels.

**The background is not an atlas.** It is ~20k numbers per species and it is
tissue-general. Three ways to get one:

| | how | independence |
|---|---|---|
| **best** | `--store` from `scanno calibrate` over annotated atlases | full |
| **workable** | one store built once from *all* your samples, reused for each | full, within your cohort |
| **fallback** | `--background-from-clusters` | **none** — prints `REVIEW` |

The fallback is honest about what it costs and is fine for a single object you are not
comparing to anything. **Do not use it per-sample and then compare samples** — that is
the exact failure above.

**A background is also per assay.** It is measured on a transcriptome, and a nuclear one is not a
whole-cell one: most of the cytoplasm's mRNA is missing and unspliced RNA is over-represented. A
store records the assay it was built from, and `annotate --store` refuses a store of the other
assay, or one that records none. See the next section.

---

## The assay: what depends on it, and what does not

`--assay` is required on `background`, `annotate`, `calibrate` and `agent`, and it has no default.
`scrna` means whole cells and `snrna` means nuclei. `sc` and `sn`, this tool's original spellings,
are accepted as the same two. A value is recorded as you declared it and compared canonically, so a
store recorded as `sn` serves a run declared `snrna`. `scanno report` takes `--assay` only for objects
annotated before the assay was recorded in `uns['scanno_context']`.

`report`, `scope` and `rescue` read several annotated objects, and read the assay each recorded.
Objects annotated under two assays are refused by name (`assay_mixed`), each assay with its objects,
before any work: one document, one scope vote or one rescue covers one kind of observation. An
assay this tool does not know is refused too (`assay_unknown`). Objects that record no assay are
accepted, because they predate the record. `scope` and `rescue` take no `--assay`, so they then
name the unit "cells or nuclei".

This table is `scanno/assay.py`'s `RULES`, and `scanno describe` prints it as `assay_rules`.

| rule | what | `scrna` | `snrna` | where its numbers came from |
|---|---|---|---|---|
| `background` | the gene background: per-gene mean and spread across cell types | built from whole cells | built from nuclei | each cohort's own data. A store records its assay, and `annotate --store` refuses the other assay or none. `--background-from-clusters` is the annotated object's own |
| `words` | the unit every print and every document names: the two reports, annotate's summaries, and the prints and documents of every other command | cell, cells | nucleus, nuclei | the declared assay, or the assay the annotated objects recorded. With neither (help text, or objects that record none) the words are "cells or nuclei". Strings recorded in data products keep theirs, because reproductions compare them, and are listed in [KNOWN_ISSUES](../KNOWN_ISSUES.md) |
| `agent_prompt` | what `scanno agent` tells the model it is annotating | single-cell RNA-seq | single-nucleus RNA-seq | the declared assay. A prompt with none is refused |
| `corpus_weights` | marker weights from the CellMarker-schema corpus | the same | the same | the corpus records no assay, and its assertions come mostly from whole-cell and tissue studies. **On nuclei, a marker whose mRNA sits in the cytoplasm can be under-detected, and nothing corrects for it** |
| `gap_min` | the descent bar: 0.30 on the corpus path, 0.15 on profiles | the same | the same | declared, and chosen on human blood (PBMC), which is whole cells. **Not calibrated on nuclei** |
| `detect_floor` | a gene is scored for a cluster only where at least 1% of the cluster detects it | the same | the same | declared in the first release. No calibration on either assay is recorded |
| `ood_floor` | the background must cover 30% of the genes the object expresses | the same | the same | a check that two gene lists overlap. It measures neither transcriptome |
| `walk` | the rooted walk: sibling contrast, truncation, the scope vote, `--resolve` | the same | the same | compares siblings' scores with each other. Its only number is `gap_min` |
| `scope_vote` | a node is sealed unless every sample that reached it descended, and voted on only when two or more samples reached it | the same | the same | declared principles, unanimity and a quorum. They were not fitted to data |
| `normalisation` | counts to 10,000 per observation then log1p; `cluster`'s 2,000 variable genes, 50 components, 15 neighbours, Leiden | the same | the same | scanpy's conventional defaults, not calibrated on either assay |
| `store_sizes` | a cell type enters a store at 10 observations and is graded at 50 | the same | the same | set on PBMC, which is whole cells: at 50 the entry floor excluded the two rarest PBMC types ([CLASSIFIER](CLASSIFIER.md)) |
| `reporting_floors` | what a report shows, never a label: 30 withheld per sample before a per-sample comparison, a 3x ratio between arms, 80% of a cluster from one sample, a 0.1-point floor on the sweep's tolerance | the same | the same | declared. Each is reported and none changes a label |
| `exclusion` | withholding exactly what upstream QC flagged | the same | the same | scAnno reads the flag and decides nothing. Upstream's own rules are per assay |
| `species_check` | the declared species against the gene identifiers | the same | the same | identifiers name a species, not an assay |

**"The same" is a statement about the code, not a validation.** Three of those rules were set
on whole cells and are used on nuclei unvalidated (`corpus_weights`, `gap_min`, `store_sizes`),
which is recorded in [KNOWN_ISSUES](../KNOWN_ISSUES.md). The family's rule is that nothing
calibrated on nuclei is lent to cells. No row here records a calibration on nuclei. That is a
statement about the record, and two rows, `detect_floor` and `reporting_floors`, record no
calibration of any kind.

---

## `scanno annotate`

Label the clusters of one object.

```bash
scanno annotate --h5ad sample.h5ad --cluster-key leiden_1p0 --tree tree.json \
                --species Human --tissue Blood --assay scrna|snrna \
                --db corpus.db  |  --store calib/store.npz \
                [--background-from-clusters] [--use-raw] [--gap-min 0.30] \
                [--min-tier 4] [--out labels.tsv] \
                [--out-h5ad annotated.h5ad] [--label-prefix scanno]
```

**Two outputs, and they are not the same thing.** `--out` is the per-CLUSTER table, for reading
and for a run log. `--out-h5ad` is the object with the label written onto every CELL, which is
what anything downstream opens. Give neither and the labels exist only in the terminal; scanno
says so rather than exiting quietly as though it had saved something.

**Weights come from `--db` (corpus) or `--store` (atlas profiles).** If both are given the
corpus wins; they are alternative sources for the same matrix, not layers. The corpus path
is the default in practice because most tissues have no atlas.

**Expression must be log1p-normalised.** scAnno measures rather than assumes: integer-looking
values above 50 are normalised for you with a printed note, and negative values are refused
because they mean the matrix is scaled. `--use-raw` reads `.raw`, which is where most
processed objects keep their log-normalised counts.

### Output columns

| | |
|---|---|
| `path` | the label, e.g. `Lymphoid/T cell`. `UNRESOLVED` only if the root decision failed |
| `depth` | how far down the tree the evidence reached. Part of the answer, not a quality score |
| `gap` | `(top − runner-up) / max|score|` at the last decision made |
| `support` | curated (tier ≤2) assertions behind the winning node's panel |

**`gap` and `support` answer different questions and you need both.** `gap` says how far the
winner beat its siblings *on this data*. `support` says how much curated evidence the panel rests
on. A small, concentrated panel can beat a large diffuse one with a perfectly healthy gap — so a
node with `gap 0.9` and `support 7` is not the same claim as `gap 0.9, support 50`. Nodes under 10
are starred and listed.

**This bites hardest deep in a tree.** Level-1 nodes usually rest on tens of curated assertions;
level-3 nodes often rest on a handful, and nothing in the score reflects that.

### In the object (`--out-h5ad`)

The same answer, per cell, under `--label-prefix` (default `scanno`):

| column | is |
|---|---|
| `scanno_cell_type` | the label, as a categorical — the column a reader will look for |
| `scanno_path` | the full root-to-leaf path |
| `scanno_depth` | how deep the walk got |
| `scanno_gap` | the gap of the **last** step: the accepted one when the walk reached a leaf or ran out of scorable children, the declining one (below the bar) when it truncated on the gap. NaN when no step was taken |
| `scanno_survival` | share of the node's evidence surviving the sibling contrast |
| `scanno_support` | curated tier ≤2 assertions, where `--db` was given |

`X` and `obsm` are the input's. scAnno adds columns; it does not rebuild the object around its
answer, so an embedding or a symbol column that went in comes out. `var` is re-keyed by
`var['gene_symbol']` when there is one, with every accession kept in `var['gene_id']` and every
duplicated symbol kept as its own row. `--out-gene-key ""` leaves `var_names` alone.
`uns['scanno_context']` records the species, tissue and assay the run declared.

Three things this join gets right, each asserted in `tests/test_emit.py`:

- **A flagged nucleus is `EXCLUDED` whatever its cluster was called**, because the exclusion is
  per nucleus. A cell keeping its cluster's label would quietly undo that.
- **NaN, never 0, for a call that was not made.** A gap of `0.0` sorts first and averages into
  every summary while looking like a marginal call.
- **A cluster with no call raises.** `classify()` returns one row per cluster, in order; a caller
  that filtered or reindexed it would otherwise mislabel a whole population plausibly.

Numeric columns are plain floats rather than nullable pandas integers on purpose: AnnData writes
a nullable integer as an HDF5 *group*, and readers that expect `categories`/`codes` in a group
skip it. A float that is read beats an integer that is dropped.

### The readiness report

`--out-h5ad` finishes by saying what a viewer will need and scAnno cannot supply — an embedding,
expression without negatives, gene symbols beside accession-named rows, and the optional sample
and condition columns. `MISSING` sorts first because it is what blocks a viewer.

It never refuses on these. An object with no embedding is not a bad annotation; it is an object
somebody still has to run UMAP on, and refusing would be scAnno deciding what the object is for.

### Refusals

| message | means | do |
|---|---|---|
| `no gene background` | neither `--store` nor `--background-from-clusters` | supply one; see above |
| `.X contains negative values` | the matrix is scaled | pass `--use-raw` |
| `the background covers only N%` | fewer than 30% of expressed genes are in the background | use a background built for this tissue and assay |
| `the corpus has nothing for X/Y` | no assertions at `--min-tier` | check `scanno panel`; try a neighbouring tissue name |
| `the store S was built from snrna data ... this run declares scrna` (and the reverse) | the store's recorded assay is the other one | build a store from this cohort's own data with `scanno background --assay ...`, or `--background-from-clusters` |
| `the store S records no assay` | the store cannot be checked against the run | rebuild it with `scanno background --assay ...` |
| `X was annotated before as snrna ... this run declares scrna` | the object's `uns['scanno_context']` says the other assay | annotate the object that earlier run started from |

### Withholding what upstream QC flagged

```bash
scanno annotate ... --exclude-flag cluster_FLAG          # boolean obs column
```

| | |
|---|---|
| `--exclude-flag COL` | a boolean `obs` column. **Exactly** its nuclei are withheld from the walk and labelled `EXCLUDED` |

There is one option and no mode. The withheld set is the column you named: not a function of it,
not a re-projection of it through your clustering, not a threshold applied to it. scAnno computes
no QC metric and cannot decide that a nucleus is technical, so it has nothing with which to widen
or narrow what you handed it.

**Nothing is deleted.** Every nucleus keeps its place in the object and its counts; only its label
changes, and re-running without the flag restores it. `EXCLUDED` is upper case and is not a cell
type in any taxonomy, so a consumer that treats it as one is making an obvious error rather than
a quiet one.

The withheld nuclei are dropped from the cluster **profile**, so they contribute to no mean and no
detection rate and cannot influence any other nucleus's label. One cluster-level consequence
remains and is unavoidable: a cluster whose every member was flagged has no profile at all and is
not walked. Every cell in such a cluster was flagged anyway, so no unflagged nucleus is affected,
and it is reported rather than silent.

The run prints a **mask digest** — a fingerprint of the exact set that was withheld — and
`exclusion_record_cells` returns it. Put it in your record: a count cannot show that the set which
ran is the set your QC handed over, because two different masks of the same size agree on every
number in a summary table.

**`--exclude-mode` and `--exclude-share` were removed in 0.3.0** and are refused by name, with what
they did. `--exclude-mode cluster` withheld a whole cluster once a share of it was flagged, which
meant withholding nuclei upstream QC had *passed*, beside their flagged neighbours, while
*keeping* flagged nuclei wherever their cluster stayed under the share. It also made the withheld
set a property of your resolution: a coarser or a finer clustering put different clusters over
the share, from one flag that never changed. If you want a cluster-level exclusion, derive it
upstream where it can be assessed, and hand the resulting column, one value per observation, to
`--exclude-flag`.

**What it cannot tell you** is whether the flag was right. scAnno takes that decision as input and
demands a `reason` with it.

---

## `scanno panel`

What the corpus knows about a context, before you rely on it.

```bash
scanno panel --db corpus.db --species Mouse --tissue Heart --top 20 [--min-tier 4]
```

Prints each cell name, how many genes are claimed for it, and the strongest few. **Run
this first.** Corpus coverage is wildly uneven — plenty of contexts have thousands of
claims and a handful have almost none, and the difference decides whether annotation is
possible at all.

---

## `scanno calibrate`

Learn marker reliability from annotated atlases. Offline, run once by whoever maintains a
corpus; the atlases are consumed and never redistributed.

```bash
scanno calibrate --manifest atlases.tsv --db corpus.db --tree tree.json \
                 --species Human --tissue Blood --assay scrna --out calib/ \
                 [--harmonise] [--min-shared-genes 2000] [--use-raw]
```

Manifest is a TSV:

```
path	source_id	label_key	provenance
/data/a.h5ad	studyA	cell_type	sorted
/data/b.h5ad	studyB	annotation	marker_derived
```

- **`source_id`** groups releases that are *not* independent — same consortium, same
  donors. Counting them separately inflates every grade.
- **`provenance`** is how the labels were obtained. `sorted`, `facs`, `genetic`,
  `hashing`, `multimodal`, `cite-seq` and `curated` count as **label-clean**; everything
  else, **including unrecorded**, is treated as marker-derived. Promotion to `C1` needs
  ≥3 clean sources, because learning from labels assigned by the markers being graded
  measures agreement with prior practice rather than truth.

### Outputs

| | |
|---|---|
| `store.npz` | profiles + the gene background + grades. Feed to `--store`. |
| **`reliability.tsv`** | **the scientific output** — every claim with its citation weight, learned `L`, posterior, both ranks, and a verdict |
| `panels.tsv` | each node's top 50 markers reordered by measured power |
| `calibration.json` | digest, census, bounds |

`L` is bounded to **[0.25, 4.0] and never zero**: a weight driven to zero deletes a marker
from every future annotation, and the ones at risk are the rare, lightly cited markers
that are the only validated marker for their type.

### It only earns its keep with several sources

With one source the pooling correctly shrinks node weights toward a gene-level prior, so
a single-atlas calibration barely reorders anything and reports `C1 0`. The command says
so when it happens. That is the evidence being thin, not a defect.

### Refusals

| message | do |
|---|---|
| `the datasets do not share a gene space` | pass `--harmonise` to intersect explicitly; it is then reported |
| `only N genes are shared` | your thinnest atlas is deciding coverage — drop it, or lower `--min-shared-genes` deliberately |

---

## `scanno store-info` · `scanno selftest`

```bash
scanno store-info --store calib/store.npz     # what a store contains
scanno selftest                               # the adversarial suite
```

`selftest` re-runs every attack that found a defect in this design. A **SKIP is not a
pass** — the suite needs scanpy and will say so.

---

## Reproducibility

A result is determined by three things: the object, the tree, and the background. Nothing
is fitted at runtime.

**Record the store digest** (printed by `annotate`, stored in `calibration.json`). The
database is designed to keep learning, which means the same classifier and the same data
will give different labels against a newer store — correct behaviour, and a hazard if the
digest is not written down beside the result.

**The same input gives the same labels, and that is tested.** tests/test_consistency.py runs the
synthetic cohort from the start twice, each command in its own process: `cluster`, `background`,
`annotate` at two resolutions with `--resolve`, and `rescue`. It requires every obs column of every
object to be identical by value between the two runs. It shows this on one machine. A reproduction
on another machine is a separate measurement.

**Every run records its time.** Each `STATUS.<cmd>.json` carries `timings`: the command's wall time
(`wall_s`), the observations it read (`n_obs`), and its steps in order, each with its own `wall_s`.
`annotate`'s steps are `import` (the analysis stack, often most of a small run), `read`,
`normalise`, `profile`, `background`, `standardise`, `corpus`, `walk`, and `force`, `l1`, `resolve`,
one `sweep <key>` per further resolution, `write` and `report` when they run. `cluster` times its
`import` and then each object, and `rescue` times `import`, `read`, `search` and `write`. Whatever
ran after the last named step is `finish`, so the steps add up to the whole. Several runs that share
an output directory overwrite one another's record. `--status-dir` keeps each one.

---

## Things worth knowing before you trust an answer

- **A coarse label is a real answer.** `Lymphoid` at depth 1 means the evidence supported
  that and not more. Treating it as a failure loses information.
- **`UNRESOLVED` is rarer than it should be.** Novelty detection is unsolved: a cluster
  whose type is absent from both corpus and store may be assigned to a sibling rather than
  withheld. See [../KNOWN_ISSUES.md](../KNOWN_ISSUES.md).
- **Validated on human PBMC only**, two datasets, common types, single-cell. No
  single-nucleus validation exists.
- **The tree is yours and it is the biggest lever.** Sibling sets decide every comparison.
  A node whose members are biologically heterogeneous will be scored on a diluted panel
  and will lose to a tighter competitor.
