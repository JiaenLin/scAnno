# scAnno

**Hierarchical cell-type annotation that truncates rather than guesses.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-0.10.0-blue.svg)](#status)

Most annotators return a label for every cluster. scAnno returns a label **at the deepest level
the evidence supports, and no deeper** — `Lymphoid` when it cannot separate T from NK, and
`Lymphoid/T cell` when it can. A truncated label is a true statement; a confident wrong one is not.

> **Read [Status](#status) before planning anything.** At `0.9.0` this is a classifier, not a
> pipeline. `annotate`, `calibrate`, `resolution` and `agent` work and are tested,
> `--out-h5ad` writes the annotation back into the object **per cell** — the form anything
> downstream can actually read — `--report` writes a self-contained document beside it, and an
> object carrying scQC's declaration arms the exclusion itself. `scanno cluster` produces the
> partition and `scanno compare` checks it against a second route. There is still no ingest step
> and no task graph.
> **What has been validated is human
> blood** — two PBMC datasets, 18 populations, zero errors — and nothing else: not another
> tissue, not another species, and **not single-nucleus data**, on which it is nevertheless being
> used. See [Status](#status) before trusting it beyond that.

---

## Why

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


## Running scAnno on a cluster

Copy `jobs/TEMPLATE.pbs` into your project and edit the marked block. It carries three things
that are easy to omit and expensive to omit:

- **`set -euo pipefail`.** Without `-e` a failed step is logged, the job continues, and the exit
  trap reads the status of the last `echo` — sealing a failed run as a successful one.
- **A seal that checks products, not just exit status.** A step can exit 0 having written
  nothing, so the trap lists the files the run must have and fails if any is missing.
- **Scratch redirected into the run directory.** R, pip, matplotlib and numba write under
  `$HOME` by default.

Create the run directory with a plain `mkdir` before `qsub`, never `mkdir -p`: a run-key
collision should abort rather than give two jobs one directory. PBS does not create the `-o`
destination and loses the log silently if it is missing.


## The core idea

> **The database learns. The classifier is a fixed function.**

```
classify : (query counts, store-or-corpus, declared tree) → labelled clusters
```

Nothing is fitted at runtime, so a result is reproducible from the store digest plus the tree, and
every failure is attributable to one of three inputs rather than to hidden training state.

## Two ways to run it

**Without an atlas** — the default, and the one most users are in. Node weights come from a curated
marker corpus you supply, with each gene charged its best competing sibling claim.

**With an atlas** — node weights come from expression profiles built from annotated datasets.
Better where you have them; the corpus covers far more tissues than atlases do.

Both share the same **gene background** (per-gene mean/sd, ~20k numbers, tissue-general) and the
same rooted walk. Measured on PBMC:

| | pbmc3k | pbmc68k *(independent, FACS labels, 765 genes)* |
|---|---|---|
| **corpus only, no atlas** | 8 correct · 0 coarser · **0 wrong** | 7 correct · 3 coarser · **0 wrong** |
| atlas profiles | 8 · 0 · **0** | 8 · 2 · **0** |

The untrained path costs one exact call and one extra coarse label. It makes no errors.

## The four steps

| # | step | does | assigns | state |
|---|---|---|---|---|
| 0 | **ingest** | validate the samplesheet and the declared tree; bind it to the corpus; refuse on a node nothing can represent | — | specified |
| 1 | **cluster** | normalise → HVG → PCA → neighbours → resolution sweep, **per sample independently** | — | **built** — `scanno cluster`, `--split-by` for per-sample |
| 2 | **score** | one pass over cells, then walk the tree: at each node score only its children | proposes | **built** |
| 3 | **assign** | the only code path that writes a label into `obs` | **yes — only here** | **built** — `scanno/emit.py`, reached by `--out-h5ad` |

Four, not eight. scQC needs eight because each gates a deletion and deletions compound; nothing in
annotation compounds, and a label is replaced by re-running.

**Why the shape is this, and not the usual one:**

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

## What comes out

```bash
scanno annotate --h5ad clustered.h5ad --cluster-key leiden --tree tree.json \
                --db corpus.db --species Mouse --tissue Heart --store store.npz \
                --out-h5ad annotated.h5ad        # the object, annotated per CELL
```

`classify()` reasons about **clusters**; everything downstream consumes **cells**. Until 0.3.1
`annotate` printed the cluster table, optionally wrote it as a TSV, and stopped — so the join
back onto the object was left to every caller, and the object scAnno had just annotated still
carried no annotation. Nothing could open it.

`--out-h5ad` writes it. The input object with columns added, and **nothing else touched**:

| column | is |
|---|---|
| `scanno_cell_type` | the label, as a categorical — **this is the one a reader wants** |
| `scanno_path` | the full root-to-leaf path the walk took |
| `scanno_depth` | how deep it got before the evidence ran out |
| `scanno_gap` | the decision gap of the **accepted** step |
| `scanno_survival` | how much of the node's own evidence survived the sibling contrast |
| `scanno_support` | curated tier-1/2 assertions behind the winning node, where a corpus was given |

`X`, `var` and `obsm` are the input's. scAnno does not rebuild the object around its answer, so
an embedding or a gene-symbol column that went in comes out.

Three properties, each asserted in `tests/test_emit.py` rather than described here:

- **A flagged nucleus is `EXCLUDED` whatever its cluster was called.** The exclusion is per
  nucleus; a cell keeping its cluster's label because the cluster survived would quietly undo
  that.
- **A statistic of a call that was not made is `NaN`, never `0`.** A gap of `0.0` sorts first
  and averages into every summary while looking like a genuinely marginal call.
- **A cluster with no call raises.** `classify()` returns one row per cluster in order; a caller
  that has filtered or reindexed it would otherwise label a whole population something plausible.

### It tells you what a viewer will still ask for

Writing the file is not the same as the file being usable, so `--out-h5ad` ends with a
readiness report — the annotation, an embedding, expression a viewer will accept, gene symbols
beside accession-named rows, and the optional sample and condition columns:

```
  what a viewer will find in it
  MISSING no 2-D embedding in obsm - a viewer needs one to draw cells. scAnno does
          not compute embeddings; run UMAP upstream
  ok      cell annotation: scanno_cell_type, 7 levels
  ok      gene symbols: var['gene_symbol'] beside accession row names
```

It never refuses on these. An object with no embedding is not a bad annotation — it is an
object somebody still has to run UMAP on, and deciding otherwise would be scAnno deciding what
the object is for.

### Where it goes next

The annotated `.h5ad` is what [scRNA-seq Lab](https://github.com/JiaenLin/scrnaseq-lab) opens.
The lab converts it to the bundle that [scRNA-seq Studio](https://github.com/JiaenLin/scrnaseq-studio)
reads, and **scAnno deliberately does not write that bundle** — one job each, and the bundle
format is the lab's to keep current.

`scanno_cell_type` is named the way it is for this: the lab has to *guess* which column holds
the annotation, and every convention for that guess keys on the substring `cell_type`. Verified
end to end against the lab's own converter, which picked
`{"cluster":"scanno_cell_type","sample":"sample","condition":"condition","embedding":"X_umap"}`
with nothing configured.

## The report

```bash
scanno annotate ... --report reports/annotation.html \
                    --sample-key sample --condition-key condition
```

One self-contained HTML file and a `report.json` carrying every number in it. No CDN, no fonts to
fetch — openable from a filesystem in five years, which is longer than any link survives.

| section | is |
|---|---|
| composition | level 1 and level 2, per sample where the object says which sample a cell came from |
| the labels on the embedding | the picture, because a composition table cannot show *where* a population sits |
| reliability | median decision gap by tree depth, against the share of cells whose node rests on fewer than 10 curated assertions |
| the markers behind the calls | the corpus panels the classifier actually scored on, against the object they were applied to |
| what was withheld | how many, on whose authority, where they sit, and how unevenly they fall across samples and design arms |
| every cluster call | the full table, with gap and support |
| provenance | the store digest, the taxonomy, the corpus, the gap threshold, the columns used |

Two properties it is built to hold, both asserted in `tests/test_report.py`:

- **Every section states what it cannot show**, in the same place as its numbers — and the report
  audits itself, counting a missing limit as a defect on its own front page. A section with no
  limit reads as a section with nothing to qualify, which is the more confident claim and the
  wrong one.
- **A figure that cannot be drawn is a NAMED absence** saying what would produce it. A blank space
  reads as "there was nothing to show".

It degrades rather than refusing. No sample column means no per-sample panel and a line saying so;
no embedding means `A3` is an absence naming UMAP as the fix; no matplotlib means every figure is
an absence and the document still writes.

`pip install 'scanno[report]'` adds matplotlib.

## Excluding what upstream QC flagged

Upstream QC often marks nuclei it considers technical. Annotating them produces a label, and a
label is indistinguishable downstream from one anybody should believe.

```bash
scanno annotate ...                                    # armed by the object's own declaration
scanno annotate ... --exclude-flag cluster_FLAG        # or name the column yourself
scanno annotate ... --no-exclude                       # or annotate everything
```

### It arms itself when the object declares one — and never guesses

An object from [scQC](https://github.com/JiaenLin/scQC) carries `uns["scqc"]`: which column holds
the flag, what it means, how many nuclei carry it, and a digest of the exact set. scAnno reads
that and arms the exclusion, printing what it found before it walks anything:

```
scQC declaration found -> exclusion ARMED on 'cluster_FLAG'
    18 of 300 nuclei (6.00%) are withheld and labelled EXCLUDED
    digest c80a7099e4f46799 verified against the column in this file
    run f82f60bf56ef  commit 2de8c34
    scAnno did not choose these nuclei and cannot widen the set.
```

**This does not weaken the rule that scAnno never decides what is technical.** The distinction is
the whole design and is easy to get backwards:

- **Sniffing** — *"there is a column called `cluster_FLAG`, I will exclude on it"* — is scAnno
  guessing what a column means. That would break the rule, and scAnno does not do it: an object
  with the column and **no declaration gets nothing**, with a line saying so.
- **Reading a declaration** — *"scQC says it wrote this column, here is what it means, here is a
  digest"* — is upstream deciding, in its own record, and scAnno obeying.

A declaration that does not check out is a **refusal**, not a warning: a flag rewritten since
upstream wrote it is not upstream's decision any more, and acting on it while citing that
provenance would attribute a choice to a pipeline that did not make it. Same for an object that
has been subset since, and for a schema this version does not understand.

Precedence is `--no-exclude` > `--exclude-flag` > the declaration. The person at the keyboard
outranks the file; the file outranks the default.

**Why armed is the default, when 0.3.0 concluded the opposite for a different capability.** The
one removed then *widened* a flag — it withheld nuclei upstream QC had passed, 783 of 2,680 on the
cohort it was written for. This one narrows to exactly what QC rejected, and the digest proves it.
The asymmetry is in which error is silent: annotating a flagged nucleus mints a label for a cell
QC refused, and that label is indistinguishable downstream from a good one, while withholding it
produces a visible `EXCLUDED` that `--no-exclude` undoes in full. The louder error is the
recoverable one.

**Exactly those nuclei are withheld.** They are dropped from the cluster **profile** — they
contribute to no mean and no detection rate, so they cannot influence any other nucleus's label —
and each is labelled `EXCLUDED`, a sentinel that is not a cell type in any taxonomy. **Nothing is
deleted:** every nucleus keeps its place in the object, and the exclusion is undone by re-running
without the flag.

There is one option and no mode, because **scAnno does not decide which nuclei are technical.** It
computes no QC metric, applies no threshold, and has no code that turns the flag into a different
set of cells. The excluded set is the column you named — identical at every clustering resolution,
and fingerprinted in the record (a **mask digest**) so a reader can check that the set which ran
is the set your QC handed over. A count cannot show that: two different masks of the same size
agree on every number in a summary table.

The only cluster-level consequence is arithmetic rather than a threshold: a cluster whose every
member was flagged has no profile at all, so it cannot be walked. Every cell in it was flagged
anyway, so no unflagged nucleus is affected, and it is reported rather than silent.

**What this cannot do** is tell you whether the flag was right. scAnno takes the decision as
input, demands a reason with it, and reports what it cost. Whether a flagged nucleus is damaged or
is a cell type your QC does not expect is not a question this tool can answer.

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

## Install

```bash
git clone https://github.com/JiaenLin/scAnno.git && cd scAnno
pip install -e '.[run]'               # '.[run]' adds anndata + scanpy, for reading .h5ad
scanno selftest                       # or run it from the clone: python bin/scanno selftest
python tests/test_calibrate.py        # synthetic; no data needed
python tests/test_adversarial.py      # needs scanpy + the PBMC datasets
```

The decision layer is **numpy + scipy only**, deliberately: a tool that is hard to install is a
tool that gets skipped.

To learn marker reliability from atlases you already have:

```bash
scanno calibrate --manifest atlases.tsv --db corpus.db --tree tree.json \
                 --species Human --tissue Blood --out calib/
```

The marker corpus is **not** distributed, the same way scQC ships a reference registry and not a
genome. Download a CellMarker release and build the SQLite database with your own ingest, or
point `--db` at any database with an `assertion` table carrying `species`, `tissue_class`,
`cell_name`, `symbol_norm`, `evidence_tier` and `n_pmids`. Check what the corpus knows about your
tissue before anything else — if `scanno panel` refuses, no amount of tuning will help:

```bash
scanno panel --db corpus.db --species Human --tissue Blood --top 10
```


### If you do not already have an environment

`pip install -e '.[run]'` installs into whatever environment you are *already in*. If you have
none, or you are reproducing a published number:

```bash
setup/install_env.sh --prefix ~/envs/scanno    # build the locked environment
setup/install_env.sh --check                   # audit the one you have; changes nothing
```

`setup/environment.lock.yml` is captured from an environment that ran a real cohort end to end,
not composed from bounds. The distinction matters: `pyproject.toml` says `scanpy>=1.10` because
scAnno's decision layer genuinely tolerates a range, but **its results do not** — clustering is
not bit-reproducible across versions, and on one cohort a cluster flag moved by 47 nuclei between
two commits of the upstream tool on identical input. Bounds are what scAnno needs to import; the
lock is what a result needs to reproduce.

`--check` grades what it finds rather than failing on the first absence: a missing `anndata` means
scAnno cannot read `.h5ad` at all, a missing `matplotlib` means every figure becomes a named
absence and the report is still written, and a version that merely *differs* means it will run
but may not match a published number.


## Documentation

| | |
|---|---|
| **[docs/QUICKSTART.md](docs/QUICKSTART.md)** | **start here** — install, tree, annotate, read the output |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | every command, every refusal, and the gene background |
| [docs/PRINCIPLES.md](docs/PRINCIPLES.md) | the four rules the code enforces, and what each cost to learn |
| [docs/CLASSIFIER.md](docs/CLASSIFIER.md) | the design as built: `Z`, the walk, both weight sources |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | how the store learns from atlases, and the promotion ladder |
| [docs/TRAINING.md](docs/TRAINING.md) | whether trained weights help, measured — and the reordered panels |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | measured, reproduced, not yet fixed |

## Status

**0.10.0.** Precise, because a tool that overstates itself does damage quietly. Every row was
checked against the tree rather than remembered.

| | |
|---|---|
| ✅ **built and tested** | the classifier: store, gene background, corpus weights, rooted walk with truncation. `tests/test_adversarial.py` re-runs every attack that found a defect. |
| ✅ **validated — on human blood, and only there** | PBMC, two datasets, 18 populations, zero errors on both paths. One of the two is independent with FACS labels. The scope is part of the claim: read the next two rows before quoting this one. |
| ⚠️ **one tissue, one species** | human blood. Every number is an existence proof, not a range. |
| ⚠️ **no single-nucleus validation** | nuclear and whole-cell transcriptomes differ systematically; the gene background would have to be built for the right assay. **It is nevertheless being used on single-nucleus data**, which is a limitation of that use and not a property this tool has earned. |
| ❌ **novelty detection unsolved** | a cluster whose type is absent from the store may be assigned to a sibling. Two formulations failed; see KNOWN_ISSUES. |
| ✅ **calibration** | `scanno calibrate` builds the store, learns bounded marker reliability and emits the reordered panels. Tested on synthetic data in `tests/test_calibrate.py`. |
| ✅ **exclusion** | `--exclude-flag` withholds **exactly** the nuclei upstream QC flagged, without deleting anything, and there is no mode or threshold with which to widen that set. `tests/test_exclude.py` asserts equivalence to deletion and — §5 — that the retired cluster-share symbols are gone; `tests/test_exclude_cell.py` asserts the excluded set is exactly the flag at any clustering granularity, by digest and not only by count. |
| ✅ **resolution** | `scanno resolution` picks a clustering resolution from the annotation rather than from the geometry, with a derived tolerance. |
| ✅ **kNN diagnostic** | `cluster_neighbourhood` / `label_flow` ask whether the annotation respects the manifold. It changes no call. |
| ✅ **agentic second opinion** | `scanno agent` — optional, bring your own key or command. Never replaces `annotate`; it is a second column to read beside it. |
| ✅ **CLI** | `annotate`, `calibrate`, `panel`, `store-info`, `resolution`, `agent`, `selftest`. Exit code 2 is a refusal. |
| ✅ **assign** | `--out-h5ad` writes the annotation into the object per CELL — label, path, depth, gap, survival and support — leaving `X`, `var` and `obsm` untouched. `tests/test_emit.py` asserts the join, the flag override, NaN-not-zero for calls that were not made, and the h5ad round trip down to the `categories`/`codes` encoding a reader looks for. |
| ✅ **cluster** | `scanno cluster` is step 1: normalise, variable genes over EVERY gene, PCA, neighbours, UMAP, Leiden at every resolution asked for. It selects nothing - every resolution is kept, because a sweep that discarded the evidence for its own stopping point would be unfalsifiable - and it REFUSES rather than proceeding when it cannot find raw counts to preserve. `--split-by` clusters each sample independently: no shared variable genes, no joint embedding, no batch key. |
| ✅ **two-route check** | `scanno compare` scores two annotated objects against each other, excluding from the denominator any cell one route withheld, naming the confused PAIRS rather than only a percentage, and reporting how much of route B is one sample - because a joint clustering of an un-integrated cohort can group by library, and then disagreement indicts B. |
| ✅ **report** | `--report` writes one self-contained HTML file plus a `report.json` carrying every number in it: composition, the labels on the embedding, reliability by tree depth, the corpus markers behind the calls, what was withheld and how unevenly, every cluster call, and provenance. Every section states what it cannot show, and the report counts a missing limit as a defect on its own front page. |
| ✅ **scope: SEAL, KEEP and FORCE** | `scanno scope` votes each internal node across a cohort and `annotate --scope` acts on the verdict. A SEAL deletes a child set, so the walk stops there for every sample alike. A FORCE says the cohort agreed the split is admissible, so **no cell may terminate on an internal node**: a stranded cluster is pushed to the child the unchanged walk already measured, and if that child is itself internal the push REPEATS — each further node really scored, by the same weights over the same data — until a leaf, at any depth. Nothing is invented: a chain that cannot reach a leaf is recorded and the run refuses rather than delivering a half-pushed cell. A forced call is not a gap-cleared call and the two are never pooled silently: `<prefix>_assignment` says how each cell was assigned, `<prefix>_force_depth` through how many steps outside the walk, and `uns` carries the chain and every step's margin — because only the FIRST step is below the bar by construction and a chain is only as strong as its weakest step. `tests/test_force.py`. |
| ✅ **upstream provenance** | an object carrying scQC's `uns["scqc"]` declaration arms the exclusion automatically, after verifying the flag against its digest. A declaration that does not check out REFUSES. An object with a flag column and no declaration gets nothing - scAnno reads declarations and never guesses from a column name. |
| ❌ **no ingest, no task graph** | step 0 is specified and not built. Step 1 exists only as `scanno resolution` over a sweep somebody else computed. |
| ❌ **one evidence stream** | reference label transfer and de-novo marker lookup are designed and not built. Cross-stream agreement is what a single stream's errors are for. |

## Licence

MIT — see [LICENSE](LICENSE).
