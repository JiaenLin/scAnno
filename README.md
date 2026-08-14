# scAnno

**Hierarchical cell-type annotation that truncates rather than guesses.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-0.3.0-blue.svg)](#status)

Most annotators return a label for every cluster. scAnno returns a label **at the deepest level
the evidence supports, and no deeper** — `Lymphoid` when it cannot separate T from NK, and
`Lymphoid/T cell` when it can. A truncated label is a true statement; a confident wrong one is not.

> **Read [Status](#status) before planning anything.** At `0.3.0` this is a classifier, not a
> pipeline. `annotate`, `calibrate`, `resolution` and `agent` work and are tested, and
> `--exclude-flag` withholds what upstream QC marked without deleting it. There is still no
> ingest step, no assign step, no report and no task graph. **What has been validated is human
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
| 1 | **cluster** | normalise → HVG → PCA → neighbours → resolution sweep, **per sample independently** | — | partial — `scanno resolution` judges a sweep somebody else computed |
| 2 | **score** | one pass over cells, then walk the tree: at each node score only its children | proposes | **built** |
| 3 | **assign** | the only code path that writes a label into `obs` | **yes — only here** | specified |

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

## Excluding what upstream QC flagged

Upstream QC often marks nuclei it considers technical. Annotating them produces a label, and a
label is indistinguishable downstream from one anybody should believe.

```bash
scanno annotate ... --exclude-flag cluster_FLAG        # an obs column of booleans
```

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

**0.3.0.** Precise, because a tool that overstates itself does damage quietly. Every row was
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
| ❌ **no ingest, no assign step, no report, no task graph** | steps 0 and 3 are specified and not built. Step 1 exists only as `scanno resolution` over a sweep somebody else computed. |
| ❌ **one evidence stream** | reference label transfer and de-novo marker lookup are designed and not built. Cross-stream agreement is what a single stream's errors are for. |

## Licence

MIT — see [LICENSE](LICENSE).
