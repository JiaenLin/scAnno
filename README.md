# scAnno

**Hierarchical cell-type annotation that truncates rather than guesses.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-0.2.0-blue.svg)](#status)

Most annotators return a label for every cluster. scAnno returns a label **at the deepest level
the evidence supports, and no deeper** — `Lymphoid` when it cannot separate T from NK, and
`Lymphoid/T cell` when it can. A truncated label is a true statement; a confident wrong one is not.

> **Read [Status](#status) before planning anything.** At `0.2.0` this is a validated
> prototype, not a pipeline. `annotate`, `calibrate`, `resolution` and `agent` work and are
> tested, and `--exclude-flag` withholds what upstream QC marked without deleting it. There is
> still no ingest step, no assign step, no report and no task graph, and the validation remains
> human blood — see [Status](#status) before trusting it on anything else.

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

| # | step | does | assigns |
|---|---|---|---|
| 0 | **ingest** | validate the samplesheet and the declared tree; bind it to the corpus; refuse on a node nothing can represent | — |
| 1 | **cluster** | normalise → HVG → PCA → neighbours → resolution sweep, **per sample independently** | — |
| 2 | **score** | one pass over cells, then walk the tree: at each node score only its children | proposes |
| 3 | **assign** | the only code path that writes a label into `obs` | **yes — only here** |

Four, not eight. scQC needs eight because each gates a deletion and deletions compound; nothing in
annotation compounds, and a label is replaced by re-running.

## Excluding what upstream QC flagged

Upstream QC often marks nuclei it considers technical. Annotating them produces a label, and a
label is indistinguishable downstream from one anybody should believe.

```bash
scanno annotate ... --exclude-flag cluster_FLAG        # an obs column of booleans
```

The flagged nuclei are dropped from the cluster **profile** — they contribute to no mean and no
detection rate, so they cannot influence any other nucleus's label — and each is labelled
`EXCLUDED`, a sentinel that is not a cell type in any taxonomy. **Nothing is deleted:** every
nucleus keeps its place in the object, and the exclusion is undone by re-running without the flag.

**`--exclude-mode cell` is the default, and the alternative is kept only because it existed.**
`cluster` excludes a whole cluster once `--exclude-share` of it is flagged, which has two costs
that the cell mode does not:

| | `cell` (default) | `cluster` |
|---|---|---|
| excludes | exactly the flagged nuclei | whole clusters ≥ share flagged |
| unflagged nuclei removed | **none** | on one cohort, **525 of 2,244** — a quarter of the exclusion |
| depends on your clustering | no | yes: the same flags gave **42** nuclei at resolution 0.25 and **4,080** at 2.0 |
| covers the flag | fully | no — 2,154 of 3,873 flagged nuclei were *kept*, being in clusters under the share |

A flag computed once, upstream, should not change meaning because something downstream chose a
different resolution. The only cluster-level exclusion `cell` performs is a cluster whose every
member was flagged: it has no profile at all, so it cannot be walked — arithmetic, not a
threshold.

**What this cannot do** is tell you whether the flag was right. scAnno takes the decision as
input, demands a reason with it, and reports which clusters were emptied and what it cost. Whether
a flagged nucleus is damaged or is a cell type your QC does not expect is not a question this tool
can answer.

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

**Five proposed additions measurably made this worse.** Hence the standing rule:

> **No statistic gates an output until it has been shown to separate correct from incorrect calls
> on held-out data, reported as an AUC beside the gate it justifies.**

## Install

```bash
git clone <this repo> && cd scAnno
pip install -e .
python tests/test_calibrate.py        # synthetic; no data needed
python tests/test_adversarial.py      # needs scanpy + the PBMC datasets
```

To learn marker reliability from atlases you already have:

```bash
scanno calibrate --manifest atlases.tsv --db corpus.db --tree tree.json \n                 --species Human --tissue Blood --out calib/
```

The marker corpus is **not** distributed — `scanno build-markers` ingests a release you fetch
yourself, the same way scQC ships a reference registry and not a genome.

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

**0.2.0.** Precise, because a tool that overstates itself does damage quietly. Every row was
checked against the tree rather than remembered.

| | |
|---|---|
| ✅ **built and tested** | the classifier: store, gene background, corpus weights, rooted walk with truncation. `tests/test_adversarial.py` re-runs every attack that found a defect. |
| ✅ **validated** | PBMC, two datasets, 18 populations, zero errors on both paths. One of the two is independent with FACS labels. |
| ⚠️ **one tissue, one species** | human blood. Every number is an existence proof, not a range. |
| ⚠️ **no single-nucleus validation** | nuclear and whole-cell transcriptomes differ systematically; the gene background would have to be built for the right assay. **It is nevertheless being used on single-nucleus data**, which is a limitation of that use and not a property this tool has earned. |
| ❌ **novelty detection unsolved** | a cluster whose type is absent from the store may be assigned to a sibling. Two formulations failed; see KNOWN_ISSUES. |
| ✅ **calibration** | `scanno calibrate` builds the store, learns bounded marker reliability and emits the reordered panels. Tested on synthetic data in `tests/test_calibrate.py`. |
| ✅ **exclusion** | `--exclude-flag` withholds what upstream QC flagged, per nucleus by default, without deleting anything. `tests/test_exclude.py` asserts equivalence to deletion; `tests/test_exclude_cell.py` asserts the excluded set is exactly the flag at any clustering granularity. |
| ✅ **resolution** | `scanno resolution` picks a clustering resolution from the annotation rather than from the geometry, with a derived tolerance. |
| ✅ **kNN diagnostic** | `cluster_neighbourhood` / `label_flow` ask whether the annotation respects the manifold. It changes no call. |
| ✅ **agentic second opinion** | `scanno agent` — optional, bring your own key or command. Never replaces `annotate`; it is a second column to read beside it. |
| ✅ **CLI** | `annotate`, `calibrate`, `panel`, `store-info`, `resolution`, `agent`, `selftest`. Exit code 2 is a refusal. |
| ❌ **no ingest, no assign step, no report, no task graph** | steps 0 and 3 are specified and not built. Step 1 exists only as `scanno resolution` over a sweep somebody else computed. |
| ❌ **one evidence stream** | reference label transfer and de-novo marker lookup are designed and not built. Cross-stream agreement is what a single stream's errors are for. |

## Licence

MIT — see [LICENSE](LICENSE).
