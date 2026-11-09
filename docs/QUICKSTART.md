# Quickstart

From nothing to labelled clusters. Ten minutes, no atlas required.

---

## 1 · Install

```bash
git clone https://github.com/JiaenLin/scAnno.git && cd scAnno
pip install -e '.[run]'          # '.[run]' adds anndata + scanpy
scanno selftest                  # or: python bin/scanno selftest
```

The decision layer itself is numpy + scipy only. `[run]` is needed to read `.h5ad`.

## 2 · Get a marker corpus

scAnno **does not ship one**, and the database is never redistributed. Download the CellMarker 3.0
release (`all_cell_marker.zip`) and build it:

```bash
scanno build-markers --release all_cell_marker.zip --out corpus.db
```

It keeps every row, folds `Cd3d` and `CD3D` into one gene, grades each assertion E1 (two or more
experimental papers) to E5 (a marker detector, no paper), and records the release's sha256 in the
database. Add `--reference-genes genes.txt` (or your `.h5ad`) to mark which markers your data can
measure. `--db` also takes an existing SQLite file with an `assertion` table carrying `species`,
`tissue_class`, `cell_name`, `symbol_norm`, `evidence_tier` and `n_pmids`.

Check what the corpus knows about your tissue before anything else:

```bash
scanno panel --db corpus.db --species Human --tissue Blood --top 10
```

The species and tissue are matched to the corpus's own spellings (`human`, `Homo sapiens` and
`Human` are one species; `blood` is `Blood`). If the corpus names no such tissue for the species,
`panel` refuses (exit 2) and lists the tissues it does name, most curated first: pick one, or pass
`--tissue-fallback species` to read every tissue of the species. That flag, and `--marker-species`
for a species the corpus has no rows for, are escapes: printed as REVIEW and recorded in STATUS
and in the annotated object.

## 3 · Declare a tree

Write it yourself, or start from a draft the corpus proposes - root, the corpus's lineages,
their cell names - and edit it: `scanno tree propose --db corpus.db --species Human --tissue
Blood --out tree.draft.json`. A draft says it is one (`proposed.approved_by` is empty), and
every `annotate` run that uses it prints REVIEW and records `tree: proposed draft` until you
set your name there. The tree is yours either way.

Save it as `tree.json`: a JSON file with three keys. **The tree must be rooted** — that is what
lets scAnno return `Lymphoid` when it cannot separate T from NK, instead of giving up.

```json
{
  "children": {
    "root":     ["Lymphoid", "Myeloid"],
    "Lymphoid": ["T cell", "B cell", "NK cell"],
    "Myeloid":  ["Monocyte", "Dendritic cell"]
  },
  "patterns": {
    "Lymphoid":       ["t cell", "b cell", "nk cell", "natural killer"],
    "Myeloid":        ["monocyte", "macrophage", "dendritic"],
    "T cell":         ["t cell", "thymocyte"],
    "B cell":         ["b cell", "plasma cell"],
    "NK cell":        ["natural killer", "nk cell"],
    "Monocyte":       ["monocyte", "macrophage"],
    "Dendritic cell": ["dendritic"]
  },
  "members": {}
}
```

- **`children`** — the tree. Every internal node lists its children; `root` is required.
- **`patterns`** — substrings of corpus `cell_name` values. A cell name joins a node's panel when
  **any** of the node's patterns occurs in it, so the order within a list does not matter. Names
  are matched lower-cased: write patterns in lower case, or they match nothing. One name can match
  two siblings' patterns (`endothelial` and `lymphatic endothelial`), and then it feeds both
  panels.
- **`members`** — corpus/atlas cell-type names per node. Needed only for the atlas path;
  leave `{}` when scoring from the corpus.

Nodes are scored only **against their own siblings**, so a node needs to be
distinguishable from its brothers, not from everything.

## 4 · Cluster, then annotate

`sample.h5ad` is one QC'd sample with raw counts in `X` (or in `layers['counts']`). A sample
delivered log-normalised, with no counts anywhere, is clustered as it is and recorded as
`counts: absent`.

```bash
scanno cluster  --h5ad sample.h5ad --out clustered.h5ad --resolutions 1.0
scanno annotate --h5ad clustered.h5ad --cluster-key leiden_1p0 \
                --tree tree.json --db corpus.db \
                --species Human --tissue Blood --assay scrna \
                --background-from-clusters \
                --out labels.tsv \
                --out-h5ad annotated.h5ad
```

`cluster` writes one `leiden_<resolution>` column per resolution: `leiden_1p0` for 1.0.
**`--assay` is required and has no default**: `scrna` for whole cells, `snrna` for nuclei. It
decides which gene backgrounds a run accepts, since a store of the other assay is refused, and
whether the report and annotate's summary say cells or nuclei. See the
[user guide](USER_GUIDE.md#the-assay-what-depends-on-it-and-what-does-not) for what it does and
does not change.

`--out` is the per-CLUSTER table. **`--out-h5ad` is the object**, with the label written onto
every cell — that is the file anything downstream opens. Without it the labels exist only in
the terminal output below, and `annotate` says so rather than exiting quietly.

```
   cluster        n  label                              depth    gap  support
   B cells      342  Lymphoid/B cell                        2   1.37
  NK cells      154  Lymphoid/NK cell                       2   0.97
CD8 T cells      316  Lymphoid/T cell                        2   0.72
Dendritic cells   37  Myeloid/Dendritic cell                 2   0.36

UNRESOLVED 0 clusters = 0 cells (0.0%)
```

## 5 · Read the output

**`path` is the answer, and its `depth` is part of it.** `Lymphoid` at depth 1 is not a
failure — it means the evidence supported that call and not a finer one. Only
`UNRESOLVED` means no call at all, and it happens only when the decision fails at the
root.

**`gap`** is how far the winning child beat its best sibling, relative to the score
scale. It is the only statistic that gates anything here, and the only one shown to
predict correctness. Descent stops below `0.30` on the corpus path.

**In the object**, the same answer is per cell: `scanno_cell_type` (the label),
`scanno_path`, `scanno_depth`, `scanno_gap`, `scanno_survival` and — with a corpus —
`scanno_support`. `X` and `obsm` are untouched. `var` is re-keyed by `var['gene_symbol']` when the
object has one, with the accession kept in `var['gene_id']`. Pass `--out-gene-key ""` to leave
`var_names` alone.

`annotate` then reports what a viewer will still need and scAnno cannot supply. An embedding is
the usual one: scAnno does not compute embeddings, so if `obsm` has none the report says
`MISSING` and names the fix. It is a report, not a refusal — the object is written either way.

## 6 · Open it

The annotated `.h5ad` goes to [scRNA-seq Lab](https://github.com/JiaenLin/scrnaseq-lab), which
converts it to the bundle [scRNA-seq Studio](https://github.com/JiaenLin/scrnaseq-studio) reads.
Nothing needs configuring: the lab guesses which column is the annotation, and
`scanno_cell_type` is named so that guess lands on it.

scAnno does **not** write that bundle. One job each, and the format belongs to the lab.

## What to fix first if it looks wrong

| symptom | usual cause |
|---|---|
| everything `UNRESOLVED` | `patterns` match no corpus `cell_name` — check with `scanno panel` |
| a node never wins | its siblings claim the same markers; the corpus cannot separate them |
| refuses on `.X` | the object is scaled; pass `--use-raw` |
| coarser labels than expected | correct behaviour on thin evidence, not a bug |

## Then read

- [PRINCIPLES.md](PRINCIPLES.md) — the four rules, and what each cost to learn
- [USER_GUIDE.md](USER_GUIDE.md) — every command, every refusal, and the gene background
- [../KNOWN_ISSUES.md](../KNOWN_ISSUES.md) — before quoting any number
