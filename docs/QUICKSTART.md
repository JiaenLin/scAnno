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

scAnno **does not ship one**. Download a CellMarker release
(`all_cell_marker.zip`) and build the database with your own ingest, or point `--db` at
an existing SQLite file with an `assertion` table carrying `species`, `tissue_class`,
`cell_name`, `symbol_norm`, `evidence_tier` and `n_pmids`.

Check what the corpus knows about your tissue before anything else:

```bash
scanno panel --db corpus.db --species Human --tissue Blood --top 10
```

If that refuses, the corpus has nothing for your context and no amount of tuning will
help. Exit code 2.

## 3 · Declare a tree

A JSON file with three keys. **The tree must be rooted** — that is what lets scAnno
return `Lymphoid` when it cannot separate T from NK, instead of giving up.

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
- **`patterns`** — substrings matching corpus `cell_name` values. First match wins, so
  order matters within a node's list: put `lymphatic endothelial` before `endothelial`.
- **`members`** — corpus/atlas cell-type names per node. Needed only for the atlas path;
  leave `{}` when scoring from the corpus.

Nodes are scored only **against their own siblings**, so a node needs to be
distinguishable from its brothers, not from everything.

## 4 · Annotate

```bash
scanno annotate --h5ad sample.h5ad --cluster-key leiden_1.0 \
                --tree tree.json --db corpus.db \
                --species Human --tissue Blood \
                --background-from-clusters \
                --out labels.tsv \
                --out-h5ad annotated.h5ad
```

`--out` is the per-CLUSTER table. **`--out-h5ad` is the object**, with the label written onto
every cell — that is the file anything downstream opens. Without it the labels exist only in
the terminal output below, and `annotate` says so rather than exiting quietly.

```
   cluster        n  label                              depth    gap
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
`scanno_support`. `X`, `var` and `obsm` are untouched.

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
