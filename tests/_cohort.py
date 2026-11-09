"""The repository's synthetic cohort: three samples, five cell types, one corpus and one tree.

WHAT IT IS FOR

The documentation's examples and the assay rules are tested on it, through the real CLI. It is
built here so that no suite carries a real cohort. Every name in it is invented or is a textbook
marker: no sample name, factor level, count or path of any study this tool has been run on.

WHAT IT IS NOT

It is not evidence that a label is correct. The five populations are separated by construction,
so the annotator finding them shows that the commands run end to end. It does not show that they
annotate well.

THE CONTEXT IS HUMAN BLOOD because that is the only context the tool was validated on (README,
Status), and the tree is the one docs/QUICKSTART.md declares, so the example a reader copies and
the one this suite runs are the same. Gene symbols are upper case, which is the human
convention, and `var_names` are Ensembl-shaped accessions with the symbols in
`var["gene_symbol"]`, which is the ordinary delivered layout. The accessions are not real.

    from tests._cohort import write_cohort, write_corpus, write_tree
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SPECIES, TISSUE = "Human", "Blood"

#: The tree docs/QUICKSTART.md section 3 declares. tests/test_docs_run.py reads that block from
#: the document itself and checks that it equals this one, so neither can drift from the other.
TREE = {
    "children": {"root": ["Lymphoid", "Myeloid"],
                 "Lymphoid": ["T cell", "B cell", "NK cell"],
                 "Myeloid": ["Monocyte", "Dendritic cell"]},
    "patterns": {"Lymphoid": ["t cell", "b cell", "nk cell", "natural killer"],
                 "Myeloid": ["monocyte", "macrophage", "dendritic"],
                 "T cell": ["t cell", "thymocyte"],
                 "B cell": ["b cell", "plasma cell"],
                 "NK cell": ["natural killer", "nk cell"],
                 "Monocyte": ["monocyte", "macrophage"],
                 "Dendritic cell": ["dendritic"]},
    "members": {},
}

#: Leaf -> (the corpus cell name, its markers). Textbook markers, twelve each, so the corpus and
#: the object share 60 symbols, above the 50 that `check_gene_space` requires.
PANELS = {
    "T cell": ("T cell", ["CD3D", "CD3E", "CD3G", "CD2", "CD5", "CD7", "IL7R", "LCK", "TRAC",
                          "CD28", "CD6", "LAT"]),
    "B cell": ("B cell", ["CD79A", "CD79B", "MS4A1", "CD19", "CD22", "PAX5", "BANK1", "CD72",
                          "FCRL1", "VPREB3", "BLK", "TCL1A"]),
    "NK cell": ("Natural killer cell", ["NKG7", "GNLY", "KLRD1", "KLRF1", "NCAM1", "PRF1", "GZMB",
                                        "KLRC1", "FCGR3A", "SH2D1B", "NCR1", "KIR2DL4"]),
    "Monocyte": ("Monocyte", ["CD14", "LYZ", "S100A8", "S100A9", "FCN1", "VCAN", "CSF1R", "CD68",
                              "ITGAM", "MNDA", "CTSS", "FCGR1A"]),
    "Dendritic cell": ("Dendritic cell", ["FCER1A", "CD1C", "CLEC10A", "CLEC9A", "XCR1", "IRF8",
                                          "ITGAX", "LAMP3", "CCR7", "HLA-DQA1", "CD1E", "FLT3"]),
}
N_FILLER = 140
SAMPLES = ("S1", "S2", "S3")
#: Two arms, invented. A study's own factor levels never appear in this repository.
ARMS = {"S1": "arm1", "S2": "arm2", "S3": "arm1"}


def symbols() -> list:
    out = [g for _name, gs in PANELS.values() for g in gs]
    return out + [f"FILLER{i:03d}" for i in range(N_FILLER)]


def _counts(n_per_type, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    genes = symbols()
    leaves = list(PANELS)
    y = np.repeat(np.arange(len(leaves)), n_per_type)
    X = rng.poisson(0.6, size=(len(y), len(genes))).astype("float32")
    for li, leaf in enumerate(leaves):
        idx = [genes.index(g) for g in PANELS[leaf][1]]
        rows = np.flatnonzero(y == li)
        X[np.ix_(rows, idx)] += rng.poisson(9.0, size=(len(rows), len(idx)))
    return X, y


def make_object(sample="S1", n_per_type=30, seed=None, log_normalised=False, embedding=False):
    """One sample: raw counts in X (or log1p CP10k), `sample`, `group`, `cluster` = the truth.

    `cluster` is the generating population, so an object built here can be annotated without
    running `scanno cluster` first. A suite that wants the real clustering runs the command.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    if seed is None:
        seed = SAMPLES.index(sample)
    X, y = _counts(n_per_type, seed)
    if log_normalised:
        X = np.log1p(X / np.maximum(X.sum(axis=1, keepdims=True), 1) * 1e4).astype("float32")
    A = ad.AnnData(X=sp.csr_matrix(X))
    A.obs_names = [f"{sample}_{i:04d}" for i in range(A.n_obs)]
    A.var_names = [f"ENSG{90000000000 + i:011d}" for i in range(A.n_vars)]
    A.var["gene_symbol"] = symbols()
    A.obs["sample"] = pd.Categorical([sample] * A.n_obs)
    A.obs["group"] = pd.Categorical([ARMS[sample]] * A.n_obs)
    A.obs["cluster"] = pd.Categorical([str(v) for v in y])
    if embedding:
        rng = np.random.default_rng(7)
        A.obsm["X_umap"] = (np.column_stack([y * 4.0, (y % 2) * 3.0])
                            + rng.normal(0, 0.4, size=(A.n_obs, 2))).astype("float32")
    return A


def write_cohort(path, samples=SAMPLES, n_per_type=30):
    """Every sample in ONE object - the QC'd cohort a reader brings to `scanno cluster`."""
    import anndata as ad
    objs = [make_object(s, n_per_type=n_per_type) for s in samples]
    A = ad.concat(objs, merge="same")
    for col in ("sample", "group"):
        A.obs[col] = A.obs[col].astype("category")
    A.var["gene_symbol"] = objs[0].var["gene_symbol"].values
    A.obs = A.obs.drop(columns=["cluster"])          # a delivered object has no clustering yet
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    A.write_h5ad(path)
    return Path(path)


def write_sample(path, sample="S1", n_per_type=30):
    """One sample's QC'd object, with no clustering on it."""
    A = make_object(sample, n_per_type=n_per_type)
    A.obs = A.obs.drop(columns=["cluster"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    A.write_h5ad(path)
    return Path(path)


def write_corpus(path, contexts=((SPECIES, TISSUE),)):
    """A CellMarker-schema SQLite: the `assertion` table and the six columns scAnno reads."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE assertion (species TEXT, tissue_class TEXT, cell_name TEXT, "
                "symbol_norm TEXT, evidence_tier INT, n_pmids INT)")
    con.executemany("INSERT INTO assertion VALUES (?,?,?,?,?,?)",
                    [(sp_, ti, name, g, 1, 12)
                     for sp_, ti in contexts for name, gs in PANELS.values() for g in gs])
    con.commit()
    con.close()
    return p


def write_tree(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(TREE, indent=2), encoding="utf-8")
    return p
