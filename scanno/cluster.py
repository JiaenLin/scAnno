"""Step 1 - cluster, and select nothing.

This produces the partition the classifier scores. It is deliberately the dullest module in the
package: normalise, find variable genes, reduce, build a graph, and run Leiden at every
resolution asked for. There is no stopping rule, no chosen resolution and no filtering, because
every one of those is a DESIGN and designs belong to the person running the study.

WHAT IS KEPT, AND WHY EACH ONE MATTERS LATER

  - EVERY RESOLUTION. A sweep that discarded the evidence for its own stopping point would be
    unfalsifiable, and keeping them all is what lets `scanno resolution` judge the LABEL rather
    than the partition afterwards. They cost one Leiden call each over a graph already built.
  - RAW COUNTS, in `layers["counts"]`, asserted after every destructive step. Pseudobulk
    differential expression needs them, and a gene lost here is unrecoverable there.
  - EVERY GENE. No class is excluded from variable-gene selection - not mitochondrial, not
    ribosomal, not haemoglobin. The flagged ones that get SELECTED are reported instead. This is
    not squeamishness: an exclusion written as `^Rp[sl]` once dropped Rps6ka2 - ribosomal protein
    S6 KINASE A2, an mTOR signalling enzyme whose correlation with the actual ribosome module is
    r = +0.046 - out of a high-fat-diet study, and nothing objected because the exclusion looked
    routine.

DEFAULTS ARE NOT DESIGNS

`n_top_genes=2000`, `n_pcs=50`, `n_neighbors=15`, `target_sum=1e4` are the neutral conventions
that GENERATE the picture. A design is a rule that SELECTS AN ANSWER FROM it. Leaving the first
at their conventional values is what makes the second visible as a choice; there is no
parameter-free measurement, and a rule that forbade defaults would forbid computing anything.

PER SAMPLE, INDEPENDENTLY

`--split-by` clusters each group on its own - no shared variable genes, no joint embedding, no
batch key. Whether a cluster that appears in one sample and not another is real is a question
about identity, and a pooled clustering cannot answer it because it has already decided.
"""
from __future__ import annotations

DEFAULTS = {
    "n_top_genes": 2000,
    "n_pcs": 50,
    "n_neighbors": 15,
    "target_sum": 1e4,
    "max_value": 10.0,
    "seed": 0,
}

#: 0.25 to 2.0 in eight steps. Wide enough that the answer is inside it, coarse enough that
#: adjacent points differ: on the calibration cohort 130 of 190 adjacent pairs of a 20-point
#: sweep agreed at ARI > 0.90, so the finer sweep was measuring its own step size.
DEFAULT_RESOLUTIONS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)

COUNTS_LAYER = "counts"


def parse_resolutions(spec) -> list:
    """`0.25,0.5,1` or `start:stop:step`. Returns a sorted list of floats."""
    if not spec:
        return list(DEFAULT_RESOLUTIONS)
    s = str(spec).strip()
    if ":" in s:
        a, b, c = (float(x) for x in s.split(":"))
        if c <= 0:
            raise ValueError("resolution step must be positive")
        out, v = [], a
        while v <= b + 1e-9:
            out.append(round(v, 6))
            v += c
        return out
    return sorted({round(float(x), 6) for x in s.split(",") if str(x).strip()})


def res_tag(res) -> str:
    """`1.0 -> '1p0'`. A dot in an obs column name survives h5ad and trips up formula parsers."""
    return str(res).replace(".", "p")


def looks_like_counts(X) -> bool:
    """Integer-valued and not tiny. Measured, because the slot name lies routinely."""
    import numpy as np
    import scipy.sparse as sp

    head = X[:200]
    head = head.toarray() if sp.issparse(head) else np.asarray(head)
    if head.size == 0:
        return False
    return bool(np.all(head >= 0) and np.allclose(head, np.round(head)) and head.max() > 1)


def cluster(adata, *, resolutions=None, n_top_genes=None, n_pcs=None, n_neighbors=None,
            seed=None, target_sum=None, max_value=None, log=print) -> dict:
    """Cluster one object in place at every resolution. Returns what it did.

    Nothing is removed and nothing is chosen. The object comes back with `layers["counts"]`,
    `obsm["X_pca"]`, `obsm["X_umap"]`, a `highly_variable` column in `var`, and one
    `leiden_<tag>` column per resolution.
    """
    import numpy as np
    import scanpy as sc

    p = dict(DEFAULTS)
    for k, v in (("n_top_genes", n_top_genes), ("n_pcs", n_pcs), ("n_neighbors", n_neighbors),
                 ("seed", seed), ("target_sum", target_sum), ("max_value", max_value)):
        if v is not None:
            p[k] = v
    res = list(resolutions or DEFAULT_RESOLUTIONS)

    # RAW COUNTS FIRST, before anything destructive. Checked rather than assumed: an object whose
    # .X has silently been normalised upstream reads exactly like one that has not, and every
    # number after this point would inherit it.
    if COUNTS_LAYER not in adata.layers:
        if not looks_like_counts(adata.X):
            raise ValueError(
                "adata.X does not look like raw counts (not integer-valued), and there is no "
                f"layers[{COUNTS_LAYER!r}] to fall back on. Clustering would proceed, and stage-5 "
                "pseudobulk would later have no counts to work from. Supply raw counts in .X or "
                f"in layers[{COUNTS_LAYER!r}].")
        adata.layers[COUNTS_LAYER] = adata.X.copy()
    elif not looks_like_counts(adata.X):
        # THE LAYER IS THERE AND .X HAS ALREADY BEEN NORMALISED. Until this branch existed the
        # presence of the layer skipped the check entirely and normalize_total + log1p then ran
        # on whatever .X happened to be - a SECOND time, on already-logged values, with no error
        # and no warning. The clustering that came out was of a matrix nobody meant to compute.
        #
        # It is not an exotic input. `scanno embed` writes exactly this - X = log1p(counts per
        # 10,000), raw integers in layers['counts'] - it is the ordinary scanpy convention, and
        # it is the ONLY cohort object this tool can produce. So the joint route in
        # compare.py's own docstring, `scanno cluster --h5ad cohort.h5ad`, could not be followed
        # correctly by anyone who built its input with this package.
        #
        # Start from the counts, which is what the layer is for, and SAY SO: a run that silently
        # substitutes its own input is a run whose log does not describe it.
        log(f"    .X is not counts and layers[{COUNTS_LAYER!r}] is - clustering from the counts "
            f"layer, not from .X")
        adata.X = adata.layers[COUNTS_LAYER].copy()
    n_counts_genes = int(adata.n_vars)

    sc.pp.normalize_total(adata, target_sum=p["target_sum"])
    sc.pp.log1p(adata)
    _assert_counts(adata, n_counts_genes, "after normalise/log1p")

    # Over ALL genes. `flavor="seurat"` on log-normalised values is the conventional pairing.
    sc.pp.highly_variable_genes(adata, n_top_genes=p["n_top_genes"], flavor="seurat")
    n_hv = int(adata.var["highly_variable"].sum())
    _assert_counts(adata, n_counts_genes, "after variable-gene selection")

    # Scaling and PCA on a TEMPORARY hvg-restricted copy, so the object keeps every gene and its
    # log-normalised values. Scaling in place would leave z-scores in .X, which a viewer refuses
    # and a marker dotplot would draw as if they were expression.
    sub = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(sub, max_value=p["max_value"])
    sc.tl.pca(sub, n_comps=min(p["n_pcs"], min(sub.shape) - 1), svd_solver="arpack",
              random_state=p["seed"])
    adata.obsm["X_pca"] = sub.obsm["X_pca"]
    del sub

    sc.pp.neighbors(adata, n_neighbors=p["n_neighbors"],
                    n_pcs=adata.obsm["X_pca"].shape[1], random_state=p["seed"])
    sc.tl.umap(adata, random_state=p["seed"])
    _assert_counts(adata, n_counts_genes, "after the embedding")

    written = []
    for r in res:
        key = f"leiden_{res_tag(r)}"
        sc.tl.leiden(adata, resolution=float(r), key_added=key, random_state=p["seed"],
                     flavor="igraph", n_iterations=2, directed=False)
        n = int(adata.obs[key].nunique())
        written.append({"resolution": float(r), "key": key, "n_clusters": n})
        log(f"    resolution {r:<5} {n:>3} clusters  -> obs[{key!r}]")

    adata.uns["scanno_cluster"] = {
        "resolutions": [float(x) for x in res],
        "keys": [w["key"] for w in written],
        "n_clusters": [w["n_clusters"] for w in written],
        "n_top_genes": int(p["n_top_genes"]), "n_pcs": int(adata.obsm["X_pca"].shape[1]),
        "n_neighbors": int(p["n_neighbors"]), "seed": int(p["seed"]),
        "target_sum": float(p["target_sum"]), "max_value": float(p["max_value"]),
        "genes_total": n_counts_genes, "genes_highly_variable": n_hv,
        "gene_classes_excluded": "none - every gene was eligible for selection",
        "counts_layer": COUNTS_LAYER,
        "note": ("every resolution computed is kept and none is chosen; `scanno resolution` "
                 "judges them on the LABEL afterwards"),
    }
    return {"resolutions": written, "n_highly_variable": n_hv, "n_genes": n_counts_genes}


def _assert_counts(adata, n_genes, where):
    """The counts layer is still there, still that shape, still integer. Checked three times.

    Not paranoia about scanpy: it is the one property this module can destroy irrecoverably, and
    the failure is silent - a normalised layer named `counts` looks exactly like a raw one.
    """
    import numpy as np
    import scipy.sparse as sp

    if COUNTS_LAYER not in adata.layers:
        raise AssertionError(f"layers[{COUNTS_LAYER!r}] disappeared {where}")
    L = adata.layers[COUNTS_LAYER]
    if int(L.shape[1]) != int(n_genes):
        raise AssertionError(
            f"layers[{COUNTS_LAYER!r}] has {L.shape[1]:,} genes {where}, was {n_genes:,}")
    head = L[:50]
    head = head.toarray() if sp.issparse(head) else np.asarray(head)
    if head.size and not np.allclose(head, np.round(head)):
        raise AssertionError(f"layers[{COUNTS_LAYER!r}] is no longer integer {where}")


def split(adata, key):
    """Yield `(group, view)` for each level of `obs[key]`, in sorted order.

    The upstream declaration is DROPPED from each piece, deliberately. It describes the cohort -
    its `n_obs` and its digest are the cohort's - so carrying it onto a subset would make every
    piece fail verification downstream, which reads as tampering rather than as splitting. The
    flag COLUMN travels with the cells, so an exclusion is still available by naming it
    explicitly, and the caller is told so.
    """
    import numpy as np

    if key not in adata.obs:
        raise KeyError(f"no obs column {key!r} to split on")
    lab = np.asarray(adata.obs[key].astype(str))
    for g in sorted(set(lab)):
        piece = adata[lab == g].copy()
        piece.uns.pop("scqc", None)
        yield g, piece
