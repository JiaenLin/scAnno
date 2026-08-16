"""ONE embedding computed over every sample together, and the check that it really is one.

WHY THIS EXISTS

A cohort object is often assembled by concatenating per-sample objects, and concatenation carries
each sample's `X_umap` across unchanged. Some assemblers then translate each block so the clouds
do not overlap. What comes out has one `obsm['X_umap']`, one row per cell, and every property a
joint embedding appears to have - and it is not one. Each sample occupies its own territory
because it was PUT there.

That difference is invisible in a figure and fatal to what the figure is for. A panel coloured by
library, drawn on a stitched embedding, shows perfect separation by library; read as evidence, it
says the cohort is dominated by batch. It says nothing of the kind. It is a picture of the
assembly step.

    scanno embed --h5ad *_annotated.h5ad --out joint.h5ad

WHAT THIS IS NOT

It is not integration. The embedding is computed over the pooled cells with no batch correction,
and the report says so wherever it is shown. Whether integration is needed - and which method -
is a decision that wants its own evidence and its own figures; an annotator quietly harmonising
its input would remove exactly the signal that decision is made on.

NO GENE IS EXCLUDED

Highly-variable selection runs over ALL genes. Gene classes convention treats as disposable -
mitochondrial, ribosomal, haemoglobin - are counted and REPORTED, never dropped: a
`^Rp[sl]\\d` pattern was "ribosomal genes" until someone printed it, at which point it contained
a kinase. Reporting influence is useful; removing it silently is not the tool's decision to make.
"""
from __future__ import annotations

import sys

#: Patterns whose selection is worth REPORTING. Nothing here is removed - this is a counter.
NOTABLE = {
    "mitochondrial": ("mt-", "MT-"),
    "ribosomal": ("RPS", "RPL", "Rps", "Rpl"),
    "haemoglobin": ("HB", "Hb"),
}


def notable_counts(genes):
    """How many selected genes match each notable class. For the record, not for a filter."""
    out = {}
    for name, pats in NOTABLE.items():
        out[name] = sorted(g for g in genes if str(g).startswith(pats))
    return out


def build(objects, *, sample_key="sample", n_hvg=2000, n_pcs=50, n_neighbors=15,
          min_dist=0.5, seed=0, gene_key=None, log=print):
    """Concatenate, normalise, select HVGs over all genes, PCA, neighbours, UMAP.

    `objects` is [(name, AnnData), ...] with raw counts in `layers['counts']` or in `.X`.
    Returns the joint AnnData. Every step prints what it did, because an embedding whose
    parameters are not recorded cannot be reproduced or argued with.
    """
    import anndata as ad
    import numpy as np
    import scanpy as sc

    # ---- PREFLIGHT. Check every input BEFORE spending anything ----------------------
    #
    # This command reads ten objects, concatenates 109,140 cells and only then normalises. A
    # defect in input three used to surface half an hour in, inside `normalize_total`, as
    # `'NoneType' object has no attribute 'dtype'` - naming neither the object nor the cause.
    # Everything checkable is checked here, in seconds, and refuses by NAME.
    bad = []
    for name, A in objects:
        X = A.layers["counts"] if "counts" in getattr(A, "layers", {}) else A.X
        if X is None:
            bad.append(f"{name}: no expression matrix (.X is None and no counts layer)")
        elif getattr(X, "shape", (0, 0))[1] == 0:
            bad.append(f"{name}: matrix has zero genes")
        if A.n_obs == 0:
            bad.append(f"{name}: zero cells")
    if bad:
        raise SystemExit("scanno embed: REFUSE - " + str(len(bad)) + " input(s) cannot be used:\n  "
                         + "\n  ".join(bad))
    log(f"  preflight: {len(objects)} object(s), all carry an expression matrix")

    parts = []
    for name, A in objects:
        B = A.copy()
        if "counts" in getattr(B, "layers", {}):
            B.X = B.layers["counts"].copy()
        if sample_key not in B.obs:
            B.obs[sample_key] = str(name)
        # Gene SYMBOLS as the shared axis where they exist: two objects indexed by accession
        # concatenate fine and two indexed differently do not, and the failure is a silent
        # inner join down to whatever they happen to share.
        col = gene_key
        if col is None:
            for cand in ("gene_symbol", "gene_symbols", "symbol", "feature_name"):
                if cand in B.var:
                    col = cand
                    break
        if col and col in B.var:
            B.var_names = [str(v) for v in B.var[col]]
        B.var_names_make_unique()
        # Build a MINIMAL object rather than stripping a full one. `B.layers` can enumerate a
        # `None` key that IS `.X` - the deprecation warning says so in as many words - so
        # `for k in list(B.layers): del B.layers[k]` deletes the matrix it was meant to protect,
        # and the failure surfaces 30 minutes later inside `normalize_total` as
        # `'NoneType' object has no attribute 'dtype'`, naming neither the layer nor the step
        # that removed it.
        import anndata as _ad
        B = _ad.AnnData(X=B.X, obs=B.obs.copy(), var=B.var.copy())
        if B.X is None:
            raise SystemExit(f"scanno embed: {name} has no expression matrix (.X is None) "
                             f"after selecting counts. Nothing downstream can be computed.")
        parts.append(B)
        log(f"  read {name}: {B.n_obs:,} cells x {B.n_vars:,} genes")

    J = ad.concat(parts, join="inner", label=None, index_unique=None)
    lost = [p.n_vars - J.n_vars for p in parts]
    log(f"  concatenated: {J.n_obs:,} cells x {J.n_vars:,} genes"
        + (f"   (inner join dropped up to {max(lost):,} genes from a sample)" if max(lost) else ""))
    del parts

    sc.pp.normalize_total(J, target_sum=1e4)
    sc.pp.log1p(J)
    log("  normalised: counts per 10,000, log1p")

    # OVER ALL GENES. Nothing is excluded; what was selected is reported.
    sc.pp.highly_variable_genes(J, n_top_genes=n_hvg, flavor="seurat")
    hv = [g for g, v in zip(J.var_names, J.var["highly_variable"]) if v]
    log(f"  {len(hv):,} highly-variable genes selected over all {J.n_vars:,} — none excluded")
    for cls, hits in notable_counts(hv).items():
        if hits:
            log(f"    {len(hits)} {cls}: {', '.join(hits[:6])}"
                + (" ..." if len(hits) > 6 else ""))

    try:
        sc.pp.pca(J, n_comps=min(n_pcs, min(J.n_obs, len(hv)) - 1),
                  mask_var="highly_variable", svd_solver="arpack", random_state=seed)
    except TypeError:                       # scanpy < 1.10 spells it differently
        sc.pp.pca(J, n_comps=min(n_pcs, min(J.n_obs, len(hv)) - 1),
                  use_highly_variable=True, svd_solver="arpack", random_state=seed)
    log(f"  PCA: {J.obsm['X_pca'].shape[1]} components")
    sc.pp.neighbors(J, n_neighbors=n_neighbors, n_pcs=J.obsm["X_pca"].shape[1],
                    random_state=seed)
    log(f"  kNN graph: k={n_neighbors}")
    sc.tl.umap(J, min_dist=min_dist, random_state=seed)
    log(f"  UMAP: min_dist={min_dist}, seed={seed}")

    xy = np.asarray(J.obsm["X_umap"])
    if sample_key in J.obs:
        ss = J.obs[sample_key].astype(str).values
        # A joint embedding whose samples still do not overlap is worth SAYING, not hiding: it
        # may be a real batch effect, and it is the reading the label-and-library figure exists
        # for. What it is not is an artefact of stitching, which is the thing just ruled out.
        boxes = {s: (xy[ss == s, 0].min(), xy[ss == s, 0].max()) for s in sorted(set(ss))}
        spans = sorted(boxes.values())
        overlap = sum(1 for i in range(len(spans) - 1) if spans[i][1] > spans[i + 1][0])
        log(f"  computed over all samples together; {overlap} of {len(spans) - 1} adjacent "
            f"sample ranges overlap on the first axis")
    J.uns["scanno_embed"] = {
        "n_hvg": int(len(hv)), "n_pcs": int(J.obsm["X_pca"].shape[1]),
        "n_neighbors": int(n_neighbors), "min_dist": float(min_dist), "seed": int(seed),
        "integrated": False,
        "note": "computed over the pooled cells with NO batch correction; whether integration "
                "is needed is a separate decision with its own evidence",
    }
    return J
