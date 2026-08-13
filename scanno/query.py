"""Turning a clustered query into `Z` — the only pass over the cells.

One sparse matmul against a one-hot cluster indicator, then standardisation against the
store's gene background. Everything after this is milliseconds.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .store import safe_scale

DETECT_FLOOR = 0.01          # per CLUSTER, never per dataset
OOD_MIN_COVERED = 0.30


def cluster_profile(X, labels: np.ndarray, n_clusters: int):
    """Cluster means and per-cluster detection rates. The single pass over cells."""
    n = X.shape[0]
    ind = sp.csr_matrix((np.ones(n, dtype=np.float64), (labels, np.arange(n))),
                        shape=(n_clusters, n))
    counts = np.asarray(ind.sum(axis=1)).ravel()
    M = np.asarray((ind @ X).todense() if sp.issparse(X) else ind @ X, dtype=np.float64)
    B = (X > 0).astype(np.float64)
    D = np.asarray((ind @ B).todense() if sp.issparse(B) else ind @ B, dtype=np.float64)
    denom = np.maximum(counts, 1)[:, None]
    return M / denom, D / denom, counts


def standardise(M, D, query_genes, store, exclude=None):
    """`Z` in the STORE's gene space, standardised against the store's gene background.

    Returns (Z, usable, stats).

    TWO PROPERTIES THIS FUNCTION EXISTS TO GUARANTEE

    1. Nothing here depends on which other clusters are present. Standardising against the
       run's own clusters made a label depend on what else was sequenced: deleting 2% of an
       object shifted every score by a median 19.4% and flipped a call.

    2. `Z` is returned in the STORE's gene space. It was once returned in the query's,
       while every consumer indexed by the store's - which coincide only when the query
       carries exactly the store's genes. True of a self-test, false of every real query.

    The detection floor is per CLUSTER. A dataset-wide floor cannot be cleared by a
    population smaller than the floor, so it deleted rare cell types by construction.

    `exclude` - a per-cluster boolean mask, or cluster indices - is the ONE place property 1
    does not already hold. The usable-gene set is `any` over clusters, so without it a cluster
    the caller had excluded could still be the sole reason a gene was admitted, and the labels
    of the kept clusters would depend on the one that was removed. Excluded rows are still
    standardised and returned; `classify` decides not to walk them. See `scanno/exclude.py`.
    """
    from .exclude import as_mask

    qg = np.array([str(g).upper() for g in query_genes])
    pos = {str(g): i for i, g in enumerate(store.genes)}
    in_store = np.array([g in pos for g in qg])
    drop = as_mask(exclude, D.shape[0])
    D_admit = D[~drop] if drop.any() else D
    if not D_admit.size:
        raise ValueError(
            "every cluster is excluded - there is nothing left to admit genes from, so no "
            "annotation is possible. Check the flag before the clustering.")
    detected = (D_admit >= DETECT_FLOOR).any(axis=0)
    usable_q = in_store & detected
    sidx = np.array([pos[g] for g in qg[usable_q]], dtype=int)

    Z = np.zeros((M.shape[0], len(store.genes)), dtype=np.float64)
    if sidx.size:
        Z[:, sidx] = (M[:, usable_q] - store.gene_mu[sidx]) / safe_scale(store.gene_sd[sidx])
    usable = np.zeros(len(store.genes), dtype=bool)
    usable[sidx] = True

    stats = {
        "genes_query": int(len(qg)),
        "genes_detected": int(detected.sum()),
        "genes_in_store": int(in_store.sum()),
        "genes_usable": int(usable.sum()),
        # Of what the query actually expresses, how much can the store speak to?
        "ood_covered": float((in_store & detected).sum() / max(detected.sum(), 1)),
        # Named even when nothing is excluded, so a reader of the stats can tell "no clusters
        # were excluded" from "this run predates the option".
        "clusters_excluded": int(drop.sum()),
    }
    return Z, usable, stats
