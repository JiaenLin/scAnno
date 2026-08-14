"""Withholding what upstream QC flagged, without deleting anything and without deciding anything.

WHAT THIS IS FOR

Upstream QC marks nuclei it considers technical. Annotating them produces a label, and a label is
indistinguishable downstream from a label anyone should believe. The usual remedy is to delete the
cells before annotating. This module does the same job without deleting: the flagged nuclei are
dropped from the cluster PROFILE, so they influence no other nucleus's label, and each is labelled
`EXCLUDED` - a sentinel that is not a cell type in any taxonomy and is spelled so that nothing
mistakes it for one.

THE RULE THIS MODULE EXISTS TO KEEP

    scAnno never decides which nuclei are technical. It excludes EXACTLY the nuclei it was
    handed, and it cannot exclude any other.

That is not a convention, a default or a recommendation - there is no code path here that turns a
per-cell flag into a different set of cells. The excluded set is the flag. It does not depend on
the clustering, the resolution, a share threshold, a QC metric, or anything scAnno computes.

**A cluster-share mode existed until 0.3.0 and was removed rather than defaulted-off.** It
excluded a whole cluster once some share of it was flagged, which meant excluding nuclei that
upstream QC had PASSED because their neighbours had not. Measured on the cohort it was written
for, at two clusterings of the same data:

    resolution 1.0   2,680 excluded, 783 of them unflagged (29.2%), 1,918 of 3,815 flagged KEPT
    resolution 2.0   2,244 excluded, 525 of them unflagged (23.4%), 2,154 of 3,873 flagged KEPT

Neither a subset nor a superset of the flag, and the size moved by two orders of magnitude with
the caller's resolution (42 nuclei at 0.25, 4,080 at 2.0) while the flag itself never changed. A
flag computed once, upstream, must not change meaning because something downstream chose a
different granularity. A default is not a gate; the mode is gone.

WHY THE NON-DESTRUCTIVE FORM IS NOT AN APPROXIMATION OF THE DESTRUCTIVE ONE

It is exactly equal, and that is a property of how this package computes things rather than a
hope:

  * `cluster_profile` builds each cluster's mean from a one-hot indicator, so a cluster's profile
    depends on its own cells and nothing else.
  * `standardise` centres and scales against the STORE's gene background, never the run's own
    clusters - a deliberate design, because standardising against the run made a label depend on
    what else was sequenced.

One thing did leak, and it is why `standardise` takes `exclude` as well: the usable-gene set is
`(D >= DETECT_FLOOR).any(axis=0)`, a reduction ACROSS clusters. Without passing the mask there, a
cluster you had excluded could still be the reason a gene was admitted, and the annotation of the
clusters you kept would silently depend on the one you removed. Pass the mask to both and the
result is identical to having deleted the cells - which `tests/test_exclude.py` asserts against a
physically subsetted matrix rather than trusting this paragraph.

WHAT IS DELIBERATELY NOT HERE

No threshold on doublet rate, mitochondrial content, count depth, cluster composition or anything
else. **This module does not decide which nuclei are bad; it takes that decision as input and
makes it auditable.** Choosing the flag is the caller's business and belongs in the caller's
record of what it chose and why. `exclusion_record_cells` therefore demands a `reason` and
fingerprints the mask it was given, so a reader can prove which set actually ran.

    from scanno import EXCLUDED, cluster_profile, exclusion_record_cells, unprofilable

    flag = adata.obs["cluster_FLAG"].fillna(False).to_numpy(dtype=bool)
    M, D, counts = cluster_profile(X[~flag], labels[~flag], n_clusters)
    drop = unprofilable(labels, ~flag, n_clusters)
    Z, usable, stats = standardise(M, D, genes, store, exclude=drop)
    calls = classify(Z, usable, tree, assertions=asr, exclude=drop)
    path[flag] = EXCLUDED                     # the label is per NUCLEUS, not per cluster
    print(exclusion_record_cells(flag, labels, n_clusters, reason="scQC cluster_FLAG"))
"""
from __future__ import annotations

import hashlib

import numpy as np

#: The label an excluded nucleus receives. Upper case and not a cell type in any taxonomy, so a
#: consumer that treats it as one is making an obvious error rather than a quiet one.
EXCLUDED = "EXCLUDED"


class ExclusionMismatch(ValueError):
    """Raised when an exclusion mask cannot be matched to the cells it is meant to describe.

    A silently mis-sized mask excludes the wrong cells, which is worse than excluding none:
    the run completes, the report renders, and the labels are wrong in a way nothing displays.
    """


def flag_digest(flagged) -> str:
    """A short fingerprint of the exact mask that was applied.

    The point is provenance. An exclusion is auditable only if a reader can check that the set
    which ran is the set upstream QC handed over, and a count cannot do that - two different
    masks of the same size agree on every number in a summary table. This hashes the packed bits
    together with the length, so a mask of a different size can never collide with one of this
    size, and prints in a record beside the count.
    """
    m = np.asarray(flagged, dtype=bool)
    h = hashlib.sha256()
    h.update(str(m.size).encode())
    h.update(np.packbits(m).tobytes())
    return h.hexdigest()[:16]


def unprofilable(labels, keep, n_clusters: int | None = None) -> np.ndarray:
    """Clusters left with no KEPT cell, as a per-cluster mask.

    When the flagged nuclei are dropped from the profile, a cluster all of whose cells were
    flagged has no profile at all - not a weak one, none - and a walk over it would score a
    vector of zeros against the tree and return whatever node is closest to nothing. Those
    clusters are excluded from the walk; every cell in them was flagged anyway, so **no unflagged
    nucleus is affected.**

    This is the ONLY cluster-level exclusion this package performs, and it is forced by
    arithmetic rather than chosen by a threshold. There is no share and no parameter: a cluster
    qualifies when its kept count is exactly zero.
    """
    labels = np.asarray(labels)
    keep = np.asarray(keep, dtype=bool)
    if labels.shape != keep.shape:
        raise ExclusionMismatch(
            f"labels has {labels.shape[0]} cells and the keep mask has {keep.shape[0]}. "
            f"They must describe the same cells, in the same order.")
    k = int(n_clusters if n_clusters is not None else (labels.max() + 1 if labels.size else 0))
    out = np.zeros(k, dtype=bool)
    for c in range(k):
        m = labels == c
        out[c] = bool(m.any()) and not bool(keep[m].any())
    return out


def exclusion_record_cells(flagged, labels, n_clusters: int | None = None, *,
                           reason: str) -> dict:
    """What the exclusion removed, in the form a run log can keep. Never a bare count.

    A record that says only how many nuclei went is not auditable: the question asked afterwards
    is always *which*, and by then the run is over. This returns the mask's fingerprint, the
    clusters it emptied, and the share of the object it is - and it demands a `reason`, because
    an exclusion whose justification lives only in the caller's head is one nobody can review.

    `passengers` is reported, and is always 0. It is not a redundant field: it is the one number
    that distinguishes this from a cluster-share exclusion, and a record that simply omitted it
    would be indistinguishable from one written before the distinction existed.
    """
    if not str(reason).strip():
        raise ValueError(
            "exclusion_record_cells needs a `reason`. An exclusion whose justification lives "
            "only in the caller's head is one nobody can review - name the flag and the "
            "decision behind it.")
    flagged = np.asarray(flagged, dtype=bool)
    labels = np.asarray(labels)
    if flagged.shape != labels.shape:
        raise ExclusionMismatch(
            f"the flag covers {flagged.shape[0]} cells and the labels {labels.shape[0]}. "
            f"They must describe the same cells, in the same order.")
    total = int(flagged.size)
    n_out = int(flagged.sum())
    if total and n_out == total:
        raise ValueError(
            "the flag marks every nucleus in the object, so there is nothing left to annotate. "
            "scAnno refuses rather than returning an object in which every label is EXCLUDED.")
    emptied = unprofilable(labels, ~flagged, n_clusters)
    return {
        "reason": str(reason),
        "mode": "cell",
        "flag_digest": flag_digest(flagged),
        "clusters_excluded": [int(i) for i in np.flatnonzero(emptied)],
        "clusters_kept": int((~emptied).sum()),
        "cells_excluded": n_out,
        "cells_total": total,
        "fraction_excluded": float(n_out / total) if total else 0.0,
        # Always zero, by construction. See the docstring: the field earns its place by being
        # the one number a cluster-share exclusion could not have reported as zero.
        "passengers": 0,
        "per_cluster": {int(c): int((flagged & (labels == c)).sum())
                        for c in np.unique(labels) if int((flagged & (labels == c)).sum())},
        "label": EXCLUDED,
    }


def as_mask(exclude, n_clusters: int) -> np.ndarray:
    """Normalise `exclude` - a bool mask, or an iterable of cluster indices - to a bool mask.

    Refuses a mask of the wrong length and an index outside the clustering, rather than
    truncating or ignoring. Both failures exclude the wrong clusters while completing normally.
    """
    if exclude is None:
        return np.zeros(n_clusters, dtype=bool)
    arr = np.asarray(list(exclude) if not isinstance(exclude, np.ndarray) else exclude)
    if arr.dtype == bool:
        if arr.shape != (n_clusters,):
            raise ExclusionMismatch(
                f"exclusion mask has {arr.shape[0]} entries for {n_clusters} clusters. "
                f"A mask must be one entry per cluster.")
        return arr
    out = np.zeros(n_clusters, dtype=bool)
    if arr.size:
        bad = [int(i) for i in arr if not (0 <= int(i) < n_clusters)]
        if bad:
            raise ExclusionMismatch(
                f"cluster index/indices {bad} are outside a clustering with {n_clusters} "
                f"clusters (0-{n_clusters - 1}).")
        out[arr.astype(int)] = True
    return out
