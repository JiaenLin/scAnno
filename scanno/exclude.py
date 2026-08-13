"""Excluding flagged clusters from annotation, without deleting anything.

WHAT THIS IS FOR

Upstream QC often flags whole clusters rather than cells: a cluster whose doublet rate is 20%,
whose median mitochondrial content sits far above the rest, or which a reviewer has marked as
debris. Annotating such a cluster produces a label, and a label is indistinguishable downstream
from a label anyone should believe. The usual remedy is to delete the cells before annotating.

This module does the same job without deleting. The excluded clusters are not walked; their
cells keep their place in the object and receive the sentinel label `EXCLUDED`, which is not a
cell type and is spelled so that nothing mistakes it for one.

WHY THE NON-DESTRUCTIVE FORM IS NOT AN APPROXIMATION OF THE DESTRUCTIVE ONE

It is exactly equal, for a fixed clustering, and that is a property of how this package computes
things rather than a hope:

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

No threshold on doublet rate, mitochondrial content or anything else. This module does not decide
which clusters are bad; it takes that decision as input and makes it visible. Choosing the
flag is the caller's business and belongs in the caller's record of what it chose and why.

    from scanno import cluster_flags, exclusion_record, EXCLUDED

    drop = cluster_flags(labels, adata.obs["cluster_FLAG"].to_numpy(), n_clusters)
    Z, usable, stats = standardise(M, D, genes, store, exclude=drop)
    calls = classify(Z, usable, tree, assertions=asr, exclude=drop)
    print(exclusion_record(drop, counts, reason="scQC cluster_FLAG"))
"""
from __future__ import annotations

import numpy as np

#: The label an excluded cluster receives. Upper case and not a cell type in any taxonomy, so a
#: consumer that treats it as one is making an obvious error rather than a quiet one.
EXCLUDED = "EXCLUDED"

#: Default share of a cluster's cells that must carry a per-cell flag before the CLUSTER counts
#: as flagged. A majority, which is the only value that needs no justification; anything else is
#: a threshold and the caller should pass it deliberately.
FLAG_SHARE = 0.5


class ExclusionMismatch(ValueError):
    """Raised when an exclusion mask cannot be matched to the clustering it is meant to describe.

    A silently mis-sized mask excludes the wrong clusters, which is worse than excluding none:
    the run completes, the report renders, and the labels are wrong in a way nothing displays.
    """


def cluster_flags(labels, flagged, n_clusters: int | None = None, *,
                  share: float = FLAG_SHARE) -> np.ndarray:
    """Per-CLUSTER exclusion mask from a per-CELL flag.

    `labels` is the integer cluster assignment per cell; `flagged` a boolean per cell. A cluster
    is flagged when at least `share` of its cells are.

    The share exists because a per-cell flag rarely covers a cluster exactly, and the two
    degenerate readings are both wrong: `any` excludes an entire population because one cell in
    it was marked, and `all` excludes nothing. It is reported by `exclusion_record` rather than
    hidden, because a cluster at 0.49 and a cluster at 0.51 are the same cluster to everyone
    except this function.
    """
    labels = np.asarray(labels)
    flagged = np.asarray(flagged, dtype=bool)
    if labels.shape != flagged.shape:
        raise ExclusionMismatch(
            f"labels has {labels.shape[0]} cells and the flag has {flagged.shape[0]}. "
            f"They must describe the same cells, in the same order.")
    k = int(n_clusters if n_clusters is not None else (labels.max() + 1 if labels.size else 0))
    out = np.zeros(k, dtype=bool)
    for c in range(k):
        m = labels == c
        if m.any():
            out[c] = float(flagged[m].mean()) >= share
    return out


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


def exclusion_record(mask, counts, *, reason: str, share: float = FLAG_SHARE) -> dict:
    """What was excluded, in the form a run log can keep. Never a bare count.

    A record of an exclusion that says only how many cells went is not auditable: the question
    asked afterwards is always *which*, and by then the run is over. This returns the cluster
    indices, the cells each held, and the share of the object they are - and it demands a
    `reason`, because an exclusion whose justification lives only in the caller's head is one
    nobody can review.
    """
    if not str(reason).strip():
        raise ValueError(
            "exclusion_record needs a `reason`. An exclusion with no recorded justification is "
            "a bug, not a choice - name the flag and the decision behind it.")
    mask = np.asarray(mask, dtype=bool)
    counts = np.asarray(counts)
    if mask.shape != counts.shape:
        raise ExclusionMismatch(
            f"mask covers {mask.shape[0]} clusters, counts {counts.shape[0]}.")
    idx = [int(i) for i in np.flatnonzero(mask)]
    n_out = int(counts[mask].sum())
    total = int(counts.sum())
    return {
        "reason": str(reason),
        "share_threshold": float(share),
        "clusters_excluded": idx,
        "clusters_kept": int((~mask).sum()),
        "cells_excluded": n_out,
        "cells_total": total,
        "fraction_excluded": float(n_out / total) if total else 0.0,
        "per_cluster": {int(i): int(counts[i]) for i in idx},
        "label": EXCLUDED,
    }
