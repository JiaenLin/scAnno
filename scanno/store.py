"""The profile store — what scAnno learns from atlases, and the gene background.

Built once, offline, by whoever calibrates. Queried at annotation time. The classifier
never fits anything, so a result is reproducible from the store digest and the tree.

THREE THINGS COME OUT OF HERE, AND THEY HAVE DIFFERENT REQUIREMENTS

  gene background   per-gene mean/sd across CELL TYPES. ~20k numbers per species,
                    tissue-general, shippable. This is what composition-independence
                    actually needs (docs/PRINCIPLES.md 1) - NOT an atlas per context.
  node profiles     per-(celltype, gene) expression. Needs an atlas for that context.
                    Often absent, and the corpus substitutes (scanno/corpus.py).
  C grades          replication of the PROFILE across independent sources. Tree-free,
                    so it can be precomputed without knowing anyone's taxonomy.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

#: The normalisation the store and every query must share. Recorded in the digest rather
#: than assumed, because a store built under different settings is not comparable.
NORM = {"target_sum": 1e4, "transform": "log1p"}

MIN_CELLS_IN_STORE = 10      # a cell type ENTERS the store here
MIN_CELLS_PRESENT = 50       # ...and counts as PRESENT for grading only here

#: Label provenance that is NOT derived from the markers being calibrated. Anything else -
#: including unrecorded provenance - is treated as marker-derived, because a provenance
#: that defaults to clean is how a circular result gets in.
CLEAN_PROVENANCE = {"sorted", "facs", "genetic", "hashing", "multimodal", "cite-seq", "curated"}


def safe_scale(sd):
    """The ONLY route by which a scale becomes a denominator in this package.

    Three separate defects in this lineage of code were a near-zero scale in a
    denominator, including one written into the document that explains the bug. Guarding
    against exactly zero does not work - the failing values are tiny, not zero. Shrinkage
    toward the pooled scale does, has no free parameter, and cannot be forgotten if it is
    the only available route.

    tests/test_no_raw_sigma.py enforces that nothing else divides by a standard deviation.
    """
    sd = np.asarray(sd, dtype=np.float64)
    finite = sd[np.isfinite(sd) & (sd > 0)]
    pooled = float(np.median(finite)) if finite.size else 1.0
    return sd + pooled


class _Accum:
    """One-pass float64 sufficient statistics.

    float32 sums over ~1e11 nonzeros lose precision silently, which is the worst way to
    lose it. The cells are never revisited, so this pass has to be right.
    """

    def __init__(self, n_genes: int):
        self.n = 0
        self.sum = np.zeros(n_genes, dtype=np.float64)
        self.nnz = np.zeros(n_genes, dtype=np.float64)

    def add(self, block):
        d = block.toarray() if sp.issparse(block) else np.asarray(block)
        self.n += d.shape[0]
        self.sum += d.sum(axis=0, dtype=np.float64)
        self.nnz += (d > 0).sum(axis=0)

    def mean(self):
        return self.sum / max(self.n, 1)

    def detect(self):
        return self.nnz / max(self.n, 1)


@dataclass
class ProfileStore:
    context: dict
    genes: np.ndarray
    celltypes: list
    mean: np.ndarray                                   # celltypes x genes
    detect: np.ndarray
    n_cells: np.ndarray
    n_present: np.ndarray                              # datasets with >= MIN_CELLS_PRESENT
    n_sources: np.ndarray                              # INDEPENDENT sources, not datasets
    between_sd: np.ndarray
    n_clean: np.ndarray = None                         # sources whose labels are not marker-derived
    gene_mu: np.ndarray = field(default=None)
    gene_sd: np.ndarray = field(default=None)
    digest: str = ""

    def __post_init__(self):
        if self.gene_mu is None:
            # Over CELL TYPES, never over a query's clusters. This is the whole of
            # composition-independence: a cluster's score must not depend on what else
            # was sequenced beside it.
            self.gene_mu = self.mean.mean(axis=0)
            self.gene_sd = safe_scale(self.mean.std(axis=0))
        if not self.digest:
            h = hashlib.sha256()
            h.update(json.dumps({"ctx": self.context, "norm": NORM}, sort_keys=True).encode())
            h.update(np.ascontiguousarray(self.mean, dtype=np.float32).tobytes())
            h.update("|".join(map(str, self.celltypes)).encode())
            h.update("|".join(map(str, self.genes)).encode())
            self.digest = h.hexdigest()[:16]

    def grade(self, i: int) -> str:
        """C ladder — replication of the profile. Tree-free.

        C0 is reserved for a cell type with NO profile at all and never reaches here; a
        type in the store but never well sampled is a real, weak profile.
        """
        src = int(self.n_sources[i])
        clean = int(self.n_clean[i]) if self.n_clean is not None else 0
        if int(self.n_present[i]) == 0:
            return "C3"
        # C1 additionally requires label-clean sources. Atlas labels are usually
        # marker-derived, and learning from labels assigned by the very markers being
        # graded measures agreement with prior practice rather than truth.
        if src >= 5 and clean >= 3:
            return "C1"
        if src >= 3:
            return "C2"
        return "C3"


def build_store(datasets, context: dict) -> ProfileStore:
    """One streaming pass per dataset. Cells are discarded; only statistics survive.

    `datasets` yields (source_id, gene_symbols, X, celltype_labels[, provenance]).

    `source_id` groups releases that are NOT independent - same consortium, same donors -
    because public atlases reuse samples and counting them separately inflates every C
    grade. `provenance` says how the labels were obtained; anything other than sorted,
    multimodal or curated is treated as marker-derived, and UNKNOWN is never assumed
    clean.
    """
    genes = None
    per_ct = {}
    for entry in datasets:
        src, gsym, X, lab = entry[:4]
        prov = (entry[4] if len(entry) > 4 else "unknown")
        g = np.array([str(s).upper() for s in gsym])
        if genes is None:
            genes = g
        elif not np.array_equal(genes, g):
            raise SystemExit(
                f"scanno: dataset '{src}' has a different gene space. Harmonise gene "
                f"spaces before building - silently intersecting would make the store's "
                f"coverage depend on load order.")
        lab = np.asarray(lab).astype(str)
        for ct in np.unique(lab):
            m = lab == ct
            n = int(m.sum())
            if n < MIN_CELLS_IN_STORE:
                continue
            # Presence gates GRADING, never entry. It gated entry once, at 50 cells, and
            # excluded the two rarest populations from the store entirely - both were then
            # confidently mislabelled. That is the rare-population bug living inside the
            # fix for the rare-population bug.
            acc = _Accum(len(g))
            acc.add(X[m])
            per_ct.setdefault(ct, []).append(
                (src, acc.mean(), acc.detect(), n, n >= MIN_CELLS_PRESENT,
                 str(prov).lower() in CLEAN_PROVENANCE))

    celltypes = sorted(per_ct)
    ng = len(genes)
    mean = np.zeros((len(celltypes), ng))
    detect = np.zeros((len(celltypes), ng))
    bsd = np.zeros((len(celltypes), ng))
    n_cells = np.zeros(len(celltypes))
    n_present = np.zeros(len(celltypes))
    n_sources = np.zeros(len(celltypes))
    n_clean = np.zeros(len(celltypes))
    for i, ct in enumerate(celltypes):
        rows = per_ct[ct]
        Ms = np.vstack([r[1] for r in rows])
        Ds = np.vstack([r[2] for r in rows])
        # Unweighted across datasets: one dataset is one vote, so a 5M-cell atlas cannot
        # outvote two hundred small studies.
        mean[i] = Ms.mean(axis=0)
        detect[i] = Ds.mean(axis=0)
        bsd[i] = Ms.std(axis=0) if len(rows) > 1 else np.nan     # unknown, not zero
        n_cells[i] = sum(r[3] for r in rows)
        n_present[i] = sum(1 for r in rows if r[4])
        n_sources[i] = len({r[0] for r in rows if r[4]})
        n_clean[i] = len({r[0] for r in rows if r[4] and r[5]})
    return ProfileStore(context, genes, celltypes, mean, detect,
                        n_cells, n_present, n_sources, bsd, n_clean=n_clean)
