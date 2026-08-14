"""Withholding flagged nuclei must equal having deleted them. Asserted, not assumed.

The claim `scanno/exclude.py` makes is a strong one: for a fixed clustering, not walking an
excluded cluster gives every OTHER cluster exactly the label it would have received had those
cells never been in the object. Section 1 checks it against a physically subsetted matrix rather
than against the argument for it.

The first version of the code was wrong in one specific way - the usable-gene set is `any` over
clusters, so an excluded cluster could still be the sole reason a gene was admitted - and section
2 is the test that found it. It is the reason `standardise` takes the mask at all.

**Section 5 tests an absence**, which is unusual and is the point: the cluster-share exclusion was
removed in 0.3.0 because it excluded nuclei upstream QC had passed. A removal that is only a
default comes back; a removal with a test against its return does not. If someone re-adds
`cluster_flags`, this file fails.

    python tests/test_exclude.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp

import scanno
from scanno import (EXCLUDED, ExclusionMismatch, classify, cluster_profile,
                    exclusion_record_cells, flag_digest, standardise)
from scanno.store import ProfileStore

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def raises(exc, fn, *a, needle="", **k):
    try:
        fn(*a, **k)
    except exc as e:
        return needle in str(e)
    except Exception:
        return False
    return False


GENES = [f"G{i}" for i in range(16)]
TREE = {
    "children": {"root": ["A", "B"], "A": ["A1", "A2"]},
    "patterns": {"A": ["a cell"], "B": ["b cell"], "A1": ["a1 cell"], "A2": ["a2 cell"]},
    "genes": GENES,
}
#: THREE markers per node, not two: `node_weights(min_markers=3)` drops a node with fewer, and
#: a parent left with under two scoreable children returns no weights at all. The first version
#: of this file gave two each, so every cluster truncated to UNRESOLVED - and section 1 compared
#: three UNRESOLVEDs against three UNRESOLVEDs and passed. A test whose fixture cannot produce a
#: label cannot detect a change in labelling.
ASR = {
    "a cell": {"G0": 30.0, "G1": 30.0, "G2": 30.0},
    "b cell": {"G3": 30.0, "G4": 30.0, "G5": 30.0},
    "a1 cell": {"G6": 30.0, "G7": 30.0, "G8": 30.0},
    "a2 cell": {"G9": 30.0, "G10": 30.0, "G11": 30.0},
}
EXCLUSIVE = 12          # expressed ONLY by the cluster that gets flagged


def store():
    """Only the gene background is used - `classify` runs the corpus route with store=None.
    gene_mu/gene_sd are set so standardisation is the identity, and any difference between two
    runs therefore comes from the exclusion rather than from the background."""
    g = len(GENES)
    return ProfileStore(
        context={"species": "Test", "tissue": "Test"}, genes=np.array(GENES),
        celltypes=["a cell", "b cell"], mean=np.zeros((2, g)), detect=np.zeros((2, g)),
        n_cells=np.array([10, 10]), n_present=np.array([1, 1]), n_sources=np.array([1, 1]),
        between_sd=np.zeros((2, g)), gene_mu=np.zeros(g), gene_sd=np.ones(g), digest="test")


#: cluster -> the genes it expresses. 0 is A/A1, 1 is A/A2, 2 is B, 3 is the one to be flagged.
ON = {0: [0, 1, 2, 6, 7, 8], 1: [0, 1, 2, 9, 10, 11], 2: [3, 4, 5],
      3: [13, 14, 15, EXCLUSIVE]}
EXPECT = {0: "A/A1", 1: "A/A2", 2: "B"}


def toy(seed=0, n_per=40):
    """Four clusters with real, checkable labels. Cluster 3 is the one that will be flagged,
    and it is the ONLY cluster expressing G12 - which is what makes the usable-gene leak
    observable at all.

    The background covers every gene EXCEPT G12, exactly zero there. An earlier version gave
    every gene a small positive background, so all of them cleared the 1% detection floor in
    every cluster and section 2 could not fail even with the bug present. A fixture that cannot
    express the failure is not a test of it.
    """
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for c in range(4):
        X = rng.random((n_per, len(GENES))) * 0.05
        X[:, EXCLUSIVE] = 0.0
        X[:, ON[c]] += 3.0
        blocks.append(X)
        labels += [c] * n_per
    return sp.csr_matrix(np.vstack(blocks)), np.array(labels)


def run(X, labels, n_clusters, exclude=None):
    M, D, counts = cluster_profile(X, labels, n_clusters)
    Z, usable, stats = standardise(M, D, GENES, store(), exclude=exclude)
    return classify(Z, usable, TREE, store=None, assertions=ASR, exclude=exclude), stats, counts


print("\n0 - the fixture produces real labels, so a change in labelling is detectable")
X, labels = toy()
base, _, _ = run(X, labels, 4)
check("the three unflagged clusters are labelled as designed",
      [c["path"] for c in base[:3]] == [EXPECT[i] for i in range(3)],
      f"{[c['path'] for c in base[:3]]}")

print("\n1 - withholding a cluster equals deleting its cells, for a fixed clustering")
kept, _, _ = run(X, labels, 4, exclude=np.array([False, False, False, True]))
keep_cells = labels != 3
deleted, _, _ = run(X[keep_cells], labels[keep_cells], 3)
check("the surviving clusters get identical paths",
      [c["path"] for c in kept[:3]] == [c["path"] for c in deleted],
      f"{[c['path'] for c in kept[:3]]}")
check("...and identical gaps, not merely similar ones",
      all(abs(a["gap"] - b["gap"]) < 1e-12 for a, b in zip(kept, deleted)))

print("\n2 - the usable-gene leak is real, which is why standardise takes the mask")
M, D, _ = cluster_profile(X, labels, 4)
_, _, without = standardise(M, D, GENES, store())
_, with_mask, _ = run(X, labels, 4, exclude=np.array([False, False, False, True]))
check("not passing the mask admits a gene only the excluded cluster expresses",
      without["genes_usable"] > with_mask["genes_usable"],
      f"{without['genes_usable']} usable without the mask, {with_mask['genes_usable']} with")

print("\n3 - excluded clusters stay in place and carry no statistics")
calls, _, _ = run(X, labels, 4, exclude=[1])
check("every cluster is still in the output, in order",
      [c["cluster"] for c in calls] == [0, 1, 2, 3])
check("the excluded one is labelled EXCLUDED and flagged as such",
      calls[1]["label"] == EXCLUDED and calls[1]["path"] == EXCLUDED
      and calls[1]["excluded"] is True)
check("its gap and survival are NaN, not zero",
      np.isnan(calls[1]["gap"]) and np.isnan(calls[1]["survival"]),
      "a zero would sort and average alongside real decisions")
check("no other cluster is marked excluded",
      all(not c["excluded"] for c in calls if c["cluster"] != 1))

print("\n4 - excluding nothing changes nothing")
a, _, _ = run(X, labels, 4)
b, _, _ = run(X, labels, 4, exclude=np.zeros(4, dtype=bool))
check("an empty mask is the same run", [c["path"] for c in a] == [c["path"] for c in b])
check("the stats say so explicitly rather than omitting the field",
      run(X, labels, 4)[1]["clusters_excluded"] == 0)

print("\n5 - the cluster-share exclusion is GONE, and cannot return by default")
# This section tests an absence. `--exclude-mode cluster` excluded nuclei upstream QC had
# passed - 783 of 2,680 on the cohort it was written for - and was removed in 0.3.0 rather
# than left switched off. A capability that is merely defaulted-off is one command-line
# argument away from returning; one whose symbols are gone fails at import.
for name in ("cluster_flags", "exclusion_record", "FLAG_SHARE", "CELL", "CLUSTER", "MODES"):
    check(f"scanno exports no {name!r}", not hasattr(scanno, name),
          "a cluster-share exclusion is a QC decision and is not scAnno's to make")
check("nothing in the exclusion module takes a `share`",
      not any("share" in n for n in dir(scanno.exclude)),
      str([n for n in dir(scanno.exclude) if "share" in n]))
check("the record names its mode, so a reader can tell which code wrote it",
      exclusion_record_cells(np.array([True, False]), np.array([0, 1]), 2,
                             reason="t")["mode"] == "cell")

print("\n6 - every way of naming the wrong clusters is refused, not absorbed")
check("a mask of the wrong length",
      raises(ExclusionMismatch, run, X, labels, 4, exclude=np.zeros(3, dtype=bool),
             needle="one entry per cluster"))
check("an index outside the clustering",
      raises(ExclusionMismatch, run, X, labels, 4, exclude=[7],
             needle="outside a clustering"))
check("excluding everything refuses rather than returning an empty annotation",
      raises(ValueError, run, X, labels, 4, exclude=np.ones(4, dtype=bool),
             needle="every cluster is excluded"))
check("a flag covering a different number of cells than the labels",
      raises(ExclusionMismatch, exclusion_record_cells, np.zeros(9, dtype=bool),
             np.zeros(10, dtype=int), 1, reason="t", needle="same cells"))
check("a flag that marks every nucleus",
      raises(ValueError, exclusion_record_cells, np.ones(6, dtype=bool),
             np.zeros(6, dtype=int), 1, reason="t", needle="nothing left to annotate"))

print("\n7 - the record demands a reason and fingerprints the set that ran")
lab = np.array([0] * 10 + [1] * 10)
flag = np.zeros(20, dtype=bool)
flag[:6] = True
check("a blank reason is refused",
      raises(ValueError, exclusion_record_cells, flag, lab, 2, reason="  ",
             needle="needs a `reason`"))
r = exclusion_record_cells(flag, lab, 2, reason="scQC cluster_FLAG")
check("it reports the cost against the whole",
      r["cells_excluded"] == 6 and r["cells_total"] == 20
      and abs(r["fraction_excluded"] - 0.3) < 1e-12)
check("it says WHICH cluster the nuclei came from, not only how many",
      r["per_cluster"] == {0: 6})
check("passengers are reported as zero rather than omitted", r["passengers"] == 0)
# Two masks of the same SIZE must not share a digest, or the fingerprint proves nothing:
# a summary table cannot tell them apart and that is exactly what it is there to do.
other = np.zeros(20, dtype=bool)
other[6:12] = True
check("the digest distinguishes two different masks of identical size",
      flag_digest(flag) != flag_digest(other),
      f"{flag_digest(flag)} vs {flag_digest(other)}")
check("...and is stable for the same mask", flag_digest(flag) == flag_digest(flag.copy()))
check("a mask of a different length cannot collide",
      flag_digest(np.zeros(8, dtype=bool)) != flag_digest(np.zeros(16, dtype=bool)))

print("\n" + "=" * 66)
if FAILED:
    print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
    raise SystemExit(1)
print("exclusion OK - equals deletion, keeps the cells, records what went, cannot widen")
