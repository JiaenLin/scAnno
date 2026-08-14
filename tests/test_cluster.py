"""Step 1 selects nothing, keeps everything, and refuses to destroy the counts.

Three properties, each one a thing that cannot be recovered later if it goes wrong:

  1. EVERY resolution asked for is computed and kept. A sweep that discarded the evidence for
     its own stopping point would be unfalsifiable.
  2. Raw counts survive, and the module REFUSES rather than proceeding when it cannot find
     them. Stage-5 pseudobulk needs them and a gene lost here is unrecoverable there.
  3. `--split-by` clusters each group independently, and drops the upstream declaration from
     each piece - carrying a cohort's declaration onto a subset would make every piece fail
     verification downstream, which reads as tampering rather than as splitting.

    python tests/test_cluster.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


from scanno.cluster import DEFAULT_RESOLUTIONS, parse_resolutions, res_tag  # noqa: E402

print("\n1 - the resolution spec, both forms")
check("default is the eight-point sweep", parse_resolutions(None) == list(DEFAULT_RESOLUTIONS))
check("a range expands", parse_resolutions("0.25:1.0:0.25") == [0.25, 0.5, 0.75, 1.0])
check("a list is sorted and deduplicated", parse_resolutions("1.0,0.5,1.0") == [0.5, 1.0])
check("a single value works", parse_resolutions("1.0") == [1.0])
try:
    parse_resolutions("0:1:0")
    check("a zero step raises", False, "it returned")
except ValueError:
    check("a zero step raises", True)

print("\n2 - the tag makes a column name that survives a formula parser")
check("the dot becomes p", res_tag(1.0) == "1p0" and res_tag(0.25) == "0p25")

try:
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import scanpy  # noqa: F401
    import igraph  # noqa: F401
except ImportError as e:
    print(f"\nSKIP the clustering checks: needs {e.name}")
    print("\n" + "=" * 64)
    if fails:
        print(f"cluster: {len(fails)} FAILED - " + ", ".join(fails))
        raise SystemExit(1)
    print("cluster OK (clustering checks skipped - a skip is not a pass)")
    raise SystemExit(0)

from scanno.cluster import COUNTS_LAYER, cluster, looks_like_counts, split  # noqa: E402


def toy(n=120, g=60, counts=True, declare=False):
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n, g)).astype("float32")
    grp = np.array([i % 3 for i in range(n)])
    for c in range(3):                      # three genuinely separable populations
        X[grp == c, c * 8:(c + 1) * 8] += 30
    if not counts:
        X = X / 3.7
    A = ad.AnnData(X=sp.csr_matrix(X))
    A.obs_names = [f"c{i}" for i in range(n)]
    A.var_names = [f"G{i}" for i in range(g)]
    A.obs["sample"] = pd.Categorical(["s1"] * (n // 2) + ["s2"] * (n - n // 2))
    A.obs["cluster_FLAG"] = pd.array([i % 20 == 0 for i in range(n)], dtype="boolean")
    if declare:
        A.uns["scqc"] = {"schema": "scqc/provenance@1", "tool": "scQC",
                         "flag_column": "cluster_FLAG", "n_obs": n, "n_flagged": 6,
                         "flag_digest": "x" * 16}
    return A


print("\n3 - counts are detected, not assumed")
check("integer counts read as counts", looks_like_counts(toy().X))
check("divided values do not", not looks_like_counts(toy(counts=False).X))

print("\n4 - it REFUSES when there are no counts to keep")
try:
    cluster(toy(counts=False), resolutions=[1.0], log=lambda *_: None)
    check("refused", False, "it clustered anyway")
except ValueError as e:
    check("refused", True)
    check("...and says what breaks later", "pseudobulk" in str(e), str(e)[:70])

print("\n5 - every resolution asked for is computed and kept")
A = toy()
info = cluster(A, resolutions=[0.5, 1.0], log=lambda *_: None)
check("one column per resolution",
      {"leiden_0p5", "leiden_1p0"} <= set(A.obs.columns), str(list(A.obs.columns)))
check("both are recorded in uns", A.uns["scanno_cluster"]["resolutions"] == [0.5, 1.0])
check("with their cluster counts", len(A.uns["scanno_cluster"]["n_clusters"]) == 2)
check("it found the three populations", A.obs["leiden_1p0"].nunique() >= 3,
      str(A.obs["leiden_1p0"].nunique()))
check("nothing was selected: no resolution is marked chosen",
      "chosen" not in str(A.uns["scanno_cluster"]).lower() or
      "none is chosen" in str(A.uns["scanno_cluster"]))

print("\n6 - what survives the clustering")
check("raw counts are in the layer", COUNTS_LAYER in A.layers)
head = A.layers[COUNTS_LAYER][:50].toarray()
check("and are still integer", bool(np.allclose(head, np.round(head))))
check("every gene is still there", A.layers[COUNTS_LAYER].shape[1] == 60)
check("X is log-normalised, not scaled (no negatives)", float(A.X.min()) >= 0)
check("an embedding was produced", "X_umap" in A.obsm and A.obsm["X_umap"].shape[1] == 2)
check("PCA is there too", "X_pca" in A.obsm)
check("no gene class was excluded, and it says so",
      "none" in A.uns["scanno_cluster"]["gene_classes_excluded"])
check("every gene was eligible for selection", A.uns["scanno_cluster"]["genes_total"] == 60)

print("\n7 - the same seed gives the same partition")
B = toy()
cluster(B, resolutions=[1.0], seed=0, log=lambda *_: None)
C = toy()
cluster(C, resolutions=[1.0], seed=0, log=lambda *_: None)
check("reproducible", list(B.obs["leiden_1p0"]) == list(C.obs["leiden_1p0"]))

print("\n8 - --split-by clusters each group on its own")
D = toy(declare=True)
pieces = list(split(D, "sample"))
check("one piece per group", [g for g, _ in pieces] == ["s1", "s2"])
check("and they partition the cells", sum(p.n_obs for _, p in pieces) == D.n_obs)
check("the upstream declaration is dropped from each piece",
      all("scqc" not in p.uns for _, p in pieces))
check("...but the flag column travels with the cells",
      all("cluster_FLAG" in p.obs for _, p in pieces))
check("the original is untouched", "scqc" in D.uns)

for _g, piece in pieces:
    cluster(piece, resolutions=[1.0], log=lambda *_: None)
check("each piece clustered independently", all("leiden_1p0" in p.obs for _, p in pieces))

print("\n" + "=" * 64)
if fails:
    print(f"cluster: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print("cluster OK - every resolution kept, counts survive, nothing selected")
