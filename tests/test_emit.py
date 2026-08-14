"""The per-cell join, and what a viewer needs from the object it lands in.

`classify()` returns one row per CLUSTER; a viewer needs one label per CELL. Between the two
sits a join that has three ways to be quietly wrong, and each is asserted here:

  1. a cluster in the cell assignment with no call must RAISE, not label the population
     something plausible;
  2. the per-cell flag must OVERRIDE the cluster's call, because the exclusion is per nucleus;
  3. a statistic of a call that was not made must be NaN, never 0.

The last one is the reason this file exists at all rather than four lines in the CLI. A gap of
0.0 sorts first, averages into every summary, and looks exactly like a genuinely marginal call.

    python tests/test_emit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fails = []
skips = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def skip(name, why):
    print(f"  SKIP  {name}   {why}")
    skips.append(name)


try:
    import numpy as np
except ImportError:
    print("SKIP: needs numpy")
    raise SystemExit(0)

from scanno.emit import (annotate_obs, format_readiness, lab_readiness,  # noqa: E402
                         per_cell, support_per_cell)
from scanno.exclude import EXCLUDED  # noqa: E402


def calls(*labels):
    """One classify()-shaped row per cluster, in order, as classify() promises."""
    return [{"cluster": i, "label": lab, "path": f"root/{lab}", "depth": 2,
             "gap": 0.5 + i / 100, "survival": 0.8, "cover": 0.9, "excluded": False,
             "trace": []}
            for i, lab in enumerate(labels)]


print("\n0 - the digest contract with scQC")
# scQC stamps uns["scqc"]["flag_digest"] onto every object it delivers, and scAnno verifies the
# flag against it before acting on it. The two implementations are deliberately NOT shared code -
# neither repo depends on the other - so they are held together by this vector, asserted in both
# suites. If it fails here, THIS implementation changed: scQC's tests/test_declaration.py pins
# the same five bits to the same value.
from scanno.exclude import flag_digest  # noqa: E402

check("the CONTRACT vector still hashes to the agreed value",
      flag_digest([True, False, True, True, False]) == "3ba679de109f5333",
      f"got {flag_digest([True, False, True, True, False])}, agreed 3ba679de109f5333")

print("\n1 - the join is by cluster index, not by row order")
res = calls("Myeloid", "Lymphoid", "Stromal")
# Deliberately out of order and unbalanced: cell 0 is in cluster 2.
y = np.array([2, 0, 0, 1, 2, 2])
got = per_cell(res, y)
check("every cell gets its own cluster's label",
      list(got["cell_type"]) == ["Stromal", "Myeloid", "Myeloid", "Lymphoid",
                                 "Stromal", "Stromal"],
      str(list(got["cell_type"])))
check("path travels with it", got["path"][0] == "root/Stromal", got["path"][0])
check("one value per cell", all(len(v) == y.shape[0] for v in got.values()))

print("\n2 - a cluster with no call RAISES rather than being labelled something plausible")
try:
    per_cell(calls("Myeloid"), np.array([0, 1, 2]))
    check("missing cluster raises", False, "it returned")
except KeyError as e:
    check("missing cluster raises KeyError", True)
    check("...and names the clusters that are missing", "1" in str(e) and "2" in str(e),
          str(e)[:90])

print("\n3 - the per-cell flag overrides the cluster's call")
res = calls("Myeloid", "Lymphoid")
y = np.array([0, 0, 1, 1])
flag = np.array([False, True, False, True])
got = per_cell(res, y, flag=flag)
check("flagged cells are EXCLUDED", list(got["cell_type"]) ==
      ["Myeloid", EXCLUDED, "Lymphoid", EXCLUDED], str(list(got["cell_type"])))
check("unflagged cells in the SAME cluster keep their label",
      got["cell_type"][0] == "Myeloid" and got["cell_type"][2] == "Lymphoid")
check("path is EXCLUDED too, not a real path", got["path"][1] == EXCLUDED, got["path"][1])

print("\n4 - a statistic of a call that was not made is NaN, never 0")
check("gap is NaN where excluded", bool(np.isnan(got["gap"][1])), str(got["gap"][1]))
check("survival is NaN where excluded", bool(np.isnan(got["survival"][3])))
check("gap is NOT zero where excluded (0.0 would sort and average as a real gap)",
      not (got["gap"][1] == 0.0))
check("kept cells keep a real gap", got["gap"][0] > 0 and not np.isnan(got["gap"][0]))
check("depth is 0 where excluded", got["depth"][1] == 0)

print("\n5 - a mis-sized flag is refused")
try:
    per_cell(calls("A"), np.array([0, 0, 0]), flag=np.array([True, False]))
    check("mis-sized flag raises", False, "it returned")
except ValueError as e:
    check("mis-sized flag raises ValueError", True, str(e)[:60])

print("\n6 - unknown support is NaN, because zero would mean 'counted, and none'")
sup = support_per_cell(np.array(["Myeloid", EXCLUDED, "Nowhere"], dtype=object),
                       {"Myeloid": 42})
check("a known label gets its count", sup[0] == 42.0, str(sup[0]))
check("the sentinel gets NaN", bool(np.isnan(sup[1])))
check("a label absent from the corpus gets NaN, not 0", bool(np.isnan(sup[2])), str(sup[2]))

# ----------------------------------------------------------------- pandas / anndata
try:
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
except ImportError as e:
    skip("obs writing and the h5ad round trip", f"needs {e.name}")
    print("\n" + "=" * 64)
    if fails:
        print(f"emit: {len(fails)} FAILED - " + ", ".join(fails))
        raise SystemExit(1)
    print(f"emit OK ({len(skips)} skipped - a skip is not a pass)")
    raise SystemExit(0)


def toy(n=60, g=25, accession=False, embedding=True, negative=False):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 8, size=(n, g)).astype("float32")
    if negative:
        X[0, 0] = -1.0
    A = ad.AnnData(X=sp.csr_matrix(X))
    A.obs_names = [f"cell{i}" for i in range(n)]
    A.var_names = ([f"ENSMUSG{i:011d}" for i in range(g)] if accession
                   else [f"Gene{i}" for i in range(g)])
    A.obs["leiden"] = pd.Categorical([str(i % 3) for i in range(n)])
    A.obs["sample"] = pd.Categorical(["s1"] * (n // 2) + ["s2"] * (n - n // 2))
    A.obs["condition"] = pd.Categorical(["ctrl"] * (n // 2) + ["hfd"] * (n - n // 2))
    if embedding:
        A.obsm["X_umap"] = rng.normal(size=(n, 2)).astype("float32")
    return A


print("\n7 - annotate_obs writes a Categorical, which is the encoding readers understand")
A = toy()
y = np.array([int(v) for v in A.obs["leiden"]])
written = annotate_obs(A, calls("Myeloid", "Lymphoid", "Stromal"), y)
check("the label column is named so a reader can guess it",
      "scanno_cell_type" in written, str(written))
check("it is a pandas Categorical",
      isinstance(A.obs["scanno_cell_type"].dtype, pd.CategoricalDtype),
      str(A.obs["scanno_cell_type"].dtype))
check("the evidence columns travel with it",
      {"scanno_path", "scanno_gap", "scanno_depth", "scanno_survival"} <= set(written))
check("support is absent when no corpus was consulted", "scanno_support" not in written)

print("\n8 - the object's own contents are untouched")
B = toy()
before_X = B.X.copy()
before_var = list(B.var_names)
before_obsm = B.obsm["X_umap"].copy()
annotate_obs(B, calls("A", "B", "C"), np.array([int(v) for v in B.obs["leiden"]]))
check("X is unchanged", (B.X != before_X).nnz == 0)
check("var_names are unchanged", list(B.var_names) == before_var)
check("obsm survives", np.array_equal(B.obsm["X_umap"], before_obsm))

print("\n9 - the h5ad round trip: written as categories + codes, and read back intact")
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "annotated.h5ad"
    A.write_h5ad(p, compression="gzip")
    back = ad.read_h5ad(p)
    check("the label column survives the round trip",
          list(back.obs["scanno_cell_type"].astype(str)) ==
          list(A.obs["scanno_cell_type"].astype(str)))
    check("the embedding survives", "X_umap" in back.obsm)

    # The encoding matters, not just the values: a reader looks inside obs/<key> for
    # `categories` and `codes`. A nullable pandas integer would be a group WITHOUT them and
    # gets skipped, which is why the numeric columns here are plain floats.
    import h5py  # noqa: E402
    with h5py.File(p, "r") as f:
        node = f["obs/scanno_cell_type"]
        check("obs/scanno_cell_type is a group", isinstance(node, h5py.Group))
        check("...holding categories and codes",
              "categories" in node and "codes" in node, str(list(node)))
        for num in ("scanno_gap", "scanno_depth", "scanno_survival"):
            check(f"obs/{num} is a plain dataset a reader will not skip",
                  isinstance(f[f"obs/{num}"], h5py.Dataset))

print("\n10 - readiness reports what a viewer needs and scAnno cannot supply")
lv = dict((m.split(":")[0], l) for l, m in lab_readiness(toy(), "scanno_cell_type"))
missing = [m for l, m in lab_readiness(toy(embedding=False), "scanno_cell_type")
           if l == "missing"]
check("a missing embedding is reported as MISSING",
      any("embedding" in m for m in missing), str(missing))
ok_embed = [m for l, m in lab_readiness(toy(), "scanno_cell_type") if l == "ok"]
check("a present UMAP is reported ok", any("X_umap" in m for m in ok_embed), str(ok_embed))

neg = [m for l, m in lab_readiness(toy(negative=True), "scanno_cell_type") if l == "warn"]
check("negative .X is flagged, because a viewer plots expression not z-scores",
      any("negative" in m for m in neg), str(neg))

acc = [m for l, m in lab_readiness(toy(accession=True), "scanno_cell_type") if l == "warn"]
check("accession var_names with no symbol column are flagged",
      any("symbol" in m for m in acc), str(acc))

Awith = toy(accession=True)
Awith.var["gene_symbol"] = [f"Gene{i}" for i in range(Awith.n_vars)]
ok = [m for l, m in lab_readiness(Awith, "scanno_cell_type") if l == "ok"]
check("...and are NOT flagged once var['gene_symbol'] is there",
      any("gene_symbol" in m for m in ok), str(ok))

A2 = toy()
A2.obs["scanno_cell_type"] = pd.Categorical([f"c{i}" for i in range(A2.n_obs)])
idish = [m for l, m in lab_readiness(A2, "scanno_cell_type") if l == "warn"]
check("one level per cell reads as an identifier, and is flagged",
      any("identifier" in m for m in idish), str(idish))

print("\n11 - the report puts what blocks a viewer first")
lines = format_readiness(lab_readiness(toy(embedding=False), "scanno_cell_type"))
check("MISSING sorts above ok", "MISSING" in lines[0], lines[0] if lines else "")

print("\n" + "=" * 64)
if fails:
    print(f"emit: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print(f"emit OK - {len(skips)} skipped" if skips else "emit OK")
