"""`scanno embed`: the matrix must survive, and a bad input must refuse before it costs anything.

WHY THIS FILE EXISTS

scAnno had 16 test suites and `embed.py` had none. The untested module is the one that shipped a
bug which deleted the expression matrix it was built to embed, and the failure surfaced 30
minutes into a PBS job as `'NoneType' object has no attribute 'dtype'`.

The bug is worth stating exactly, because the shape of it is more common than the bug:
`B.layers` enumerates a `None` key that IS `.X`. anndata's own deprecation warning says so in
as many words. The code that removed layers was written to SILENCE that warning, and it removed
the matrix instead. **Reading the warning and acting on it are different things.**
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
fails = []
def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: fails.append(name)

try:
    import anndata as ad, scipy.sparse as sp
except ImportError:
    print("SKIP: needs anndata/scanpy.  pip install -e '.[run]'"); raise SystemExit(0)

from scanno.embed import build, notable_counts, NOTABLE  # noqa: E402

rng = np.random.default_rng(0)
def obj(n=60, g=80, counts_layer=True, x=True):
    X = sp.csr_matrix(rng.poisson(0.6, (n, g)).astype(np.float32))
    import pandas as pd
    A = ad.AnnData(X=X if x else None,
                   obs=pd.DataFrame({"sample": ["s"] * n}, index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame({"gene_symbol": [f"G{j}" for j in range(g)]},
                                    index=[f"ENSG{j}" for j in range(g)]))
    if counts_layer and x: A.layers["counts"] = X.copy()
    return A

print("A. the matrix survives the layer strip")
objs = [(f"S{i}", obj()) for i in range(3)]
J = build(objs, sample_key="sample", n_hvg=20, n_pcs=5, n_neighbors=5, log=lambda *_: None)
check("X is not None after build", J.X is not None)
check("X has the right shape", J.X.shape == (180, 80), str(None if J.X is None else J.X.shape))
check("an embedding was computed", "X_umap" in J.obsm)
check("HVGs were selected over ALL genes", int(J.var["highly_variable"].sum()) == 20)
check("the run records that it is NOT integration",
      J.uns["scanno_embed"]["integrated"] is False)

print("\nB. `.X` under the None key is NOT deleted with the layers")
# The regression, stated as the property rather than the implementation: whatever build() does
# internally, an object whose ONLY matrix is .X must still embed.
objs = [(f"S{i}", obj(counts_layer=False)) for i in range(2)]
J2 = build(objs, sample_key="sample", n_hvg=10, n_pcs=4, n_neighbors=4, log=lambda *_: None)
check("no counts layer: .X still carries through", J2.X is not None and J2.X.shape[0] == 120)

print("\nC. a bad input REFUSES up front, by name, before anything is spent")
def refuses(objs):
    try:
        build(objs, sample_key="sample", n_hvg=5, n_pcs=3, n_neighbors=3, log=lambda *_: None)
    except SystemExit as e:
        return str(e)
    return ""
msg = refuses([("good", obj()), ("BROKEN", obj(x=False))])
check("a None matrix refuses", bool(msg), msg[:70])
check("the refusal NAMES the object", "BROKEN" in msg, msg[:70])
check("the refusal does not name the healthy one", "good:" not in msg)
check("it is a refusal, not a crash 30 minutes in", "REFUSE" in msg)

print("\nD. nothing is excluded from HVG selection, and notable classes are counted")
n = notable_counts(["mt-Co1", "Rps6ka2", "Hba-a1", "Actb"])
check("mitochondrial counted", n["mitochondrial"] == ["mt-Co1"], str(n["mitochondrial"]))
check("haemoglobin counted", n["haemoglobin"] == ["Hba-a1"], str(n["haemoglobin"]))
check("ribosomal counted", n["ribosomal"] == ["Rps6ka2"], str(n["ribosomal"]))
check("counting is REPORTING, not filtering — no remove/exclude in the module",
      not any(w in (Path(__file__).resolve().parents[1] / "scanno/embed.py").read_text()
              for w in ("hv = hv[~", ".drop(", "exclude_genes")))

print("")
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}"); raise SystemExit(1)
print("embed OK - the matrix survives, a bad input refuses by name before it costs anything")
