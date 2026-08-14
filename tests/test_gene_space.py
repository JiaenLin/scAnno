"""The naming mismatch that returns UNRESOLVED for everything and exits 0.

A corpus is keyed by SYMBOL. An object is very often keyed by ACCESSION with the symbols beside
it in `var` - which is the right way round for the object, because symbols are not unique. Match
one against the other and every marker panel comes out empty, every node is dropped for having
too few markers, the walk finds fewer than two children at the root, and every cluster is
UNRESOLVED. The run exits 0 with a full table.

`check_gene_space` was written for exactly this and documents the incident that caused it. It
was called on the `agent` path and NOT on the `annotate` path, which is the one everybody uses -
so the guard existed, was correct, and did not run where it mattered. Found by reproducing a real
stage: ten libraries, eight resolutions, 100% UNRESOLVED, no error anywhere.

Two things are asserted, and the second is the one that matters:

  1. the symbols are picked up when they are there
  2. when they are NOT - and the names genuinely do not match the corpus - the run REFUSES
     rather than returning a confident table of UNRESOLVED

    python tests/test_gene_space.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


try:
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
except ImportError as e:
    print(f"SKIP: needs {e.name}")
    raise SystemExit(0)

from scanno.corpus import GeneSpaceMismatch, check_gene_space  # noqa: E402

# Twenty markers each, not five. `check_gene_space` refuses below 50 SHARED symbols, so a
# three-by-five toy corpus cannot clear the floor and the passing case could never pass - the
# fixture would have been testing the guard against itself.
PANELS = {
    "Cardiomyocyte": ["TNNT2", "MYH6", "ACTC1", "MYL2", "TTN"] + [f"CM{i}" for i in range(15)],
    "Endothelial": ["PECAM1", "CDH5", "VWF", "KDR", "TIE1"] + [f"EC{i}" for i in range(15)],
    "Fibroblast": ["COL1A1", "DCN", "PDGFRA", "LUM", "POSTN"] + [f"FB{i}" for i in range(15)],
}

print("\n1 - the guard itself")
asr = {c.lower() + " cell": {g: 8.0 for g in gs} for c, gs in PANELS.items()}
symbols = [g for gs in PANELS.values() for g in gs] + [f"GENE{i}" for i in range(80)]
accessions = [f"ENSMUSG{i:011d}" for i in range(len(symbols))]
try:
    check_gene_space(asr, np.array(symbols))
    check("symbols pass", True)
except GeneSpaceMismatch as e:
    check("symbols pass", False, str(e)[:80])
try:
    check_gene_space(asr, np.array(accessions))
    check("accessions RAISE", False, "it accepted them")
except GeneSpaceMismatch as e:
    check("accessions RAISE", True, str(e)[:70])


def build(tmp, with_symbols):
    """An accession-keyed object, with or without the symbol column beside it."""
    rng = np.random.default_rng(0)
    n = 150
    X = rng.poisson(1.0, size=(n, len(symbols))).astype("float32")
    y = np.array([i % 3 for i in range(n)])
    for ci, gs in enumerate(PANELS.values()):
        for g in gs:
            X[y == ci, symbols.index(g)] += 40
    A = ad.AnnData(X=sp.csr_matrix(X))
    A.obs_names = [f"c{i}" for i in range(n)]
    A.var_names = accessions
    if with_symbols:
        A.var["gene_symbol"] = symbols
    A.obs["cluster"] = pd.Categorical([str(v) for v in y])
    p = Path(tmp) / f"obj_{'sym' if with_symbols else 'nosym'}.h5ad"
    A.write_h5ad(p)
    return p


def corpus(tmp):
    import sqlite3
    p = Path(tmp) / "corpus.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE assertion (species TEXT, tissue_class TEXT, cell_name TEXT, "
                "symbol_norm TEXT, evidence_tier INT, n_pmids INT)")
    con.executemany("INSERT INTO assertion VALUES (?,?,?,?,?,?)",
                    [("Mouse", "Heart", c, g, 1, 20) for c, gs in PANELS.items() for g in gs])
    con.commit(); con.close()
    return p


def tree(tmp):
    import json
    p = Path(tmp) / "tree.json"
    p.write_text(json.dumps({
        "children": {"root": list(PANELS)},
        "patterns": {c: [c.lower()] for c in PANELS}, "members": {}}), encoding="utf-8")
    return p


def run(*args):
    return subprocess.run([sys.executable, str(ROOT / "bin" / "scanno"), *map(str, args)],
                          capture_output=True, text=True)


print("\n2 - end to end: symbols present, so the corpus can address the object")
with tempfile.TemporaryDirectory() as tmp:
    obj, db, tr = build(tmp, True), corpus(tmp), tree(tmp)
    r = run("annotate", "--h5ad", obj, "--cluster-key", "cluster", "--tree", tr, "--db", db,
            "--species", "Mouse", "--tissue", "Heart", "--background-from-clusters")
    out = r.stdout + r.stderr
    check("it says which var column it read",
          "gene_symbol" in out and "gene names from" in out,
          [ln for ln in out.splitlines() if "gene names" in ln][:1])
    check("exit 0", r.returncode == 0, f"rc={r.returncode}")
    check("and it RESOLVED the clusters, rather than returning UNRESOLVED",
          "UNRESOLVED 0 clusters" in out,
          [ln for ln in out.splitlines() if "UNRESOLVED" in ln][:1])

print("\n3 - end to end: no symbols, names cannot match - REFUSE, not a table of UNRESOLVED")
with tempfile.TemporaryDirectory() as tmp:
    obj, db, tr = build(tmp, False), corpus(tmp), tree(tmp)
    r = run("annotate", "--h5ad", obj, "--cluster-key", "cluster", "--tree", tr, "--db", db,
            "--species", "Mouse", "--tissue", "Heart", "--background-from-clusters")
    out = r.stdout + r.stderr
    check("exit code is 2 (REFUSE), not 0", r.returncode == 2, f"rc={r.returncode}")
    check("and it says so", "REFUSE" in out, [ln for ln in out.splitlines() if "REFUSE" in ln][:1])
    # The refusal MESSAGE mentions UNRESOLVED - that is the failure it is describing. What must
    # not appear is the per-cluster table, which is what a completed run prints.
    # The per-cluster table is what a completed run prints; its header is the unambiguous
    # marker. Searching for the word "cluster" instead matches the refusal's own prose and the
    # --background-from-clusters REVIEW, neither of which is a result.
    check("it did NOT print the per-cluster table instead",
          "label" not in out.split("REFUSE")[0].split("\n")[-3:][0], out.splitlines()[-1][:80])

print("\n4 - --gene-key names the column explicitly, and a wrong one is refused")
with tempfile.TemporaryDirectory() as tmp:
    obj, db, tr = build(tmp, True), corpus(tmp), tree(tmp)
    r = run("annotate", "--h5ad", obj, "--cluster-key", "cluster", "--tree", tr, "--db", db,
            "--species", "Mouse", "--tissue", "Heart", "--background-from-clusters",
            "--gene-key", "nope")
    out = r.stdout + r.stderr
    check("a var column that does not exist is refused", r.returncode != 0, f"rc={r.returncode}")
    check("and the available columns are listed", "gene_symbol" in out)

print("\n" + "=" * 64)
if fails:
    print(f"gene space: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print("gene space OK - symbols are found, and a mismatch refuses instead of resolving nothing")
