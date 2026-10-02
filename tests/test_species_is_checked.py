"""The declared species, against the data (single-cell-harness ADR-0027, N2).

`--species` filtered the marker corpus and was never compared with the object. Symbols are
upper-cased on both sides before matching, so a corpus of one species scored another's genes in
silence: `--species Mouse` carried onto a human cohort would label every cluster from mouse
markers and exit 0. Ensembl identifiers name the species outright, so a contradiction there is
refused; symbol case is only a convention, so a disagreement there is a note.

    python tests/test_species_is_checked.py
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

from scanno.species_check import check as species_check  # noqa: E402

print("\n1 - the check")
human_ids = pd.DataFrame({"gene_ids": [f"ENSG{i:011d}" for i in range(50)]})
mouse_ids = pd.DataFrame({"gene_ids": [f"ENSMUSG{i:011d}" for i in range(50)]})
upper = [f"GENE{i}" for i in range(50)]
title = [f"Gene{i}x" for i in range(50)]
why, note = species_check("Mouse", human_ids, upper)
check("Mouse declared, human identifiers: REFUSED", why is not None and "ENSG" not in (why or "")[:0], why)
check("and the refusal names both", why is not None and "'Mouse'" in why and "human" in why)
check("Human declared, human identifiers: nothing", species_check("Human", human_ids, upper) == (None, None))
check("Homo sapiens is human", species_check("Homo sapiens", human_ids, upper) == (None, None))
check("Mouse declared, mouse identifiers: nothing", species_check("Mouse", mouse_ids, title) == (None, None))
why, note = species_check("Mouse", pd.DataFrame(index=range(50)), upper)
check("no identifiers, a convention mismatch: a NOTE, not a refusal", why is None and note is not None, note)
check("no identifiers, the declared convention: nothing",
      species_check("Mouse", pd.DataFrame(index=range(50)), title) == (None, None))

# Three panels of twenty: scAnno refuses below 50 symbols shared with the corpus, and the control
# case must be refused for nothing but the species.
PANELS = {"Neuron": ["SNAP25", "SYT1", "RBFOX3", "STMN2", "GAP43"] + [f"NE{i}" for i in range(15)],
          "Glia": ["SOX2", "VIM", "NES", "PAX6", "HES1"] + [f"GL{i}" for i in range(15)],
          "Choroid": ["TTR", "CLIC6", "HTR2C", "KCNJ13", "OTX2"] + [f"CP{i}" for i in range(15)]}
symbols = [g for gs in PANELS.values() for g in gs] + [f"GENE{i}" for i in range(60)]


def build(tmp, prefix):
    rng = np.random.default_rng(0)
    n = 120
    X = rng.poisson(1.0, size=(n, len(symbols))).astype("float32")
    y = np.array([i % 3 for i in range(n)])
    for ci, gs in enumerate(PANELS.values()):
        for g in gs:
            X[y == ci, symbols.index(g)] += 40
    A = ad.AnnData(X=sp.csr_matrix(X))
    A.obs_names = [f"c{i}" for i in range(n)]
    A.var_names = [f"{prefix}{i:011d}" for i in range(len(symbols))]
    A.var["gene_symbol"] = symbols
    A.obs["cluster"] = pd.Categorical([str(v) for v in y])
    p = Path(tmp) / f"obj_{prefix}.h5ad"
    A.write_h5ad(p)
    return p


def corpus(tmp):
    import sqlite3
    p = Path(tmp) / "corpus.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE assertion (species TEXT, tissue_class TEXT, cell_name TEXT, "
                "symbol_norm TEXT, evidence_tier INT, n_pmids INT)")
    con.executemany("INSERT INTO assertion VALUES (?,?,?,?,?,?)",
                    [("Mouse", "Brain", c, g, 1, 20) for c, gs in PANELS.items() for g in gs])
    con.commit(); con.close()
    return p


def tree(tmp):
    import json
    p = Path(tmp) / "tree.json"
    p.write_text(json.dumps({"children": {"root": list(PANELS)},
                             "patterns": {c: [c.lower()] for c in PANELS}, "members": {}}))
    return p


def run(*args):
    return subprocess.run([sys.executable, str(ROOT / "bin" / "scanno"), *map(str, args)],
                          capture_output=True, text=True)


print("\n2 - end to end: the same object, under each species' identifiers")
with tempfile.TemporaryDirectory() as tmp:
    db, tr = corpus(tmp), tree(tmp)
    for prefix, want in (("ENSMUSG", 0), ("ENSG", 2)):
        r = run("annotate", "--h5ad", build(tmp, prefix), "--cluster-key", "cluster", "--tree", tr,
                "--db", db, "--species", "Mouse", "--tissue", "Brain", "--assay", "sc",
                "--background-from-clusters")
        out = r.stdout + r.stderr
        check(f"{prefix} identifiers with --species Mouse: exit {want}", r.returncode == want,
              f"rc={r.returncode} {[l for l in out.splitlines() if 'REFUSE' in l][:1]}")
    check("the refusal says why", "Ensembl identifiers" in out and "human" in out)

print("\n" + "=" * 64)
if fails:
    print(f"species: {len(fails)} FAILED - " + "; ".join(fails))
    raise SystemExit(1)
print("species: a declared species the identifiers contradict is refused")
