#!/usr/bin/env python3
"""annotate, background and build-markers read an object's gene names through ONE reader.

THE DEFECT (single-cell-harness GAP-C83). `build-markers --reference-genes <h5ad>` read the
object's `var_names`; `annotate` and `background` read `var['gene_symbol']` when it was present.
On an object keyed by accession with its symbols in `var` - the usual layout of a QC deliverable -
every marker in the corpus read as unmeasurable (`in_reference` 0 for every assertion), and the
corpus was written with STATUS ok. `background` also differed from `annotate` on a column that was
named and absent: `annotate` refused it, `background` read `var_names` without a word.

THE OBJECT is invented: three genes keyed by made-up accessions, their symbols in
var['gene_symbol'] and var['feature_name'], spelled as tests/test_build_markers.py's text list
spells them (Cd3d, CD3E, GFAP), so the .h5ad and the text list describe the same gene axis.

WHAT IS CHECKED, through the real CLI:
  1. build-markers reads the .h5ad's var['gene_symbol']: CD3D and CD3E measurable, CD2 not, the
     reference digest the text list's, and corpus_info says which names were read;
  2. --gene-key COLUMN reads that column; --gene-key '' reads var_names; a named column the object
     lacks is refused by name, by build-markers and by background alike;
  3. background's store holds the same names, and scanno/genes.py gives them.

    python tests/test_one_gene_reader.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests" / "fixtures" / "mini_cellmarker.txt"

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


try:
    import anndata as ad
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"SKIP the gene-reader checks: needs {e.name}. A skip is not a pass.")
    raise SystemExit(0)

from scanno import cli as C  # noqa: E402


def here(*args):
    o, e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
        try:
            rc = C.main([str(x) for x in args])
        except SystemExit as x:
            rc = x.code if isinstance(x.code, int) else 1
    return rc, o.getvalue() + e.getvalue()


def q(db, sql, *params):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def in_ref(db, sym):
    r = q(db, "SELECT in_reference FROM assertion WHERE species='Human' AND tissue_class='Blood' "
              "AND cell_name='T cell' AND symbol_norm=?", sym)
    return r[0][0] if r else None


work = Path(tempfile.mkdtemp(prefix="scanno_gene_reader_"))
SYMBOLS = ["Cd3d", "CD3E", "GFAP"]
ACCESSIONS = ["ACC000001", "ACC000002", "ACC000003"]
rng = np.random.default_rng(0)
obj = ad.AnnData(
    X=rng.poisson(2.0, size=(40, 3)).astype(np.float32),
    obs=pd.DataFrame({"cluster": ["a"] * 20 + ["b"] * 20}, index=[f"c{i}" for i in range(40)]),
    var=pd.DataFrame({"gene_symbol": SYMBOLS, "feature_name": SYMBOLS}, index=ACCESSIONS))
H5 = work / "keyed_by_accession.h5ad"
obj.write_h5ad(H5)
text_digest = hashlib.sha256("\n".join(sorted(SYMBOLS)).encode("utf-8")).hexdigest()

print("\n1 - build-markers reads the symbols annotate reads")
db = work / "a" / "corpus.db"
rc, out = here("build-markers", "--release", FIXTURE, "--out", db, "--reference-genes", H5)
check("build-markers with an .h5ad keyed by accession exits 0", rc == 0, out[-300:])
info = dict(q(db, "SELECT key, value FROM corpus_info")) if db.exists() else {}
check("CD3D and CD3E are measurable (in_reference 1): read from var['gene_symbol'], not var_names",
      (in_ref(db, "CD3D"), in_ref(db, "CD3E")) == (1, 1) if db.exists() else False,
      str((in_ref(db, "CD3D"), in_ref(db, "CD3E"))) if db.exists() else "no corpus")
check("a gene the object lacks is not (CD2: in_reference 0)",
      db.exists() and in_ref(db, "CD2") == 0)
check("the reference digest is the text list's: the same gene axis, read the same way",
      info.get("reference_sha256") == text_digest, str(info.get("reference_sha256"))[:16])
check("corpus_info says which names were read",
      info.get("reference_gene_names") == f"{H5.name}: var['gene_symbol']",
      str(info.get("reference_gene_names")))

print("\n2 - --gene-key, the same flag and the same rule as annotate's")
db2 = work / "b" / "corpus.db"
rc, out = here("build-markers", "--release", FIXTURE, "--out", db2, "--reference-genes", H5,
               "--gene-key", "feature_name")
info2 = dict(q(db2, "SELECT key, value FROM corpus_info")) if db2.exists() else {}
check("--gene-key feature_name reads that column",
      rc == 0 and info2.get("reference_gene_names") == f"{H5.name}: var['feature_name']"
      and info2.get("reference_sha256") == text_digest, out[-200:])
db3 = work / "c" / "corpus.db"
rc, out = here("build-markers", "--release", FIXTURE, "--out", db3, "--reference-genes", H5,
               "--gene-key", "")
info3 = dict(q(db3, "SELECT key, value FROM corpus_info")) if db3.exists() else {}
check("--gene-key '' reads var_names, says so, and the accessions match no marker",
      rc == 0 and info3.get("reference_gene_names") == f"{H5.name}: var_names"
      and in_ref(db3, "CD3D") == 0, out[-200:])
db4 = work / "d" / "corpus.db"
rc, out = here("build-markers", "--release", FIXTURE, "--out", db4, "--reference-genes", H5,
               "--gene-key", "not_a_column")
check("build-markers refuses a named column the object lacks, by name, and writes nothing",
      rc != 0 and "no var column 'not_a_column'" in out and not db4.exists(), out[-200:])
assay = C._ASSAY_CHOICES[0]
rc, out = here("background", "--h5ad", H5, "--cluster-key", "cluster", "--out", work / "bg_bad.npz",
               "--species", "Human", "--tissue", "Blood", "--assay", assay,
               "--gene-key", "not_a_column")
check("background refuses it the same way (it used to read var_names without a word)",
      rc != 0 and "no var column 'not_a_column'" in out and not (work / "bg_bad.npz").exists(),
      f"rc {rc}: {out[-200:]}")

print("\n3 - one reader: background's store and scanno/genes.py name the same genes")
rc, out = here("background", "--h5ad", H5, "--cluster-key", "cluster", "--out", work / "bg.npz",
               "--species", "Human", "--tissue", "Blood", "--assay", assay)
stored = [str(g) for g in np.load(work / "bg.npz", allow_pickle=True)["genes"]] if rc == 0 else []
check("background's store holds the symbols, upper-cased",
      sorted(stored) == sorted(s.upper() for s in SYMBOLS), f"rc {rc}: {stored or out[-200:]}")
try:
    from scanno.genes import GeneKeyMissing, gene_names
except ImportError as e:
    check("scanno/genes.py, the one reader, exists", False, str(e))
else:
    check("gene_names: var['gene_symbol'] by default, as stored",
          gene_names(obj) == (SYMBOLS, "var['gene_symbol']"))
    check("gene_names: '' is var_names", gene_names(obj, "") == (ACCESSIONS, "var_names"))
    bare = ad.AnnData(X=obj.X, var=pd.DataFrame(index=ACCESSIONS))
    check("gene_names: no symbol column falls back to var_names",
          gene_names(bare) == (ACCESSIONS, "var_names"))
    try:
        gene_names(obj, "not_a_column")
        check("gene_names refuses a named, absent column", False, "returned")
    except GeneKeyMissing as e:
        check("gene_names refuses a named, absent column", e.key == "not_a_column")
    src = (ROOT / "scanno" / "cli.py").read_text(encoding="utf-8") + \
        (ROOT / "scanno" / "markers.py").read_text(encoding="utf-8")
    check("annotate, background and build-markers all call it (three call sites, no other copy "
          "of the default)",
          src.count("gene_names(src, a.gene_key)") == 1 and src.count("gene_names(A, a.gene_key)") == 1
          and src.count("gene_names(A, gene_key)") == 1
          and '"gene_symbol" in src.var' not in src and '"gene_symbol" in A.var:' not in src)

print("\n" + "=" * 64)
if fails:
    print(f"one gene reader: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print("one gene reader OK - annotate, background and build-markers read the same names")
