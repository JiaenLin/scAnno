#!/usr/bin/env python3
"""`scanno build-markers`: the corpus is built by the tool, from a release the user fetched.

THE DEFECT (single-cell-harness B1-030). No annotation could run on a fresh install: the CellMarker
ingest that builds the evidence tiers lived only in one cohort's project, as that project's script.

THE FIXTURE is tests/fixtures/mini_cellmarker.txt: an INVENTED release in CellMarker 3.0's
22-column layout - the textbook markers tests/_cohort.py uses, made-up PMIDs, titles and journals -
laid out so that every case the ingest must get right occurs once: each evidence tier, a gene
written in two cases, a gene claimed by two cell types, a fetal tissue type and a disease state, a
row with no symbol, a malformed row, a second tissue and a second species.

WHAT IS CHECKED, through the real CLI:
  1. it builds; STATUS is ok and names the corpus; corpus_info records the release's sha256, the
     rows and what was skipped; every delivered row is in `marker`;
  2. the assertion table: one row per UPPER(symbol), the E1-E5 ladder, both specificities, the
     fetal and disease counts, the empty-symbol row kept and never read as a marker;
  3. the reference genes fill in_reference, exact_reference and the display spelling, and are NULL
     without them; the ortholog table bridges mouse and human by symbol case;
  4. a zip read in place gives the same table as its member; two builds give the same table;
  5. it refuses an existing --out, a release missing a column, and an empty release, by name;
  6. `panel` and `annotate` run on the corpus it built.

    python tests/test_build_markers.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
FIXTURE = ROOT / "tests" / "fixtures" / "mini_cellmarker.txt"

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


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


def row(db, cell, sym, species="Human", tissue="Blood"):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        r = con.execute("SELECT * FROM assertion WHERE species=? AND tissue_class=? AND cell_name=? "
                        "AND symbol_norm=?", (species, tissue, cell, sym)).fetchone()
        return dict(r) if r else {}
    finally:
        con.close()


def same_assertions(a, b):
    con = sqlite3.connect(f"file:{a}?mode=ro", uri=True)
    con.execute(f"ATTACH DATABASE 'file:{b}?mode=ro' AS b")
    try:
        x = con.execute("SELECT COUNT(*) FROM (SELECT * FROM main.assertion EXCEPT "
                        "SELECT * FROM b.assertion)").fetchone()[0]
        y = con.execute("SELECT COUNT(*) FROM (SELECT * FROM b.assertion EXCEPT "
                        "SELECT * FROM main.assertion)").fetchone()[0]
        return x == 0 and y == 0
    finally:
        con.close()


work = Path(tempfile.mkdtemp(prefix="scanno_build_markers_"))
db = work / "c" / "corpus.db"

print("\n0 - the fixture is what its generator writes")
import subprocess  # noqa: E402
subprocess.run([sys.executable, str(ROOT / "tests" / "fixtures" / "make_mini_cellmarker.py"),
                str(work / "regenerated.txt")], check=True, capture_output=True)
check("tests/fixtures/mini_cellmarker.txt is exactly make_mini_cellmarker.py's output",
      (work / "regenerated.txt").read_bytes() == FIXTURE.read_bytes())

print("\n1 - it builds, says what it built, and keeps every row")
rc, out = here("build-markers", "--release", FIXTURE, "--out", db)
check("build-markers exits 0", rc == 0, out[-300:])
st = json.loads((work / "c" / "STATUS.build-markers.json").read_text()) if rc == 0 else {}
check("STATUS is ok and lists the corpus as its product",
      st.get("status") == "ok" and any(p["path"] == "corpus.db" for p in st.get("products", [])))
info = dict(q(db, "SELECT key, value FROM corpus_info")) if db.exists() else {}
check("corpus_info records the release's sha256",
      info.get("source_sha256") == hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
check("and its rows: 105 written, 1 malformed skipped, none removed",
      (info.get("rows_written"), info.get("rows_malformed_skipped"), info.get("rows_removed"))
      == ("105", "1", "0"), str({k: info.get(k) for k in ("rows_written", "rows_malformed_skipped")}))
check("every delivered row is in `marker`", q(db, "SELECT COUNT(*) FROM marker")[0][0] == 105)
check("the version that built it is recorded", info.get("built_by", "").startswith("scanno"))

print("\n2 - the assertion table")
cd3d = row(db, "T cell", "CD3D")
check("CD3D and Cd3d are ONE assertion (UPPER(symbol)), two spellings folded",
      cd3d.get("n_case_variants") == 2 and cd3d.get("n_rows") == 3, str(cd3d)[:160])
tiers = {g: row(db, "T cell", g).get("evidence_tier")
         for g in ("CD3E", "CD5", "TRAC", "CD28", "LAT")}
check("the ladder: E1 two experimental PMIDs, E2 one, E3 two single-cell, E4 one, E5 detector only",
      tiers == {"CD3E": 1, "CD5": 2, "TRAC": 3, "CD28": 4, "LAT": 5}, str(tiers))
check("a gene one cell type claims has specificity 1.0", row(db, "T cell", "CD3E").get("specificity") == 1.0)
lck = row(db, "T cell", "LCK")
check("a gene two cell types claim has specificity below 1, and n_celltypes 2",
      lck.get("n_celltypes_same_tissue") == 2 and lck.get("specificity") < 1.0, str(lck)[:200])
check("...but specificity_pub reads publication-backed claims only: 1.0",
      lck.get("specificity_pub") == 1.0 and lck.get("n_celltypes_pub") == 1)
cd14 = row(db, "Monocyte", "CD14")
check("a fetal tissue type and a disease state are counted, not dropped",
      (cd14.get("n_rows"), cd14.get("n_fetal"), cd14.get("n_disease")) == (3, 1, 1), str(cd14)[:200])
check("the row with no symbol is kept, keyed ''", bool(row(db, "T cell", "")))
_, pout = here("panel", "--db", db, "--species", "Human", "--tissue", "Blood", "--top", "5")
check("and is never read as a marker", "T cell" in pout and ", ," not in pout and "  , " not in pout)

print("\n3 - the reference genes, and the ortholog bridge")
check("without --reference-genes, in_reference and exact_reference are NULL (unknown)",
      q(db, "SELECT COUNT(*) FROM assertion WHERE in_reference IS NOT NULL OR exact_reference IS NOT NULL")[0][0] == 0)
(work / "genes.txt").write_text("# the gene axis of some data\nCd3d\nCD3E\nGFAP\n", encoding="utf-8")
db_ref = work / "r" / "corpus.db"
rc, out = here("build-markers", "--release", FIXTURE, "--out", db_ref, "--reference-genes", work / "genes.txt")
check("with a reference list it builds", rc == 0, out[-200:])
r1, r2, r3 = row(db_ref, "T cell", "CD3D"), row(db_ref, "T cell", "CD3E"), row(db_ref, "T cell", "CD2")
check("measurable case-insensitively: in_reference 1", r1.get("in_reference") == 1 and r2.get("in_reference") == 1)
check("spelled as the reference spells it: exact_reference",
      r1.get("exact_reference") == 1 and r2.get("exact_reference") == 1, str((r1.get("exact_reference"), r2.get("exact_reference"))))
check("the display spelling is the reference's (Cd3d), not the most frequent",
      r1.get("symbol_display") == "Cd3d", str(r1.get("symbol_display")))
check("a gene the data lacks: in_reference 0", r3.get("in_reference") == 0)
ref_info = dict(q(db_ref, "SELECT key, value FROM corpus_info"))
check("corpus_info records the list's size and sha256",
      ref_info.get("reference_genes") == "3" and len(ref_info.get("reference_sha256", "")) == 64)
pairs = set(q(db, "SELECT symbol_mouse, symbol_human, match_method FROM ortholog"))
check("mouse and human are bridged by symbol case, and labelled so",
      ("Cd3d", "CD3D", "symbol_case") in pairs and ("Cd19", "CD19", "symbol_case") in pairs
      and not any(m == "Lyz2" for m, _h, _x in pairs), str(sorted(pairs)))

print("\n4 - a zip read in place, and two builds, give the same table")
with zipfile.ZipFile(work / "all_cell_marker.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(FIXTURE, "all_cell_marker.txt")
rc, out = here("build-markers", "--release", work / "all_cell_marker.zip", "--out", work / "z" / "corpus.db")
check("the zip builds", rc == 0, out[-200:])
check("to the same assertion table as its member", rc == 0 and same_assertions(db, work / "z" / "corpus.db"))
rc, out = here("build-markers", "--release", FIXTURE, "--out", work / "again" / "corpus.db")
check("a second build of the same release is the same table", rc == 0 and same_assertions(db, work / "again" / "corpus.db"))

print("\n5 - it refuses what it cannot use, by name")
rc, out = here("build-markers", "--release", FIXTURE, "--out", db)
check("an existing --out is refused (exit 2), never replaced", rc == 2 and "exists" in out, f"rc={rc}")
lines = FIXTURE.read_text(encoding="utf-8").splitlines()
head = lines[0].split("\t")
k = head.index("marker_source")
(work / "nosource.txt").write_text("\n".join("\t".join(c for i, c in enumerate(ln.split("\t")) if i != k)
                                             for ln in lines) + "\n", encoding="utf-8")
rc, out = here("build-markers", "--release", work / "nosource.txt", "--out", work / "x" / "corpus.db")
check("a release missing a column is refused, naming it", rc == 2 and "marker_source" in out, f"rc={rc}")
check("and leaves nothing behind", not (work / "x" / "corpus.db").exists()
      and not (work / "x" / "corpus.db.partial").exists())
(work / "empty.txt").write_text(lines[0] + "\n", encoding="utf-8")
rc, out = here("build-markers", "--release", work / "empty.txt", "--out", work / "y" / "corpus.db")
check("an empty release is refused", rc == 2 and "no row" in out, f"rc={rc}")

print("\n6 - panel and annotate run on the corpus it built")
rc, out = here("panel", "--db", db, "--species", "Human", "--tissue", "Blood")
check("panel --db <built> --species Human --tissue Blood exits 0", rc == 0, out[-200:])
try:
    import anndata as ad
    from _cohort import make_object, write_tree
    make_object("S1").write_h5ad(work / "S1.h5ad")
    rc, out = here("annotate", "--h5ad", work / "S1.h5ad", "--cluster-key", "cluster",
                   "--tree", write_tree(work / "tree.json"), "--db", db, "--species", "Human",
                   "--tissue", "Blood", "--assay", "scrna", "--background-from-clusters",
                   "--out-h5ad", work / "a" / "S1.h5ad")
    check("annotate on the built corpus exits 0", rc == 0, out[-300:])
    if rc == 0:
        A = ad.read_h5ad(work / "a" / "S1.h5ad")
        tcells = A.obs["cluster"].astype(str) == "0"            # _cohort.PANELS' first: T cell
        check("and calls the T cells T cells", set(A.obs.loc[tcells, "scanno_path"].astype(str))
              == {"Lymphoid/T cell"}, str(set(A.obs.loc[tcells, "scanno_path"].astype(str))))
except ImportError as e:
    print(f"  SKIP  annotate needs anndata ({e})")

import shutil  # noqa: E402

shutil.rmtree(work, ignore_errors=True)
print("\n" + "=" * 64)
if fails:
    print(f"build-markers: {len(fails)} FAILED - " + "; ".join(fails[:8]))
    raise SystemExit(1)
print("build-markers: every row kept, one assertion per gene, the ladder, and it annotates")
