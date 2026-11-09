#!/usr/bin/env python3
"""`scanno report`'s wiring, run through the real CLI instead of read off the source.

WHY THIS FILE EXISTS (single-cell-harness B1-034). The report's sections were guarded by reading
code: that `document.py` defines `scope_section(` and `write_cohort` calls it, that cli.py contains
`"--scope"`, `scope=scope` and `from .document import write_all`, that report.json's payload
literal mentions `"l1"`, that `_report`'s text contains `a.l1_key == path_key` and a comment about
"the good outcome". Every one of those can hold while the section is missing from the page, and
every one fails when the code is moved. What they stood for is run here, on two synthetic samples
annotated, voted into a scope, and annotated again against it:

  1. the HTML carries the common-scope section and the taxonomy section, and report.json carries
     the scope (as given) and the independent L1 column;
  2. --l1-key naming the deep walk's own column, or any column holding paths, is refused (exit 2);
  3. an L1 column that agrees with the deep walk everywhere is NOT refused: perfect agreement is a
     result, and refusing it would refuse the good outcome;
  4. `scanno lab` offers --fix.

    python tests/test_report_cli.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


try:
    import anndata as ad  # noqa: F401
    import pandas as pd
except ImportError as e:
    print(f"SKIP: needs {e.name}")
    raise SystemExit(0)

from _cohort import SPECIES, TISSUE, make_object, write_corpus, write_tree  # noqa: E402
from scanno import cli as C  # noqa: E402


def here(*args):
    """The real `scanno` entry point, in this process, with everything it prints captured."""
    o, e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
        try:
            rc = C.main([str(x) for x in args])
        except SystemExit as x:
            rc = x.code if isinstance(x.code, int) else 1
    return rc, o.getvalue() + e.getvalue()


work = Path(tempfile.mkdtemp(prefix="scanno_report_cli_"))
db, tree = write_corpus(work / "corpus.db"), write_tree(work / "tree.json")
base = ["--tree", tree, "--db", db, "--species", SPECIES, "--tissue", TISSUE, "--assay", "scrna",
        "--background-from-clusters", "--cluster-key", "leiden_1p0", "--resolve"]
for s in ("S1", "S2"):
    O = make_object(s)
    O.obs["leiden_1p0"] = O.obs["cluster"]
    O.write_h5ad(work / f"{s}.h5ad")

print("\n0 - two samples annotated, voted into a scope, annotated against it")
ok = all(here("annotate", "--h5ad", work / f"{s}.h5ad", *base,
              "--out-h5ad", work / "p1" / f"{s}.h5ad")[0] == 0 for s in ("S1", "S2"))
check("pass 1 annotates", ok)
rc, out = here("scope", "--h5ad", work / "p1" / "S1.h5ad", work / "p1" / "S2.h5ad", "--tree", tree,
               "--out", work / "scope.json", "--out-l1-tree", work / "l1.json")
check("the vote writes scope.json and the L1 tree", rc == 0 and (work / "scope.json").exists(),
      out[-300:])
ok = all(here("annotate", "--h5ad", work / f"{s}.h5ad", *base, "--scope", work / "scope.json",
              "--l1-tree", work / "l1.json", "--out-h5ad", work / "p2" / f"{s}.h5ad")[0] == 0
         for s in ("S1", "S2"))
check("pass 2 annotates against the scope, with the independent L1", ok)
p1 = [work / "p1" / "S1.h5ad", work / "p1" / "S2.h5ad"]
p2 = [work / "p2" / "S1.h5ad", work / "p2" / "S2.h5ad"]

print("\n1 - the document carries both sections, and report.json both columns")
rc, out = here("report", "--h5ad", *p2, "--out", work / "rep", "--scope", work / "scope.json",
               "--l1-key", "scAnno_L1", "--tree", tree, "--no-per-sample")
check("report exits 0", rc == 0, out[-400:])
try:
    rj = json.loads((work / "rep" / "report.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    rj = {}
page = work / "rep" / str((rj.get("reports") or {}).get("cohort") or "reports/cohort.html")
html = page.read_text(encoding="utf-8") if page.exists() else ""
check("the cohort page is where report.json says it is", page.exists(), str(page))
check("the common-scope section is on the page", "<h2>The common scope</h2>" in html)
check("and it renders the vote: a node row with its verdict",
      any(v in html for v in ("KEEP", "SEAL", "FORCE")) and "min-support" in html)
check("the taxonomy section is on the page", "<h2>The delivered annotation" in html)
given = json.loads((work / "scope.json").read_text(encoding="utf-8"))
check("report.json carries the scope it was given, unchanged",
      rj.get("scope") == json.loads(json.dumps(given)))
check("and the independent L1 column, by name", (rj.get("l1") or {}).get("column") == "scAnno_L1",
      str(rj.get("l1"))[:120])

print("\n2 - --l1-key naming the deep walk, or any path column, is refused")
rc, out = here("report", "--h5ad", *p2, "--out", work / "rep_self", "--l1-key", "scanno_path",
               "--no-per-sample")
check("the deep walk's own column: exit 2", rc == 2 and "deep walk's own column" in out,
      f"rc={rc}")
rc, out = here("report", "--h5ad", *p2, "--out", work / "rep_path", "--l1-key",
               "scanno_resolved_path", "--no-per-sample")
check("a column holding paths: exit 2, naming one", rc == 2 and "below" in out and "/" in out,
      f"rc={rc}")

print("\n3 - an L1 column that agrees everywhere is a result, not a refusal")
rc, out = here("report", "--h5ad", *p1, "--out", work / "rep_agree", "--l1-key", "scAnno_L1",
               "--no-per-sample")
check("a derived L1, which agrees with the path by construction, reports (exit 0)", rc == 0,
      out[-300:])

print("\n4 - the viewer's fixer is offered")
rc, out = here("lab", "--help")
check("`scanno lab --help` offers --fix", rc == 0 and "--fix" in out, f"rc={rc}")

import shutil  # noqa: E402

shutil.rmtree(work, ignore_errors=True)

print("\n" + "=" * 64)
if fails:
    print(f"report cli: {len(fails)} FAILED - " + "; ".join(fails[:8]))
    raise SystemExit(1)
print("report cli: the sections reach the page, and the --l1-key guard refuses what it should")
