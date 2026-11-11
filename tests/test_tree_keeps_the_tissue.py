#!/usr/bin/env python3
"""`scanno tree propose` drafts the tissue's OWN lineages, and says what it set aside and why.

THE DEFECT (single-cell-harness GAP-C85). The draft took the classes with the most markers among
everything a tissue's studies mention. A tissue's studies also report the tumours that grow in it,
the immune cells that infiltrate those tumours, and the blood and vessels every tissue carries, so a
draft for one tissue drafted lineages that belong to another: a person was asked to approve a tree
whose lineages the sample cannot hold, and cells walked to them.

THE RULE, from the corpus's own fields (each row's study, disease, tissue_class, cell_name_class) and
nothing else - no list of tissues or lineages, so it holds for any tissue. Counted in studies (here
every marker row is its own invented study), a lineage is the tissue's own when
  (a) at least two healthy studies of the tissue report it, and its disease studies do not report it
      twice as often, as a share, as its healthy studies (else it is the disease's), and
  (b) the tissue's healthy studies report it at least as often, as a share, as healthy studies across
      every tissue of the species do (else it is shared, more common elsewhere).
A name is placed under the kept lineage where it has the most markers, if two healthy studies report
it. Everything that cleared the marker floor and was not placed is recorded with why
(tests/test_tree_evidence.py checks the study counting itself).

WHAT IS CHECKED, on an invented release built by `scanno build-markers` and read by `scanno tree
propose` through the real CLI. Three invented tissues; in the first, a resident lineage and a second
resident lineage, a visitor lineage that the two other tissues report far more, a tumour lineage seen
only in disease samples, and a disease-state name under a resident lineage:
  1. the draft for the first tissue roots only the two resident lineages;
  2. a name the corpus files under the visitor AND under a resident lineage is placed under the
     resident one; a name filed only under the visitor, the tumour names and the disease-state name
     are not placed, each recorded with its lineage and reason;
  3. the set-aside lineages are recorded with their reason and numbers; stdout says what was set aside;
  4. the same rule on the second tissue, where the visitor lineage is that tissue's own, keeps it:
     nothing about any lineage is written into scAnno;
  5. the draft stays a draft: approved_by is null.

    python tests/test_tree_keeps_the_tissue.py
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scanno import cli as C  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def here(*args):
    o, e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
        try:
            rc = C.main([str(x) for x in args])
        except SystemExit as x:
            rc = x.code if isinstance(x.code, int) else 1
    return rc, o.getvalue() + e.getvalue()


HEALTHY, SICK = "Normal", "Invented carcinoma"
A, B, G = "Alpha tissue", "Beta tissue", "Gamma tissue"
# (tissue, disease, cell_name_class, cell_name, number of markers). Every marker is its own symbol.
DESIGN = [
    (A, HEALTHY, "Alpha resident cell", "Resident type one", 5),
    (A, HEALTHY, "Alpha resident cell", "Resident type two", 4),
    (A, HEALTHY, "Alpha resident cell", "Shared-filed type", 4),
    (A, SICK, "Alpha resident cell", "Resident disease state", 5),
    (A, HEALTHY, "Second resident cell", "Second type one", 4),
    (A, HEALTHY, "Second resident cell", "Second type two", 4),
    (A, HEALTHY, "Visitor cell", "Visitor type one", 6),
    (A, HEALTHY, "Visitor cell", "Visitor type two", 5),
    (A, HEALTHY, "Visitor cell", "Shared-filed type", 5),
    (A, HEALTHY, "Visitor cell", "Lost type", 4),
    (A, SICK, "Tumour cell", "Tumour type one", 8),
    (A, SICK, "Tumour cell", "Tumour type two", 6),
    (A, HEALTHY, "Rare visitor cell", "Second type one", 3),
    (B, HEALTHY, "Visitor cell", "Visitor type one", 20),
    (B, HEALTHY, "Visitor cell", "Visitor type three", 20),
    (B, HEALTHY, "Beta resident cell", "Beta type one", 6),
    (B, HEALTHY, "Beta resident cell", "Beta type two", 6),
    (G, HEALTHY, "Visitor cell", "Visitor type one", 20),
    (G, HEALTHY, "Gamma resident cell", "Gamma type one", 6),
    (G, HEALTHY, "Gamma resident cell", "Gamma type two", 6),
    (G, HEALTHY, "Rare visitor cell", "Rare type one", 10),
]


def write_release(path):
    header = (ROOT / "tests" / "fixtures" / "mini_cellmarker.txt").read_text(
        encoding="utf-8").splitlines()[0].split("\t")
    lines, pmid = ["\t".join(header)], 91000000
    for i, (ti, dis, cls, name, k) in enumerate(DESIGN):
        for j in range(k):
            pmid += 1
            sym = f"INV{i:02d}G{j}"
            rec = {"species": "Human", "tissue_class": ti, "tissue_type": ti, "disease": dis,
                   "cell_name_class": cls, "cell_name": name, "marker": sym, "symbol": sym,
                   "gene_type": "protein_coding", "gene_name": f"{sym} (invented)",
                   "technology_seq": "scRNA-seq", "marker_source": "Experiment",
                   "pmid": str(pmid), "title": f"Invented study {pmid}",
                   "journal": "Invented journal", "year": "2020.0"}
            lines.append("\t".join(rec.get(c, "") for c in header))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


work = Path(tempfile.mkdtemp(prefix="scanno_tree_tissue_"))
db = work / "corpus.db"
rc, out = here("build-markers", "--release", write_release(work / "release.txt"), "--out", db)
check("the corpus is built from the invented release", rc == 0, out[-300:])

print("\n1 - the draft for a tissue roots only the tissue's own lineages")
rc, out = here("tree", "propose", "--db", db, "--species", "Human", "--tissue", A,
               "--out", work / "a" / "tree.json")
check("tree propose exits 0", rc == 0, out[-400:])
draft = json.loads((work / "a" / "tree.json").read_text()) if rc == 0 else {}
ch, prop = draft.get("children", {}), draft.get("proposed") or {}
check("root holds the two resident lineages and nothing else",
      sorted(ch.get("root", [])) == ["Alpha resident cell", "Second resident cell"], str(ch.get("root")))
check("the visitor and the tumour lineage are not in the tree",
      not ({"Visitor cell", "Tumour cell"} & set(draft.get("patterns", {}))), str(list(draft.get("patterns", {}))))

print("\n2 - each name filed under a kept lineage is placed; the others are recorded with why")
check("a name filed under the visitor and a resident lineage is placed under the resident one",
      "Shared-filed type" in ch.get("Alpha resident cell", []), str(ch.get("Alpha resident cell")))
check("the resident lineages keep their healthy names",
      {"Resident type one", "Resident type two"} <= set(ch.get("Alpha resident cell", []))
      and sorted(ch.get("Second resident cell", [])) == ["Second type one", "Second type two"], str(ch))
np_ = {r.get("name"): r for r in prop.get("not_placed") or [] if isinstance(r, dict)}
check("the names filed only under set-aside lineages are recorded as not placed",
      {"Visitor type one", "Visitor type two", "Lost type", "Tumour type one", "Tumour type two"} <= set(np_),
      str(sorted(np_)))
check("each with the lineage the corpus files it under",
      np_.get("Lost type", {}).get("lineage") == "Visitor cell"
      and np_.get("Tumour type one", {}).get("lineage") == "Tumour cell", str(np_.get("Lost type")))
check("a disease-state name under a resident lineage is not placed, for its disease claims",
      "Resident disease state" not in json.dumps(ch)
      and "disease" in str(np_.get("Resident disease state", {}).get("reason", "")),
      str(np_.get("Resident disease state")))
check("a placed name is not also listed as not placed", "Shared-filed type" not in np_)

print("\n3 - what was set aside is recorded with its reason and numbers, and printed")
aside = {r.get("lineage"): r for r in prop.get("set_aside") or [] if isinstance(r, dict)}
check("the tumour lineage is set aside as the disease's (no healthy study)",
      "disease" in str(aside.get("Tumour cell", {}).get("reason", ""))
      and aside.get("Tumour cell", {}).get("healthy_studies") == 0, str(aside.get("Tumour cell")))
v = aside.get("Visitor cell", {})
check("the visitor lineage is set aside as shared: its enrichment is below 1",
      "shared" in str(v.get("reason", "")) and isinstance(v.get("enrichment"), (int, float))
      and v["enrichment"] < 1, str(v))
check("a set-aside lineage is recorded even when its one name is placed under a kept lineage",
      "shared" in str(aside.get("Rare visitor cell", {}).get("reason", ""))
      and "Second type one" in ch.get("Second resident cell", []), str(aside.get("Rare visitor cell")))
kept = {r.get("lineage"): r for r in prop.get("kept") or [] if isinstance(r, dict)}
check("each kept lineage carries its numbers: enrichment at least 1, two healthy studies or more, "
      "two of them healthy-only",
      set(kept) == {"Alpha resident cell", "Second resident cell"}
      and all(r.get("enrichment", 0) >= 1 and r.get("healthy_studies", 0) >= 2
              and r.get("healthy_only_studies", 0) >= 2 for r in kept.values()), str(kept))
check("the rule the draft applied is named in it", "healthy" in str(prop.get("rule", "")).lower()
      and all(v == "applied" for v in (prop.get("lineage_tests") or {"x": 0}).values()),
      str(prop.get("lineage_tests")))
check("stdout says what was set aside, and why", "SET ASIDE" in out and "Visitor cell" in out
      and "Tumour cell" in out, out[-600:])

print("\n4 - the same rule on another tissue keeps the lineage that is its own")
rc, out = here("tree", "propose", "--db", db, "--species", "Human", "--tissue", B,
               "--out", work / "b" / "tree.json")
beta = json.loads((work / "b" / "tree.json").read_text()) if rc == 0 else {}
check("in the second tissue the visitor lineage is kept, beside that tissue's resident lineage",
      rc == 0 and sorted(beta.get("children", {}).get("root", [])) == ["Beta resident cell", "Visitor cell"],
      str(beta.get("children", {}).get("root")) + out[-300:])

print("\n5 - it stays a draft")
check("approved_by is null in both drafts", "approved_by" in prop and prop["approved_by"] is None
      and (beta.get("proposed") or {}).get("approved_by", "x") is None)

shutil.rmtree(work, ignore_errors=True)
print("\n" + "=" * 64)
if fails:
    print(f"tree keeps the tissue: {len(fails)} FAILED - " + "; ".join(fails[:8]))
    raise SystemExit(1)
print("tree keeps the tissue: the draft roots the tissue's own lineages and records what it set aside")
