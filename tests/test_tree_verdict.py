#!/usr/bin/env python3
"""`scanno tree propose`: a disease cannot remove a tissue's own lineage, a release that cannot tell
health is refused, an unchecked draft is an escape wherever its labels go, and the cap never hides a
verdict.

THE DEFECTS (single-cell-harness W044, the re-check of fix round 1).
  - The disease test was a ratio: a lineage whose share of the context's disease studies was twice its
    share of the healthy ones was set aside, so a disease that expands a tissue's own cells (fat cells
    in obesity, fibroblasts in fibrosis) removed them however many healthy studies reported them.
  - A release whose disease field was blank, "Healthy" or lower-case "normal" was drafted with every
    test off and exit 0: laxer than a context with no healthy study, which is refused.
  - `--lineage-tests off` was in no STATUS `escapes`, and annotate recorded "proposed draft" for a
    tree drafted that way, so labels from an unchecked tree read as labels from a checked one.
  - A lineage that passed every test and was cut by max_lineages was filed with the lineages that
    failed one ("beyond max_lineages" under set_aside); one that passed with no name over the marker
    floor was recorded nowhere. Nothing said a parameter was not its default.

THE RULE'S DISEASE TEST NOW (scanno/propose.py): a lineage, or a name, is the disease's when fewer than
min_studies of the healthy studies reporting it in the context are HEALTHY-ONLY (a study whose rows in
that tissue all come from healthy samples). It never asks how often disease studies report a lineage.

WHAT IS CHECKED, on invented releases built by `scanno build-markers` and read through the real CLI
(no tissue, lineage or name of any real data set):
  1. a resident lineage that the context's disease studies report three times as often, as a share,
     as its healthy studies, and that six healthy-only studies report, is KEPT;
  2. a lineage whose every healthy report comes from studies that also sampled the disease there is
     set aside, and the reason says so;
  3. a release whose disease field is blank, or reads only "Healthy", is REFUSED, naming the field
     and what it holds; a corpus with no per-study rows is refused; "normal", "NORMAL" and " Normal "
     are read as the release's Normal;
  4. `--lineage-tests off` is in STATUS.tree.json `escapes` and is named on stdout; a test parameter
     off its default is named on stdout and recorded as an escape; a shape parameter is named only;
  5. annotate on an unchecked draft records the escape it inherits in STATUS.annotate.json `escapes`
     and its tree record says unchecked; on a checked draft at its defaults it records none; a draft
     that records no lineage tests reads as unchecked;
  6. every lineage that passed every test and was cut by max_lineages is in `proposed.capped` and
     printed CAPPED; one with no name over the marker floor is in `proposed.no_panel`; neither is in
     `set_aside`.

    python tests/test_tree_verdict.py
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def status(d, cmd):
    try:
        return json.loads((Path(d) / f"STATUS.{cmd}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


OK, ILL = "Normal", "Invented fibrosis"
X, Y = "Xi tissue", "Upsilon tissue"
# (tissue, disease, cell_name_class, cell_name, study, first marker, number of markers)
DESIGN = []
for s in range(20):         # the context's steady resident lineage: twenty healthy-only studies
    DESIGN.append((X, OK, "Steady resident cell", "Steady one", f"h{s}", 0, 3))
for s in range(6):          # six of them also report the resident lineage the disease expands
    DESIGN.append((X, OK, "Expanded resident cell", "Expanded one", f"h{s}", 100, 3))
for s in range(30):         # and thirty disease studies report it: r = (30/33) / (6/23) = 3.5
    DESIGN.append((X, ILL, "Expanded resident cell", "Expanded one", f"d{s}", 100, 3))
for s in range(3):          # three studies sampled both: their healthy rows report this lineage only
    DESIGN.append((X, OK, "Adjacent lesion cell", "Adjacent one", f"m{s}", 200, 3))
    DESIGN.append((X, ILL, "Steady resident cell", "Steady one", f"m{s}", 0, 3))
for s in range(6, 10):      # a third and a fourth resident lineage, healthy-only, for the cap
    DESIGN.append((X, OK, "Third resident cell", "Third one", f"h{s}", 300, 3))
for s in range(10, 13):
    DESIGN.append((X, OK, "Fourth resident cell", "Fourth one", f"h{s}", 400, 3))
for s in range(13, 16):     # a lineage the tests pass whose one name has a single marker
    DESIGN.append((X, OK, "Thin panel cell", "Thin one", f"h{s}", 500, 1))
for s in range(20):         # the rest of the species, so the wide pool is not the context alone
    DESIGN.append((Y, OK, "Upsilon cell", "Upsilon one", f"y{s}", 600, 3))
STUDIES = sorted({row[4] for row in DESIGN})


def write_release(path, healthy=OK):
    """The invented release; `healthy` is what the healthy rows' disease field says (a list cycles)."""
    header = (ROOT / "tests" / "fixtures" / "mini_cellmarker.txt").read_text(
        encoding="utf-8").splitlines()[0].split("\t")
    words = healthy if isinstance(healthy, list) else [healthy]
    lines, i = ["\t".join(header)], 0
    for ti, dis, cls, name, study, first, k in DESIGN:
        for j in range(first, first + k):
            sym = f"VRD{j:04d}"
            if dis == OK:
                dis_w, i = words[i % len(words)], i + 1
            else:
                dis_w = dis
            rec = {"species": "Human", "tissue_class": ti, "tissue_type": ti, "disease": dis_w,
                   "cell_name_class": cls, "cell_name": name, "marker": sym, "symbol": sym,
                   "gene_type": "protein_coding", "gene_name": f"{sym} (invented)",
                   "technology_seq": "scRNA-seq", "marker_source": "Experiment",
                   "pmid": str(94000000 + STUDIES.index(study)),
                   "title": f"Invented study {study}", "journal": "Invented journal",
                   "year": "2022.0"}
            lines.append("\t".join(rec.get(c, "") for c in header))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build(name, healthy=OK):
    dbp = work / f"{name}.db"
    rc, out = here("build-markers", "--release", write_release(work / f"{name}.txt", healthy),
                   "--out", dbp)
    check(f"the corpus '{name}' is built from its invented release", rc == 0, out[-300:])
    return dbp


def propose(dbp, out, *extra, tissue=X, species="Human"):
    rc, text = here("tree", "propose", "--db", dbp, "--species", species, "--tissue", tissue,
                    "--out", out, *extra)
    d = json.loads(Path(out).read_text()) if rc == 0 and Path(out).exists() else {}
    return rc, text, d


work = Path(tempfile.mkdtemp(prefix="scanno_tree_verdict_"))
db = build("main")

print("\n1 - a disease that expands a tissue's own lineage cannot remove it")
rc, out_a, d = propose(db, work / "a" / "tree.json")
check("tree propose exits 0", rc == 0, out_a[-400:])
prop = d.get("proposed") or {}
kept = {r.get("lineage"): r for r in prop.get("kept") or []}
aside = {r.get("lineage"): r for r in prop.get("set_aside") or []}
ex = kept.get("Expanded resident cell", {})
check("the expanded resident lineage is kept: six healthy-only studies report it",
      "Expanded resident cell" in d.get("children", {}).get("root", [])
      and ex.get("healthy_only_studies") == 6, str(aside.get("Expanded resident cell") or ex))
check("though its disease ratio is over 3 (reported, tested nowhere)",
      (ex.get("disease_ratio") or 0) > 3, str(ex))
check("the draft's parameters hold no disease ratio: the test adds no threshold",
      "disease_ratio" not in (prop.get("parameters") or {}), str(prop.get("parameters")))

print("\n2 - a lineage reported healthy only by studies that also sampled the disease is set aside")
adj = aside.get("Adjacent lesion cell", {})
check("set aside for the disease test, saying none of its healthy studies sampled no disease",
      adj.get("healthy_studies") == 3 and adj.get("healthy_only_studies") == 0
      and "disease: 0 of the 3 healthy studies" in str(adj.get("reason", "")), str(adj))

print("\n3 - a release that cannot tell health is refused, naming the field and what it holds")
for name, word, said in (("blank", "", "(blank)"), ("healthy_word", "Healthy", "'Healthy'")):
    dbw = build(name, word)
    rc, out, _ = propose(dbw, work / name / "tree.json")
    st = status(work / name, "tree")
    check(f"disease field {said}: refused (exit 2, tree_no_healthy_rows), naming the field and {said}",
          rc == 2 and "disease field" in out and said in out
          and (st.get("refusal") or {}).get("code") == "tree_no_healthy_rows"
          and not (work / name / "tree.json").exists(), f"rc={rc} " + out[-300:])
dbn = build("spellings", ["normal", "NORMAL", " Normal "])
rc, out, dn = propose(dbn, work / "spellings" / "tree.json")
tn = (dn.get("proposed") or {}).get("lineage_tests") or {}
check("'normal', 'NORMAL' and ' Normal ' are the release's Normal: drafted with every test applied",
      rc == 0 and tn and all(v == "applied" for v in tn.values())
      and "Expanded resident cell" in dn.get("children", {}).get("root", []), f"rc={rc} {tn} " + out[-300:])
dbv = work / "no_view.db"
shutil.copy(db, dbv)
con = sqlite3.connect(dbv)
con.execute("DROP VIEW v_marker")
con.commit()
con.close()
rc, out, _ = propose(dbv, work / "no_view" / "tree.json")
check("a corpus with no per-study rows is refused (tree_no_study_rows), naming build-markers",
      rc == 2 and "per-study rows" in out and "build-markers" in out
      and (status(work / "no_view", "tree").get("refusal") or {}).get("code") == "tree_no_study_rows",
      f"rc={rc} " + out[-300:])

print("\n4 - an escape or a moved threshold is named on stdout and recorded")
check("at the defaults, stdout names no parameter and STATUS records no escape",
      "NOT DEFAULT" not in out_a and status(work / "a", "tree").get("escapes") == []
      and prop.get("not_default") == {}, str(status(work / "a", "tree").get("escapes")))
rc, out, doff = propose(db, work / "off" / "tree.json", "--lineage-tests", "off")
esc = status(work / "off", "tree").get("escapes") or []
check("--lineage-tests off: named NOT DEFAULT and recorded in STATUS.tree.json escapes",
      rc == 0 and "NOT DEFAULT  lineage_tests=off (default on)" in out
      and any((e.get("decision") or {}).get("flag") == "--lineage-tests"
              and (e.get("decision") or {}).get("value") == "off" for e in esc), str(esc) + out[-300:])
rc, out, d3 = propose(db, work / "ms3" / "tree.json", "--min-studies", "3", "--max-types", "5")
esc3 = status(work / "ms3", "tree").get("escapes") or []
check("--min-studies 3 is named NOT DEFAULT and recorded as an escape",
      rc == 0 and "NOT DEFAULT  min_studies=3 (default 2); recorded in STATUS escapes" in out
      and any((e.get("decision") or {}).get("flag") == "--min-studies" for e in esc3), str(esc3))
check("--max-types 5 (the draft's shape) is named NOT DEFAULT and is not an escape",
      "NOT DEFAULT  max_types=5 (default 6)\n" in out
      and not any((e.get("decision") or {}).get("flag") == "--max-types" for e in esc3), out[-300:])

print("\n5 - annotate records the escape a tree carries in")
from _cohort import make_object  # noqa: E402
mini = work / "mini.db"
rc, out = here("build-markers", "--release", ROOT / "tests" / "fixtures" / "mini_cellmarker.txt",
               "--out", mini)
check("the mini corpus is built", rc == 0, out[-200:])
make_object("S1").write_h5ad(work / "S1.h5ad")
base = ["--h5ad", work / "S1.h5ad", "--cluster-key", "cluster", "--db", mini, "--species", "Human",
        "--tissue", "Blood", "--assay", "scrna", "--background-from-clusters"]
for tag, extra in (("checked", ()), ("unchecked", ("--lineage-tests", "off"))):
    rc, out = here("tree", "propose", "--db", mini, "--species", "human", "--tissue", "blood",
                   "--out", work / tag / "tree.json", *extra)
    check(f"the {tag} draft is written", rc == 0, out[-300:])
    rc, out = here("annotate", *base, "--tree", work / tag / "tree.json",
                   "--out-h5ad", work / f"ann_{tag}" / "S1.h5ad")
    st = status(work / f"ann_{tag}", "annotate")
    inh = [e for e in st.get("escapes") or [] if (e.get("decision") or {}).get("inherited")]
    tr = st.get("tree") or {}
    if tag == "checked":
        check("annotate on the checked draft: no inherited escape, tree 'proposed draft'",
              rc == 0 and not inh and tr.get("source") == "proposed draft"
              and tr.get("unchecked") == [], f"rc={rc} {inh} {tr.get('source')}")
    else:
        check("annotate on the unchecked draft: the escape is in STATUS.annotate.json escapes",
              rc == 0 and len(inh) == 1 and (inh[0].get("decision") or {}).get("flag") == "--tree"
              and (inh[0].get("decision") or {}).get("unchecked") == ["evidence", "share",
                                                                    "specificity", "disease"],
              f"rc={rc} {st.get('escapes')}")
        check("and its tree record and stdout say the draft is unchecked",
              "unchecked" in str(tr.get("source")) and "drafted without the lineage tests" in out,
              f"{tr.get('source')} " + out[-300:])
from scanno.propose import provenance  # noqa: E402
old = provenance({"proposed": {"by": "scanno tree propose", "approved_by": None}})
check("a draft that records no lineage tests (drafted before they existed) reads as unchecked",
      old.get("unchecked") == ["evidence", "share", "specificity", "disease"]
      and "unchecked" in str(old.get("source")), str(old))

print("\n6 - the cap never hides a verdict")
rc, out, dc = propose(db, work / "cap" / "tree.json", "--max-lineages", "2")
pc = dc.get("proposed") or {}
capped = {r.get("lineage"): r for r in pc.get("capped") or []}
check("every lineage that passed every test and was cut by max_lineages is listed as capped",
      rc == 0 and set(capped) == {"Third resident cell", "Fourth resident cell"}
      and all("passed every lineage test" in str(r.get("reason")) for r in capped.values()),
      str(sorted(capped)) + " set_aside: " + str([r.get("lineage") for r in pc.get("set_aside") or []]))
check("and none of them is filed under set_aside",
      not any("max_lineages" in str(r.get("reason")) for r in pc.get("set_aside") or []),
      str([(r.get("lineage"), r.get("reason")) for r in pc.get("set_aside") or []]))
check("stdout prints each capped lineage", out.count("  CAPPED  ") == 2
      and "CAPPED  Fourth resident cell" in out, out[-500:])
nop = {r.get("lineage") for r in prop.get("no_panel") or []}
check("a lineage that passed every test with no name over the marker floor is listed in no_panel",
      nop == {"Thin panel cell"} and "Thin panel cell" not in aside, str(nop))

print("\n7 - it stays a draft")
check("approved_by is null in every draft", all(
    (x.get("proposed") or {}).get("approved_by", "absent") is None for x in (d, doff, d3, dc, dn)))

shutil.rmtree(work, ignore_errors=True)
print("\n" + "=" * 64)
if fails:
    print(f"tree verdict: {len(fails)} FAILED - " + "; ".join(fails[:8]))
    raise SystemExit(1)
print("tree verdict: a disease cannot remove a tissue's own lineage, a release that cannot tell is "
      "refused, an unchecked draft is an escape, and the cap hides nothing")
