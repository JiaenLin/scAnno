#!/usr/bin/env python3
"""`scanno tree propose` counts its evidence in STUDIES, declares its thresholds, and says when the
corpus cannot tell.

THE DEFECTS (single-cell-harness W044, the verifier's findings on the first fix). The lineage and name
tests were "at least half the CLAIMS healthy", so one study listing many markers from a disease sample
outvoted several healthy studies; a class filed by one study was 100% healthy and maximally enriched,
so it passed and took a slot; the thresholds were literals no flag could move and the draft recorded
them only as prose; and a release whose disease field never reads Normal was refused as if every
lineage had been set aside for disease.

THE RULE, from the corpus's own fields (each row's study, disease, tissue_class, cell_name_class) and
nothing else - no list of tissues, lineages or names. A study counts once however many markers it
lists. A lineage is the tissue's own when at least min_studies healthy studies report it in the
context, they are at least min_share of the context's healthy studies, it is enriched there against
the species' healthy reports (min_enrichment), and the context's disease studies do not report it
disease_ratio times as often, as a share, as its healthy studies do.

WHAT IS CHECKED, on invented releases built by `scanno build-markers` and read by `scanno tree propose`
through the real CLI:
  1. one disease study with a large panel does not outvote four healthy studies: the lineage is kept
     and its name, which that study dominates, is placed;
  2. a class one study files is not drafted, and the draft says why with its numbers; every kept,
     set-aside and not-placed entry carries studies, healthy studies, disease studies and claims;
  3. the thresholds are values in the draft and the flags move them (--min-studies 1 drafts the
     one-study class, marked single_study);
  4. a context the corpus holds only in disease samples is refused for that reason; a thin context is
     drafted and says it is thin;
  5. a release whose disease field never reads Normal is drafted unchecked and says why, and
     --lineage-tests off does so on purpose;
  6. the draft stays a draft: approved_by is null.

    python tests/test_tree_evidence.py
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


OK, ILL = "Normal", "Invented carcinoma"
A, B, G, D = "Alpha tissue", "Beta tissue", "Gamma tissue", "Delta tissue"
# (tissue, disease, cell_name_class, cell_name, study, first marker, number of markers)
DESIGN = []
for s in range(4):          # four healthy studies report the resident lineage, three markers each
    DESIGN.append((A, OK, "Resident cell", "Resident one", f"h{s}", 0, 3))
for s in range(2):          # two of them also report a second resident name
    DESIGN.append((A, OK, "Resident cell", "Resident two", f"h{s}", 10, 3))
# ONE disease study lists a large panel for both resident names: claims 40 + 30 against 18 healthy
DESIGN += [(A, ILL, "Resident cell", "Resident one", "d0", 100, 40),
           (A, ILL, "Resident cell", "Resident two", "d0", 200, 30)]
for s in range(4, 7):       # a second resident lineage, healthy, three studies
    DESIGN.append((A, OK, "Second resident cell", "Second one", f"h{s}", 300, 3))
    DESIGN.append((A, OK, "Second resident cell", "Second two", f"h{s}", 310, 3))
# a class filed by ONE healthy study, with a large panel: the curator artefact
DESIGN.append((A, OK, "Filing artefact", "Artefact type", "h7", 400, 18))
# a tumour lineage, disease studies only
for s in (1, 2):
    DESIGN.append((A, ILL, "Tumour cell", "Tumour one", f"d{s}", 500, 5))
# other healthy studies of the tissue (the context's healthy pool: 10)
for s in (8, 9):
    DESIGN.append((A, OK, "Second resident cell", "Second one", f"h{s}", 300, 3))
# the rest of the species: another tissue, so the wide pool is not the context alone
for s in range(10):
    DESIGN.append((B, OK, "Beta resident cell", "Beta one", f"b{s}", 600, 3))
    DESIGN.append((B, OK, "Other beta cell", "Beta two", f"b{s}", 610, 3))
# a thin tissue: three healthy studies
DESIGN += [(G, OK, "Gamma resident cell", "Gamma one", "g0", 700, 3),
           (G, OK, "Gamma resident cell", "Gamma one", "g1", 700, 3),
           (G, OK, "Gamma other cell", "Gamma two", "g1", 710, 3),
           (G, OK, "Gamma other cell", "Gamma two", "g2", 710, 3)]
# a tissue reported only by disease studies
DESIGN += [(D, ILL, "Delta resident cell", "Delta one", "x0", 800, 4),
           (D, ILL, "Delta other cell", "Delta two", "x1", 810, 4)]


STUDIES = sorted({row[4] for row in DESIGN})      # one invented PMID per study


def write_release(path, disease=None):
    header = (ROOT / "tests" / "fixtures" / "mini_cellmarker.txt").read_text(
        encoding="utf-8").splitlines()[0].split("\t")
    lines = ["\t".join(header)]
    for ti, dis, cls, name, study, first, k in DESIGN:
        for j in range(first, first + k):
            sym = f"EVD{j:04d}"
            rec = {"species": "Human", "tissue_class": ti, "tissue_type": ti,
                   "disease": dis if disease is None else disease,
                   "cell_name_class": cls, "cell_name": name, "marker": sym, "symbol": sym,
                   "gene_type": "protein_coding", "gene_name": f"{sym} (invented)",
                   "technology_seq": "scRNA-seq", "marker_source": "Experiment",
                   "pmid": str(93000000 + STUDIES.index(study)),
                   "title": f"Invented study {study}", "journal": "Invented journal",
                   "year": "2021.0"}
            lines.append("\t".join(rec.get(c, "") for c in header))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def propose(db, tissue, out, *extra):
    rc, text = here("tree", "propose", "--db", db, "--species", "Human", "--tissue", tissue,
                    "--out", out, *extra)
    d = json.loads(Path(out).read_text()) if rc == 0 and Path(out).exists() else {}
    return rc, text, d


work = Path(tempfile.mkdtemp(prefix="scanno_tree_evidence_"))
db = work / "corpus.db"
rc, out = here("build-markers", "--release", write_release(work / "release.txt"), "--out", db)
check("the corpus is built from the invented release", rc == 0, out[-300:])

print("\n1 - one disease study's large panel does not outvote four healthy studies")
rc, out, d = propose(db, A, work / "a" / "tree.json")
check("tree propose exits 0", rc == 0, out[-400:])
ch, prop = d.get("children", {}), d.get("proposed") or {}
check("the resident lineage is kept beside the second one",
      sorted(ch.get("root", [])) == ["Resident cell", "Second resident cell"], str(ch.get("root")))
check("the name the disease study dominates is placed under it",
      "Resident two" in ch.get("Resident cell", []), str(ch.get("Resident cell")))
kept = {r.get("lineage"): r for r in prop.get("kept") or []}
r = kept.get("Resident cell", {})
check("its numbers are in studies: 4 healthy, 1 disease, 5 in all, its claims beside them",
      (r.get("healthy_studies"), r.get("disease_studies"), r.get("studies")) == (4, 1, 5)
      and r.get("claims", 0) > r.get("healthy_claims", 0) * 2, str(r))

print("\n2 - a class one study files is not drafted, and every entry carries its numbers")
aside = {x.get("lineage"): x for x in prop.get("set_aside") or []}
art = aside.get("Filing artefact", {})
check("the one-study class is set aside for its evidence",
      "evidence" in str(art.get("reason", "")) and art.get("healthy_studies") == 1, str(art))
check("the tumour lineage is set aside: no healthy study reports it",
      aside.get("Tumour cell", {}).get("healthy_studies") == 0
      and "disease" in str(aside.get("Tumour cell", {}).get("reason", "")), str(aside.get("Tumour cell")))
need = {"studies", "healthy_studies", "disease_studies", "claims", "healthy_claims"}
entries = list(kept.values()) + list(aside.values())
check("every kept and set-aside entry carries studies, healthy studies, disease studies and claims",
      entries and all(need <= set(x) for x in entries), str([sorted(x) for x in entries][:1]))
npl = prop.get("not_placed") or []
check("every not-placed entry carries its studies and claims' markers",
      npl and all({"studies", "healthy_studies", "disease_studies", "markers"} <= set(x) for x in npl),
      str(npl[:1]))

print("\n3 - the thresholds are values in the draft, and the flags move them")
par = prop.get("parameters") or {}
check("the draft records the four thresholds as values",
      (par.get("min_studies"), par.get("min_share"), par.get("min_enrichment"),
       par.get("disease_ratio")) == (2, 0.02, 1.0, 2.0), str(par))
rc, out, d1 = propose(db, A, work / "a1" / "tree.json", "--min-studies", "1", "--disease-ratio", "3")
k1 = {x.get("lineage"): x for x in (d1.get("proposed") or {}).get("kept") or []}
check("--min-studies 1 drafts the one-study class, marked single_study",
      rc == 0 and k1.get("Filing artefact", {}).get("single_study") is True, str(k1.get("Filing artefact")) + out[-200:])
check("and the draft records the values the flags set",
      ((d1.get("proposed") or {}).get("parameters") or {}).get("disease_ratio") == 3.0)

print("\n4 - a context held only in disease samples is refused for that; a thin one says it is thin")
rc, out, _ = propose(db, D, work / "d" / "tree.json")
check("a disease-only context is refused, and the refusal says it holds no healthy study",
      rc == 2 and "0 healthy studies" in out and "--lineage-tests off" in out, f"rc={rc} " + out[-300:])
rc, out, dg = propose(db, G, work / "g" / "tree.json")
cg = (dg.get("proposed") or {}).get("cannot_tell") or []
check("a thin context is drafted, and the draft and stdout say it holds only 3 healthy studies",
      rc == 0 and any("only 3 healthy studies" in x for x in cg) and "REVIEW  this context holds only 3" in out,
      f"rc={rc} {cg} " + out[-300:])

print("\n5 - a release whose disease field never reads Normal: drafted unchecked, and it says why")
for word in ("", "Healthy"):
    dbw = work / f"word_{word or 'blank'}.db"
    rc, out = here("build-markers", "--release", write_release(work / f"r_{word or 'blank'}.txt", word),
                   "--out", dbw)
    rc, out, dw = propose(dbw, A, work / f"w_{word or 'blank'}" / "tree.json")
    pw = dw.get("proposed") or {}
    check(f"disease field {word or 'blank'!r}: drafted, cannot_tell names Normal, the tests marked not applied",
          rc == 0 and any("Normal" in x for x in pw.get("cannot_tell") or [])
          and str((pw.get("lineage_tests") or {}).get("evidence", "")).startswith("not applied")
          and "REVIEW" in out, f"rc={rc} " + out[-300:])
rc, out, doff = propose(db, A, work / "off" / "tree.json", "--lineage-tests", "off")
poff = doff.get("proposed") or {}
check("--lineage-tests off drafts every lineage unchecked, and says so",
      rc == 0 and "Tumour cell" in doff.get("children", {}).get("root", [])
      and any("--lineage-tests off" in x for x in poff.get("cannot_tell") or [])
      and (poff.get("parameters") or {}).get("lineage_tests") == "off", str(doff.get("children", {}).get("root")))

print("\n6 - it stays a draft")
check("approved_by is null in every draft", all(
    "approved_by" in (x.get("proposed") or {}) and (x.get("proposed") or {})["approved_by"] is None
    for x in (d, d1, dg, doff)))

shutil.rmtree(work, ignore_errors=True)
print("\n" + "=" * 64)
if fails:
    print(f"tree evidence: {len(fails)} FAILED - " + "; ".join(fails[:8]))
    raise SystemExit(1)
print("tree evidence: counted in studies, thresholds declared, and the corpus says when it cannot tell")
