#!/usr/bin/env python3
"""`scanno tree propose`: a draft tree from the corpus, which stays a draft until approved.

THE CONSTRAINT (single-cell-harness B1-098). `annotate` needs a declared taxonomy, written by hand
("the tree is yours and it is the biggest lever"), so every new tissue started from a blank page.
The tree stays the user's; what changes is that the tool drafts one from the corpus's own grouping
and says, on every run that uses it, that nobody has approved it yet.

WHAT IS CHECKED, through the real CLI, on a corpus `scanno build-markers` builds from the invented
release in tests/fixtures:
  1. `tree propose` writes a rooted lineage -> type draft from the corpus's cell_name_class, bare
     names unique, patterns the corpus's own names, and a `proposed` block with no approver;
     STATUS says 'tree: proposed draft';
  2. `annotate --tree <draft>` runs, prints REVIEW, and records 'tree: proposed draft' in STATUS;
     once proposed.approved_by names someone, it records who and the REVIEW goes;
  3. a corpus with no cell_name_class gets a flat draft and says so (this one also has no per-study
     rows, so it is refused unless `--lineage-tests off` asks for the unchecked draft);
  4. it never overwrites a tree, and a context the corpus does not name is refused with the
     tissues it does.

    python tests/test_tree_propose.py
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


from _cohort import make_object, write_corpus  # noqa: E402
from scanno import cli as C  # noqa: E402
from scanno.scope import bare_names_unique  # noqa: E402


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


work = Path(tempfile.mkdtemp(prefix="scanno_tree_propose_"))
db = work / "corpus.db"
rc, out = here("build-markers", "--release", ROOT / "tests" / "fixtures" / "mini_cellmarker.txt",
               "--out", db)
check("the corpus is built from the invented release", rc == 0, out[-200:])

print("\n1 - a rooted lineage -> type draft, from the corpus's own grouping")
rc, out = here("tree", "propose", "--db", db, "--species", "human", "--tissue", "blood",
               "--out", work / "t" / "tree.draft.json")
check("tree propose exits 0", rc == 0, out[-300:])
draft = json.loads((work / "t" / "tree.draft.json").read_text()) if rc == 0 else {}
ch = draft.get("children", {})
check("rooted, with the corpus's two lineages", sorted(ch.get("root", [])) == ["Lymphocyte", "Myeloid cell"],
      str(ch.get("root")))
check("each lineage holds its cell names",
      sorted(ch.get("Lymphocyte", [])) == ["B cell", "Natural killer cell", "T cell"]
      and sorted(ch.get("Myeloid cell", [])) == ["Dendritic cell", "Monocyte"], str(ch))
check("no name sits at two positions", bare_names_unique(draft) == {})
check("patterns are the corpus's names, lower-cased",
      draft.get("patterns", {}).get("T cell") == ["t cell"]
      and "natural killer cell" in draft.get("patterns", {}).get("Lymphocyte", []))
prop = draft.get("proposed", {})
check("it carries a proposed block: its rule, its corpus, no approver",
      prop.get("by") == "scanno tree propose" and prop.get("approved_by") is None
      and prop.get("rule") and len(prop.get("corpus_sha256") or "") == 64, str(prop)[:200])
st = status(work / "t", "tree")
check("STATUS says tree: proposed draft", st.get("headline") == "tree: proposed draft"
      and (st.get("tree") or {}).get("source") == "proposed draft", str(st.get("headline")))

print("\n2 - annotate runs on it, and says it is a draft until someone approves it")
make_object("S1").write_h5ad(work / "S1.h5ad")
base = ["--h5ad", work / "S1.h5ad", "--cluster-key", "cluster", "--db", db, "--species", "Human",
        "--tissue", "Blood", "--assay", "scrna", "--background-from-clusters"]
rc, out = here("annotate", *base, "--tree", work / "t" / "tree.draft.json",
               "--out-h5ad", work / "a" / "S1.h5ad")
check("annotate --tree <draft> exits 0", rc == 0, out[-300:])
check("and prints REVIEW about the draft", "REVIEW" in out and "nobody has approved it" in out)
st = status(work / "a", "annotate")
check("STATUS.annotate.json records tree: proposed draft",
      (st.get("tree") or {}).get("source") == "proposed draft", str(st.get("tree")))
try:
    import anndata as ad
    A = ad.read_h5ad(work / "a" / "S1.h5ad")
    t = A.obs["cluster"].astype(str) == "0"                    # _cohort's T cells
    check("and labels by the draft: the T cells are Lymphocyte/T cell",
          set(A.obs.loc[t, "scanno_path"].astype(str)) == {"Lymphocyte/T cell"},
          str(set(A.obs.loc[t, "scanno_path"].astype(str))))
except (ImportError, OSError) as e:
    check("and labels by the draft", False, str(e))
draft["proposed"]["approved_by"] = "a reviewer"
(work / "t" / "tree.approved.json").write_text(json.dumps(draft), encoding="utf-8")
rc, out = here("annotate", *base, "--tree", work / "t" / "tree.approved.json",
               "--out-h5ad", work / "b" / "S1.h5ad")
st = status(work / "b", "annotate")
check("approved: it records who approved it", rc == 0 and (st.get("tree") or {}).get("approved_by")
      == "a reviewer", str(st.get("tree")))
check("and the draft REVIEW goes", "nobody has approved it" not in out)
plain = {"children": {"root": ["Lymphocyte", "Myeloid cell"]}, "patterns": draft["patterns"],
         "members": {}}
(work / "t" / "plain.json").write_text(json.dumps(plain), encoding="utf-8")
rc, out = here("annotate", *base, "--tree", work / "t" / "plain.json",
               "--out-h5ad", work / "c" / "S1.h5ad")
check("a tree nobody proposed records nothing about it", rc == 0
      and "tree" not in status(work / "c", "annotate"), str(status(work / "c", "annotate").get("tree")))

print("\n3 - a corpus with no cell_name_class: a flat draft, and it says so")
flat_db = write_corpus(work / "flat.db")
rc, out = here("tree", "propose", "--db", flat_db, "--species", "Human", "--tissue", "Blood",
               "--out", work / "f0" / "tree.json")
check("with no per-study rows it is refused, naming them and the escape", rc == 2
      and "per-study rows" in out and "--lineage-tests off" in out, out[-200:])
rc, out = here("tree", "propose", "--db", flat_db, "--species", "Human", "--tissue", "Blood",
               "--out", work / "f" / "tree.json", "--lineage-tests", "off")
flat = json.loads((work / "f" / "tree.json").read_text()) if rc == 0 else {}
check("it drafts root -> names", rc == 0 and len(flat.get("children", {}).get("root", [])) == 5
      and list(flat.get("children", {})) == ["root"], out[-200:])
check("and says so", "flat" in out and "flat" in (flat.get("proposed") or {}).get("lineage_from", ""))

print("\n4 - it never overwrites a tree, and a missing context names the alternatives")
rc, out = here("tree", "propose", "--db", db, "--species", "Human", "--tissue", "Blood",
               "--out", work / "t" / "tree.draft.json")
check("an existing --out is refused (exit 2)", rc == 2 and "exists" in out, f"rc={rc}")
rc, out = here("tree", "propose", "--db", db, "--species", "Human", "--tissue", "Organoid",
               "--out", work / "o" / "tree.json")
check("a tissue the corpus does not name is refused, naming Blood and Brain",
      rc == 2 and "Blood" in out and "Brain" in out, f"rc={rc}")

import shutil  # noqa: E402

shutil.rmtree(work, ignore_errors=True)
print("\n" + "=" * 64)
if fails:
    print(f"tree propose: {len(fails)} FAILED - " + "; ".join(fails[:8]))
    raise SystemExit(1)
print("tree propose: a rooted draft from the corpus, and every run says it is a draft until approved")
