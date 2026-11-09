#!/usr/bin/env python3
"""Which corpus rows a run reads: the declared species and tissue, resolved to the corpus's own.

THE DEFECT (single-cell-harness B1-031). The corpus was matched `WHERE species=? AND
tissue_class=?`, exactly and case-sensitively, with no fallback: `human` against `Human`, a tissue
the corpus does not name, and a species it has no rows for were all refused the same way - "the
corpus has nothing for this context" - naming no alternative, so the next attempt was a guess.

WHAT HOLDS NOW, through the real CLI (`panel` and `annotate`):
  1. the species through species_check.normalise and the tissue case-insensitively: `human` /
     `brain` reads Human / Brain, and annotate's labels are the same whichever spelling is used;
  2. a tissue the corpus does not name is refused (exit 2) naming the tissues it does name, and
     annotate refuses it BEFORE reading the object;
  3. `--tissue-fallback species` reads every tissue of the species instead: an escape, printed as
     REVIEW, recorded in STATUS (escapes[] and `corpus`) and in the annotated object;
  4. `--marker-species` reads another species' markers while `--species` stays the object's: an
     escape, recorded the same way;
  5. a context that resolves to itself reads exactly the rows it always did.

    python tests/test_corpus_context.py
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


from _cohort import make_object, write_corpus, write_tree  # noqa: E402
from scanno import cli as C  # noqa: E402


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


work = Path(tempfile.mkdtemp(prefix="scanno_corpus_context_"))
db = write_corpus(work / "corpus.db", contexts=(("Human", "Blood"), ("Human", "Brain")))
mouse_db = write_corpus(work / "mouse.db", contexts=(("Mouse", "Blood"),))
tree = write_tree(work / "tree.json")

print("\n1 - the spellings resolve to the corpus's own")
rc, out = here("panel", "--db", db, "--species", "human", "--tissue", "brain")
check("panel --species human --tissue brain exits 0", rc == 0, out[-200:])
check("and reads Human / Brain, saying so", "Human / Brain" in out and "declared" in out, out[:200])
rc, out = here("panel", "--db", db, "--species", "Homo sapiens", "--tissue", "BLOOD")
check("a scientific name and an upper-case tissue resolve too", rc == 0 and "Human / Blood" in out,
      out[-200:])
rc, out = here("panel", "--db", db, "--species", "Human", "--tissue", "Blood")
check("the corpus's own spelling says nothing extra", rc == 0 and "declared" not in out, out[:200])

print("\n2 - a tissue the corpus does not name is refused, naming the ones it does")
rc, out = here("panel", "--db", db, "--species", "human", "--tissue", "Organoid",
               "--status-dir", work / "p_miss")
check("panel --tissue Organoid exits 2", rc == 2, f"rc={rc}")
check("naming Brain and Blood", "Brain" in out and "Blood" in out, out[-300:])
check("STATUS says corpus_tissue, with the fix",
      (status(work / "p_miss", "panel").get("refusal") or {}).get("code") == "corpus_tissue"
      and "--tissue-fallback" in (status(work / "p_miss", "panel").get("refusal") or {}).get("fix", ""))
rc, out = here("annotate", "--h5ad", work / "no_such_object.h5ad", "--cluster-key", "cluster",
               "--tree", tree, "--db", db, "--species", "human", "--tissue", "Organoid",
               "--assay", "scrna", "--background-from-clusters", "--out-h5ad", work / "a_miss" / "x.h5ad")
check("annotate refuses it BEFORE reading the object (exit 2, not 1 for the missing file)",
      rc == 2 and "Brain" in out, f"rc={rc} {out[-200:]}")
rc, out = here("panel", "--db", db, "--species", "zebrafish", "--tissue", "Blood")
check("a species the corpus lacks is refused, naming the one it has", rc == 2 and "Human" in out,
      out[-200:])

print("\n3 - --tissue-fallback species: every tissue of the species, as an escape")
rc, out = here("panel", "--db", db, "--species", "human", "--tissue", "Organoid",
               "--tissue-fallback", "species", "--status-dir", work / "p_fb")
check("panel exits 0", rc == 0, out[-200:])
check("printed as REVIEW", "REVIEW" in out and "--tissue-fallback" in out)
st = status(work / "p_fb", "panel")
check("STATUS records the escape as an ask/decision pair",
      any(e.get("decision", {}).get("flag") == "--tissue-fallback" for e in st.get("escapes", [])),
      str(st.get("escapes")))
check("and the context it read", (st.get("corpus") or {}).get("tissues") == "every tissue",
      str(st.get("corpus")))

S1 = make_object("S1")
S1.write_h5ad(work / "S1.h5ad")
base = ["--h5ad", work / "S1.h5ad", "--cluster-key", "cluster", "--tree", tree, "--db", db,
        "--assay", "scrna", "--background-from-clusters"]
rc, out = here("annotate", *base, "--species", "human", "--tissue", "Organoid",
               "--tissue-fallback", "species", "--out-h5ad", work / "a_fb" / "S1.h5ad")
check("annotate --tissue Organoid --tissue-fallback species exits 0", rc == 0, out[-300:])
st = status(work / "a_fb", "annotate")
check("the escape is in STATUS.annotate.json",
      any(e.get("decision", {}).get("flag") == "--tissue-fallback" for e in st.get("escapes", [])))
try:
    import anndata as ad
    ctx_fb = dict(ad.read_h5ad(work / "a_fb" / "S1.h5ad").uns["scanno_context"])
    check("and the annotated object records what the labels rest on",
          dict(ctx_fb.get("corpus", {})).get("tissues") == "every tissue", str(ctx_fb))
except (ImportError, OSError, KeyError) as e:
    check("and the annotated object records what the labels rest on", False, str(e))

print("\n4 - --marker-species: another species' markers, the object's species kept")
rc, out = here("annotate", *[x if x != db else mouse_db for x in base], "--species", "Human",
               "--tissue", "Blood", "--out-h5ad", work / "a_ms0" / "S1.h5ad")
check("a corpus with no rows for the object's species is refused (exit 2), naming Mouse",
      rc == 2 and "Mouse" in out, f"rc={rc}")
rc, out = here("annotate", *[x if x != db else mouse_db for x in base], "--species", "Human",
               "--tissue", "Blood", "--marker-species", "mouse",
               "--out-h5ad", work / "a_ms" / "S1.h5ad")
check("with --marker-species mouse it runs (exit 0)", rc == 0, out[-300:])
check("printed as REVIEW, naming the heuristic", "REVIEW" in out and "upper-cased" in out)
st = status(work / "a_ms", "annotate")
check("recorded as an escape", any(e.get("decision", {}).get("flag") == "--marker-species"
                                   for e in st.get("escapes", [])))

print("\n5 - a context that resolves to itself reads the same rows")
rc1, _ = here("annotate", *base, "--species", "Human", "--tissue", "Blood",
              "--out-h5ad", work / "a_same" / "S1.h5ad")
rc2, _ = here("annotate", *base, "--species", "human", "--tissue", "blood",
              "--out-h5ad", work / "a_case" / "S1.h5ad")
check("both spellings annotate (exit 0)", rc1 == 0 and rc2 == 0)
try:
    a1 = ad.read_h5ad(work / "a_same" / "S1.h5ad").obs
    a2 = ad.read_h5ad(work / "a_case" / "S1.h5ad").obs
    cols = [c for c in a1.columns if c.startswith(("scanno_", "scAnno_"))]
    differ = [c for c in cols if not a1[c].astype(str).equals(a2[c].astype(str))]
    check("and write identical label columns", cols and not differ, str(differ))
    check("the exact spelling records no corpus note", "corpus" not in status(work / "a_same", "annotate"))
except (NameError, OSError) as e:
    check("and write identical label columns", False, str(e))

import shutil  # noqa: E402

shutil.rmtree(work, ignore_errors=True)
print("\n" + "=" * 64)
if fails:
    print(f"corpus context: {len(fails)} FAILED - " + "; ".join(fails[:8]))
    raise SystemExit(1)
print("corpus context: spellings resolve, a miss names the alternatives, escapes are recorded")
