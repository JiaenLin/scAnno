"""Single cells are not single nuclei: the assay decides the rules that depend on it, and the words.

THE DEFECT (single-cell-harness gap GAP-029). `--assay` had been required since 7502c88, and it
changed nothing. It was stored in a context and never read again:
  - a store built from nuclei was accepted as the gene background of a run declared on whole
    cells, and the reverse. A background is the per-gene mean and spread of the transcriptome it
    was measured on, and nuclear and whole-cell transcriptomes differ systematically, so lending
    one to the other is the lending the family's rule 2 forbids;
  - every document said "nuclei", whatever the assay. 171 string literals in the package named
    the nucleus. A single-cell cohort's report described its cells as nuclei;
  - the agent's prompt read an ABSENT assay as whole cells (`assay=""` meant "single-cell"). That
    is the silent default which `sch conform` S13 does not see, because `""` is not a literal value.

WHAT THIS CHECKS, THROUGH THE REAL CLI where there is one:
  A  one declared table of the two assays and of which rules depend on them
  B  `annotate --store` refuses a store of the other assay, and one that records none, BY NAME
  C  `annotate --report` words cells for scrna and nuclei for snrna: HTML, JSON and figure text
  D  `scanno report` likewise, reading the assay the objects recorded, and refusing a mixture
  E  the agent prompt requires the assay
  F  no string literal in the package names the nucleus except the table itself

WHAT IT CANNOT CHECK: that a hard-coded "cells" no longer stands in for nuclei somewhere. The word
is also the generic word of the field ("cell type"), so no scan can separate the two uses. Section
D checks the places a reader sees the unit named. A new hard-coded unit word would escape it.

    python tests/test_assay_rules.py
"""
from __future__ import annotations

import ast
import json
import re
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
    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
except ImportError as e:
    print(f"SKIP: needs {e.name}")
    raise SystemExit(0)

# By file, not through `tests`: an environment can carry a regular package of that name, which
# beats this directory on import (tests/test_portability.py, section 3).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cohort import SPECIES, TISSUE, make_object, write_corpus, write_tree  # noqa: E402

# Every figure the run saves passes through Figure.savefig; its text is read on the way out, so a
# word drawn into a PNG is checked like a word written into the HTML.
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.text import Text  # noqa: E402

FIG_TEXT: list = []
_savefig = Figure.savefig


def _spy(self, *a, **k):
    FIG_TEXT.extend(t.get_text() for t in self.findobj(Text) if t.get_text())
    return _savefig(self, *a, **k)


Figure.savefig = _spy

NUCLEUS = re.compile(r"nucle(us|i)", re.I)
DATA_URI = re.compile(r"data:[^\"')\s]+")


def cli(*args):
    """The real CLI, in a subprocess: what a user's shell runs."""
    r = subprocess.run([sys.executable, "-m", "scanno.cli", *map(str, args)], cwd=str(ROOT),
                       capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT), "PATH": "", "MPLBACKEND": "Agg"})
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def refused(rc, out):
    """Exit 2 with the tool's own REFUSE line. argparse also exits 2, for a flag it does not
    know, and that is not a refusal."""
    return rc == 2 and "REFUSE" in out


def cli_here(*args):
    """The same `main`, in this process, so the figure spy sees what it draws."""
    from scanno import cli as C
    try:
        return C.main([str(x) for x in args])
    except SystemExit as e:                     # argparse refuses with SystemExit(2)
        return int(e.code or 0)


def text_of(directory):
    """Every word a reader can see in what a command wrote: HTML (minus embedded images), JSON,
    CSV, TSV and Markdown."""
    out = []
    for p in sorted(Path(directory).rglob("*")):
        if p.suffix in (".html", ".json", ".csv", ".tsv", ".md") and p.is_file():
            out.append(DATA_URI.sub("", p.read_text(encoding="utf-8", errors="replace")))
    return "\n".join(out)


def obj(tmp, name, sample="S1", flagged=True):
    A = make_object(sample, n_per_type=30, embedding=True)
    if flagged:
        A.obs["flagged"] = np.arange(A.n_obs) % 25 == 0
    p = Path(tmp) / f"{name}.h5ad"
    A.write_h5ad(p)
    return p


def npz_with_context(src, dst, context):
    """A copy of a saved store with its context replaced - how a store with another or no
    recorded assay is made without a second background build."""
    d = dict(np.load(src, allow_pickle=True))
    d["context"] = json.dumps(context)
    np.savez_compressed(dst, **d)
    return dst


# ------------------------------------------------------------------------------------------ A
print("\nA - one declared table: the two assays, their spellings and words, and the rules")
try:
    from scanno import assay as AS
    have_table = True
except ImportError as e:
    have_table = False
    check("scanno/assay.py declares the assays", False, f"ImportError: {e}")
if have_table:
    check("the two assays are scrna and snrna", set(AS.ASSAYS) == {"scrna", "snrna"},
          str(sorted(AS.ASSAYS)))
    check("sc and sn are the same two assays", AS.canonical("sc") == "scrna"
          and AS.canonical("sn") == "snrna" and AS.canonical("snrna") == "snrna")
    for bad in (None, "", "bulk", "single"):
        try:
            AS.canonical(bad)
            check(f"an assay of {bad!r} raises: there is no default", False, "it returned")
        except ValueError:
            check(f"an assay of {bad!r} raises: there is no default", True)
    check("cells for scrna", AS.words("scrna").units == "cells" and AS.words("sc").unit == "cell")
    check("nuclei for snrna", AS.words("snrna").units == "nuclei" and AS.words("sn").unit == "nucleus")
    rules = list(getattr(AS, "RULES", []))
    check("the rule table is not empty", len(rules) >= 5, f"{len(rules)} rule(s)")
    check("every rule states both assays and whether it depends on the assay",
          all({"id", "scrna", "snrna", "depends"} <= set(r) for r in rules))
    guide = (ROOT / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")
    missing = [r["id"] for r in rules if f"`{r['id']}`" not in guide]
    check("docs/USER_GUIDE.md names every rule of the table", not missing, ", ".join(missing))
rc, out = cli("describe")
try:
    desc = json.loads(out[out.index("{"):])
except ValueError:
    desc = {}
check("`scanno describe` publishes the rule table", "assay_rules" in desc,
      ", ".join(sorted(desc)[:6]) + " ...")

with tempfile.TemporaryDirectory() as tmp:
    db, tree = write_corpus(Path(tmp) / "corpus.db"), write_tree(Path(tmp) / "tree.json")
    base = ["--cluster-key", "cluster", "--tree", tree, "--db", db,
            "--species", SPECIES, "--tissue", TISSUE]
    s1 = obj(tmp, "s1")

    # ------------------------------------------------------------------------------------ B
    print("\nB - annotate refuses a store of the other assay, and a store that records none")
    rc, out = cli("background", "--h5ad", s1, "--cluster-key", "cluster", "--out",
                  Path(tmp) / "store_sn.npz", "--species", SPECIES, "--tissue", TISSUE,
                  "--assay", "sn")
    check("a nucleus store is built (`--assay sn`, the spelling every existing job passes)",
          rc == 0, out.strip().splitlines()[-1:])
    st_sn = Path(tmp) / "store_sn.npz"
    rec = json.loads(str(np.load(st_sn, allow_pickle=True)["context"]))
    check("the store records the assay AS DECLARED, so its digest is unchanged", rec.get("assay") == "sn",
          str(rec))
    st_sc = npz_with_context(st_sn, Path(tmp) / "store_sc.npz", {**rec, "assay": "sc"})
    st_none = npz_with_context(st_sn, Path(tmp) / "store_none.npz",
                               {k: v for k, v in rec.items() if k != "assay"})

    rc, out = cli("annotate", "--h5ad", s1, *base, "--assay", "sc", "--store", st_sn)
    check("a whole-cell run on a nucleus store is REFUSED (exit 2)", refused(rc, out), f"rc={rc}")
    check("the refusal names the store, its assay and the run's",
          "store_sn.npz" in out and "snrna" in out and "scrna" in out,
          [ln for ln in out.splitlines() if "REFUSE" in ln][:1])
    rc, out = cli("annotate", "--h5ad", s1, *base, "--assay", "sn", "--store", st_sc)
    check("a nucleus run on a whole-cell store is REFUSED (exit 2)", refused(rc, out), f"rc={rc}")
    rc, out = cli("annotate", "--h5ad", s1, *base, "--assay", "sn", "--store", st_none)
    check("a store that records no assay is REFUSED: it cannot be checked", refused(rc, out),
          f"rc={rc}")
    check("and the refusal says the store records none",
          "records no assay" in out, [ln for ln in out.splitlines() if "REFUSE" in ln][:1])
    rc, out = cli("annotate", "--h5ad", s1, *base, "--assay", "sn", "--store", st_sn)
    check("CONTROL: a nucleus run on a nucleus store runs (exit 0)", rc == 0, f"rc={rc}")
    a_sn = Path(tmp) / "a_sn.h5ad"
    rc, out = cli("annotate", "--h5ad", s1, *base, "--assay", "snrna", "--store", st_sn,
                  "--out-h5ad", a_sn)
    check("`--assay snrna` and a store recorded as `sn` are the same assay (exit 0)", rc == 0,
          f"rc={rc} {out.strip().splitlines()[-1:]}")
    if a_sn.exists():
        ctx = dict(ad.read_h5ad(a_sn).uns.get("scanno_context", {}))
        check("the annotated object records the assay it was annotated under",
              ctx.get("assay") in ("snrna", "sn"), str(ctx))
        rc, out = cli("annotate", "--h5ad", a_sn, *base, "--assay", "sc",
                      "--background-from-clusters")
        check("an object annotated as nuclei is REFUSED as cells (exit 2)", refused(rc, out),
              f"rc={rc}")
    else:
        check("the annotated object records the assay it was annotated under", False, "not written")
    status = Path(tmp) / "st"
    cli("annotate", "--h5ad", s1, *base, "--assay", "sc", "--store", st_sn, "--status-dir", status)
    try:
        refusal = json.loads((status / "STATUS.annotate.json").read_text())
        code = (refusal.get("refusal") or {}).get("code", "")
    except (OSError, ValueError):
        code = ""
    check("the status file records the refusal under its own code", code == "assay_contradicted",
          repr(code))

    # ------------------------------------------------------------------------------------ C
    print("\nC - `annotate --report`: cells for scrna, nuclei for snrna, in every place it is read")
    worded = {}
    for spelling in ("sc", "scrna", "sn", "snrna"):
        out_dir = Path(tmp) / f"report_{spelling}"
        del FIG_TEXT[:]
        rc = cli_here("annotate", "--h5ad", s1, *base, "--assay", spelling,
                      "--background-from-clusters", "--exclude-flag", "flagged",
                      "--sample-key", "sample", "--report", out_dir / "report.html")
        worded[spelling] = (rc, text_of(out_dir), list(FIG_TEXT))
    for spelling in ("sc", "scrna"):
        rc, txt, figs = worded[spelling]
        check(f"--assay {spelling}: the report is written", rc == 0 and bool(txt), f"rc={rc}")
        hits = sorted({m.group(0) for m in NUCLEUS.finditer(txt + "\n".join(figs))})
        check(f"--assay {spelling}: no nucleus anywhere in the document or its figures",
              rc == 0 and bool(txt) and not hits, f"rc={rc}, found {hits}")
        check(f"--assay {spelling}: composition is a share of cells", "% of cells" in figs,
              str([t for t in figs if t.startswith("% of")][:3]))
        check(f"--assay {spelling}: the embedding is every cell",
              "Every cell, coloured by its label" in figs)
    for spelling in ("sn", "snrna"):
        rc, txt, figs = worded[spelling]
        check(f"--assay {spelling}: the report is written", rc == 0 and bool(txt), f"rc={rc}")
        check(f"--assay {spelling}: composition is a share of nuclei", "% of nuclei" in figs,
              str([t for t in figs if t.startswith("% of")][:3]))
        check(f"--assay {spelling}: the embedding is every nucleus",
              "Every nucleus, coloured by its label" in figs)
        check(f"--assay {spelling}: the withheld are nuclei", "withheld nuclei" in txt + " ".join(figs))
    rc, txt, _ = worded["scrna"]
    check("the report records the assay it describes", '"assay": "scrna"' in txt
          or '"assay": "sc"' in txt)

    # ------------------------------------------------------------------------------------ D
    print("\nD - `scanno report`: the assay the objects recorded words the cohort document")
    ann = {}
    for spelling, samples in (("scrna", ("S1", "S2")), ("snrna", ("S1", "S2"))):
        for s in samples:
            src = obj(tmp, f"{spelling}_{s}", sample=s)
            dst = Path(tmp) / f"ann_{spelling}" / f"{s}.h5ad"
            dst.parent.mkdir(exist_ok=True)
            rc, out = cli("annotate", "--h5ad", src, *base, "--assay", spelling,
                          "--background-from-clusters", "--exclude-flag", "flagged",
                          "--out-h5ad", dst)
            ann.setdefault(spelling, []).append(dst)
    for spelling, want, other in (("scrna", "cells", "nuclei"), ("snrna", "nuclei", "cells")):
        out_dir = Path(tmp) / f"cohort_{spelling}"
        del FIG_TEXT[:]
        rc = cli_here("report", "--h5ad", *ann[spelling], "--out", out_dir, "--sample-key",
                      "sample", "--condition-key", "group", "--flag-key", "flagged",
                      "--tree", tree, "--db", db, "--panels", "auto", "--species", SPECIES,
                      "--tissue", TISSUE)
        txt, figs = text_of(out_dir), list(FIG_TEXT)
        check(f"{spelling}: the cohort document is written, with no --assay given", rc == 0 and bool(txt),
              f"rc={rc}")
        if spelling == "scrna":
            hits = sorted({m.group(0) for m in NUCLEUS.finditer(txt + "\n".join(figs))})
            check("scrna: no nucleus anywhere in the documents, tables or figures",
                  rc == 0 and bool(txt) and not hits, f"rc={rc}, found {hits}")
        check(f"{spelling}: the headline tile counts {want}", f"<p class='lbl'>{want}</p>" in txt)
        check(f"{spelling}: the headline tile does not count {other}",
              rc == 0 and bool(txt) and f"<p class='lbl'>{other}</p>" not in txt)
        check(f"{spelling}: the composition axis is a share of {want}",
              any(t.startswith("%") and want in t for t in figs),
              str([t for t in figs if t.startswith("%")][:3]))
    mixed = Path(tmp) / "cohort_mixed"
    rc, out = cli("report", "--h5ad", ann["scrna"][0], ann["snrna"][1], "--out", mixed)
    check("objects annotated under different assays are REFUSED (exit 2)", refused(rc, out),
          f"rc={rc}")
    check("and the refusal names both", "scrna" in out and "snrna" in out,
          [ln for ln in out.splitlines() if "REFUSE" in ln][:1])
    rc, out = cli("report", "--h5ad", *ann["scrna"], "--out", Path(tmp) / "c2", "--assay", "snrna")
    check("an --assay that contradicts the objects' record is REFUSED (exit 2)", refused(rc, out),
          f"rc={rc}")
    if ann["scrna"][0].exists():
        bare = Path(tmp) / "bare.h5ad"
        B = ad.read_h5ad(ann["scrna"][0])
        B.uns.pop("scanno_context", None)
        B.write_h5ad(bare)
        rc, out = cli("report", "--h5ad", bare, "--out", Path(tmp) / "c3")
        check("objects that record no assay, and no --assay: REFUSED, naming --assay",
              refused(rc, out) and "--assay" in out, f"rc={rc}")
        rc, out = cli("report", "--h5ad", bare, "--out", Path(tmp) / "c4", "--assay", "sc")
        check("the same objects with --assay given: written", rc == 0, f"rc={rc}")
    else:
        check("objects annotated --assay scrna exist to report on", False, "annotate wrote none")

# ---------------------------------------------------------------------------------------- E
print("\nE - the agent's prompt requires the assay, and words it")
from scanno.agent import make_prompt  # noqa: E402

TREE = {"children": {"root": ["A", "B"]}, "patterns": {}, "members": {}}
try:
    p = make_prompt(["G1"], TREE)
    check("a prompt with no assay RAISES: an absent assay is not whole cells", False,
          f"it said {p.splitlines()[0]!r}")
except (TypeError, ValueError):
    check("a prompt with no assay RAISES: an absent assay is not whole cells", True)
for spelling, want in (("scrna", "single-cell"), ("sc", "single-cell"),
                       ("snrna", "single-nucleus"), ("sn", "single-nucleus")):
    p = make_prompt(["G1"], TREE, assay=spelling)
    check(f"assay {spelling}: the prompt says {want}", f"Annotate one {want} RNA-seq" in p,
          p.splitlines()[0])

# ---------------------------------------------------------------------------------------- F
print("\nF - no string literal in the package names the nucleus, except the assay table")
offenders = []
for f in sorted((ROOT / "scanno").glob("*.py")):
    if f.name == "assay.py":
        continue
    tree = ast.parse(f.read_text(encoding="utf-8"))
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.body and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
                and NUCLEUS.search(n.value)):
            offenders.append(f"{f.name}:{n.lineno}")
check("every unit word comes from scanno/assay.py", not offenders,
      f"{len(offenders)} literal(s), e.g. {', '.join(offenders[:6])}")

print("\n" + "=" * 64)
if fails:
    print(f"assay rules: {len(fails)} FAILED - " + "; ".join(fails[:12]))
    raise SystemExit(1)
print("assay rules: the assay decides the background a run accepts and the words of its reports")
