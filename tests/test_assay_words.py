#!/usr/bin/env python3
"""Every command words its unit by the assay it knows, and refuses a cohort of two assays.

THE DEFECTS (single-cell-harness gaps GAP-C19 and GAP-C29).
  - C19. Only the two reports and annotate's own summary took their unit from the assay. Every
    other print said "cell(s)", so on nuclei it named the wrong unit: background, calibrate,
    rescue and compare; annotate's --resolve, --scope and --l1-tree summaries, its readiness
    lines and its "nothing was written" hint; the README.md `scanno report` writes; the documents
    of rescue, compare and joint-review; embed, cluster and lab; and the help of every command.
    The hint was also false: it said "nothing was written" after `--out-h5ad` had written the
    object, because it looked only at --out.
  - C29. `scanno report` refused objects annotated under different assays. `scope` and `rescue`
    took the same mixture and exited 0, so one scope was voted, and one rescue searched, over cells
    and nuclei as though they were one kind of observation.

WHAT THIS CHECKS, through the real CLI on tests/_cohort.py's objects:
  A  scanno/assay.py has words for an UNKNOWN assay - "cells or nuclei" - and they are not an
     assay: an unrecognised value still raises
  B  `scope` and `rescue` refuse a mixture of assays and an unrecognised one, by name, exit 2,
     with the code in STATUS; one assay is accepted, and so are objects that record none
  C  every command's prints and documents under snrna: no "cell(s)" for nuclei; a command that
     knows the assay says nuclei; one whose objects record no assay says "cells or nuclei". A
     document's quoted RECORD strings are removed first, reading them back from the record itself
     (they are what reproductions compare, and KNOWN_ISSUES lists them)
  D  the summary formatters, given the words, at every branch the cohort above does not reach
  E  every literal left in the package that says cell(s) as a unit is declared below with its
     reason, and every one recorded in a data product is named in KNOWN_ISSUES.md

WHAT IT CANNOT CHECK: a word assembled at run time from pieces none of which is "cell". Section C
reads what the commands actually print, which is where such a word would show.

    python tests/test_assay_words.py
"""
from __future__ import annotations

import ast
import contextlib
import html as _html
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


try:
    import anndata as ad
    import matplotlib
    import numpy as np
    import pandas as pd
    matplotlib.use("Agg")
except ImportError as e:
    print(f"SKIP: needs {e.name}")
    raise SystemExit(0)

warnings.filterwarnings("ignore")          # a library's warning is not the tool's print
# By file, not through `tests`: an environment can carry a regular package of that name.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cohort import (PANELS, SPECIES, TISSUE, TREE, make_object, write_cohort,  # noqa: E402
                     write_corpus, write_tree)
from _words import cells_said  # noqa: E402

#: Data values, which may be called anything: every name in the tree, every corpus cell name.
LABELS = ({n for n in TREE["children"]} | {c for v in TREE["children"].values() for c in v}
          | {name for name, _g in PANELS.values()}) - {"root"}
NUCLEUS = re.compile(r"\bnucle(us|i)\b", re.I)
EITHER = re.compile(r"cells or nuclei|cell or nucleus", re.I)


def cli(*args):
    """The real CLI in a subprocess: what a user's shell runs, and its exit code."""
    r = subprocess.run([sys.executable, "-m", "scanno.cli", *map(str, args)], cwd=str(ROOT),
                       capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT), "PATH": "", "MPLBACKEND": "Agg"})
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def here(*args):
    """The same `main` in this process, with everything it prints captured."""
    from scanno import cli as C
    o, e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
        try:
            rc = C.main([str(x) for x in args])
        except SystemExit as x:                     # argparse, and embed's refusals
            rc = x.code if isinstance(x.code, int) else 1
    return rc, o.getvalue() + e.getvalue()


def page(path, records=()):
    """A document's words: tags and styles removed, entities read, and each RECORD string the
    document quotes from its own record removed."""
    t = Path(path).read_text(encoding="utf-8", errors="replace")
    t = re.sub(r"<style.*?</style>", " ", t, flags=re.S)
    t = _html.unescape(re.sub(r"<[^>]+>", " ", t))
    for r in records:
        t = t.replace(str(r), " ")
    return t


def worded(name, text, want):
    """C: no cell(s) for nuclei, and the unit the command should know, said at least once.
    `want` None: the command prints no unit at all, so only the first half applies."""
    hits = cells_said(text, also=LABELS)
    check(f"{name}: never says cell(s) of nuclei", not hits, f"{len(hits)}: " + " | ".join(hits[:3]))
    if want is None:
        return
    if want == "nuclei":
        check(f"{name}: says nuclei", bool(NUCLEUS.search(EITHER.sub(" ", text))))
    else:
        check(f"{name}: says cells or nuclei, the objects recording no assay", bool(EITHER.search(text)))


def status_code(d, cmd):
    try:
        rec = json.loads((Path(d) / f"STATUS.{cmd}.json").read_text(encoding="utf-8"))
        return (rec.get("refusal") or {}).get("code", "")
    except (OSError, ValueError):
        return ""


# ------------------------------------------------------------------------------------------ A
print("\nA - words for an assay that is not known, which are not an assay")
from scanno import assay as AS  # noqa: E402

E = getattr(AS, "EITHER", None)
check("assay.EITHER exists", E is not None)
if E is not None:
    check("it says cells or nuclei, cell or nucleus",
          E.units == "cells or nuclei" and E.unit == "cell or nucleus", f"{E.units!r}, {E.unit!r}")
    check("it is neither assay", E.assay not in AS.ASSAYS, repr(E.assay))
woe = getattr(AS, "words_or_either", None)
check("assay.words_or_either exists", woe is not None)
if woe is not None:
    check("an absent assay gets EITHER's words, not an assay's",
          all(woe(v).units == "cells or nuclei" for v in (None, "", "  ")))
    check("a known assay gets its own", woe("sn").units == "nuclei" and woe("scrna").units == "cells")
    try:
        woe("bulk")
        check("an UNRECOGNISED assay still raises: unknown is not unrecognised", False, "returned")
    except ValueError:
        check("an UNRECOGNISED assay still raises: unknown is not unrecognised", True)

work = Path(tempfile.mkdtemp(prefix="scanno_words_"))
try:
    db, tree = write_corpus(work / "corpus.db"), write_tree(work / "tree.json")
    base = ["--tree", tree, "--db", db, "--species", SPECIES, "--tissue", TISSUE]

    def sample(name, embedding=True, merge_dc=False, keep_dc=30):
        """One sample with two clusterings, leiden_1p0 and the finer leiden_2p0. With `merge_dc`
        its few dendritic cells sit inside the monocyte cluster at 1.0 and alone at 2.0, which is
        the shape a rescue searches for and a joint route proposes to correct."""
        A = make_object(name, n_per_type=30, embedding=embedding)
        c = A.obs["cluster"].astype(str).to_numpy()
        if keep_dc < 30:
            keep = np.ones(A.n_obs, bool)
            keep[np.flatnonzero(c == "4")[keep_dc:]] = False
            A = A[keep].copy()
            c = A.obs["cluster"].astype(str).to_numpy()
        A.obs["leiden_1p0"] = pd.Categorical(np.where(c == "4", "3", c) if merge_dc else c)
        A.obs["leiden_2p0"] = pd.Categorical(
            c if merge_dc else np.array([f"{x}{i % 2}" for i, x in enumerate(c)]))
        p = work / "in" / f"{name}.h5ad"
        p.parent.mkdir(exist_ok=True)
        A.write_h5ad(p)
        return p

    s1, s2 = sample("S1"), sample("S2", embedding=False, merge_dc=True, keep_dc=8)
    rungs = ["--cluster-key", "leiden_1p0", "--cluster-key", "leiden_2p0", "--resolve"]
    rescue_args = ["--label-key", "scanno_resolved_path", "--unit-key", "sample",
                   "--sweep-prefix", "scanno_resolved_path_r", "--cluster-prefix", "leiden_"]

    # ---------------------------------------------------------------- the snrna cohort, pass 1
    out_bg = here("background", "--h5ad", s1, s2, "--cluster-key", "leiden_1p0", "--out",
                  work / "store.npz", "--species", SPECIES, "--tissue", TISSUE, "--assay", "snrna")
    store = ["--store", work / "store.npz"]
    p1 = {}
    for s, src in (("S1", s1), ("S2", s2)):
        p1[s] = here("annotate", "--h5ad", src, *rungs, *base, *store, "--assay", "snrna",
                     "--out-h5ad", work / "p1" / f"{s}.h5ad")
    scope_out = here("scope", "--h5ad", work / "p1" / "S1.h5ad", work / "p1" / "S2.h5ad",
                     "--tree", tree, "--out", work / "scope.json",
                     "--out-l1-tree", work / "l1.json")

    # ------------------------------------------------------------------------------------ B
    print("\nB - scope and rescue refuse a cohort of two assays, and an unrecognised one, by name")
    sc_src = here("annotate", "--h5ad", s1, *rungs, *base, "--assay", "scrna",
                  "--background-from-clusters", "--out-h5ad", work / "p1sc" / "S1.h5ad")
    mixed = [work / "p1sc" / "S1.h5ad", work / "p1" / "S2.h5ad"]
    check("an scrna object to mix in was annotated", sc_src[0] == 0 and mixed[0].exists(),
          f"rc={sc_src[0]}")
    bulk = work / "bulk" / "S2.h5ad"
    bulk.parent.mkdir()
    B_ = ad.read_h5ad(work / "p1" / "S2.h5ad")
    B_.uns["scanno_context"] = {**dict(B_.uns.get("scanno_context", {})), "assay": "bulk"}
    B_.write_h5ad(bulk)
    for cmd, extra in (("scope", ["--tree", tree, "--out", work / "mx" / "scope.json"]),
                       ("rescue", rescue_args + ["--out", work / "mx" / "rescue.json"])):
        rc, out = cli(cmd, "--h5ad", *mixed, *extra)
        line = [ln for ln in out.splitlines() if "REFUSE" in ln][:1]
        check(f"{cmd}: objects annotated under two assays are REFUSED (exit 2)",
              rc == 2 and "REFUSE" in out, f"rc={rc}")
        check(f"{cmd}: the refusal names both assays and each object",
              "scrna" in out and "snrna" in out and "S1" in str(line) and "S2" in str(line), line)
        check(f"{cmd}: STATUS records it as assay_mixed",
              status_code(work / "mx", cmd) == "assay_mixed", repr(status_code(work / "mx", cmd)))
        rc, out = cli(cmd, "--h5ad", work / "p1" / "S1.h5ad", bulk,
                      *[str(x).replace("/mx/", "/bk/") for x in extra])
        check(f"{cmd}: an assay this tool does not know is REFUSED (exit 2), naming it",
              rc == 2 and "REFUSE" in out and "bulk" in out, f"rc={rc}")
        check(f"{cmd}: STATUS records it as assay_unknown",
              status_code(work / "bk", cmd) == "assay_unknown", repr(status_code(work / "bk", cmd)))
    rc, _ = here("scope", "--h5ad", work / "p1" / "S1.h5ad", work / "p1" / "S2.h5ad",
                 "--tree", tree, "--out", work / "one" / "scope.json")
    check("CONTROL: scope over one assay runs (exit 0)", rc == 0, f"rc={rc}")
    rc, _ = here("rescue", "--h5ad", work / "p1" / "S1.h5ad", work / "p1" / "S2.h5ad",
                 *rescue_args, "--out", work / "one" / "rescue.json")
    check("CONTROL: rescue over one assay runs (exit 0)", rc == 0, f"rc={rc}")
    bare = []
    for s in ("S1", "S2"):
        A_ = ad.read_h5ad(work / "p1" / f"{s}.h5ad")
        A_.uns.pop("scanno_context", None)
        bare.append(work / "bare" / f"{s}.h5ad")
        bare[-1].parent.mkdir(exist_ok=True)
        A_.write_h5ad(bare[-1])
    rc_bs, out_bs = here("scope", "--h5ad", *bare, "--tree", tree, "--out", work / "bare" / "scope.json")
    check("objects that record no assay: scope runs, as report accepts them (exit 0)", rc_bs == 0,
          f"rc={rc_bs}")
    rc_br, out_br = here("rescue", "--h5ad", *bare, *rescue_args, "--out", work / "bare" / "r.json")
    check("objects that record no assay: rescue runs (exit 0)", rc_br == 0, f"rc={rc_br}")

    # ---------------------------------------------------------------- the snrna cohort, pass 2
    sc = json.loads((work / "scope.json").read_text(encoding="utf-8"))
    sc["nodes"]["Myeloid"]["verdict"] = "FORCE"            # FORCE a node the walk descends through
    (work / "scope_force.json").write_text(json.dumps(sc), encoding="utf-8")
    p2 = {}
    for s, src in (("S1", s1), ("S2", s2)):
        p2[s] = here("annotate", "--h5ad", src, *rungs, "--scope", work / "scope_force.json",
                     "--l1-tree", work / "l1.json", *base, *store, "--assay", "snrna",
                     "--out-h5ad", work / "p2" / f"{s}.h5ad")
    hint = here("annotate", "--h5ad", s1, "--cluster-key", "leiden_1p0", *base, *store,
                "--assay", "snrna")
    resc = here("rescue", "--h5ad", work / "p2" / "S1.h5ad", work / "p2" / "S2.h5ad", *rescue_args,
                "--out-dir", work / "rescued", "--out", work / "rescue" / "rescue.json",
                "--out-table", work / "rescue" / "impact.csv",
                "--out-report", work / "rescue" / "rescue.html")
    # route A: the per-sample annotations as one object; route B: one joint clustering, annotated
    A1, A2 = ad.read_h5ad(work / "p1" / "S1.h5ad"), ad.read_h5ad(work / "p1" / "S2.h5ad")
    ad.concat([A1, A2], merge="same", uns_merge="same").write_h5ad(work / "routeA.h5ad")
    J = ad.concat([ad.read_h5ad(s1), ad.read_h5ad(s2)], merge="same")
    J.obs["joint"] = pd.Categorical(J.obs["cluster"].astype(str))
    J.var["gene_symbol"] = ad.read_h5ad(s1).var["gene_symbol"].values
    J.write_h5ad(work / "joint_in.h5ad")
    here("annotate", "--h5ad", work / "joint_in.h5ad", "--cluster-key", "joint", *base, *store,
         "--assay", "snrna", "--out-h5ad", work / "routeB.h5ad")
    rep = work / "cmp" / "report"
    comp = here("compare", "--a", work / "routeA.h5ad", "--b", work / "routeB.h5ad",
                "--path-key", "scanno_path", "--sample-key", "sample", "--cluster-key", "joint",
                "--out", work / "cmp" / "cmp.json", "--out-h5ad", work / "cmp" / "joint.h5ad",
                "--out-report", rep / "joint_route.html", "--verdicts", work / "cmp" / "verdicts.json")
    try:
        payload = json.loads((rep / "joint_route.json").read_text(encoding="utf-8"))
        cands = payload["compare"]["merge_candidates"]["candidates"]
    except (OSError, ValueError, KeyError):
        payload, cands = {}, []
    jrev = here("joint-review", "--payload", rep / "joint_route.json", "--verdict",
                f"{cands[0]['cluster'] if cands else '0'}=adopt:the agreement and the sweep",
                "--out", work / "cmp" / "rev.json", "--out-report", rep / "joint_route2.html")
    emb = here("embed", "--h5ad", work / "p2" / "S1.h5ad", work / "p2" / "S2.h5ad",
               "--out", work / "emb" / "joint.h5ad", "--sample-key", "sample")
    (work / "deliv" / "annotated").mkdir(parents=True)
    shutil.copy(work / "p2" / "S1.h5ad", work / "deliv" / "annotated" / "S1_annotated.h5ad")
    rdme = here("readme", "--dir", work / "deliv")
    rprt = here("report", "--h5ad", work / "p2" / "S1.h5ad", work / "p2" / "S2.h5ad",
                "--out", work / "rep", "--no-per-sample")
    lab = here("lab", "--h5ad", work / "p2" / "S2.h5ad")
    atlas = make_object("S3", n_per_type=30)
    leaf = {str(i): name for i, name in enumerate(PANELS)}
    atlas.obs["label"] = pd.Categorical([leaf[c] for c in atlas.obs["cluster"].astype(str)])
    atlas.var_names = list(atlas.var["gene_symbol"])
    atlas.write_h5ad(work / "atlas.h5ad")
    (work / "manifest.tsv").write_text("path\tsource_id\tlabel_key\tprovenance\n"
                                       f"{work / 'atlas.h5ad'}\tsrc1\tlabel\texpert\n",
                                       encoding="utf-8")
    cal = here("calibrate", "--manifest", work / "manifest.tsv", *base, "--assay", "snrna",
               "--out", work / "cal")
    q = write_cohort(work / "qc.h5ad")
    Q = ad.read_h5ad(q)
    Q.uns["scqc"] = {"declaration": "synthetic"}
    Q.write_h5ad(q)
    clus = here("cluster", "--h5ad", q, "--split-by", "sample", "--out-dir", work / "split",
                "--resolutions", "1.0")

    # ------------------------------------------------------------------------------------ C
    print("\nC - every command's prints and documents, on nuclei")
    ran = {"background": out_bg, "annotate --resolve (S1)": p1["S1"],
           "annotate --resolve (S2)": p1["S2"], "scope": scope_out,
           "annotate --scope --l1-tree (S1)": p2["S1"], "annotate --scope --l1-tree (S2)": p2["S2"],
           "annotate, nothing written": hint, "rescue": resc, "compare": comp,
           "joint-review": jrev, "embed": emb, "readme": rdme, "report": rprt,
           "calibrate": cal}
    for name, (rc, out) in ran.items():
        check(f"{name}: exits 0", rc == 0, f"rc={rc} {out.strip().splitlines()[-2:]}")
        worded(name, out, None if name == "readme" else "nuclei")   # readme prints only a path
    check("rescue renamed something, so its moved line was read",
          "renamed" in resc[1] and not re.search(r"\b0 \w+ renamed", resc[1]))
    check("compare found a candidate, so its candidate lines were read", bool(cands))
    check("annotate --out-h5ad does not say nothing was written",
          all("nothing was written" not in out for _rc, out in p2.values()))
    check("annotate with nothing written says so", "nothing was written" in hint[1])
    for name, (rc, out) in (("cluster --split-by", clus), ("lab", lab),
                            ("scope over objects that record no assay", (rc_bs, out_bs)),
                            ("rescue over objects that record no assay", (rc_br, out_br))):
        check(f"{name}: exits 0", rc == 0, f"rc={rc}")
        worded(name, out, "either")
    try:
        rj = json.loads((work / "rescue" / "rescue.json").read_text(encoding="utf-8"))
        worded("rescue.html, its record's quoted strings aside",
               page(work / "rescue" / "rescue.html", (rj.get("limit", ""), rj.get("undecided", ""))),
               "nuclei")
    except (OSError, ValueError):
        check("rescue.json and rescue.html were written", False)
    if payload:
        quoted = [payload["record"].get("gating", ""), payload["record"].get("limit", ""),
                  ((payload["compare"]["merge_candidates"].get("lost_labels") or {}).get("limit", "")),
                  ((payload.get("review") or {}).get("limit", ""))]
        worded("joint_route.html, its record's quoted strings aside",
               page(rep / "joint_route.html", quoted), "nuclei")
        worded("joint_route.html re-rendered by joint-review", page(rep / "joint_route2.html", quoted
               + [json.loads((work / "cmp" / "rev.json").read_text()).get("limit", "")]), "nuclei")
        worded("review_request.md", (rep / "review_request.md").read_text(encoding="utf-8"), "nuclei")
    else:
        check("compare wrote joint_route.json", False)
    for name, p in (("README.md written by readme", work / "deliv" / "README.md"),
                    ("README.md written by report", work / "rep" / "README.md")):
        worded(name, p.read_text(encoding="utf-8") if p.exists() else "", "nuclei")
finally:
    shutil.rmtree(work, ignore_errors=True)

# ---------------------------------------------------------------------------------------- D
print("\nD - the summary formatters, given the words, at the branches the cohort did not reach")
W = AS.words("snrna")


def lines_of(fn, *a, **k):
    try:
        return "\n".join(fn(*a, words=W, **k))
    except TypeError as e:                       # a formatter that takes no words
        return f"TypeError: {e} cell"


from scanno import emit, force  # noqa: E402

rec = {"nodes": ["Myeloid"], "n_forced": 2, "n_cells_forced": 46, "unforceable": {
           "7": {"node": "Myeloid", "reached": "Myeloid/X", "reason": "no scorer"}},
       "assigned": {"3": {"node": "Myeloid", "steps": ["Myeloid/Monocyte"], "margins": [0.12],
                          "n_cells": 38, "force_depth": 1},
                    "5": {"node": "Myeloid", "steps": ["Myeloid/Monocyte", "Myeloid/Monocyte/A"],
                          "margins": [0.1, 0.4], "n_cells": 8, "force_depth": 2}}}
txt = lines_of(force.format_force, rec, gap_min=0.3)
worded("force.format_force, two clusters forced, one twice", txt, "nuclei")
res_rec = {"by_origin": {o: 1 for o in force.RESOLVED_ORIGINS},
           "cells_by_origin": {o: 10 for o in force.RESOLVED_ORIGINS},
           "unresolvable": {"9": {"from": "root", "why": "no scorer"}}}
worded("force.format_resolved, every origin", lines_of(force.format_resolved, res_rec), "nuclei")
bad_scope = {"nodes": {"root": {"verdict": "KEEP"}, "Myeloid": {"verdict": "FORCE"},
                       "Myeloid/Monocyte": {"verdict": "FORCE"},
                       "Lymphoid/T cell": {"verdict": "FORCE"}}}
try:
    why = "\n".join(force.check_scope(bad_scope, TREE, words=W))
except TypeError as e:
    why = f"TypeError: {e} cell"
check("check_scope finds the problems it is given", "FORCE" in why, why[:120])
worded("force.check_scope, a FORCE leaf and a FORCE child", why, "nuclei")
l1 = {"column": "scAnno_L1", "tree": "l1.json", "labels": ["A", "B"], "n_cells": 100,
      "source": "independent", "replaced": "derived", "n_disagree": 4,
      "disagreements": {"A -> B": 4}}
worded("emit.format_independent_l1, the two disagreeing", lines_of(emit.format_independent_l1, l1),
       "nuclei")
O = ad.AnnData(X=np.zeros((10, 2), dtype="float32"))
O.obs["lab"] = pd.Categorical([f"t{i}" for i in range(10)])
try:
    ready = "\n".join(m for _lvl, m in emit.lab_readiness(O, "lab", words=W))
except TypeError as e:
    ready = f"TypeError: {e} cell"
worded("emit.lab_readiness, a label with a level per observation and no embedding", ready, "nuclei")

# ---------------------------------------------------------------------------------------- E
print("\nE - every literal still saying cell(s) as a unit is declared, and KNOWN_ISSUES names it")
#: (file, a piece of the literal, why it stays). RECORD: written into a data product that
#: reproductions compare, or read back as a key; changing it changes the product. NOT A UNIT: the
#: word is something else there. Each RECORD carries the words KNOWN_ISSUES.md must name it by.
KEEP = [
    ("rescue.py", "THAT CLUSTER'S CELLS take it", "RECORD", "rescue.json"),
    ("rescue.py", "a renamed cell was in a cluster", "RECORD", "rescue.json"),
    ("compare.py", "holding cells from", "RECORD", "merge_candidates"),
    ("compare.py", "these cells group with", "RECORD", "merge_candidates"),
    ("compare.py", "averaged over exactly the cells that would move", "RECORD", "merge_candidates"),
    ("joint.py", "RECOVERED cells moved onto", "RECORD", "scanno_joint_route"),
    ("joint.py", "A corrected cell GROUPS with cells", "RECORD", "scanno_joint_route"),
    ("joint.py", "cells differing from the one above", "RECORD", "cells differing from the one above"),
    ("cli.py", "cells differing from the one above", "RECORD", "cells differing from the one above"),
    ("embed.py", "computed over the pooled cells", "RECORD", "scanno_embed"),
    ("scope.py", "cells", "RECORD", "`cells`"),
    ("document.py", "cells", "RECORD", "`cells`"),
    ("exclude.py", "cell", "NOT A UNIT", "the exclusion record's mode, per observation"),
    ("emit.py", "cell", "NOT A UNIT", "a column-name hint: keep any obs column named like it"),
    ("document.py", ".cell{", "NOT A UNIT", "a CSS class"),
    ("document.py", "class='cell'", "NOT A UNIT", "a CSS class"),
]
known = (ROOT / "KNOWN_ISSUES.md").read_text(encoding="utf-8")
offenders, used = [], set()
for f in sorted((ROOT / "scanno").glob("*.py")):
    if f.name == "assay.py":                     # the table itself names both units
        continue
    tree_ = ast.parse(f.read_text(encoding="utf-8"))
    docs = {id(n.body[0].value) for n in ast.walk(tree_)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    for n in ast.walk(tree_):
        if not (isinstance(n, ast.Constant) and isinstance(n.value, str)) or id(n) in docs:
            continue
        if not cells_said(n.value):
            continue
        k = next((i for i, (ff, piece, _w, _n) in enumerate(KEEP)
                  if ff == f.name and (n.value == piece if piece in ("cells", "cell") else piece in n.value)),
                 None)
        if k is None:
            offenders.append(f"{f.name}:{n.lineno} {n.value.strip()[:60]!r}")
        else:
            used.add(k)
check("no undeclared literal says cell(s) as a unit", not offenders,
      f"{len(offenders)}, e.g. " + "; ".join(offenders[:5]))
stale = [f"{KEEP[i][0]}: {KEEP[i][1]!r}" for i in range(len(KEEP)) if i not in used]
check("every declared literal is still there (the list is exact, not a licence)", not stale,
      "; ".join(stale[:4]))
unnamed = sorted({named for _f, _p, why, named in KEEP if why == "RECORD" and named not in known})
check("KNOWN_ISSUES.md names every recorded string that still says cell(s)", not unnamed,
      ", ".join(unnamed))

print("\n" + "=" * 64)
if fails:
    print(f"assay words: {len(fails)} FAILED - " + "; ".join(fails[:10]))
    raise SystemExit(1)
print("assay words: every command names its unit by the assay it knows, or names both")
