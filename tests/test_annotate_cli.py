#!/usr/bin/env python3
"""`scanno annotate`'s guarantees, run through the real CLI instead of read off its source.

WHY THIS FILE EXISTS (single-cell-harness B1-034). These guarantees were asserted on the TEXT of
`cli._annotate`: how many times `classify(` and `apply_force(` are called and with which argument
names, that a guard's source position precedes `ad.read_h5ad(a.h5ad)`, which subparser a flag's
`add_argument` sits under, that `if force_paths:` appears. A test of code text fails when the code
is moved or reworded and the behaviour is unchanged, and passes when the text survives beside a
change that breaks the behaviour. Each property is now RUN, on the synthetic sample:

  1. an all-KEEP scope gives the labels no scope gives: the scope acts only where it says so;
  2. a scope's FORCE reaches the deep column and never the independent L1: the L1 column equals
     a separate walk of the depth-1 tree, under the same exclusion, which is also what "a second
     CALL, not a second mode" means;
  3. without --l1-tree the L1 column is derived from the path and says nothing about independence;
  4. a depth-2 --l1-tree is refused (exit 2, naming the tree) BEFORE the object is read;
  5. --l1-tree and --scope are annotate's options and not another command's;
  6. a FORCE node whose children cannot be scored: the cluster stranded on it keeps the walk's
     label with assignment `unforceable`, and the run completes (B1-033);
  7. a FORCED chain that stops on an internal node - the one refusal B1-033 keeps - still exits
     2 with nothing written (the push is broken on purpose, in-process).

The sample: tests/_cohort.py's five populations, clustered so that T and B cells share one cluster.
That cluster's profile sits between two siblings, so the walk stops on `Lymphoid` - a stranded
cluster, the shape FORCE exists for - while the other four reach their leaves.

    python tests/test_annotate_cli.py
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
    import anndata as ad
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"SKIP: needs {e.name}")
    raise SystemExit(0)

from _cohort import SPECIES, TISSUE, TREE, make_object, write_corpus  # noqa: E402
from scanno import cli as C  # noqa: E402
from scanno.scope import internal_nodes, truncate_tree  # noqa: E402


def here(*args):
    """The real `scanno` entry point, in this process, with everything it prints captured."""
    o, e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
        try:
            rc = C.main([str(x) for x in args])
        except SystemExit as x:                         # argparse: --help, a bad option
            rc = x.code if isinstance(x.code, int) else 1
    return rc, o.getvalue() + e.getvalue()


def scope_file(path, tree, force=()):
    """A `scanno scope --out` payload: every internal node KEEP, except those named FORCE."""
    nodes = {n: {"verdict": "FORCE" if n in force else "KEEP", "children_declared": kids}
             for n, kids in internal_nodes(tree).items()}
    Path(path).write_text(json.dumps({"nodes": nodes, "n_samples": 1, "removed_labels": {},
                                      "rule": {}}), encoding="utf-8")
    return path


work = Path(tempfile.mkdtemp(prefix="scanno_annotate_cli_"))
O = make_object("S1")
O.obs["cl"] = pd.Categorical(O.obs["cluster"].astype(str).map(
    {"0": "TB", "1": "TB", "2": "NK", "3": "MO", "4": "DC"}))
O.obs["qc_flag"] = pd.array([i % 10 == 0 for i in range(O.n_obs)], dtype="boolean")
O.write_h5ad(work / "S1.h5ad")
db = write_corpus(work / "corpus.db")
(work / "tree.json").write_text(json.dumps(TREE), encoding="utf-8")
(work / "l1.json").write_text(json.dumps(truncate_tree(TREE, 1)), encoding="utf-8")
BASE = ["--h5ad", work / "S1.h5ad", "--cluster-key", "cl", "--db", db, "--species", SPECIES,
        "--tissue", TISSUE, "--assay", "scrna", "--background-from-clusters",
        "--exclude-flag", "qc_flag"]
LABELS = ["scanno_cell_type", "scanno_path", "scanno_depth", "scanno_gap", "scanno_survival",
          "scanno_support", "scAnno_L1"]


def annotate(name, *extra, tree="tree.json"):
    rc, out = here("annotate", *BASE, "--tree", work / tree, *extra,
                   "--out-h5ad", work / f"{name}.h5ad")
    A = ad.read_h5ad(work / f"{name}.h5ad") if rc == 0 else None
    return rc, out, A


def same(a, b, cols):
    return [c for c in cols if c in a.obs or c in b.obs
            if c not in a.obs or c not in b.obs
            or a.obs[c].astype(str).tolist() != b.obs[c].astype(str).tolist()]


print("\n0 - the fixture strands one cluster on an internal node, and only one")
rc0, out0, A0 = annotate("plain")
check("annotate without a scope exits 0", rc0 == 0, out0[-300:])
if A0 is not None:
    tb = A0.obs["cl"].astype(str) == "TB"
    kept = tb & (A0.obs["scanno_path"].astype(str) != "EXCLUDED")
    check("the T+B cluster stops on Lymphoid", set(A0.obs.loc[kept, "scanno_path"].astype(str))
          == {"Lymphoid"}, str(set(A0.obs.loc[kept, "scanno_path"].astype(str))))
    check("the other four reach a leaf", all("/" in p for p in
                                            A0.obs.loc[~tb, "scanno_path"].astype(str)
                                            if p != "EXCLUDED"))

print("\n1 - an all-KEEP scope gives the labels no scope gives")
rc1, out1, A1 = annotate("keep", "--scope", scope_file(work / "keep.json", TREE))
check("exits 0", rc1 == 0, out1[-300:])
if A0 is not None and A1 is not None:
    check("every label column is identical", not same(A0, A1, LABELS), str(same(A0, A1, LABELS)))
    check("and so is the set of columns: a scope that forces nothing writes nothing more",
          list(A0.obs.columns) == list(A1.obs.columns),
          str(set(A0.obs.columns) ^ set(A1.obs.columns)))

print("\n2 - FORCE reaches the deep column and never the independent L1")
rc2, out2, A2 = annotate("forced", "--scope", scope_file(work / "force.json", TREE, ("Lymphoid",)),
                         "--l1-tree", work / "l1.json")
rcL, outL, AL = annotate("l1walk", tree="l1.json")
check("the forced run and the separate L1 walk exit 0", rc2 == 0 and rcL == 0,
      (out2 + outL)[-300:])
if A2 is not None and AL is not None and A0 is not None:
    tb = (A2.obs["cl"].astype(str) == "TB") & (A2.obs["scanno_path"].astype(str) != "EXCLUDED")
    check("the stranded cluster was pushed below Lymphoid",
          all(p.startswith("Lymphoid/") for p in A2.obs.loc[tb, "scanno_path"].astype(str)),
          str(set(A2.obs.loc[tb, "scanno_path"].astype(str))))
    check("and is marked forced", set(A2.obs.loc[tb, "scanno_assignment"].astype(str)) == {"forced"})
    check("the L1 column equals a separate walk of the depth-1 tree, cell for cell",
          A2.obs["scAnno_L1"].astype(str).tolist() == AL.obs["scanno_path"].astype(str).tolist())
    check("including the same withheld set",
          (A2.obs["scAnno_L1"].astype(str) == "EXCLUDED").equals(
              A0.obs["scanno_path"].astype(str) == "EXCLUDED"))
    check("and the object says the L1 is independent",
          dict(A2.uns.get("scAnno_L1_provenance", {})).get("source") == "independent",
          str(dict(A2.uns.get("scAnno_L1_provenance", {}))))

print("\n3 - without --l1-tree the L1 column is derived, and does not claim independence")
if A0 is not None:
    derived = [p.split("/")[0] for p in A0.obs["scanno_path"].astype(str)]
    check("scAnno_L1 is the path's first level", A0.obs["scAnno_L1"].astype(str).tolist() == derived)
    check("and no provenance calls it independent", "scAnno_L1_provenance" not in A0.uns)

print("\n4 - a depth-2 --l1-tree is refused before the object is read")
rc4, out4 = here("annotate", "--h5ad", work / "no_such_object.h5ad", "--cluster-key", "cl",
                 "--tree", work / "tree.json", "--l1-tree", work / "tree.json", "--db", db,
                 "--species", SPECIES, "--tissue", TISSUE, "--assay", "scrna",
                 "--background-from-clusters", "--status-dir", work / "deep",
                 "--out-h5ad", work / "deep" / "x.h5ad")
check("exit 2, a refusal - not exit 1 for the missing object", rc4 == 2, f"rc={rc4} {out4[-200:]}")
check("naming the tree and its depth", "tree.json" in out4 and "depth-2" in out4, out4[-300:])
try:
    st = json.loads((work / "deep" / "STATUS.annotate.json").read_text(encoding="utf-8"))
    check("STATUS records the refusal", st.get("status") == "refused", st.get("status"))
except (OSError, ValueError) as e:
    check("STATUS records the refusal", False, str(e))

print("\n5 - --l1-tree and --scope are annotate's options, and not another command's")
_r, ann = here("annotate", "--help")
_r, cmp_ = here("compare", "--help")
check("annotate --help offers --l1-tree and --scope", "--l1-tree" in ann and "--scope" in ann)
check("compare --help offers neither", "--l1-tree" not in cmp_ and "--scope" not in cmp_)

print("\n6 - a FORCE node whose children cannot be scored")
# Lymphoid's children match nothing in the corpus, so the walk can score no child of Lymphoid and
# the T+B cluster (with the T- and B-only ones' cells) has nowhere measured to go.
blind = json.loads(json.dumps(TREE))
for kid in blind["children"]["Lymphoid"]:
    blind["patterns"][kid] = ["no-such-cell-name"]
(work / "blind.json").write_text(json.dumps(blind), encoding="utf-8")
rc6, out6 = here("annotate", *BASE, "--tree", work / "blind.json",
                 "--scope", scope_file(work / "blind_force.json", blind, ("Lymphoid",)),
                 "--status-dir", work / "blind", "--out-h5ad", work / "blind" / "x.h5ad")
# It refused the whole object until single-cell-harness B1-033: one cluster FORCE could not push
# withheld every label of the sample. The walk's own truncation is a true label.
check("the run completes (exit 0)", rc6 == 0, f"rc={rc6} {out6[-300:]}")
check("and says so: REVIEW, naming the assignment", "REVIEW" in out6 and "unforceable" in out6)
st6 = {}
try:
    st6 = json.loads((work / "blind" / "STATUS.annotate.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    pass
check("STATUS is ok", st6.get("status") == "ok", str(st6.get("status")))
if (work / "blind" / "x.h5ad").exists():
    B6 = ad.read_h5ad(work / "blind" / "x.h5ad")
    lym = B6.obs["scanno_path"].astype(str) == "Lymphoid"
    check("the stranded cells keep the walk's label", bool(lym.any()))
    check("with assignment `unforceable`, and no other cell has it",
          set(B6.obs.loc[lym, "scanno_assignment"].astype(str)) == {"unforceable"}
          and set(B6.obs.loc[~lym, "scanno_assignment"].astype(str)) <= {"gap", "EXCLUDED"})
    check("and the object's provenance names them",
          dict(dict(B6.uns["scanno_assignment_provenance"])["clusters_by_assignment"]).get(
              "unforceable", 0) > 0)
else:
    check("the object is written", False)

print("\n7 - a forced chain that stops on an internal node is still REFUSED, through the CLI")
# The one refusal B1-033 keeps: a FORCED row ending on an internal node is a recursion that stopped
# short, and would deliver a compartment name one level down. The recursion cannot produce one, so
# `force.push_to_leaf` is replaced, in this process, by a broken one: a single step that lands back
# on the FORCE node and reports no refusal. Before B1-034 this was an AST read of `if stuck:` in
# cli._annotate; the verifier showed `if False and stuck:` passing every suite (W021 fix-up).
import scanno.force as _force  # noqa: E402

_real_push = _force.push_to_leaf


def _broken_push(cid, path, first, tree=None, scorer=None, sep="/"):
    return [{"to": path.split(sep)[-1], "path": path, "margin": 0.0, "survival": 1.0,
             "cover": 1.0}], ""


_force.push_to_leaf = _broken_push
try:
    rc7, out7 = here("annotate", *BASE, "--tree", work / "tree.json",
                     "--scope", scope_file(work / "force7.json", TREE, ("Lymphoid",)),
                     "--status-dir", work / "short", "--out-h5ad", work / "short" / "x.h5ad")
finally:
    _force.push_to_leaf = _real_push
check("the run refuses (exit 2)", rc7 == 2 and "REFUSE" in out7, f"rc={rc7} {out7[-300:]}")
st7 = {}
try:
    st7 = json.loads((work / "short" / "STATUS.annotate.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    pass
check("STATUS records the refusal", st7.get("status") == "refused", str(st7.get("status")))
check("and nothing was written", not (work / "short" / "x.h5ad").exists())
check("the real push is back for whatever runs next", _force.push_to_leaf is _real_push)

import shutil  # noqa: E402

shutil.rmtree(work, ignore_errors=True)

print("\n" + "=" * 64)
if fails:
    print(f"annotate cli: {len(fails)} FAILED - " + "; ".join(fails[:8]))
    raise SystemExit(1)
print("annotate cli: scope, force, the independent L1 and the deep-tree refusal hold when run")
