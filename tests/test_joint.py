"""The joint route: a third column that corrects the second and replaces nothing.

Four properties, each one a way this could quietly become something else:

  1. it corrects EXACTLY the cells a candidate names, and never a withheld nucleus - a cell the
     upstream flag withheld was never annotated, so there is no call to move.
  2. the column it corrects is UNCHANGED, so reverting is a column drop rather than a re-run.
  3. the summary is read off the two delivered arrays, not predicted from the candidate rows,
     so it is an independent check on the reconciliation rather than a restatement of it.
  4. nothing in the module names a project, a tissue, a species or a design factor.

    python tests/test_joint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"SKIP: needs {e.name}")
    raise SystemExit(0)

from scanno.compare import compare  # noqa: E402
from scanno.joint import CORRECTED, KEPT, document, reconcile, summarise  # noqa: E402

DC, MP, FB = "Immune/Myeloid/Dendritic cell", "Immune/Myeloid/Macrophage", "Stromal/Fibroblast"

# s1 resolved six dendritic cells; s2 and s3 had theirs absorbed and carry no dendritic call
# anywhere. One withheld nucleus sits in the same cluster and must not be touched.
lab_a = [DC] * 6 + [MP] * 4 + ["EXCLUDED"] + [MP] * 4 + [FB] * 6
lab_b = [DC] * 15 + [FB] * 6
sams = ["s1"] * 6 + ["s2"] * 5 + ["s3"] * 4 + ["s1", "s1", "s2", "s2", "s3", "s3"]
clus = ["J1"] * 15 + ["J0"] * 6
ids = [f"c{i}" for i in range(21)]
A = pd.DataFrame({"lab": lab_a}, index=ids)
B = pd.DataFrame({"jp": lab_b, "sample": sams, "cl": clus}, index=ids)
res = compare(A, B, path_key="lab", path_key_b="jp", sample_key="sample", cluster_key="cl")
cands = res["merge_candidates"]["candidates"]

print("\n1 - it corrects exactly the cells the candidate names")
new, origin, rec = reconcile(np.array(lab_a), np.array(clus), np.array(sams), cands)
check("one candidate was found to apply", len(cands) == 1, str(cands))
check("the eight absorbed cells are corrected", rec["n_corrected"] == 8, str(rec["n_corrected"]))
check("they now carry the joint route's label",
      list(new[6:10]) == [DC] * 4 and list(new[11:15]) == [DC] * 4)
check("s1's own dendritic calls are untouched", list(new[0:6]) == [DC] * 6)
check("and the other cluster is untouched", list(new[15:]) == [FB] * 6)

print("\n2 - a withheld nucleus is never corrected")
# It was never annotated, so there is no call to move. Correcting it would invent one.
check("the EXCLUDED cell keeps its sentinel", new[10] == "EXCLUDED")
check("and is marked as kept, not corrected", origin[10] == KEPT)

print("\n3 - origin names every corrected cell and nothing else")
check("the origin column counts what the record claims",
      int((origin == CORRECTED).sum()) == rec["n_corrected"])
check("every other cell is kept",
      int((origin == KEPT).sum()) == len(lab_a) - rec["n_corrected"])
check("only two origin values exist", set(origin.tolist()) == {KEPT, CORRECTED})

print("\n4 - the column it corrects is UNCHANGED, so reverting is a column drop")
check("the input array was not mutated", lab_a[6] == MP and lab_a[11] == MP)
check("the record says so in words", "beside" in rec["limit"] or "column drop" in rec["limit"])

print("\n5 - the summary is read off the arrays, not predicted from the candidates")
summ = summarise(np.array(lab_a), new, np.array(sams))
check("it counts the same number of changed cells", summ["n_changed"] == rec["n_corrected"])
per = {r["label"]: r for r in summ["per_label"]}
check("the label that gained is named with its delta", per[DC]["n_delta"] == 8)
check("and the label that lost it too", per[MP]["n_delta"] == -8)
check("no nucleus is created or destroyed",
      sum(r["n_before"] for r in summ["per_label"])
      == sum(r["n_after"] for r in summ["per_label"]))
ps = {(r["sample"], r["label"]): r for r in summ["per_sample"]}
check("a per-sample denominator is the sample's own total",
      ps[("s2", DC)]["n_sample_total"] == sams.count("s2"))
check("and the percentage-point delta is the difference of its own two shares",
      abs(ps[("s2", DC)]["pct_after"] - ps[("s2", DC)]["pct_before"]
          - ps[("s2", DC)]["pct_delta"]) < 0.002)

print("\n6 - nothing is gated, and the page says it is not")
check("every candidate was applied", rec["n_candidates_applied"] == len(cands))
check("and the record states that dominance does not decide",
      "does not gate" in rec["gating"] or "not gate" in rec["gating"])

print("\n7 - the document renders, and carries its limits")
html = document({"generated": "2026-01-01T00:00:00", "version": "0.0.0",
                 "a_name": "A.h5ad", "b_name": "B.h5ad", "forced_key": "lab",
                 "out_key": "lab_joint",
                 "columns": [{"column": "lab", "what it is": "x",
                              "cells differing from the one above": ""}],
                 "record": rec, "summary": summ, "compare": res})
check("it is one self-contained page", html.startswith("<!doctype html>") and "<style>" in html)
check("it fetches nothing", "http://" not in html and "https://" not in html)
check("it says co-membership is not identity", "co-membership" in html)
check("it reports what the joint route ABSORBED", "could not resolve" in html)

print("\n8 - it ADDS a column and refuses to replace one")
try:
    import anndata as ad

    from scanno.emit import annotate_joint
    X = np.zeros((len(lab_a), 3), dtype="float32")
    Ad = ad.AnnData(X=X)
    Ad.obs_names = ids
    Ad.obs["lab"] = pd.Categorical(lab_a)
    info = annotate_joint(Ad, new, origin, rec, key="lab_joint")
    check("the new column is written", "lab_joint" in Ad.obs)
    check("with its origin beside it", "lab_joint_origin" in Ad.obs)
    check("the corrected column is still there and still itself",
          list(Ad.obs["lab"].astype(str)) == lab_a)
    check("provenance lands in uns", Ad.uns["scanno_joint_route"]["key"] == "lab_joint")
    try:
        annotate_joint(Ad, new, origin, rec, key="lab")
        check("writing over an existing column REFUSES", False, "it did not")
    except ValueError:
        check("writing over an existing column REFUSES", True)

    # ROUND-TRIP THROUGH h5ad. Everything above passes on an in-memory object, and the first
    # real run still died at write: `uns` held a LIST OF DICTS, which anndata cannot represent
    # and reports as "Can't implicitly convert non-string objects to strings" - naming the key
    # and not the shape. A record that cannot be written is not provenance.
    import tempfile

    from scanno.emit import write_h5ad
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "rt.h5ad"
        write_h5ad(Ad, fp)
        R = ad.read_h5ad(fp)
        check("the object writes and reads back", R.n_obs == Ad.n_obs)
        check("the joint column survives the round trip",
              list(R.obs["lab_joint"].astype(str)) == list(Ad.obs["lab_joint"].astype(str)))
        check("so does the origin column",
              list(R.obs["lab_joint_origin"].astype(str)) == list(origin))
        u = R.uns["scanno_joint_route"]
        check("and the provenance survives it", int(u["n_corrected"]) == rec["n_corrected"])
        check("including every cluster it moved",
              len([k for k in u["moved"] if not k.startswith("_")]) == len(rec["moved"]))
        # The labels this package writes are PATHS. A dict keyed by one cannot be an HDF5
        # group, so the key has to travel as a value - and the round trip must still be able
        # to say what moved to what.
        mv = u["moved"]["0000"]
        got = {e["key"]: int(e["value"]) for k, e in mv["from"].items() if not k.startswith("_")}
        check("a label-keyed record survives with its slashes intact",
              got == rec["moved"][0]["from"], f"{got} vs {rec['moved'][0]['from']}")
        check("and says how it was encoded", "_encoding" in mv["from"])
except ImportError:
    print("  SKIP the h5ad checks: needs anndata")

print("\n9 - nothing in the module names a project, a tissue or a design")
import scanno.joint as _j  # noqa: E402
src = Path(_j.__file__).read_text(encoding="utf-8").lower()
check("no species, tissue or study vocabulary",
      not any(w in src for w in ("mouse", "human", "heart", "sambo", "hfd", "aging",
                                 "cardiomyocyte", "pbmc")),
      "found project vocabulary")
import inspect  # noqa: E402
check("reconcile cannot be given a design",
      not any(w in p_ for p_ in inspect.signature(reconcile).parameters
              for w in ("factor", "design", "condition", "arm", "group")))

print("\n10 - the predicted summary and the delivered column agree, label for label")
# Two routes to one quantity: `compare._impact` predicts from the candidate rows, and
# `joint.summarise` counts the arrays that were actually written. They disagreed on the first
# real object - 552 predicted for a label the column held 455 at - because a label that GAINS
# from one candidate and LOSES to another was only ever credited with the gain. Binding them
# here is what stops that coming back, and the fixture makes one label do both.
lab2 = [DC] * 6 + [MP] * 4 + [MP] * 4 + [DC] * 5 + [FB] * 3
lb2 = [DC] * 14 + [FB] * 8
sm2 = ["s1"] * 6 + ["s2"] * 4 + ["s3"] * 4 + ["s1"] * 5 + ["s2"] * 3
cl2 = ["J1"] * 14 + ["J2"] * 8
i2 = [f"d{i}" for i in range(22)]
Ax = pd.DataFrame({"lab": lab2}, index=i2)
Bx = pd.DataFrame({"jp": lb2, "sample": sm2, "cl": cl2}, index=i2)
rx = compare(Ax, Bx, path_key="lab", path_key_b="jp", sample_key="sample", cluster_key="cl")
cx = rx["merge_candidates"]["candidates"]
nx, ox, recx = reconcile(np.array(lab2), np.array(cl2), np.array(sm2), cx)
sx = summarise(np.array(lab2), nx, np.array(sm2))
impx = rx["merge_candidates"]["impact"]
truth = {r["label"]: r["n_after"] for r in sx["per_label"]}
pred = {r["label"]: r["n_route_a_after"] for r in impx["labels"]}
bad = {k: (pred[k], truth.get(k)) for k in pred if truth.get(k) != pred[k]}
check("a label in this fixture both gains and loses",
      any(r["n_gained"] and r["n_lost"] for r in impx["labels"]),
      str([(r["label"].split(chr(47))[-1], r["n_gained"], r["n_lost"])
           for r in impx["labels"]]))
check("every PREDICTED after-count equals the DELIVERED one", not bad, str(bad))
check("gains and losses balance to zero",
      sum(r["n_delta"] for r in impx["labels"]) == 0,
      str([(r["label"].split(chr(47))[-1], r["n_delta"]) for r in impx["labels"]]))
check("and the total moved is the number of corrected cells",
      impx["n_cells_total"] == recx["n_corrected"],
      f'{impx["n_cells_total"]} vs {recx["n_corrected"]}')

print("\n" + "=" * 64)
if fails:
    print(f"joint: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print("joint OK - a third column that corrects the second, replaces nothing, gates nothing")
