"""The resolution sweep — evidence about an annotation, and nothing that changes one.

A joint clustering annotated at ONE granularity cannot say whether a correction is a property of
the cells or of the resolution that was chosen. Walking every resolution can. The whole mechanism
is that measurement: extra label columns, one agreement number per cell, and one more field on
each merge candidate.

WHAT THIS SUITE EXISTS TO STOP, and it is a defect that shipped for one afternoon. The sweep
first VOTED a consensus label and made it route B's. That column was per CELL, where an
annotation is per CLUSTER, and `joint.reconcile` reads "route B delivers L" off the label column
on the assumption that the two are the same set — true for any single-resolution annotation and
false for a vote. The result: a label with no cluster of its own anywhere at the reconciled
resolution was spared ABSORPTION, while being invisible to RECOVERY, and survived at exactly the
per-sample route's own count because a column mentioned the string. Nothing decided that; two
definitions of one phrase did.

So the sweep reports and does not decide. Which is also what `docs/PRINCIPLES.md` §3 asks of any
statistic whose agreement with correctness has not been shown, and this one's has not.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scanno.compare import compare
from scanno.context import sweep_stem
from scanno.joint import review_prompt
from scanno.resolution import sweep_agreement

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


print("sweep_agreement: a number about a call, not a new call")

coarse = np.array(["Big"] * 10)
fine = np.array(["Big"] * 8 + ["Rare"] * 2)
SWEEP = {0.25: coarse, 0.5: coarse, 0.75: coarse, 1.0: coarse,
         1.25: coarse, 1.5: coarse, 1.75: fine, 2.0: fine}

# Measured against the resolution the run DELIVERED. Two references, two honest answers about
# the same sweep - which is the point: the number describes a call, so it needs one.
a_coarse = sweep_agreement(SWEEP, coarse)
a_fine = sweep_agreement(SWEEP, fine)
check("a delivered coarse call is agreed by 6 of 8 on the contested cells",
      np.allclose(a_coarse[-2:], 6 / 8), f"got {a_coarse[-2:]}")
check("a delivered fine call is agreed by 2 of 8 on the same cells",
      np.allclose(a_fine[-2:], 2 / 8), f"got {a_fine[-2:]}")
check("cells nothing disagrees about read 1.0 under either reference",
      np.allclose(a_coarse[:8], 1.0) and np.allclose(a_fine[:8], 1.0))
check("it returns numbers only - no label is produced",
      a_coarse.dtype.kind == "f" and a_coarse.shape == (10,), str(a_coarse.dtype))

# 0.0 is a real state and must not be confused with an absent measurement: it says the delivered
# call is unique to the resolution that made it.
lonely = sweep_agreement({0.5: coarse, 1.0: coarse}, np.array(["Odd"] * 10))
check("a call no resolution shares reads 0.0, which is a finding", np.allclose(lonely, 0.0))

d = sweep_agreement({0.5: np.array(["S/Fib/Matri"]), 1.0: np.array(["S/Fib/Quiescent"])},
                    np.array(["S/Fib/Matri"]), depth=1)
check("depth truncates before comparing", np.allclose(d, 1.0), f"got {d}")

for bad, why in (({1.0: coarse}, "one-resolution sweep"),
                 ({0.5: np.array(["A"]), 1.0: coarse}, "ragged sweep")):
    try:
        sweep_agreement(bad, coarse)
        check(f"REFUSES a {why}", False, "returned instead of raising")
    except ValueError as e:
        check(f"REFUSES a {why}", True, str(e)[:46])

print("")
print("the sweep votes on NOTHING - asserted on the source, not assumed")

import ast
_src = (Path(__file__).resolve().parents[1] / "scanno" / "resolution.py").read_text("utf-8")
_mod = ast.parse(_src)
_names = {n.name for n in _mod.body if isinstance(n, ast.FunctionDef)}
check("no consensus/vote function survives in resolution.py",
      not {n for n in _names if "consensus" in n or "vote" in n}, str(sorted(_names)))
_cli = (Path(__file__).resolve().parents[1] / "scanno" / "cli.py").read_text("utf-8")
check("annotate declares no vote-shaped option",
      "--consensus-weight" not in _cli and "--consensus-plain-majority" not in _cli)
_emit = (Path(__file__).resolve().parents[1] / "scanno" / "emit.py").read_text("utf-8")
_ef = {n.name for n in ast.walk(ast.parse(_emit)) if isinstance(n, ast.FunctionDef)}
check("emit has one sweep writer and it writes no label",
      "sweep_agreement_column" in _ef and "consensus_columns" not in _ef, str(sorted(_ef)[:6]))

print("")
print("sweep_stem: finding the sweep that was written")

check("an ordinary key with `_r` inside a word is NOT split",
      sweep_stem("scanno_resolved_path_scope") == "scanno_resolved_path_scope",
      sweep_stem("scanno_resolved_path_scope"))
check("a key that IS a sweep column is split", sweep_stem("scanno_path_r1p0") == "scanno_path")
check("a key with no `_r` is unchanged", sweep_stem("cell_type_forced") == "cell_type_forced")
for key in ("scanno_resolved_path_scope", "scanno_path", "cell_type_forced"):
    col = f"{key}_r1p0"
    check(f"a column written beside {key!r} is found",
          col.startswith(sweep_stem(key) + "_r") and col[len(sweep_stem(key)) + 2:] == "1p0", col)

print("")
print("compare: agreement is carried, and gates nothing")

n = 12
idx = [f"c{i}" for i in range(n)]
sample = ["S1"] * 6 + ["S2"] * 6
a_path = ["Rare", "Rare", "Big", "Big", "Big", "Big"] + ["Big"] * 6
b_path = ["Rare"] * 4 + ["Big"] * 2 + ["Rare"] * 4 + ["Big"] * 2
b_clu = ["c1"] * 4 + ["c2"] * 2 + ["c1"] * 4 + ["c2"] * 2
agree = [1.0] * 6 + [1.0, 1.0, 0.5, 0.5] + [1.0, 1.0]
A = pd.DataFrame({"p": a_path}, index=idx)
B = pd.DataFrame({"p": b_path, "s": sample, "c": b_clu, "ag": agree}, index=idx)

no_ag = compare(A, B, path_key="p", path_key_b="p", sample_key="s", cluster_key="c")
with_ag = compare(A, B, path_key="p", path_key_b="p", sample_key="s", cluster_key="c",
                  agreement_key="ag")


def ident(res):
    return sorted((c["cluster"], c["label_absent"], c["n_cells"])
                  for c in res["merge_candidates"]["candidates"])


check("the candidate set is IDENTICAL with and without the agreement column",
      ident(no_ag) == ident(with_ag), f"{ident(no_ag)} vs {ident(with_ag)}")
cands = with_ag["merge_candidates"]["candidates"]
check("a candidate is produced at all", len(cands) == 1, f"{len(cands)} candidates")
if cands:
    check("pct_sweep_agrees averages EXACTLY the cells that would move",
          cands[0]["pct_sweep_agrees"] == 75.0, str(cands[0]["pct_sweep_agrees"]))
check("without the column the field is None, not 0",
      no_ag["merge_candidates"]["candidates"][0].get("pct_sweep_agrees") is None)
check("and the absence is stated in the result",
      "NOT MEASURED" in no_ag["merge_candidates"]["sweep"],
      no_ag["merge_candidates"]["sweep"][:60])

p_meas = review_prompt(cands[0], brief=False)
p_unmeas = review_prompt(no_ag["merge_candidates"]["candidates"][0], brief=False)
check("the review prompt reports a measured sweep", "SWEEP AGREEMENT - 75.0%" in p_meas)
check("and NAMES an unmeasured one rather than leaving it blank", "NOT MEASURED" in p_unmeas)

print("")
print("emit: the column survives a round trip through disk")

try:
    import tempfile

    import anndata as ad

    from scanno.emit import sweep_agreement_column, sweep_path, write_h5ad

    rows = [{"cluster": 0, "label": "Big", "path": "Root/Big", "depth": 2, "gap": 0.5},
            {"cluster": 1, "label": "Rare", "path": "Root/Rare", "depth": 2, "gap": 0.2}]
    y = np.array([0, 0, 1, 1])
    Ad = ad.AnnData(X=np.zeros((4, 3), dtype="float32"))
    Ad.obs_names = [f"c{i}" for i in range(4)]
    k05 = sweep_path(Ad, rows, y, prefix="scanno", suffix="_scope", tag="0p5")
    k10 = sweep_path(Ad, rows, np.array([0, 0, 0, 1]), prefix="scanno", suffix="_scope",
                     tag="1p0")
    check("sweep_path puts `_r<tag>` LAST", k05.endswith("_r0p5") and k10.endswith("_r1p0"))
    check("and only ONE column per resolution", len(Ad.obs.columns) == 2,
          list(Ad.obs.columns))
    ag = sweep_agreement({0.5: np.asarray(Ad.obs[k05].astype(str)),
                          1.0: np.asarray(Ad.obs[k10].astype(str))},
                         np.asarray(Ad.obs[k10].astype(str)))
    info = sweep_agreement_column(Ad, ag, {"resolutions": [0.5, 1.0], "reference": k10,
                                           "by_label": {"Root/Rare": 1}},
                                  prefix="scanno", suffix="_scope")
    try:
        sweep_agreement_column(Ad, ag, {}, prefix="scanno", suffix="_scope")
        check("REFUSES to overwrite", False, "overwrote instead of raising")
    except ValueError:
        check("REFUSES to overwrite", True)
    check("the sweep adds NO label column",
          not [c for c in Ad.obs.columns if "cell_type" in c or "consensus" in c],
          list(Ad.obs.columns))
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "x.h5ad"
        write_h5ad(Ad, fp)
        back = ad.read_h5ad(fp)
        check("the column survives the write", info["key"] in back.obs)
        check("and comes back numeric, not string",
              np.issubdtype(np.asarray(back.obs[info["key"]]).dtype, np.floating))
        check("the provenance survives, slash-bearing label keys and all",
              "scanno_sweep_provenance_scope" in back.uns)
except ImportError as e:                                   # noqa: BLE001
    print(f"  SKIP  anndata not importable here ({e})")

print("")
if FAILED:
    print(f"FAILED {len(FAILED)}: {', '.join(FAILED)}")
    raise SystemExit(1)
print("all sweep checks pass")
