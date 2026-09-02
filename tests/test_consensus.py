"""The sweep consensus — the vote a single resolution cannot hold.

The mechanism under test says: walk the joint object at every resolution against the same tree,
the same scope and the same background, then read off, per CELL, the label the sweep agrees on
and how much of it agreed. Every case here is a way that vote can be wrong while still returning
a full column of plausible labels — which is the only failure mode that matters, because a
consensus always produces one.

Three of these are defects the surrounding code actually shipped with:

  * `context.sweep_stem` split `path_key` on the bare substring `_r`, so
    `scanno_resolved_path_scope` became `scanno` and every sweep column written beside that key
    was looked for under the wrong stem. The per-resolution figures then reported an object that
    had never been swept — an absence indistinguishable from a run that never swept.
  * an agreement column on a one-resolution sweep is 1.0 for every cell BY CONSTRUCTION and
    reads exactly like one that was measured.
  * `compare` had no way to say "not measured" apart from "measured low", and they are opposite
    findings about a candidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scanno.compare import compare
from scanno.context import sweep_stem
from scanno.joint import reconcile, review_prompt
from scanno.resolution import cluster_weights, consensus

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


print("consensus: what the sweep agrees on")

# A rare population that only the FINE resolution separates. Coarse and medium pool it into the
# neighbour it sits next to; fine splits it out. This is the whole reason the sweep exists, and
# a consensus that reports it as `Rare` on a 1-of-3 vote is reporting the granularity, not the
# cells.
coarse = np.array(["Big"] * 8 + ["Big", "Big"])
medium = np.array(["Big"] * 8 + ["Big", "Big"])
fine = np.array(["Big"] * 8 + ["Rare", "Rare"])
lab, agr = consensus({0.5: coarse, 1.0: medium, 2.0: fine})
check("a 1-of-3 call does NOT become the consensus label",
      list(lab[-2:]) == ["Big", "Big"], f"got {list(lab[-2:])}")
check("and its agreement says how thin it was",
      np.allclose(agr[-2:], 2 / 3), f"got {agr[-2:]}")
check("cells every resolution agreed on read 1.0",
      np.allclose(agr[:8], 1.0))

# Two of three, the other way round: the population IS separated by most of the sweep.
lab2, agr2 = consensus({0.5: coarse, 1.0: fine, 2.0: fine})
check("a 2-of-3 call DOES become the consensus label",
      list(lab2[-2:]) == ["Rare", "Rare"], f"got {list(lab2[-2:])}")
# AGREEMENT IS THE SHARE OF THE WEIGHT THAT COULD HAVE VOTED, not of the whole sweep. Two of
# three resolutions deliver `Rare`, which meets min_support, so the denominator is those two and
# both carried it: 1.0. Under a plain majority the same cells read 2/3 - the two numbers answer
# different questions and the column has to say which it is.
check("agreement is measured against what COULD have voted", np.allclose(agr2[-2:], 1.0),
      f"got {agr2[-2:]}")
check("...and against the whole sweep under a plain majority",
      np.allclose(consensus({0.5: coarse, 1.0: fine, 2.0: fine}, eligible=False)[1][-2:], 2 / 3))

# Sentinels are not special-cased. A cell the sweep mostly could not place reads UNRESOLVED,
# because promoting a minority named call over it is the guess `classify` exists not to make.
u = {0.5: np.array(["UNRESOLVED"]), 1.0: np.array(["UNRESOLVED"]), 2.0: np.array(["Rare"])}
lu, au = consensus(u)
check("a mostly-UNRESOLVED cell stays UNRESOLVED", lu[0] == "UNRESOLVED", f"got {lu[0]}")

# The refusals.
try:
    consensus({1.0: np.array(["A", "B"])})
    check("REFUSES a one-resolution sweep", False, "returned instead of raising")
except ValueError as e:
    check("REFUSES a one-resolution sweep", True, str(e)[:52])
try:
    consensus({0.5: np.array(["A"]), 1.0: np.array(["A", "B"])})
    check("REFUSES ragged sweeps", False, "returned instead of raising")
except ValueError as e:
    check("REFUSES ragged sweeps", True, str(e)[:52])

# Agreement is a fraction of the sweep, and its floor is 1/K, never 0 — the modal label is
# always carried by at least one resolution. A 0 would mean a label no resolution produced.
many = {float(i): np.array([f"L{i}"]) for i in range(4)}
_, a4 = consensus(many)
check("agreement floor is 1/K, never 0", np.allclose(a4, 0.25), f"got {a4}")

# `depth` truncates before voting, so two resolutions that differ only below level 1 agree at
# level 1. Without this a deep tree makes every sweep look unstable.
d = {0.5: np.array(["S/Fib/Matri"]), 1.0: np.array(["S/Fib/Quiescent"])}
l1, ag1 = consensus(d, depth=1)
check("depth truncates before voting", l1[0] == "S" and ag1[0] == 1.0, f"{l1[0]} {ag1[0]}")

print("")
print("the vote: a coarse partition cannot outvote what it could not see")

# THE DEFECT THIS SECTION EXISTS FOR, at the shape it was measured in. A rare population is
# separated by two resolutions of eight and merged by the other six. A plain majority deletes
# it - and it is exactly the population a resolution sweep was added to recover, so the
# mechanism would have been reporting the failure it was built to fix.
#
# Measured on 100,713 nuclei: `Dendritic cell` at 457 and 561 cells in the two finest of eight
# resolutions, 0 cells after a plain majority.
_fine = np.array(["Big"] * 8 + ["Rare"] * 2)
_coarse = np.array(["Big"] * 10)
SWEEP = {0.25: _coarse, 0.5: _coarse, 0.75: _coarse, 1.0: _coarse,
         1.25: _coarse, 1.5: _coarse, 1.75: _fine, 2.0: _fine}

lp, ap = consensus(SWEEP, eligible=False)
check("a PLAIN MAJORITY deletes a population 2 of 8 resolutions separate",
      list(lp[-2:]) == ["Big", "Big"], f"got {list(lp[-2:])}")
le, ae = consensus(SWEEP)
check("the ELIGIBLE vote keeps it", list(le[-2:]) == ["Rare", "Rare"], f"got {list(le[-2:])}")
check("and its agreement is 1.0 - every resolution that COULD assert it did",
      np.allclose(ae[-2:], 1.0), f"got {ae[-2:]}")
check("cells nothing disagrees about are untouched by either rule",
      list(le[:8]) == ["Big"] * 8 and list(lp[:8]) == ["Big"] * 8)

# A WEIGHT ALONE IS NOT THE FIX, and the test says so rather than leaving it to be assumed.
# Weighting by cluster count still lost that population on the real cohort; weighting by the
# resolution parameter rescued ONE cell of 457.
w_res = {r: r for r in SWEEP}
lw, _ = consensus(SWEEP, weights=w_res, eligible=False)
check("weighting by resolution does NOT rescue it on its own",
      list(lw[-2:]) == ["Big", "Big"], f"got {list(lw[-2:])}")

# ...and the eligible vote does not DEPEND on the weight, which is what makes it the mechanism
# rather than a second knob.
for _name, _w in (("equal", None), ("resolution", w_res),
                  ("clusters", {r: 10 + 20 * r for r in SWEEP})):
    _l, _ = consensus(SWEEP, weights=_w)
    check(f"eligible vote survives weight={_name}", list(_l[-2:]) == ["Rare", "Rare"],
          f"got {list(_l[-2:])}")

# cluster_weights reads the partition, not the parameter.
cw = cluster_weights({0.5: np.array([0, 0, 1, 1]), 2.0: np.array([0, 1, 2, 3])})
check("cluster_weights counts the clusters a partition PRODUCED", cw == {0.5: 2.0, 2.0: 4.0},
      str(cw))

# Weights must not silently drop a resolution.
try:
    consensus(SWEEP, weights={0.25: 1.0})
    check("REFUSES a weight map missing a resolution", False, "returned instead of raising")
except ValueError as e:
    check("REFUSES a weight map missing a resolution", True, str(e)[:44])
try:
    consensus(SWEEP, weights={r: 0.0 for r in SWEEP})
    check("REFUSES a zero weight", False, "returned instead of raising")
except ValueError as e:
    check("REFUSES a zero weight", True, str(e)[:44])

# THE NO-OP CASE. Where every resolution delivers every label, eligibility changes nothing and
# neither does the weight - so this generalises the plain vote rather than replacing it.
_flat = {r: np.array(["A", "B", "A"]) for r in (0.5, 1.0, 2.0)}
_a, _ = consensus(_flat)
_b, _ = consensus(_flat, eligible=False)
_c, _ = consensus(_flat, weights={0.5: 1.0, 1.0: 5.0, 2.0: 9.0})
check("with no disagreement all three rules agree", list(_a) == list(_b) == list(_c))

print("")
print("sweep_stem: finding the sweep that was written")

check("an ordinary key with `_r` inside a word is NOT split",
      sweep_stem("scanno_resolved_path_scope") == "scanno_resolved_path_scope",
      sweep_stem("scanno_resolved_path_scope"))
check("a key that IS a sweep column is split", sweep_stem("scanno_path_r1p0") == "scanno_path")
check("a key with no `_r` is unchanged", sweep_stem("cell_type_forced") == "cell_type_forced")
# The round trip that matters: the name emit writes must be found by the stem context derives.
for key in ("scanno_resolved_path_scope", "scanno_path", "cell_type_forced"):
    col = f"{key}_r1p0"
    check(f"a column written beside {key!r} is found",
          col.startswith(sweep_stem(key) + "_r")
          and col[len(sweep_stem(key)) + 2:] == "1p0", col)

print("")
print("compare: agreement is carried, and gates nothing")

# Two samples. `Rare` exists in S1 under route A and nowhere in S2, and route B's cluster c1 is
# called Rare — the candidate shape. Half of c1's movable cells have a sweep that agrees, half
# do not; the candidate must report the mean over EXACTLY those cells and nothing else.
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
    # S2's four c1 cells are the movable ones: two at 1.0, two at 0.5 -> 75%.
    check("pct_sweep_agrees averages EXACTLY the cells that would move",
          cands[0]["pct_sweep_agrees"] == 75.0, str(cands[0]["pct_sweep_agrees"]))
check("without the column the field is None, not 0",
      no_ag["merge_candidates"]["candidates"][0].get("pct_sweep_agrees") is None)
check("and the absence is stated in the result",
      "NOT MEASURED" in no_ag["merge_candidates"]["sweep"],
      no_ag["merge_candidates"]["sweep"][:60])

# The prompt a reviewer reads must distinguish the two.
p_meas = review_prompt(cands[0], brief=False)
p_unmeas = review_prompt(no_ag["merge_candidates"]["candidates"][0], brief=False)
check("the review prompt reports a measured sweep", "SWEEP AGREEMENT - 75.0%" in p_meas,
      [l for l in p_meas.splitlines() if "SWEEP" in l])
check("and NAMES an unmeasured one rather than leaving it blank",
      "NOT MEASURED" in p_unmeas, [l for l in p_unmeas.splitlines() if "SWEEP" in l])

print("")
print("reconcile: a consensus route B is still a route B")

# THE INVARIANCE. When every resolution agrees, the consensus column IS the single-resolution
# column, so the joint route must deliver exactly what it delivered before the sweep existed.
# Without this the feature could silently change the answer on data it was meant not to touch.
flat = {0.5: np.array(b_path), 1.0: np.array(b_path), 2.0: np.array(b_path)}
cons, _ = consensus(flat)
r_single = reconcile(np.array(a_path), np.array(b_path), np.array(b_clu), np.array(sample),
                     with_ag["merge_candidates"]["candidates"])
r_cons = reconcile(np.array(a_path), cons, np.array(b_clu), np.array(sample),
                   with_ag["merge_candidates"]["candidates"])
check("an all-agreeing sweep reconciles IDENTICALLY to the single resolution",
      list(r_single[0]) == list(r_cons[0]) and list(r_single[1]) == list(r_cons[1]))
check("and the sweep number rides on the moved record",
      r_cons[2]["moved"] and "pct_sweep_agrees" in r_cons[2]["moved"][0])

print("")
print("emit: the columns survive a round trip through disk")

# The suite that only ever built objects IN MEMORY passed while `uns` held a structure anndata
# refuses to write — twice, once for a list of dicts and once for slash-bearing keys, and labels
# here ARE slash-bearing paths. So this writes and reads back.
try:
    import anndata as ad
    import tempfile

    from scanno.context import sweep_stem as _stem
    from scanno.emit import consensus_columns, sweep_path, write_h5ad

    res_rows = [{"cluster": 0, "label": "Big", "path": "Root/Big", "depth": 2, "gap": 0.5},
                {"cluster": 1, "label": "Rare", "path": "Root/Rare", "depth": 2, "gap": 0.2}]
    y = np.array([0, 0, 1, 1])
    Adata = ad.AnnData(X=np.zeros((4, 3), dtype="float32"))
    Adata.obs_names = [f"c{i}" for i in range(4)]
    k05 = sweep_path(Adata, res_rows, y, prefix="scanno", suffix="_scope", tag="0p5")
    k10 = sweep_path(Adata, res_rows, np.array([0, 0, 0, 1]), prefix="scanno",
                     suffix="_scope", tag="1p0")
    check("sweep_path puts `_r<tag>` LAST", k05.endswith("_r0p5") and k10.endswith("_r1p0"),
          f"{k05} {k10}")
    check("and only ONE column per resolution", len(Adata.obs.columns) == 2,
          list(Adata.obs.columns))
    stem = _stem("scanno_path_scope")
    check("the written names are found under the stem context derives",
          all(c.startswith(stem + "_r") for c in (k05, k10)), f"stem {stem}")

    cons, agr = consensus({0.5: np.asarray(Adata.obs[k05].astype(str)),
                           1.0: np.asarray(Adata.obs[k10].astype(str))})
    info = consensus_columns(Adata, cons, agr,
                             {"resolutions": [0.5, 1.0], "columns": [k05, k10],
                              "by_label": {"Root/Rare": 1}},
                             prefix="scanno", suffix="_scope")
    try:
        consensus_columns(Adata, cons, agr, {}, prefix="scanno", suffix="_scope")
        check("consensus_columns REFUSES to overwrite", False, "overwrote instead of raising")
    except ValueError as e:
        check("consensus_columns REFUSES to overwrite", True, str(e)[:44])

    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "x.h5ad"
        write_h5ad(Adata, fp)
        back = ad.read_h5ad(fp)
        check("both consensus columns survive the write",
              info["key"] in back.obs and info["agreement_key"] in back.obs)
        check("the agreement comes back numeric, not string",
              np.issubdtype(np.asarray(back.obs[info["agreement_key"]]).dtype, np.floating),
              str(np.asarray(back.obs[info["agreement_key"]]).dtype))
        check("the provenance survives, slash-bearing label keys and all",
              "scanno_consensus_provenance_scope" in back.uns)
except ImportError as e:                                   # noqa: BLE001
    print(f"  SKIP  anndata not importable here ({e})")

print("")
if FAILED:
    print(f"FAILED {len(FAILED)}: {', '.join(FAILED)}")
    raise SystemExit(1)
print("all consensus checks pass")
