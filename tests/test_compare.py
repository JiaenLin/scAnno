"""Two routes, and what their agreement is and is not evidence of.

The comparison exists to ask whether the labels survive a different clustering. Three things it
has to get right, because each would quietly overstate the result:

  1. a cell one route WITHHELD is not a disagreement, and must leave the denominator. Counting
     an exclusion as an error makes a careful run look worse than a careless one.
  2. the disagreements are reported as PAIRS. One confusable boundary and a route that disagrees
     everywhere give the same single percentage.
  3. sample dominance of route B travels with the answer. On an un-integrated cohort a joint
     clustering can group by library, and then disagreement indicts B rather than A.

    python tests/test_compare.py
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
    import pandas as pd
except ImportError as e:
    print(f"SKIP: needs {e.name}")
    raise SystemExit(0)

from scanno.compare import DOMINANCE, compare, format_report, level  # noqa: E402


def obs(paths, samples=None, clusters=None, ids=None):
    n = len(paths)
    d = {"scanno_path": paths}
    if samples:
        d["sample"] = samples
    if clusters:
        d["cl"] = clusters
    return pd.DataFrame(d, index=ids or [f"c{i}" for i in range(n)])


print("\n1 - truncation to a level, with sentinels passing through whole")
# `classify()` returns the path WITHOUT the root - "Lymphoid/T cell", not "root/Lymphoid/T cell"
# - so level 1 is the top-level type and level 2 is its child. Getting this wrong is easy and
# silent: with a spurious `root/` prefix every depth shifts by one and level 2 compares the
# quantity level 1 was meant to.
check("level 1 is the top-level type", level(["Lymphoid/T cell"], 1) == ["Lymphoid"])
check("level 2 is its child", level(["Lymphoid/T cell"], 2) == ["Lymphoid/T cell"])
check("EXCLUDED is not truncated into nonsense", level(["EXCLUDED"], 1) == ["EXCLUDED"])
check("nor is UNRESOLVED", level(["UNRESOLVED"], 2) == ["UNRESOLVED"])

print("\n2 - perfect agreement reads as perfect")
A = obs(["A/x", "B/y", "A/x"])
res = compare(A, A.copy())
check("scored every cell", res["n_scored"] == 3)
check("100% at level 1", res["levels"][0]["agreement_pct"] == 100.0)
check("100% at level 2", res["levels"][1]["agreement_pct"] == 100.0)

print("\n3 - a withheld cell is NOT a disagreement")
A = obs(["A/x", "A/x", "A/x", "A/x"])
B = obs(["A/x", "EXCLUDED", "A/x", "UNRESOLVED"])
res = compare(A, B)
check("the sentinels leave the denominator", res["n_scored"] == 2, str(res["n_scored"]))
check("and are counted separately", res["n_sentinel_either"] == 2)
check("agreement is over what was actually compared",
      res["levels"][0]["agreement_pct"] == 100.0, str(res["levels"][0]))

print("\n4 - disagreement at level 2 that vanishes at level 1")
A = obs(["Lymphoid/T cell", "Lymphoid/T cell", "Myeloid/Mono"])
B = obs(["Lymphoid/NK cell", "Lymphoid/T cell", "Myeloid/Mono"])
res = compare(A, B)
check("level 1 agrees fully", res["levels"][0]["agreement_pct"] == 100.0)
check("level 2 does not", res["levels"][1]["agreement_pct"] < 100.0,
      str(res["levels"][1]["agreement_pct"]))
check("and the PAIR is named, not just a count",
      any("T cell -> Lymphoid/NK cell" in p
          for p, _ in res["levels"][1]["top_disagreements"]),
      str(res["levels"][1]["top_disagreements"]))

print("\n5 - only the shared cells are compared")
A = obs(["A/x", "A/x"], ids=["c0", "c1"])
B = obs(["A/x", "B/y"], ids=["c1", "c2"])
res = compare(A, B)
check("the intersection is scored", res["n_shared"] == 1, str(res["n_shared"]))
check("and both totals are reported", (res["n_a"], res["n_b"]) == (2, 2))

print("\n6 - route B's sample dominance travels with the answer")
paths = ["A/x"] * 8
# cluster 0 is entirely s1; cluster 1 is evenly mixed
sams = ["s1", "s1", "s1", "s1", "s1", "s2", "s1", "s2"]
clus = ["0", "0", "0", "0", "1", "1", "1", "1"]
A = obs(paths)
B = obs(paths, samples=sams, clusters=clus)
res = compare(A, B, sample_key="sample", cluster_key="cl")
dom = res["b_dominance"]
check("every cluster is profiled", dom["n_clusters"] == 2)
check("the one-sample cluster is flagged as dominated", dom["n_dominated"] == 1,
      str(dom["clusters"]))
check("the threshold is stated", dom["threshold_pct"] == round(100 * DOMINANCE))
lines = format_report(res, "per_sample", "joint")
check("and the report says B cannot arbitrate",
      any("cannot arbitrate" in ln for ln in lines), str(lines[-4:]))

print("\n7 - the limit is attached to the answer, not left to a footnote")
lines = format_report(compare(obs(["A/x"]), obs(["A/x"])), "A", "B")
joined = " ".join(lines)
check("agreement is not presented as correctness",
      "does NOT" in joined and "mean they are correct" in joined, str(lines[-3:]))
check("and the shared-inputs reason is given",
      any("corpus" in ln and "identically" in ln for ln in lines))

print("\n8 - two routes annotated under DIFFERENT column names can be compared at all")
# The routes are separate objects, and the ordinary way to stop a second annotation colliding
# with the first is `--label-suffix`. `compare` had ONE key for both, so the only comparable
# pair was two routes sharing a column NAME - the one thing you cannot do when both annotations
# live in one object. Cost, measured on SAMBO: the joint-vs-per-sample comparison this module
# was written for could not address the promoted per-sample column (`cell_type_forced`) and the
# joint route's own (`scanno_resolved_path_scope`) in one call, and so was never run.
A = obs(["A/x", "A/y"])
B = pd.DataFrame({"other_path": ["A/x", "A/y"]}, index=["c0", "c1"])
res = compare(A, B, path_key="scanno_path", path_key_b="other_path")
check("route B is read from its own column", res["levels"][0]["agreement_pct"] == 100.0)
check("and the key it used is recorded", res.get("path_key_b") == "other_path")
check("omitting it still means one key for both",
      compare(obs(["A/x"]), obs(["A/x"])).get("path_key_b") == "scanno_path")

print("\n9 - a rare population the per-sample clustering merged is NAMED")
# s1 resolved six dendritic cells. s2 and s3 had theirs absorbed into a macrophage cluster and
# carry no dendritic call ANYWHERE. The joint clustering puts all fourteen in one cluster.
# No threshold states this: the label is a property of WHICH SAMPLE DID THE CLUSTERING rather
# than of the cell, and that is a structural fact about the crosstab.
DC, MP = "Immune/Myeloid/Dendritic cell", "Immune/Myeloid/Macrophage"
a_lab = [DC] * 6 + [MP] * 4 + [MP] * 4 + [MP] * 6
b_lab = [DC] * 14 + [MP] * 6
sams = ["s1"] * 6 + ["s2"] * 4 + ["s3"] * 4 + ["s1", "s1", "s2", "s2", "s3", "s3"]
clus = ["J1"] * 14 + ["J0"] * 6
ids = [f"c{i}" for i in range(20)]
A = pd.DataFrame({"scanno_path": a_lab}, index=ids)
B = pd.DataFrame({"jp": b_lab, "sample": sams, "cl": clus}, index=ids)
res = compare(A, B, path_key="scanno_path", path_key_b="jp",
              sample_key="sample", cluster_key="cl")
cands = res["merge_candidates"]["candidates"]
check("exactly one candidate is named", len(cands) == 1, str(cands))
c = cands[0] if cands else {}
check("it is the joint cluster holding both", c.get("cluster") == "J1")
check("the label the merge hid is named", c.get("label_absent") == DC)
check("so is the label it was hidden under", c.get("label_carried") == MP)
check("the samples that could not resolve it are NAMED, not counted",
      c.get("samples_lacking") == ["s2", "s3"], str(c.get("samples_lacking")))
check("and so are the ones that could", c.get("samples_with") == ["s1"])
check("the cells that would move are counted", c.get("n_cells") == 8, str(c.get("n_cells")))
check("the cluster's sample dominance travels with it",
      c.get("top_share_pct") is not None and c["top_share_pct"] < 50, str(c.get("top_share_pct")))
check("every cluster keeps a crosstab, candidate or not",
      set(res["merge_candidates"]["crosstab"]) == {"J0", "J1"})

print("\n10 - the two negative controls, and BOTH halves are load-bearing")
# (a) labels that disagree inside a cluster where every sample carries BOTH cohort-wide. That is
#     an ordinary boundary disagreement - most of them are - and a check that fires on it is
#     noise. With only the positive half a rule can be made to catch everything by loosening,
#     which is how a gate becomes decoration.
FB, PC = "Stromal/Fibroblast", "Stromal/Mural/Pericyte"
ids = [f"c{i}" for i in range(4)]
A = pd.DataFrame({"scanno_path": [FB, PC, FB, PC]}, index=ids)
B = pd.DataFrame({"jp": [FB] * 4, "sample": ["s1", "s1", "s2", "s2"],
                  "cl": ["J0"] * 4}, index=ids)
res = compare(A, B, path_key="scanno_path", path_key_b="jp",
              sample_key="sample", cluster_key="cl")
check("a mixed cluster with no absent label is NOT a candidate",
      res["merge_candidates"]["candidates"] == [],
      str(res["merge_candidates"]["candidates"]))

# (b) a structurally valid candidate whose cluster is five sixths one animal. It is REPORTED,
#     never removed: DOMINANCE is a fact about the comparison's weaker arm and the reader's to
#     weigh (compare.DOMINANCE). But it must travel with the candidate, or a reader adopts a
#     library as a population.
# s2 must carry MP SOMEWHERE or the rule fires in both directions and the fixture, not the
# code, is what makes the count two. J1 gives both samples an MP cell outside the candidate
# cluster, which is also what a real object looks like.
ids = [f"c{i}" for i in range(8)]
A = pd.DataFrame({"scanno_path": [DC] * 5 + [MP] + [MP, MP]}, index=ids)
B = pd.DataFrame({"jp": [DC] * 6 + [MP, MP],
                  "sample": ["s1"] * 5 + ["s2"] + ["s1", "s2"],
                  "cl": ["J0"] * 6 + ["J1", "J1"]}, index=ids)
res = compare(A, B, path_key="scanno_path", path_key_b="jp",
              sample_key="sample", cluster_key="cl")
cands = res["merge_candidates"]["candidates"]
check("a library-dominated cluster is still reported", len(cands) == 1, str(cands))
check("but carries a dominance above the threshold, so it can be refused",
      bool(cands) and cands[0]["top_share_pct"] > 100 * DOMINANCE,
      str(cands[0]["top_share_pct"]) if cands else "")

print("\n11 - the candidate rule reads the SAMPLES and never the design")
# A design-differential gate was built here once, refused a real comparison in which two
# libraries of ten held 94% of the unresolved nuclei, and was removed (KNOWN_ISSUES). So this
# names which samples lack the label and stops. Deciding that a study's arms differ is not the
# tool's call, and a block that cannot see the design cannot make it.
# A word scan over the source was written here first and was the wrong check: `group` is what
# a clustering does to cells, and a check that fires on correct code is a check somebody
# switches off. What matters is STRUCTURAL - the function cannot be given a design, and a
# candidate names samples.
import inspect  # noqa: E402
_params = set(inspect.signature(compare).parameters)
check("compare cannot be given a design at all",
      not any(w in p_ for p_ in _params for w in ("factor", "design", "condition", "arm")),
      str(sorted(_params)))
check("a candidate names the samples and nothing above them",
      set(c) >= {"samples_with", "samples_lacking"}
      and not any(("factor" in k) or ("arm" in k) for k in c), str(sorted(c)))
check("and the limit says whose judgement the pattern is",
      "reader" in res["merge_candidates"]["limit"])

print("\n12 - AT REALISTIC SCALE: a big cluster with a few stray rare cells is NOT a candidate")
# The fixtures above are fourteen cells and clean. Real clusters are thousands of cells holding
# one or two of a rare label, and on the first run against a real cohort the rule fired on every
# rare label in every cluster: 87 candidates over 23 clusters, the largest claiming 8,749 cells
# should become `Neural` on the evidence of THREE Neural cells, and 21 of the 87 naming the
# sentinel `EXCLUDED` as the label somebody was missing.
#
# Both are fixed by the same thing - the candidate is anchored on ROUTE B'S OWN CALL for the
# cluster - and neither would have been caught by a fourteen-cell fixture. A convenient fixture
# hides the bug it was built to catch, so this one is built at the scale the defect lives at.
MAC, NEU = "Immune/Myeloid/Macrophage", "Neural"
lab_a, lab_b, sams, clus = [], [], [], []
for i in range(1, 7):                       # six samples, ~500 macrophages each, one cluster
    n = 500
    lab_a += [MAC] * n
    sams += [f"s{i}"] * n
    clus += ["J0"] * n
lab_a[0:3] = [NEU] * 3                      # s1 alone resolved THREE neural cells in there
lab_a[500:520] = ["EXCLUDED"] * 20          # and s2 alone carries some withheld nuclei
lab_b = [MAC] * len(lab_a)                  # the joint route calls the whole cluster Macrophage
ids = [f"c{i}" for i in range(len(lab_a))]
A = pd.DataFrame({"scanno_path": lab_a}, index=ids)
B = pd.DataFrame({"jp": lab_b, "sample": sams, "cl": clus}, index=ids)
res = compare(A, B, path_key="scanno_path", path_key_b="jp",
              sample_key="sample", cluster_key="cl")
cands = res["merge_candidates"]["candidates"]
check("three stray cells do not condemn a 3,000-cell cluster", cands == [],
      str([(r["label_absent"], r["n_cells"]) for r in cands]))
check("and the sentinel is never a label somebody is missing",
      not any(r["label_absent"] == "EXCLUDED" for r in cands))

# The same cluster, now ANNOTATED as Neural by the joint route, is a candidate - and its
# credibility is reported rather than assumed: route A agrees on 3 of 3,000.
B2 = B.copy(); B2["jp"] = [NEU] * len(lab_a)
res2 = compare(A, B2, path_key="scanno_path", path_key_b="jp",
               sample_key="sample", cluster_key="cl")
c2 = res2["merge_candidates"]["candidates"]
check("with route B calling it Neural it IS a candidate", len(c2) == 1, str(c2))
check("the withheld nuclei are not counted as cells that would move",
      bool(c2) and c2[0]["n_cells"] == 2480, str(c2[0]["n_cells"]) if c2 else "")
check("and how little route A agrees is on the row",
      bool(c2) and c2[0]["pct_route_a_agrees"] < 1.0,
      str(c2[0]["pct_route_a_agrees"]) if c2 else "")

print("\n13 - the design is tabulated, and CANNOT change what is a candidate")
# Rule one's third question - is the change differential across the design - has to be answered,
# and answering it means touching a column that names the arms. That is the same shape as
# --sample-key: the caller names a column, the tool counts its levels and does not know what they
# mean. What must never happen is the design reaching the DECISION. A design-differential gate
# was built here once, refused a comparison in which two libraries of ten held 94% of the
# unresolved nuclei, and was removed.
grp = ["aged" if x in ("s1", "s2") else "young" for x in sams]
B3 = B2.copy(); B3["grp"] = grp
with_g = compare(A, B3, path_key="scanno_path", path_key_b="jp", sample_key="sample",
                 cluster_key="cl", group_key="grp")
without_g = compare(A, B3, path_key="scanno_path", path_key_b="jp", sample_key="sample",
                    cluster_key="cl")


def strip(rows):
    return [{k: v for k, v in r.items() if k != "moving_by_group"} for r in rows]


check("the candidate set is IDENTICAL with and without a design column",
      strip(with_g["merge_candidates"]["candidates"])
      == strip(without_g["merge_candidates"]["candidates"]))
cg = with_g["merge_candidates"]["candidates"][0]
check("the moving cells are counted per arm",
      sum(cg["moving_by_group"].values()) == cg["n_cells"], str(cg["moving_by_group"]))
check("and the arms are NAMED, not ranked or judged",
      set(cg["moving_by_group"]) <= {"aged", "young"})
imp = with_g["merge_candidates"]["impact"]
check("the impact is derived from the rows, so it cannot disagree with them",
      imp["n_cells_total"] == sum(r["n_cells"]
                                  for r in with_g["merge_candidates"]["candidates"]))
check("it says what adopting everything would do, and that it is not a recommendation",
      "not a recommendation" in imp["limit"])

print("\n14 - the per-sample impact conserves cells and keeps its denominator")
# Adoption RELABELS; it never adds or removes a nucleus. If the per-sample table does not
# conserve, it is describing something other than a relabelling - and the percentage-point
# deltas, which are the thing a reader compares across samples of different size, would be
# measured against a denominator that moved underneath them.
ips = with_g["merge_candidates"]["impact_per_sample"]
by_s = {}
for r in ips["rows"]:
    by_s.setdefault(r["sample"], []).append(r)
ok_tot, ok_den, ok_pct = True, True, True
for x, rs in by_s.items():
    if sum(r["n_before"] for r in rs) != sum(r["n_after"] for r in rs):
        ok_tot = False
    if len({r["n_sample_total"] for r in rs}) != 1:
        ok_den = False
    if sum(r["n_before"] for r in rs) != rs[0]["n_sample_total"]:
        ok_den = False
    for r in rs:
        if abs(r["pct_after"] - r["pct_before"] - r["pct_delta"]) > 0.002:
            ok_pct = False
check("no nucleus is created or destroyed in any sample", ok_tot)
check("the denominator is every nucleus of the sample and is the same before and after", ok_den)
check("and the percentage-point delta is the difference of the two shares", ok_pct)
moved = sum(r["n_delta"] for r in ips["rows"] if r["n_delta"] > 0)
check("the cells gained equal the candidates' n_cells",
      moved == sum(r["n_cells"] for r in with_g["merge_candidates"]["candidates"]),
      f"{moved} moved")
check("a sentinel row is marked as one rather than silently mixed in",
      all(isinstance(r["is_sentinel"], bool) for r in ips["rows"]))
check("the denominator is DECLARED, not left to be inferred",
      "every nucleus" in ips["denominator"])

print("\n" + "=" * 64)
if fails:
    print(f"compare: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print("compare OK - withheld is not disagreement, pairs are named, B's weakness is reported")
