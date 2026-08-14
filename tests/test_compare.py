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

print("\n" + "=" * 64)
if fails:
    print(f"compare: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print("compare OK - withheld is not disagreement, pairs are named, B's weakness is reported")
