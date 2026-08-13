"""The resolution finder — every way it can pick the wrong thing quietly.

A resolution chooser is unusually easy to get wrong without noticing: it always returns a
resolution, that resolution always looks plausible, and nothing downstream can tell that a
different one would have been better. So the cases below are about the SILENT failures, not
about whether it runs.

Two of them are defects this module actually shipped with, caught by running it on a real sweep:
  * it reported the first tie-break that narrowed the field rather than the one that decided,
    crediting completeness for a pick that rare-population retention made;
  * it counted distinct LABELS and called them clusters, printing "19 clusters" for a sweep of
    156.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scanno.resolution import (MIN_TOL, derived_tolerance, format_report, pick_resolution,
                               sweep_stability)

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


TREE = {"children": {"root": ["A", "B", "Rare"], "A": ["A1", "A2"], "B": ["B1", "B2"],
                     "A1": [], "A2": [], "B1": [], "B2": [], "Rare": []}}


def sweep(spec, n=1000):
    """{res: [(path, count), ...]} -> {res: array of paths}, all the same length."""
    out = {}
    for r, parts in spec.items():
        lab = []
        for p, k in parts:
            lab += [p] * k
        assert len(lab) == n, f"{r}: {len(lab)} != {n}"
        out[r] = np.array(lab)
    return out


print("\n1 · splitting a node does not disturb the level ABOVE it")
# 2.0 splits A into A1/A2. At level 1 that is not a change at all, and the finder must say so -
# this is the commonest real case and the reason level 1 and level 2 need separate answers.
S = sweep({0.5: [("A/A1", 500), ("B/B1", 480), ("Rare", 20)],
           1.0: [("A/A1", 500), ("B/B1", 480), ("Rare", 20)],
           2.0: [("A/A1", 300), ("A/A2", 200), ("B/B1", 480), ("Rare", 20)]})
by1 = {r["resolution"]: r for r in sweep_stability(S, TREE, depth=1)}
by2 = {r["resolution"]: r for r in sweep_stability(S, TREE, depth=2)}
check("level 1 is untouched by a level-2 split",
      all(by1[r]["modal"] == 100.0 for r in (0.5, 1.0, 2.0)),
      f"{[round(by1[r]['modal'], 1) for r in (0.5, 1.0, 2.0)]}")
check("...and level 2 sees it", by2[2.0]["modal"] < by2[1.0]["modal"],
      f"{by2[2.0]['modal']:.0f}% vs {by2[1.0]['modal']:.0f}%")

print("\n1b · a genuine level-1 relabelling IS penalised")
S = sweep({0.5: [("A/A1", 500), ("B/B1", 480), ("Rare", 20)],
           1.0: [("A/A1", 500), ("B/B1", 480), ("Rare", 20)],
           2.0: [("A/A1", 700), ("B/B1", 280), ("Rare", 20)]})
by = {r["resolution"]: r for r in sweep_stability(S, TREE, depth=1)}
check("the outlier scores lower than the two that agree",
      by[0.5]["modal"] == by[1.0]["modal"] > by[2.0]["modal"],
      f"{by[0.5]['modal']:.0f} / {by[1.0]['modal']:.0f} / {by[2.0]['modal']:.0f}")
check("the end resolutions have no neighbour score",
      by[0.5]["neighbour"] is None and by[2.0]["neighbour"] is None
      and by[1.0]["neighbour"] is not None)

print("\n2 · a resolution that SHATTERS a rare population must not win on stability alone")
# 1.0 and 1.25 relabel nothing at level 1 - identical stability - but 1.25 splits Rare into a
# 20-cell and a 4-cell fragment. Only rare retention can see it.
S = sweep({0.75: [("A/A1", 600), ("B/B1", 376), ("Rare", 24)],
           1.0: [("A/A1", 600), ("B/B1", 376), ("Rare", 24)],
           1.25: [("A/A1", 600), ("B/B1", 376), ("Rare", 20), ("UNRESOLVED", 4)]})
out = pick_resolution(S, TREE, depths=(1,))
# 0.75 and 1.0 are IDENTICAL here, so there is no ground to prefer either and the assertion is
# only that the shatterer loses. Asserting `pick == 1.0` was the first version of this test and
# it was wrong: it demanded a preference the evidence does not contain.
check("the shattering resolution is rejected",
      out["pick"] != 1.25 and 1.25 not in out["tied"],
      f"picked {out['pick']}, tied {out['tied']}")

print("\n3 · a resolution that is stable by TRUNCATING everything must not win")
S = sweep({0.5: [("A", 500), ("B", 500)],            # everything stops at a non-leaf
           1.0: [("A/A1", 500), ("B/B1", 500)],      # same cells, resolved to leaves
           1.5: [("A/A1", 480), ("A", 20), ("B/B1", 500)]})
out = pick_resolution(S, TREE, depths=(1,))
check("completeness beats a fully truncated sweep", out["pick"] == 1.0,
      f"picked {out['pick']}; complete "
      + ", ".join(f"{r['resolution']}:{r['complete']:.0f}%"
                  for r in out["per_depth"][1]))

print("\n4 · the reason names the step that DECIDED, not the first that narrowed")
# stability leaves {1.0, 1.25}; completeness cannot separate them; rare retention does.
S = sweep({0.5: [("A/A1", 400), ("B/B1", 560), ("Rare", 40)],
           1.0: [("A/A1", 500), ("B/B1", 460), ("Rare", 40)],
           1.25: [("A/A1", 500), ("B/B1", 495), ("Rare", 5)]})
out = pick_resolution(S, TREE, depths=(1,))
check("rare retention is credited when it is what decided",
      out["pick"] == 1.0 and out["reason"] == "rare-population retention",
      f"picked {out['pick']} for {out['reason']!r}")

print("\n5 · the parsimony unit is CLUSTERS when clusters are supplied, and says which")
S = sweep({0.5: [("A/A1", 500), ("B/B1", 500)], 1.0: [("A/A1", 500), ("B/B1", 500)]})
cl = {0.5: np.repeat(np.arange(8), 125), 1.0: np.repeat(np.arange(40), 25)}
with_cl = sweep_stability(S, TREE, depth=1, clusters_by_res=cl)
without = sweep_stability(S, TREE, depth=1)
check("clusters are counted when given",
      [r["n_units"] for r in with_cl] == [8, 40]
      and all(r["units_are"] == "clusters" for r in with_cl))
check("...and labels are NOT silently called clusters",
      all(r["units_are"] == "labels" for r in without)
      and [r["n_units"] for r in without] == [2, 2],
      "a label count printed as a cluster count is how '19 clusters' described a sweep of 156")

print("\n6 · the tolerance is derived from the sweep, and has a floor")
check("a noisy sweep gets a wider tolerance than a clean one",
      derived_tolerance([90.0, 95.0, 91.0, 96.0]) > derived_tolerance([99.0, 99.1, 99.2]))
check("an identical sweep cannot get a zero tolerance",
      derived_tolerance([99.0, 99.0, 99.0]) == MIN_TOL)

print("\n7 · malformed input is refused, not guessed at")
for name, bad in (("one resolution", {1.0: np.array(["A"] * 10)}),
                  ("ragged lengths", {0.5: np.array(["A"] * 10),
                                      1.0: np.array(["A"] * 9)})):
    try:
        sweep_stability(bad, TREE, depth=1)
        check(f"{name} is refused", False, "it returned instead of raising")
    except ValueError:
        check(f"{name} is refused", True)

print("\n8 · the report shows every candidate, not just the winner")
S = sweep({0.5: [("A/A1", 500), ("B/B1", 500)],
           1.0: [("A/A1", 500), ("B/B1", 500)],
           1.5: [("A/A1", 400), ("A/A2", 100), ("B/B1", 500)]})
txt = format_report(pick_resolution(S, TREE, depths=(1,)), depths=(1,))
check("all resolutions appear in the report",
      all(str(r) in txt for r in (0.5, 1.0, 1.5)) and "PICK" in txt)
check("the derived tolerance is stated", "tolerance" in txt)

print("\n" + "=" * 62)
if FAILED:
    print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
    raise SystemExit(1)
print("resolution finder OK")
