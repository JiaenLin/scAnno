"""Targeted rescue — a label a unit lacks, looked for in that unit alone.

The guarantee this suite exists to hold is narrow and absolute: **only the located cluster's
cells are renamed**. Everything else in the unit keeps the label it was delivered with. A
mechanism that quietly adopted the finer annotation would be re-annotating a whole unit to
recover one population, which is a different and far larger claim than the one being made.

Two defects are pinned here because both happened:

  * a numpy fixed-width string array silently TRUNCATED the rescued label - `<U3` holding "Big"
    turned "Rare" into "Rar", with no error. Every rescue writes a label that is by construction
    absent from the array, so it is quite likely longer than anything in it.
  * "not found" has two meanings and printing them the same way is the failure. Where the finest
    clustering still makes clusters larger than the population would be, a search that found
    nothing has established nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scanno.rescue import KEPT, RESCUED, find, imbalanced, reach, rescue, summarise

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


# A: carries Rare. B: does not, and at the finer rung its cluster "3" comes back Rare.
A = np.array(["Big"] * 8 + ["Rare"] * 2)
B = np.array(["Big"] * 6 + ["Mid"] * 4)
SWEEP = {"A": {1.0: A, 2.0: A},
         "B": {1.0: B, 2.0: np.array(["Big"] * 6 + ["Mid"] * 1 + ["Rare"] * 3)}}
CLUS = {"A": {1.0: np.array(["0"] * 10), 2.0: np.array(["0"] * 10)},
        "B": {1.0: np.array(["0"] * 10), 2.0: np.array(["0"] * 7 + ["3"] * 3)}}
RUNGS = [1.0, 2.0]

print("the trigger set")
t = imbalanced({"A": A, "B": B})
check("a label one unit carries and another lacks IS a target", "Rare" in t, str(sorted(t)))
check("and the units are named the right way round", t["Rare"] == (["A"], ["B"]), str(t["Rare"]))
check("a label BOTH carry is not a target", "Big" not in t, str(sorted(t)))
check("a label only one unit has, that no one lacks, cannot arise", "Mid" in t)
check("no rarity threshold is consulted - imbalance is the whole criterion",
      set(t) == {"Rare", "Mid"}, str(sorted(t)))
# `Rare` given CHILDREN is an internal node, so a cell resting on it carries a compartment
# name rather than a population, and it must not be a target. A fixture writing `"Rare": []`
# declares a leaf and the code was right to keep it - the first version of this check had that
# backwards and the failure was the test's.
tl = imbalanced({"A": A, "B": B},
                tree={"children": {"root": ["Big", "Mid", "Rare"], "Rare": ["Sub1", "Sub2"]}})
check("with a tree, an INTERNAL node is not a target", "Rare" not in tl, str(sorted(tl)))
check("...and its leaf siblings still are", "Mid" in tl, str(sorted(tl)))

print("")
print("the search")
f = find("Rare", SWEEP["B"], CLUS["B"], RUNGS)
check("finds the first rung where a cluster comes back as the target", f["rung"] == 2.0)
check("and names the cluster, not just the cells", f["clusters"] == ["3"], str(f["clusters"]))
check("a target no rung produces returns None",
      find("Nope", SWEEP["B"], CLUS["B"], RUNGS) is None)

print("")
print("ONLY THE LOCATED CLUSTER MOVES - the guarantee")
new, org, rec = rescue({"A": A, "B": B}, SWEEP, CLUS, RUNGS)
check("the searched unit gains exactly the located cells",
      list(new["B"]) == ["Big"] * 6 + ["Mid"] + ["Rare"] * 3, str(list(new["B"])))
check("THE LABEL IS NOT TRUNCATED - `<U3` holding 'Big' must not make 'Rare' into 'Rar'",
      set(new["B"]) == {"Big", "Mid", "Rare"}, str(sorted(set(new["B"]))))
check("the unit that CARRIED the label is untouched", list(new["A"]) == list(A))
check("origin marks exactly the renamed cells",
      [i for i, x in enumerate(org["B"]) if x == RESCUED] == [7, 8, 9], str(list(org["B"])))
check("every other cell is kept", sum(1 for x in org["B"] if x == KEPT) == 7)
# THE INVARIANT, stated as an equality rather than a count: a cell changed label if and only if
# it is marked rescued. A count of each would pass while naming different cells.
changed = np.asarray(new["B"]) != np.asarray(B).astype(str)
check("changed and rescued are the SAME CELLS, not merely the same number",
      list(changed) == [x == RESCUED for x in org["B"]])
check("the record's count agrees with the arrays", rec["n_renamed"] == int(changed.sum()) == 3)
check("and it names where the cells came from", rec["moved"][0]["from"] == {"Mid": 3},
      str(rec["moved"][0]["from"]))

print("")
print("nothing found - and the two kinds of nothing")
r = reach(n_unit=10000, n_clusters_finest=20, rate_pct=0.5)
check("a 50-cell population against 500-cell clusters COULD NOT have been found",
      r["could_form"] is False and round(r["expected_cells"]) == 50, str(r))
r2 = reach(n_unit=10000, n_clusters_finest=20, rate_pct=8.0)
check("an 800-cell population against 500-cell clusters could have been", r2["could_form"] is True)
check("the comparison has no free parameter - only measured quantities",
      set(r) == {"expected_cells", "mean_cluster_finest", "could_form"}, str(sorted(r)))
_, _, rec2 = rescue({"A": A, "B": np.array(["Big"] * 10)},
                    {"A": {1.0: A, 2.0: A}, "B": {1.0: np.array(["Big"] * 10),
                                                  2.0: np.array(["Big"] * 10)}},
                    CLUS, RUNGS)
check("a miss is recorded with its reach, not as a bare zero",
      rec2["not_found"] and "could_form" in rec2["not_found"][0], str(rec2["not_found"][:1]))
check("and the rungs it searched are named", rec2["not_found"][0]["searched"] == ["1.0", "2.0"])

print("")
print("no threshold anywhere in the module")
src = (Path(__file__).resolve().parents[1] / "scanno" / "rescue.py").read_text("utf-8")
import ast
nums = [n.value for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Constant) and isinstance(n.value, float)]
check("the module declares no floating-point constant to compare against",
      not [x for x in nums if x not in (0.0, 1.0, 100.0)], str(nums))

print("")
print("summarise reads the arrays, not the record")
summ = summarise({"A": A, "B": B}, new)
rows = {(r["unit"], r["label"]): r for r in summ["rows"]}
check("only labels whose count changed are listed", set(u for u, _l in rows) == {"B"}, str(rows))
check("Rare in B goes 0 -> 3", rows[("B", "Rare")]["n_before"] == 0
      and rows[("B", "Rare")]["n_after"] == 3)
check("Mid in B goes 4 -> 1", rows[("B", "Mid")]["n_delta"] == -3)
check("percentages use every cell of the unit", rows[("B", "Rare")]["pct_after"] == 30.0)

print("")
print("the document renders and names what it cannot show")
try:
    from scanno.rescue import document
    html = document({"record": rec, "summary": summ, "label_key": "cell_type_forced",
                     "version": "0.0.0", "generated": "now"})
    check("it renders", "<h1>" in html and "Targeted rescue" in html)
    check("it states the co-membership limit", "co-membership is not identity" in html)
    check("it separates UNDECIDED misses from real absences", "UNDECIDED" in html or
          "could have found it" in html)
except Exception as e:                                    # noqa: BLE001
    check("it renders", False, repr(e))

print("")
if FAILED:
    print(f"FAILED {len(FAILED)}: {', '.join(FAILED)}")
    raise SystemExit(1)
print("all rescue checks pass")
