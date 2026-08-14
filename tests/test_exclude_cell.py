# Exercises per-nucleus exclusion. It removes nothing from disk; every array here is synthetic.
"""The excluded set is the flag: exactly, at any clustering granularity, with no passengers.

This is the property that makes scAnno an annotation tool rather than a QC tool. It withholds the
nuclei it was handed and it has no code that can widen or narrow that set.

Section 2 is the one that matters. It re-projects one fixed flag through three different
clusterings and asserts the excluded set is identical every time. Beside it, `_retired_share_rule`
reimplements the cluster-share exclusion that scAnno carried until 0.3.0 - **it is defined here,
in the test, precisely because the package no longer ships it** - so the contrast that retired it
stays measurable rather than becoming a claim in a changelog nobody can check.

    python tests/test_exclude_cell.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from scanno import (EXCLUDED, exclusion_record_cells,  # noqa: E402
                    flag_digest, unprofilable)

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def _retired_share_rule(labels, flagged, share=0.5):
    """The cluster-share exclusion scAnno carried until 0.3.0, kept ONLY as a reference here.

    A cluster goes once `share` of it is flagged, taking its unflagged members with it. Defined
    in the test rather than in the package so the two can be compared without the package being
    able to do it.
    """
    labels = np.asarray(labels)
    flagged = np.asarray(flagged, dtype=bool)
    out = np.zeros(labels.shape, dtype=bool)
    for c in np.unique(labels):
        m = labels == c
        if m.any() and float(flagged[m].mean()) >= share:
            out |= m
    return out


print("\n1 - the retired rule takes passengers; the flag itself cannot")
# cluster 0: 6 of 10 flagged -> the share rule excludes all 10, four of them unflagged
labels = np.array([0] * 10 + [1] * 10)
flag = np.array([True] * 6 + [False] * 4 + [False] * 10)

retired = _retired_share_rule(labels, flag)
check("the retired rule excludes the whole cluster", int(retired.sum()) == 10,
      f"{int(retired.sum())} cells")
check("...taking 4 unflagged passengers with it",
      int((retired & ~flag).sum()) == 4, f"{int((retired & ~flag).sum())}")

rec = exclusion_record_cells(flag, labels, 2, reason="test")
check("scAnno excludes exactly the flagged", rec["cells_excluded"] == int(flag.sum()),
      f"{rec['cells_excluded']}")
check("...and says so: zero passengers", rec["passengers"] == 0)
check("no cluster is emptied here", rec["clusters_excluded"] == [],
      str(rec["clusters_excluded"]))

print("\n2 - one fixed flag, three clusterings: the excluded set does not move")
# The flag is CONTIGUOUS and the clusterings are nested blocks, deliberately. A uniformly
# scattered flag is the one fixture on which the retired rule looks harmless: at 20% overall it
# never reaches a majority anywhere, so it excludes nothing at every granularity and the contrast
# this section exists to measure cannot appear. A fixture that cannot express the failure is not
# a test of it. Concentrated, one flag produces all three pathologies at once:
#     k=2   excludes NOTHING though 80 nuclei are flagged   (the flag is not honoured)
#     k=8   excludes 100, of which 20 carry no flag         (passengers)
#     k=40  excludes 80                                     (the size moved with granularity)
n, n_flagged = 400, 80
flag2 = np.zeros(n, dtype=bool)
flag2[:n_flagged] = True                                 # one fixed set of flagged nuclei
sizes, digests = [], set()
for k in (2, 8, 40):                                     # coarse, medium, fine
    lab = np.arange(n) // (n // k)                       # nested block clusterings
    retired_mask = _retired_share_rule(lab, flag2)
    n_retired = int(retired_mask.sum())
    passengers = int((retired_mask & ~flag2).sum())
    missed = int((flag2 & ~retired_mask).sum())
    r = exclusion_record_cells(flag2, lab, k, reason="t")
    sizes.append((k, n_retired, r["cells_excluded"], passengers, missed))
    digests.add(r["flag_digest"])
    print(f"     k={k:<3} retired rule {n_retired:>4} "
          f"({passengers:>3} unflagged taken, {missed:>3} flagged missed)"
          f"   scAnno {r['cells_excluded']:>4}")
check("the same count at every granularity", len({c for _, _, c, _, _ in sizes}) == 1,
      str([c for _, _, c, _, _ in sizes]))
check("and it is exactly the flag", {c for _, _, c, _, _ in sizes} == {n_flagged})
check("the digest is the same set, not merely the same size", len(digests) == 1, str(digests))
check("the retired rule, by contrast, moved with the clustering",
      len({r for _, r, _, _, _ in sizes}) > 1, str([r for _, r, _, _, _ in sizes]))
check("...took nuclei that carried no flag",
      max(p for _, _, _, p, _ in sizes) > 0, str([p for _, _, _, p, _ in sizes]))
check("...and at one granularity honoured none of the flag at all",
      max(m for _, _, _, _, m in sizes) == n_flagged, str([m for _, _, _, _, m in sizes]))

print("\n3 - a cluster whose every cell is flagged has no profile and must not be walked")
lab3 = np.array([0, 0, 0, 1, 1, 1])
flag3 = np.array([True, True, True, False, False, False])
emptied = unprofilable(lab3, ~flag3, 2)
check("the all-flagged cluster is unprofilable", bool(emptied[0]))
check("the untouched cluster is not", not bool(emptied[1]))
check("it is reported in the record",
      exclusion_record_cells(flag3, lab3, 2, reason="t")["clusters_excluded"] == [0])
# The emptied cluster is the only cluster-level exclusion left, and it must never reach an
# unflagged nucleus. Asserted rather than argued: every cell of an emptied cluster is flagged.
r3 = exclusion_record_cells(flag3, lab3, 2, reason="t")
in_emptied = np.isin(lab3, r3["clusters_excluded"])
check("no unflagged nucleus sits in an emptied cluster",
      int((in_emptied & ~flag3).sum()) == 0, f"{int((in_emptied & ~flag3).sum())}")

print("\n4 - the record refuses to exist without a reason, or without anything to keep")
try:
    exclusion_record_cells(flag3, lab3, 2, reason="  ")
    check("blank reason refused", False)
except ValueError:
    check("blank reason refused", True)
try:
    exclusion_record_cells(np.ones(6, dtype=bool), lab3, 2, reason="t")
    check("a flag covering everything refused", False)
except ValueError:
    check("a flag covering everything refused", True)

print("\n5 - there is no mode to choose, so a caller cannot select the retired behaviour")
import scanno  # noqa: E402
check("no MODES tuple exists", not hasattr(scanno, "MODES"))
check("no cluster_flags exists", not hasattr(scanno, "cluster_flags"))
check("the record still names its mode for the reader", rec["mode"] == "cell")
check("the sentinel is spelled so nothing mistakes it for a cell type", EXCLUDED == "EXCLUDED")
check("the digest is short enough to print in a report", len(flag_digest(flag)) == 16)

print("\n" + "=" * 70)
if fails:
    print(f"per-nucleus exclusion: {len(fails)} FAILED - " + ", ".join(fails))
    sys.exit(1)
print("per-nucleus exclusion OK - exactly the flagged nuclei, at any clustering granularity")
