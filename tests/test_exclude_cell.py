# Exercises per-CELL exclusion. It removes nothing from disk; every array here is synthetic.
"""Per-cell exclusion: exactly the flagged nuclei, no passengers, no dependence on granularity.

The cluster mode and the cell mode answer different questions, and the difference is not a
refinement - it is a change of what gets excluded:

  cluster  a whole cluster that is >= share flagged, INCLUDING its unflagged members
  cell     the flagged nuclei, and nothing else

Section 2 is the one that matters. It re-projects one fixed flag through three different
clusterings and asserts that the cell mode returns the same set every time while the cluster mode
does not. That is the property the mode was added for: a flag computed once, upstream, must not
change meaning because something downstream chose a different resolution.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from scanno import (CELL, CLUSTER, MODES, cluster_flags,  # noqa: E402
                    exclusion_record_cells, unprofilable)

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


print("\n1 - the cluster mode takes passengers; the cell mode does not")
# cluster 0: 6 of 10 flagged -> cluster mode excludes all 10, four of them unflagged
labels = np.array([0] * 10 + [1] * 10)
flag = np.array([True] * 6 + [False] * 4 + [False] * 10)

drop = cluster_flags(labels, flag, 2, share=0.5)
cluster_excluded = np.isin(labels, np.flatnonzero(drop))
check("cluster mode excludes the whole cluster", int(cluster_excluded.sum()) == 10,
      f"{int(cluster_excluded.sum())} cells")
check("cluster mode takes 4 unflagged passengers",
      int((cluster_excluded & ~flag).sum()) == 4, f"{int((cluster_excluded & ~flag).sum())}")

rec = exclusion_record_cells(flag, labels, 2, reason="test")
check("cell mode excludes exactly the flagged", rec["cells_excluded"] == 6,
      f"{rec['cells_excluded']}")
check("cell mode reports zero passengers", rec["passengers"] == 0)
check("cell mode empties no cluster here", rec["clusters_excluded"] == [],
      str(rec["clusters_excluded"]))

print("\n2 - one fixed flag, three clusterings: the cell mode is invariant")
rng = np.random.default_rng(0)
n = 400
flag2 = np.zeros(n, dtype=bool)
flag2[rng.choice(n, 80, replace=False)] = True          # one fixed set of flagged nuclei
sizes = []
for k in (2, 8, 40):                                     # coarse, medium, fine
    lab = rng.integers(0, k, size=n)
    drop_k = cluster_flags(lab, flag2, k, share=0.5)
    n_cluster_mode = int(np.isin(lab, np.flatnonzero(drop_k)).sum())
    n_cell_mode = int(exclusion_record_cells(flag2, lab, k, reason="t")["cells_excluded"])
    sizes.append((k, n_cluster_mode, n_cell_mode))
    print(f"     k={k:<3} cluster mode {n_cluster_mode:>4}   cell mode {n_cell_mode:>4}")
check("cell mode returns the same count at every granularity",
      len({c for _, _, c in sizes}) == 1, str([c for _, _, c in sizes]))
check("cell mode returns exactly the flag", {c for _, _, c in sizes} == {int(flag2.sum())})

print("\n3 - a cluster whose every cell is flagged has no profile and must not be walked")
lab3 = np.array([0, 0, 0, 1, 1, 1])
flag3 = np.array([True, True, True, False, False, False])
emptied = unprofilable(lab3, ~flag3, 2)
check("the all-flagged cluster is unprofilable", bool(emptied[0]))
check("the untouched cluster is not", not bool(emptied[1]))
check("it is reported in the record",
      exclusion_record_cells(flag3, lab3, 2, reason="t")["clusters_excluded"] == [0])

print("\n4 - the record refuses to exist without a reason")
try:
    exclusion_record_cells(flag3, lab3, 2, reason="  ")
    check("blank reason refused", False)
except ValueError:
    check("blank reason refused", True)

print("\n5 - the modes are named, so a caller cannot invent a third")
check("MODES is exactly (cell, cluster)", MODES == (CELL, CLUSTER), str(MODES))

print("\n" + "=" * 66)
if fails:
    print(f"per-cell exclusion: {len(fails)} FAILED - " + ", ".join(fails))
    sys.exit(1)
print("per-cell exclusion OK - exactly the flagged nuclei, at any clustering granularity")
