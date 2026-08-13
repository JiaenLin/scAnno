"""The kNN diagnostic — it must find a mixed cluster, and must not invent one.

A diagnostic that never fires is decoration; one that fires on correct behaviour gets switched
off. So the cases are a clean partition (must score near 1), a deliberately mixed cluster (must
be caught by `foreign`, which is the metric that exists for it), and the arithmetic traps that
would quietly inflate every number: the self-loop, an isolated cell, and a group being counted
as bordering itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp

from scanno.neighbours import cluster_neighbourhood, format_report, label_flow

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def ring(blocks, cross=0):
    """Block-diagonal adjacency: each block fully connected, plus `cross` edges between them."""
    n = sum(blocks)
    G = np.zeros((n, n))
    off = 0
    for b in blocks:
        G[off:off + b, off:off + b] = 1.0
        off += b
    np.fill_diagonal(G, 0.0)
    for k in range(cross):
        i, j = k % blocks[0], blocks[0] + (k % blocks[1])
        G[i, j] = G[j, i] = 1.0
    return sp.csr_matrix(G)


print("\n1 · a clean partition scores near 1 and borders on nothing")
G = ring([20, 20])
lab = np.array(["A"] * 20 + ["B"] * 20)
rows = {r["group"]: r for r in cluster_neighbourhood(G, lab)}
check("agreement is 1.0 for both", all(abs(rows[g]["agreement"] - 1.0) < 1e-9 for g in "AB"),
      f"A {rows['A']['agreement']:.3f}  B {rows['B']['agreement']:.3f}")
check("no cell sits in a foreign neighbourhood",
      all(rows[g]["foreign"] == 0.0 for g in "AB"))
check("a group is not counted as bordering itself",
      all(rows[g]["borders_on"] == [] for g in "AB"),
      "self-mass must be removed or every group borders itself first")

print("\n2 · the self-loop is removed, or every agreement is inflated")
G2 = G.tolil()
G2.setdiag(1.0)
G2 = G2.tocsr()
_, share_a, _ = label_flow(G, lab)
_, share_b, _ = label_flow(G2, lab)
check("adding a diagonal changes nothing", np.allclose(share_a, share_b),
      "counting a cell as its own neighbour inflates most where a cell has fewest of them")

print("\n3 · a MIXED cluster is caught by `pieces`, and `foreign` CANNOT catch it")
# One partition ('M') holding two populations, each in a coherent neighbourhood of its own
# label. This is the case that motivated a second metric: the first version of this test
# asserted `foreign` would fire and it does not - correctly, because every cell here IS among
# its own kind. The cluster is wrong; no individual cell is misplaced.
G = ring([15, 15])
grp = np.array(["M"] * 30)
lab = np.array(["A"] * 15 + ["B"] * 15)
r = {x["group"]: x for x in cluster_neighbourhood(G, lab, clusters=grp)}["M"]
check("foreign is silent, and is right to be", r["foreign"] == 0.0,
      "every cell's plurality neighbour carries its own label")
check("agreement is silent too", abs(r["agreement"] - 1.0) < 1e-9,
      f"{r['agreement']:.3f} - a mean of 1.0 over a cluster that is two things")
check("`pieces` catches it", r["pieces"] == 2 and abs(r["largest_piece"] - 0.5) < 1e-9,
      f"{r['pieces']} components, largest {100*r['largest_piece']:.0f}%")
check("...and it has no single label", r["label"] is None)

print("\n3b · a genuinely single population is ONE piece — the metric must not cry wolf")
r = {x["group"]: x for x in cluster_neighbourhood(ring([20, 20]),
                                                  np.array(["A"] * 20 + ["B"] * 20))}["A"]
check("a coherent cluster reports one piece", r["pieces"] == 1 and r["largest_piece"] == 1.0)

print("\n4 · a cluster half-inside another's territory is caught by `foreign`, not by the mean")
# 18 cells labelled A: 12 sit among A, 6 sit among B. Mean agreement stays respectable.
G = ring([12, 6, 12])
lab = np.array(["A"] * 12 + ["A"] * 6 + ["B"] * 12)
G = G.tolil()
for i in range(12, 18):                          # wire the middle block into the B block
    for j in range(18, 30):
        G[i, j] = G[j, i] = 1.0
G = G.tocsr()
rows = {r["group"]: r for r in cluster_neighbourhood(G, lab)}
check("mean agreement alone looks acceptable", rows["A"]["agreement"] > 0.6,
      f"{100*rows['A']['agreement']:.0f}% - the mean does not report the problem")
check("...but `foreign` reports the third that is misplaced", rows["A"]["foreign"] > 0.25,
      f"{100*rows['A']['foreign']:.0f}% of A's cells have a B-plurality neighbourhood")
check("and A is reported as bordering B",
      rows["A"]["borders_on"] and rows["A"]["borders_on"][0][0] == "B")

print("\n5 · an isolated cell is reported, never averaged over")
G = ring([10, 10]).tolil()
G[0, :] = 0
G[:, 0] = 0
G = G.tocsr()
lab = np.array(["A"] * 10 + ["B"] * 10)
rows = {r["group"]: r for r in cluster_neighbourhood(G, lab)}
check("it is counted", rows["A"]["isolated"] == 1)
check("...and excluded from the mean rather than scored 0 or 1",
      abs(rows["A"]["agreement"] - 1.0) < 1e-9,
      f"{rows['A']['agreement']:.3f} - the nine connected A cells all agree")
_, share, _ = label_flow(G, lab)
check("its own share is NaN, not a number", np.isnan(share[0]))

print("\n6 · weighted and unweighted are different questions, and both are available")
G = sp.csr_matrix(np.array([[0.0, 9.0, 1.0], [9.0, 0.0, 1.0], [1.0, 1.0, 0.0]]))
lab = np.array(["A", "A", "B"])
_, w, _ = label_flow(G, lab, weighted=True)
_, u, _ = label_flow(G, lab, weighted=False)
check("weights change the answer", not np.allclose(w, u),
      f"weighted {w[0]:.2f} vs unweighted {u[0]:.2f} for cell 0")
check("the weighted answer follows the strong edge", w[0] > u[0])

print("\n7 · the report is ordered worst-first and says what it is not")
txt = format_report(cluster_neighbourhood(ring([12, 6, 12]),
                                          np.array(["A"] * 18 + ["B"] * 12)))
check("it states the caveat", "not a quality score" in txt and "THIS object" in txt)
check("it reports the range", "median" in txt)

print("\n" + "=" * 62)
if FAILED:
    print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
    raise SystemExit(1)
print("kNN diagnostic OK - finds a mixed cluster, invents none, changes no call")
