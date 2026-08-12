"""The adversarial suite — every attack that found a defect, re-run against the fix.

This is not a unit-test suite and is not trying to be. Every case below is a defect that
shipped in a prototype, survived reading, and was caught only by attacking the code with a
hostile input. A fix nobody re-attacked is a hypothesis.

Needs PBMC 3k and PBMC 68k, which scanpy downloads. Skips cleanly without them.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import scanno as sa

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


TREE = {
    "children": {"root": ["Lymphoid", "Myeloid", "Megakaryocyte"],
                 "Lymphoid": ["T cell", "B cell", "NK cell"],
                 "Myeloid": ["Monocyte", "Dendritic cell"]},
    "members": {"Lymphoid": ["CD4 T cells", "CD8 T cells", "B cells", "NK cells"],
                "Myeloid": ["CD14+ Monocytes", "FCGR3A+ Monocytes", "Dendritic cells"],
                "Megakaryocyte": ["Megakaryocytes"],
                "T cell": ["CD4 T cells", "CD8 T cells"], "B cell": ["B cells"],
                "NK cell": ["NK cells"],
                "Monocyte": ["CD14+ Monocytes", "FCGR3A+ Monocytes"],
                "Dendritic cell": ["Dendritic cells"]},
}
TRUTH = {"CD4 T cells": "Lymphoid/T cell", "CD8 T cells": "Lymphoid/T cell",
         "B cells": "Lymphoid/B cell", "NK cells": "Lymphoid/NK cell",
         "CD14+ Monocytes": "Myeloid/Monocyte", "FCGR3A+ Monocytes": "Myeloid/Monocyte",
         "Dendritic cells": "Myeloid/Dendritic cell", "Megakaryocytes": "Megakaryocyte"}


def prep(store, X, G, L):
    cats = sorted(set(L))
    y = np.array([cats.index(x) for x in L])
    M, D, counts = sa.cluster_profile(X, y, len(cats))
    Z, usable, st = sa.standardise(M, D, G, store)
    return cats, Z, usable, st, counts


def main():
    try:
        import scanpy as sc
    except ImportError:
        print("SKIP — scanpy not installed. A SKIP is not a pass.")
        return 0
    sc.settings.datasetdir = str(Path(__file__).parent / "_data")
    A = sc.datasets.pbmc3k_processed()
    G = np.array([str(v).upper() for v in A.raw.var_names])
    L = A.obs["louvain"].astype(str).values
    store = sa.build_store([("pbmc3k", G, A.raw.X, L)],
                           {"species": "Human", "tissue": "Blood", "assay": "sc"})

    print("\nA1 · a cluster's score must not depend on what else is in the sample")
    keep = ~np.isin(L, ("Megakaryocytes", "Dendritic cells"))
    shifts = []
    full = {}
    for tag, msk in (("full", np.ones(len(L), bool)), ("dropped", keep)):
        cats, Z, usable, _, _ = prep(store, A.raw.X[msk], G, L[msk])
        W, order, _ = sa.profile_weights(store, TREE["members"],
                                         TREE["children"]["root"], usable)
        for i, c in enumerate(cats):
            full.setdefault(c, {})[tag] = float(np.max(Z[i] @ W))
    for c, d in full.items():
        if "dropped" in d and abs(d["full"]) > 1e-9:
            shifts.append(abs(d["dropped"] - d["full"]) / abs(d["full"]) * 100)
    check("store-standardised Z is composition-stable", max(shifts) < 10.0,
          f"max shift {max(shifts):.1f}% (was 34.9% against the run's own clusters)")

    print("\nA2 · a rare population must clear the detection floor on its own genes")
    cats, Z, usable, st, counts = prep(store, A.raw.X, G, L)
    gi = {str(g): i for i, g in enumerate(store.genes)}
    rare = [g for g in ("ITGA2B", "GP9") if g in gi]
    glob = np.asarray((A.raw.X > 0).mean(axis=0)).ravel()
    qi = {str(g): i for i, g in enumerate(G)}
    check("markers of a 0.57% population survive", all(usable[gi[g]] for g in rare),
          f"{', '.join(rare)} — global detection "
          f"{', '.join(f'{glob[qi[g]]*100:.2f}%' for g in rare)}, below a dataset-wide floor")

    print("\nB2 · no raw sigma may be a denominator anywhere in the package")
    bad = []
    for f in (Path(__file__).resolve().parents[1] / "scanno").glob("*.py"):
        for n, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            s = ln.strip()
            if s.startswith("#") or "safe_scale" in s:
                continue
            if "/" in s and (".std(" in s or "_sd[" in s or "_sd)" in s):
                bad.append(f"{f.name}:{n}")
    check("every scale goes through safe_scale()", not bad, ", ".join(bad))

    print("\nE1 · a declared node the store cannot represent must be named, not dropped")
    miss = sa.missing_nodes(store, {**TREE["members"], "Ghost": ["NoSuchType"]})
    check("missing nodes are detected", "Ghost" in miss and "Megakaryocyte" not in miss,
          f"found {list(miss)}")

    print("\nD4 · the rooted walk truncates rather than abstaining")
    res = sa.classify(Z, usable, TREE, store=store)
    ex = sum(1 for r in res if r["path"] == TRUTH[cats[r["cluster"]]])
    wrong = sum(1 for r in res
                if r["path"] != "UNRESOLVED"
                and not TRUTH[cats[r["cluster"]]].startswith(r["path"]))
    check("pbmc3k annotates with no errors", wrong == 0, f"{ex}/{len(res)} exact, {wrong} wrong")

    print("\nD1 · the gating statistic must separate correct from incorrect calls")
    ok = [TRUTH[cats[r['cluster']]].startswith(r["path"]) for r in res]
    auc = sa.gate_auc([r["gap"] for r in res], ok)
    check("gap AUC is not measured as noise", np.isnan(auc) or auc >= 0.6,
          "no incorrect calls to score against" if np.isnan(auc) else f"AUC {auc:.2f}")

    print(f"\n{'=' * 60}")
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all adversarial checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
