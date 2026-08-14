"""Two routes to the same labels, and how far they agree.

Every label rests on one clustering, and nothing in the classifier tests whether the labels would
survive a different one. This compares two annotated objects over the cells they share.

The intended pair is a PER-SAMPLE route against a JOINT one - the same tree, the same corpus, the
same classifier and the same gene background, so the only thing that differs is how the cells
were grouped before scoring. Build it out of the verbs that already exist:

    scanno cluster  --h5ad cohort.h5ad --split-by sample --out-dir per_sample/   # route A
    scanno cluster  --h5ad cohort.h5ad --out joint.h5ad                          # route B
    scanno annotate ... each of them ...
    scanno compare  --a per_sample_annotated.h5ad --b joint_annotated.h5ad

WHAT AGREEMENT MEANS HERE, AND WHAT IT DOES NOT

Agreement means the labels do not depend on the clustering scheme, which is the strongest
statement available without a truth set. It does NOT mean the labels are correct: both routes
share the tree, the corpus and the classifier, so a corpus that is wrong about this tissue is
wrong identically in both, and they will agree beautifully.

THE JOINT ROUTE IS USUALLY THE WEAKER ONE, AND THIS SAYS SO

On an un-integrated cohort a joint clustering may group cells by library rather than by cell
type. If it does, disagreement indicts the joint clustering rather than the per-sample one - so
sample dominance of each joint cluster is measured and reported beside the agreement, and a route
whose clusters are mostly one animal cannot arbitrate anything.
"""
from __future__ import annotations

#: A cluster this much one sample is a library, not a population. Reported, never acted on: it
#: is a fact about the comparison's weaker arm and the reader's to weigh.
DOMINANCE = 0.80


def level(paths, depth):
    """Truncate `root/a/b` to `depth` components. Sentinels pass through whole."""
    out = []
    for p in paths:
        s = str(p)
        out.append(s if s in ("EXCLUDED", "UNRESOLVED") else "/".join(s.split("/")[:depth]))
    return out


def compare(a_obs, b_obs, *, path_key="scanno_path", sample_key=None, cluster_key=None,
            sentinels=("EXCLUDED", "UNRESOLVED")) -> dict:
    """Agreement between two annotations over the cells both actually annotated.

    `a_obs` and `b_obs` are DataFrames indexed by cell id. Only the intersection is scored, and
    cells carrying a sentinel in either route are excluded from the denominator - a route that
    withheld a nucleus has not disagreed about it, and counting that as a disagreement would make
    an exclusion look like an error.
    """
    import numpy as np
    import pandas as pd

    shared = a_obs.index.intersection(b_obs.index)
    A, B = a_obs.loc[shared], b_obs.loc[shared]
    pa = np.asarray(A[path_key].astype(str))
    pb = np.asarray(B[path_key].astype(str))

    scorable = ~(np.isin(pa, sentinels) | np.isin(pb, sentinels))
    out = {
        "n_a": int(len(a_obs)), "n_b": int(len(b_obs)),
        "n_shared": int(len(shared)), "n_scored": int(scorable.sum()),
        "n_sentinel_either": int((~scorable).sum()),
        "levels": [],
    }
    for depth in (1, 2):
        la = np.asarray(level(pa[scorable], depth))
        lb = np.asarray(level(pb[scorable], depth))
        agree = float((la == lb).mean()) if la.size else float("nan")
        # Where they differ, WHICH pairs - a single confusable pair is a different finding from
        # a route that disagrees everywhere, and one number cannot tell them apart.
        pairs = {}
        for x, y in zip(la[la != lb], lb[la != lb]):
            pairs[f"{x} -> {y}"] = pairs.get(f"{x} -> {y}", 0) + 1
        out["levels"].append({
            "depth": depth,
            "agreement_pct": (None if agree != agree else round(100 * agree, 2)),
            "n_disagree": int((la != lb).sum()),
            "top_disagreements": sorted(pairs.items(), key=lambda kv: -kv[1])[:8],
        })

    # How much of route B is one sample. The control against reading the agreement as a verdict.
    if sample_key and cluster_key and sample_key in B and cluster_key in B:
        sam = np.asarray(B[sample_key].astype(str))
        clu = np.asarray(B[cluster_key].astype(str))
        rows, dominated = [], 0
        for c in sorted(set(clu)):
            m = clu == c
            vals, cnt = np.unique(sam[m], return_counts=True)
            top = float(cnt.max() / cnt.sum())
            dominated += top > DOMINANCE
            rows.append({"cluster": str(c), "n": int(m.sum()),
                         "top_sample": str(vals[cnt.argmax()]),
                         "top_share_pct": round(100 * top, 1)})
        out["b_dominance"] = {
            "threshold_pct": round(100 * DOMINANCE),
            "n_clusters": len(rows), "n_dominated": int(dominated),
            "clusters": sorted(rows, key=lambda r: -r["top_share_pct"]),
        }
    return out


def format_report(res, a_name="A", b_name="B") -> list:
    """The comparison as lines, with the limit attached rather than left to a footnote."""
    L = [f"{a_name} {res['n_a']:,} cells   {b_name} {res['n_b']:,} cells   "
         f"shared {res['n_shared']:,}   scored {res['n_scored']:,}"]
    if res["n_sentinel_either"]:
        L.append(f"    {res['n_sentinel_either']:,} excluded from the denominator: one route or "
                 f"the other withheld them, which is not a disagreement")
    for lv in res["levels"]:
        pct = "n/a" if lv["agreement_pct"] is None else f"{lv['agreement_pct']}%"
        L.append(f"  level {lv['depth']}: {pct} agreement   {lv['n_disagree']:,} differ")
        for pair, n in lv["top_disagreements"][:4]:
            L.append(f"      {n:>6,}  {pair}")
    dom = res.get("b_dominance")
    if dom:
        L.append(f"  {b_name} sample dominance: {dom['n_dominated']} of {dom['n_clusters']} "
                 f"clusters are more than {dom['threshold_pct']}% one sample")
        if dom["n_dominated"]:
            L.append(f"      so {b_name} is the weaker of the two routes and cannot arbitrate "
                     f"{a_name}")
    L.append("  Agreement means the labels do not depend on the clustering scheme. It does NOT")
    L.append("  mean they are correct: both routes share the tree, the corpus and the")
    L.append("  classifier, so a corpus wrong about this tissue is wrong identically in both.")
    return L
