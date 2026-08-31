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


def compare(a_obs, b_obs, *, path_key="scanno_path", path_key_b=None, sample_key=None,
            cluster_key=None, sentinels=("EXCLUDED", "UNRESOLVED")) -> dict:
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
    # The routes are separate objects and are normally annotated under different suffixes, so
    # they normally carry different column names. One key for both meant the only comparable
    # pair was two routes sharing a NAME - which is the one thing you cannot do when both
    # annotations live in one object. Measured on SAMBO: the promoted per-sample column is
    # `cell_type_forced` and the joint route's own is `scanno_resolved_path_scope`, so the
    # joint-vs-per-sample comparison this module exists for could not be expressed, and in ten
    # runs of that stage it was never once run.
    key_b = path_key_b or path_key
    pa = np.asarray(A[path_key].astype(str))
    pb = np.asarray(B[key_b].astype(str))

    scorable = ~(np.isin(pa, sentinels) | np.isin(pb, sentinels))
    out = {
        "n_a": int(len(a_obs)), "n_b": int(len(b_obs)),
        "n_shared": int(len(shared)), "n_scored": int(scorable.sum()),
        "n_sentinel_either": int((~scorable).sum()),
        "path_key": str(path_key), "path_key_b": str(key_b),
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

        # Which samples carry no cell of a label ANYWHERE, over the whole comparison rather than
        # within a cluster. A label missing from one cluster but present elsewhere in that sample
        # is an ordinary boundary disagreement; a label the sample does not have AT ALL is the
        # only shape that can be a population its own clustering could not separate.
        all_samples = sorted(set(sam.tolist()))
        lacking = {}
        for lab in sorted(set(pa.tolist())):
            have = set(sam[pa == lab].tolist())
            lacking[lab] = [x for x in all_samples if x not in have]

        rows, dominated, crosstab, candidates = [], 0, {}, []
        for c in sorted(set(clu)):
            m = clu == c
            vals, cnt = np.unique(sam[m], return_counts=True)
            top = float(cnt.max() / cnt.sum())
            dominated += top > DOMINANCE
            rows.append({"cluster": str(c), "n": int(m.sum()),
                         "top_sample": str(vals[cnt.argmax()]),
                         "top_share_pct": round(100 * top, 1)})

            # Route A's labels inside route B's cluster, BY SAMPLE. The pair count reported
            # above is flat over the whole object, so it can say `M -> L: 200` and cannot say
            # which cluster that happened in or whether it was structured by sample - and those
            # two facts are the whole difference between two routes disagreeing and a population
            # one clustering could not separate.
            xt = {}
            for x in sorted(set(sam[m].tolist())):
                labs, ns = np.unique(pa[m & (sam == x)], return_counts=True)
                xt[x] = {str(a_): int(b_) for a_, b_ in zip(labs, ns)}
            crosstab[str(c)] = xt

            present = set()
            for d in xt.values():
                present |= set(d)
            for L in sorted(present):
                s_with = sorted(x for x, d in xt.items() if L in d)
                s_lack = [x for x in sorted(xt) if x in lacking.get(L, ())]
                if not s_with or not s_lack:
                    continue
                carried = {}
                for x in s_lack:
                    for lab, n in xt[x].items():
                        carried[lab] = carried.get(lab, 0) + n
                if not carried:
                    continue
                M = max(sorted(carried), key=lambda k: carried[k])
                candidates.append({
                    "cluster": str(c), "n_cluster": int(m.sum()),
                    "label_absent": str(L), "label_carried": str(M),
                    "samples_with": s_with, "samples_lacking": s_lack,
                    "n_cells": int(sum(carried.values())),
                    "n_label_absent_in_cluster": int(sum(d.get(L, 0) for d in xt.values())),
                    "top_sample": str(vals[cnt.argmax()]),
                    "top_share_pct": round(100 * top, 1),
                })
        out["b_dominance"] = {
            "threshold_pct": round(100 * DOMINANCE),
            "n_clusters": len(rows), "n_dominated": int(dominated),
            "clusters": sorted(rows, key=lambda r: -r["top_share_pct"]),
        }
        out["merge_candidates"] = {
            "rule": "within one route-B cluster, some samples carry a label that other samples "
                    "in the same cluster do not carry ANYWHERE in route A. No threshold: the "
                    "label is behaving as a property of which sample was clustered rather than "
                    "of the cell.",
            "limit": "co-membership is not a label. A candidate says these cells group with "
                     "cells called X, not that they score as X, and a cluster that is mostly "
                     "one animal cannot arbitrate anything - read top_share_pct beside every "
                     "row. Which samples are named here is a fact about the samples; whether "
                     "that pattern follows a study's arms is the reader's to judge.",
            "n_candidates": len(candidates),
            "candidates": sorted(candidates, key=lambda r: -r["n_cells"]),
            "crosstab": crosstab,
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
    mc = res.get("merge_candidates")
    if mc is not None:
        L.append(f"  merge candidates: {mc['n_candidates']} cluster/label pairs where a label is "
                 f"absent from a sample ENTIRELY")
        for r in mc["candidates"][:4]:
            L.append(f"      {r['n_cells']:>6,}  cluster {r['cluster']}: "
                     f"{r['label_carried']} -> {r['label_absent']}   "
                     f"lacking {','.join(r['samples_lacking'])}   "
                     f"cluster is {r['top_share_pct']}% {r['top_sample']}")
        if mc["n_candidates"]:
            L.append("      co-membership is not a label: these cells GROUP with cells called")
            L.append("      that, which is not the same as scoring as it.")
    L.append("  Agreement means the labels do not depend on the clustering scheme. It does NOT")
    L.append("  mean they are correct: both routes share the tree, the corpus and the")
    L.append("  classifier, so a corpus wrong about this tissue is wrong identically in both.")
    return L
