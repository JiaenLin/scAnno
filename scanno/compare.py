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
            cluster_key=None, group_key=None, sentinels=("EXCLUDED", "UNRESOLVED")) -> dict:
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
        grp = (np.asarray(B[group_key].astype(str))
               if group_key and group_key in B else None)

        # Which samples carry no cell of a label ANYWHERE, over the whole comparison rather than
        # within a cluster. A label missing from one cluster but present elsewhere in that sample
        # is an ordinary boundary disagreement; a label the sample does not have AT ALL is the
        # only shape that can be a population its own clustering could not separate.
        all_samples = sorted(set(sam.tolist()))
        lacking = {}
        for lab in sorted(set(pa.tolist())):
            if lab in sentinels:
                continue      # EXCLUDED is not a cell type and a sample having none of it is
                              # not evidence of anything. Reading them here produced 21 of 87
                              # candidates on the first real run, and the samples they named
                              # were exactly the four the upstream flag never touched.
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

            # THE CANDIDATE IS ANCHORED ON ROUTE B'S OWN CALL FOR THIS CLUSTER, not on any
            # label that happens to appear inside it. Without that anchor the rule fired on
            # every rare label in every cluster - 87 candidates over 23 clusters on the first
            # real run, the largest claiming 8,749 cells should become `Neural` on the evidence
            # of THREE Neural cells. A joint clustering is only evidence about a population if
            # the joint route ANNOTATED it as that population.
            labs_b, ns_b = np.unique(pb[m], return_counts=True)
            L = str(labs_b[ns_b.argmax()])
            if L in sentinels:
                continue
            s_lack = [x for x in sorted(xt) if x in lacking.get(L, ())]
            if not s_lack:
                continue
            s_with = sorted(x for x, d in xt.items() if L in d)

            # Only the cells that would actually move: in this cluster, from a sample with no
            # L anywhere, and not already sentinel. A withheld nucleus was never annotated, so
            # there is no call to move.
            moving, moving_by_sample = {}, {}
            for x in s_lack:
                per = {}
                for lab, n in xt[x].items():
                    if lab in sentinels or lab == L:
                        continue
                    moving[lab] = moving.get(lab, 0) + n
                    per[lab] = per.get(lab, 0) + n
                if per:
                    moving_by_sample[x] = per
            if not moving:
                continue
            M = max(sorted(moving), key=lambda k: moving[k])

            # How much of this cluster route A ALREADY calls L. This is the credibility of the
            # joint call and it is a measurement, not a threshold: a cluster the two routes
            # mostly agree on is a population one of them resolved better, and one they mostly
            # disagree on is the joint route asserting something the per-sample route denies.
            n_agree = int(sum(d.get(L, 0) for d in xt.values()))

            # WHERE the moving cells sit across a caller-named column. This is rule one's third
            # question - is the change differential across the design - and it is REPORTED, never
            # acted on: it takes no part in deciding whether a cluster is a candidate, and a test
            # asserts the candidate set is identical with and without it. The tool names the
            # levels it was given and does not know what they mean; a design-differential GATE
            # was built here once, refused a real comparison, and was removed.
            by_group = {}
            if grp is not None:
                for x in s_lack:
                    sel = m & (sam == x) & ~np.isin(pa, list(sentinels)) & (pa != L)
                    for g, n in zip(*np.unique(grp[sel], return_counts=True)):
                        by_group[str(g)] = by_group.get(str(g), 0) + int(n)
            candidates.append({
                "cluster": str(c), "n_cluster": int(m.sum()),
                "label_absent": L, "label_carried": str(M),
                "samples_with": s_with, "samples_lacking": s_lack,
                "n_cells": int(sum(moving.values())),
                "n_route_a_agrees": n_agree,
                "pct_route_a_agrees": round(100.0 * n_agree / max(1, int(m.sum())), 1),
                "top_sample": str(vals[cnt.argmax()]),
                "top_share_pct": round(100 * top, 1),
                "moving_by_group": by_group,
                "moving_by_sample": moving_by_sample,
            })
        out["b_dominance"] = {
            "threshold_pct": round(100 * DOMINANCE),
            "n_clusters": len(rows), "n_dominated": int(dominated),
            "clusters": sorted(rows, key=lambda r: -r["top_share_pct"]),
        }
        out["merge_candidates"] = {
            "rule": "a route-B cluster whose OWN delivered label is L, holding cells from "
                    "samples that carry no L anywhere in route A. No threshold. Sentinels are "
                    "not labels and take no part.",
            "limit": "co-membership is not a label. A candidate says these cells group with "
                     "cells called X, not that they score as X, and a cluster that is mostly "
                     "one animal cannot arbitrate anything - read top_share_pct beside every "
                     "row. Which samples are named here is a fact about the samples; whether "
                     "that pattern follows a study's arms is the reader's to judge.",
            "n_candidates": len(candidates),
            "candidates": sorted(candidates, key=lambda r: -r["n_cells"]),
            "impact": _impact(candidates, pa, group_key),
            "impact_per_sample": _impact_per_sample(candidates, pa, sam, sentinels),
            "crosstab": crosstab,
        }
    return out


def _impact(candidates, pa, group_key):
    """What adopting every candidate would do to route A's composition, per label.

    Derived FROM the candidate rows, never computed beside them: a summary that recomputes its
    own numbers can disagree with the table it summarises, and nothing on the page says which
    half is right.
    """
    import numpy as np

    rows = {}
    for r in candidates:
        L = r["label_absent"]
        d = rows.setdefault(L, {"label": L, "n_would_move": 0, "n_clusters": 0,
                                "from_labels": {}, "by_group": {}})
        d["n_would_move"] += r["n_cells"]
        d["n_clusters"] += 1
        d["from_labels"][r["label_carried"]] = (
            d["from_labels"].get(r["label_carried"], 0) + r["n_cells"])
        for g, n in (r.get("moving_by_group") or {}).items():
            d["by_group"][g] = d["by_group"].get(g, 0) + n
    out = []
    for L, d in rows.items():
        now = int((pa == L).sum())
        d["n_route_a_now"] = now
        d["n_route_a_after"] = now + d["n_would_move"]
        d["fold_change"] = (round((now + d["n_would_move"]) / now, 2) if now else None)
        out.append(d)
    out.sort(key=lambda r: -r["n_would_move"])
    return {"group_key": group_key, "labels": out,
            "n_cells_total": int(sum(r["n_would_move"] for r in out)),
            "limit": "this is what adopting EVERY candidate would do. It is an arithmetic "
                     "consequence of the table above, not a recommendation, and a candidate "
                     "whose cluster is mostly one animal or whose route-A agreement is low "
                     "should not be adopted at all."}


def _impact_per_sample(candidates, pa, sam, sentinels):
    """Every label's share of every sample, before and after adopting all candidates.

    THE DENOMINATOR IS EVERY NUCLEUS OF THAT SAMPLE, sentinels included, and it does not change:
    adoption moves cells between labels and adds or removes none. So a percentage-point delta
    here is directly comparable across samples of different size, which a share of the moving
    set would not be.

    Derived from the candidate rows, like the cohort summary, so the two cannot disagree.
    """
    import numpy as np

    samples = sorted(set(sam.tolist()))
    before = {x: {} for x in samples}
    total = {x: 0 for x in samples}
    for x in samples:
        mask = sam == x
        total[x] = int(mask.sum())
        labs, ns = np.unique(pa[mask], return_counts=True)
        before[x] = {str(a_): int(b_) for a_, b_ in zip(labs, ns)}

    after = {x: dict(d) for x, d in before.items()}
    for r in candidates:
        L = r["label_absent"]
        for x, per in (r.get("moving_by_sample") or {}).items():
            for frm, n in per.items():
                after[x][frm] = after[x].get(frm, 0) - n
                after[x][L] = after[x].get(L, 0) + n

    rows = []
    for x in samples:
        for lab in sorted(set(before[x]) | set(after[x])):
            b, a_ = int(before[x].get(lab, 0)), int(after[x].get(lab, 0))
            if not b and not a_:
                continue
            t = max(1, total[x])
            rows.append({
                "sample": x, "label": lab, "n_sample_total": total[x],
                "n_before": b, "n_after": a_, "n_delta": a_ - b,
                "pct_before": round(100.0 * b / t, 3),
                "pct_after": round(100.0 * a_ / t, 3),
                "pct_delta": round(100.0 * (a_ - b) / t, 3),
                "is_sentinel": lab in sentinels,
            })
    return {"denominator": "every nucleus of that sample, sentinels included; unchanged by "
                           "adoption, so a percentage-point delta is comparable across samples",
            "n_samples": len(samples), "rows": rows}


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
            L.append(f"      {r['n_cells']:>6,}  cluster {r['cluster']} = "
                     f"{r['label_absent']} ({r['n_cluster']:,} cells, route A already agrees on "
                     f"{r['pct_route_a_agrees']}%)   {r['label_carried']} -> "
                     f"{r['label_absent']}   lacking {','.join(r['samples_lacking'])}   "
                     f"cluster is {r['top_share_pct']}% {r['top_sample']}")
        if mc["n_candidates"]:
            L.append("      co-membership is not a label: these cells GROUP with cells called")
            L.append("      that, which is not the same as scoring as it.")
        imp = mc.get("impact") or {}
        if imp.get("labels"):
            L.append(f"  IF EVERY CANDIDATE WERE ADOPTED - {imp['n_cells_total']:,} cells move:")
            for r in imp["labels"]:
                g = ("   " + ", ".join(f"{k} {v}" for k, v in sorted(r["by_group"].items()))
                     if r["by_group"] else "")
                L.append(f"      {r['label']}: {r['n_route_a_now']:,} -> "
                         f"{r['n_route_a_after']:,}  (x{r['fold_change']}){g}")
            L.append("      an arithmetic consequence of the rows above, not a recommendation.")
    L.append("  Agreement means the labels do not depend on the clustering scheme. It does NOT")
    L.append("  mean they are correct: both routes share the tree, the corpus and the")
    L.append("  classifier, so a corpus wrong about this tissue is wrong identically in both.")
    return L
