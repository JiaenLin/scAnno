"""THE JOINT ROUTE — a third annotation, and a document saying exactly what it changed.

WHAT THIS IS FOR

A per-sample clustering cannot separate a population that is too small inside any one sample.
A joint clustering of the whole cohort can, because the same cells pooled across samples reach
a size a partition can hold. Annotating that second clustering and reading the first one's
labels through it says, per cluster, where the two disagree in a way that is STRUCTURED BY
SAMPLE — some samples calling a cluster's cells one thing while others, which carry no such
label anywhere, call them another.

WHY A THIRD COLUMN AND NOT A CORRECTION IN PLACE

Because the joint route is not the authority and must not be made to look like one. It is
COARSER: it recovers populations the per-sample route merged AND merges populations the
per-sample route recovered, and `compare.lost_labels` reports the second so the first cannot
be read as a strict improvement. A tool that overwrote the label would present one route's
losses as the other's gains, with nothing on the page to say so.

So all three annotations ship side by side and the reader chooses:

    <path_key>          what the walk was willing to assert; may be UNRESOLVED
    <forced_key>        the same walk with every walked cell pushed to a leaf; no holes
    <out_key>           the forced label with the joint route's corrections applied

`<out_key>_origin` names, per cell, which of the two it came from — so the third column is a
view over the second rather than a replacement for it, and reverting is a column drop.

WHAT IS DELIBERATELY NOT GATED

Every candidate is applied. A cluster that is mostly one animal cannot arbitrate anything, and
its dominance travels with it on every row of the document — but it is REPORTED and does not
decide, because `docs/PRINCIPLES.md` §3 forbids a statistic gating an output until it has been
shown to separate correct from incorrect calls, and four statistics given veto power in this
codebase's history all made results worse. The reader weighs dominance; the tool does not.
"""
from __future__ import annotations

import numpy as np

KEPT = "kept"
CORRECTED = "joint_corrected"


def reconcile(labels, clusters, samples, candidates, sentinels=("EXCLUDED", "UNRESOLVED")):
    """Apply every merge candidate to `labels`. Returns (new_labels, origin, record).

    Derived from the SAME candidate rows the document reports, so the column and the page
    describing it cannot disagree — the failure mode where a summary recomputes its own numbers
    and nothing says which half is right.

    A cell is corrected when it sits in a candidate's cluster, came from one of the samples that
    carry none of that label anywhere, and is not already carrying that label or a sentinel. A
    withheld nucleus is never corrected: it was never annotated, so there is no call to move.
    """
    lab = np.asarray(labels).astype(str).copy()
    clu = np.asarray(clusters).astype(str)
    sam = np.asarray(samples).astype(str)
    origin = np.full(lab.shape, KEPT, dtype=object)
    sent = set(sentinels)

    moved = []
    for r in candidates:
        L = str(r["label_absent"])
        lack = set(r.get("samples_lacking") or ())
        if not lack:
            continue
        m = ((clu == str(r["cluster"])) & np.isin(sam, list(lack))
             & (lab != L) & ~np.isin(lab, list(sent)))
        n = int(m.sum())
        if not n:
            continue
        froms, counts = np.unique(lab[m], return_counts=True)
        lab[m] = L
        origin[m] = CORRECTED
        moved.append({"cluster": str(r["cluster"]), "to": L, "n": n,
                      "from": {str(a): int(b) for a, b in zip(froms, counts)},
                      "samples": sorted(lack),
                      "pct_route_a_agrees": r.get("pct_route_a_agrees"),
                      "top_share_pct": r.get("top_share_pct"),
                      "top_sample": r.get("top_sample")})

    n_corr = int((origin == CORRECTED).sum())
    record = {
        "schema": "scanno/joint@1",
        "n_cells": int(lab.size), "n_corrected": n_corr,
        "pct_corrected": round(100.0 * n_corr / max(1, lab.size), 3),
        "n_candidates_applied": len(moved),
        "moved": moved,
        "origin_values": [KEPT, CORRECTED],
        "gating": "none. Every candidate is applied and its cluster's sample dominance travels "
                  "with it. A statistic does not gate an output here until it has been shown to "
                  "separate correct from incorrect calls (docs/PRINCIPLES.md 3).",
        "limit": "co-membership is not identity. A corrected cell GROUPS with cells the joint "
                 "route called that label; it was not itself scored as one. The uncorrected "
                 "column sits beside this one and reverting is a column drop.",
    }
    return lab, origin.astype(str), record


def summarise(before, after, samples, sentinels=("EXCLUDED", "UNRESOLVED")):
    """Per-label and per-sample counts before and after, from the two arrays themselves.

    Read off the delivered columns rather than predicted from the candidate rows, so this is an
    independent check on `reconcile` rather than a restatement of it: if the two disagree, one
    of them is wrong and the document shows both.
    """
    b = np.asarray(before).astype(str)
    a = np.asarray(after).astype(str)
    sam = np.asarray(samples).astype(str)

    labels = sorted(set(b.tolist()) | set(a.tolist()))
    per_label = []
    for L in labels:
        nb, na = int((b == L).sum()), int((a == L).sum())
        if nb or na:
            per_label.append({"label": L, "n_before": nb, "n_after": na, "n_delta": na - nb,
                              "is_sentinel": L in sentinels})
    per_label.sort(key=lambda r: -abs(r["n_delta"]))

    per_sample = []
    for s in sorted(set(sam.tolist())):
        m = sam == s
        tot = int(m.sum())
        for L in labels:
            nb, na = int((b[m] == L).sum()), int((a[m] == L).sum())
            if not nb and not na:
                continue
            per_sample.append({
                "sample": s, "label": L, "n_sample_total": tot,
                "n_before": nb, "n_after": na, "n_delta": na - nb,
                "pct_before": round(100.0 * nb / max(1, tot), 3),
                "pct_after": round(100.0 * na / max(1, tot), 3),
                "pct_delta": round(100.0 * (na - nb) / max(1, tot), 3),
                "is_sentinel": L in sentinels})
    return {"per_label": per_label, "per_sample": per_sample,
            "n_changed": int((a != b).sum()),
            "denominator": "every nucleus of that sample, sentinels included. Correction "
                           "relabels and never adds or removes, so it does not move."}


def document(payload) -> str:
    """One self-contained page: the three annotations, what moved, and what it cost."""
    from .report import _CSS, _esc, _table

    cmp_ = payload["compare"]
    mc = cmp_.get("merge_candidates") or {}
    rec, summ = payload["record"], payload["summary"]
    P = [f'<div class="wrap"><h1>The joint route</h1>',
         f'<div class="sub">{_esc(payload["a_name"])} corrected against '
         f'{_esc(payload["b_name"])} &middot; scAnno {_esc(payload.get("version", ""))} '
         f'&middot; generated {_esc(payload["generated"])}</div>']

    P.append("<h2>Three annotations, and what separates them</h2>")
    P.append(_table(["column", "what it is", "cells differing from the one above"],
                    payload["columns"], numeric=("cells differing from the one above",)))
    P.append('<div class="cannot"><b>What this cannot show.</b> That any of the three is '
             'correct. They are three readings of the same evidence, and the joint route is '
             'the COARSER clustering of the two - it recovers populations the per-sample route '
             'merged and merges populations the per-sample route recovered.</div>')

    P.append(f"<h2>What the joint route changed</h2>"
             f'<p><b>{rec["n_corrected"]:,} of {rec["n_cells"]:,} cells '
             f'({rec["pct_corrected"]}%)</b> carry a different label in '
             f'<code>{_esc(payload["out_key"])}</code> than in '
             f'<code>{_esc(payload["forced_key"])}</code>, from '
             f'{rec["n_candidates_applied"]} candidate(s).</p>')
    rows = [{"cluster": m["cluster"], "to": m["to"], "cells": m["n"],
             "from": ", ".join(f"{k} {v}" for k, v in sorted(m["from"].items(),
                                                             key=lambda kv: -kv[1])[:3]),
             "route A agrees": f'{m["pct_route_a_agrees"]}%',
             "most one sample": f'{m["top_share_pct"]}% {m["top_sample"]}',
             "samples lacking it": ", ".join(m["samples"])}
            for m in sorted(rec["moved"], key=lambda r: -r["n"])]
    P.append(_table(["cluster", "to", "cells", "from", "route A agrees", "most one sample",
                     "samples lacking it"], rows, numeric=("cells",)))
    P.append('<div class="cannot"><b>What this cannot show.</b> ' + _esc(rec["gating"]) + " "
             + _esc(rec["limit"]) + "</div>")

    P.append("<h2>What it cost, per label</h2>")
    P.append(_table(["label", "n_before", "n_after", "n_delta"],
                    [r for r in summ["per_label"] if r["n_delta"]],
                    numeric=("n_before", "n_after", "n_delta")))

    lost = mc.get("lost_labels") or {}
    P.append("<h2>What the joint route could not resolve</h2>")
    if lost.get("labels"):
        P.append(_table(["label", "n", "absorbed into", "lost from"],
                        [{"label": r["label"], "n": r["n"],
                          "absorbed into": ", ".join(f"{k} {v}" for k, v in
                                                     list(r["absorbed_into"].items())[:3]),
                          "lost from": ", ".join(f"{k} {v}" for k, v in
                                                 sorted(r["by_sample"].items()))}
                         for r in lost["labels"]], numeric=("n",)))
        P.append('<div class="cannot"><b>What this cannot show.</b> ' + _esc(lost["limit"])
                 + "</div>")
    else:
        P.append("<p>No label the first route delivered is absent from the joint route.</p>")

    P.append("<h2>Per sample</h2>")
    P.append(_table(["sample", "label", "n_before", "n_after", "pct_before", "pct_after",
                     "pct_delta"],
                    [r for r in summ["per_sample"] if r["n_delta"]],
                    numeric=("n_before", "n_after", "pct_before", "pct_after", "pct_delta")))
    P.append(f'<div class="cannot"><b>The denominator.</b> {_esc(summ["denominator"])}</div>')

    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>The joint route</title>"
            f"<style>{_CSS}</style></head><body>{''.join(P)}</div></body></html>")
