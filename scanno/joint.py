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
RECOVERED = "joint_recovered"
ABSORBED = "joint_absorbed"


def reconcile(labels, labels_b, clusters, samples, candidates,
              sentinels=("EXCLUDED", "UNRESOLVED")):
    """Apply the joint route to `labels`, IN BOTH DIRECTIONS. Returns (new_labels, origin, record).

    A joint clustering differs from a per-sample one in two ways and the column has to carry
    both, or it contradicts the document describing it:

      RECOVERED  a joint cluster's own label is one that some samples carry nowhere, so their
                 cells in it are corrected to it. The joint partition separated a population the
                 per-sample partition could not.
      ABSORBED   a label route A delivers that route B delivers NOWHERE. The joint partition
                 could not separate that population, and its cells take their own cluster's
                 route-B call.

    Applying only the first makes the joint route look strictly better than the per-sample one,
    which is the thing `lost_labels` exists to deny. Measured on a real cohort: 47 dendritic
    nuclei that route B absorbed into a macrophage cluster stayed labelled dendritic in the
    joint column while the report beside it said they had been absorbed. The report and the
    column disagreed, and nothing on the page said which was right.

    `<key>_origin` names which of the three a cell is, so the two directions never pool.

    Derived from the SAME candidate rows the document reports, so the column and the page
    describing it cannot disagree — the failure mode where a summary recomputes its own numbers
    and nothing says which half is right.

    A cell is corrected when it sits in a candidate's cluster, came from one of the samples that
    carry none of that label anywhere, and is not already carrying that label or a sentinel. A
    withheld nucleus is never corrected: it was never annotated, so there is no call to move.
    """
    lab = np.asarray(labels).astype(str).copy()
    lb = np.asarray(labels_b).astype(str)
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
        origin[m] = RECOVERED
        moved.append({"cluster": str(r["cluster"]), "to": L, "n": n,
                      "from": {str(a): int(b) for a, b in zip(froms, counts)},
                      "samples": sorted(lack),
                      "pct_route_a_agrees": r.get("pct_route_a_agrees"),
                      "pct_sweep_agrees": r.get("pct_sweep_agrees"),
                      "top_share_pct": r.get("top_share_pct"),
                      "top_sample": r.get("top_sample")})

    # DIRECTION TWO. A label route B delivers nowhere is one its partition could not separate;
    # its cells take their own cluster's route-B call, per cell, so cells of one absorbed label
    # may land on different targets - which is what the joint clustering actually says about
    # them, and pooling them onto one target would be a claim it never made.
    delivered_b = {x for x in set(lb.tolist()) if x not in sent}
    absorbed = {}
    for L in sorted(set(np.asarray(labels).astype(str).tolist())):
        if L in sent or L in delivered_b:
            continue
        m = (lab == L) & (origin == KEPT) & ~np.isin(lb, list(sent)) & (lb != L)
        n = int(m.sum())
        if not n:
            continue
        into, cnt = np.unique(lb[m], return_counts=True)
        absorbed[L] = {"n": n, "into": {str(a): int(b) for a, b in zip(into, cnt)},
                       "samples": sorted(set(sam[m].tolist()))}
        lab[m] = lb[m]
        origin[m] = ABSORBED

    n_rec = int((origin == RECOVERED).sum())
    n_abs = int((origin == ABSORBED).sum())
    n_corr = n_rec + n_abs
    record = {
        "schema": "scanno/joint@1",
        "n_cells": int(lab.size), "n_corrected": n_corr,
        "n_recovered": n_rec, "n_absorbed": n_abs,
        "pct_corrected": round(100.0 * n_corr / max(1, lab.size), 3),
        "n_candidates_applied": len(moved),
        "moved": moved,
        "absorbed": absorbed,
        "origin_values": [KEPT, RECOVERED, ABSORBED],
        "gating": "none. Every candidate is applied and its cluster's sample dominance travels "
                  "with it. A statistic does not gate an output here until it has been shown to "
                  "separate correct from incorrect calls (docs/PRINCIPLES.md 3).",
        "directions": "RECOVERED cells moved onto a label the joint partition separated and the "
                      "per-sample one did not; ABSORBED cells moved OFF a label the joint "
                      "partition could not separate at all. A coarser partition does both, and "
                      "a column carrying only the first would make it look strictly better.",
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


GRADES = ("adopt", "refuse", "undecided")

#: What the reviewer is told, once, before the candidates. It states the criteria and the two
#: things a verdict cannot do, because an agent that believes it is editing the annotation will
#: grade differently from one that knows it is annotating the annotation.
BRIEF = """You are reviewing corrections that a JOINT clustering proposes to a per-sample
annotation. A candidate is a joint cluster whose own delivered label is L, holding cells from
samples that carry no L anywhere in the first route.

Your verdict changes NO label. Every candidate is applied to the third column either way; a
verdict is a recorded note saying whether a reader should build on that correction.

Grade each on the evidence given, and nothing else:

1. AGREEMENT - the share of the cluster the first route already calls L. High means the joint
   route resolved a population the first route mostly agreed on. Low means the joint route is
   asserting something the first route DENIES on most of the cluster.
2. SAMPLE DOMINANCE - how much of the cluster is one sample. A joint clustering of an
   un-integrated cohort can group cells by library rather than by cell type, and a cluster that
   is mostly one sample cannot arbitrate anything, whatever its agreement.
3. WHERE THE CORRECTED CELLS FALL across the design levels given. A correction landing entirely
   in one level, or giving one level none of it, cannot be told apart from a technical effect
   when that level is confounded with something technical. This is the judgement the measuring
   tool deliberately does not make.
4. WHAT THE JOINT ROUTE LOST. It is the coarser partition and destroys populations as well as
   recovering them.
5. SWEEP AGREEMENT - how much of the joint route's OWN resolution sweep calls these same cells
   L. A population too rare to form a cluster at one granularity forms one at another, so a call
   every resolution makes and a call one resolution makes are different evidence even when their
   agreement and dominance are identical. Low sweep agreement means the correction is a property
   of the granularity that was chosen rather than of the cells. NOT MEASURED means route B was
   annotated at one resolution and this cannot be judged - which is not the same as low, and
   must not be graded as though it were.

"This would make the groups differ" is not a reason to refuse, and "this would make them agree"
is not a reason to adopt. The reason must be about whether the evidence SEPARATES a merged
population from an artefact.

Answer for THIS candidate only, in two lines:
GRADE: one of adopt, refuse, undecided
REASON: one sentence, citing the numbers you used."""


def review_prompt(cand, *, lost=None, group_key=None, per_sample=None, brief=True):
    """The evidence for one candidate, in words. Numbers only; no conclusion is offered.

    `brief=False` omits the standing instructions. They were repeated per candidate because the
    first design put each candidate to a provider as its own call, where every call needs them.
    A reviewer reads ONE document, and seven copies of the same nine paragraphs made an 18 KB
    file that is mostly duplication - which is not a formatting complaint: a reader who scrolls
    past the same block six times stops reading it, and the criteria are the part that matters.
    """
    L = ([BRIEF, ""] if brief else []) + [f"CANDIDATE - joint cluster {cand['cluster']}",
         f"  the joint route calls this cluster: {cand['label_absent']}",
         f"  cluster size: {cand.get('n_cluster', 0):,} cells",
         f"  cells that would be corrected: {cand['n_cells']:,}",
         f"  they currently carry: {cand['label_carried']}",
         f"  AGREEMENT - the first route already calls "
         f"{cand.get('pct_route_a_agrees')}% of this cluster {cand['label_absent']}",
         f"  SAMPLE DOMINANCE - {cand.get('top_share_pct')}% of it is one sample "
         f"({cand.get('top_sample')})",
         # How much of route B's OWN resolution sweep carries this label, over exactly the cells
         # that would move. An absence is stated rather than left blank: a candidate measured
         # without a sweep and one measured across a sweep that mostly disagreed would otherwise
         # read identically, and they are opposite findings.
         (f"  SWEEP AGREEMENT - {cand['pct_sweep_agrees']}% of the joint route's own resolution "
          f"sweep calls these cells {cand['label_absent']}"
          if cand.get("pct_sweep_agrees") is not None else
          "  SWEEP AGREEMENT - NOT MEASURED; route B was annotated at one resolution only, so "
          "how much this call depends on the granularity is unknown"),
         f"  samples carrying no {cand['label_absent']} anywhere: "
         f"{', '.join(cand.get('samples_lacking') or [])}"]
    if per_sample or cand.get("moving_by_sample"):
        d = per_sample or {k: sum(v.values()) for k, v in cand["moving_by_sample"].items()}
        L.append("  corrected cells per sample: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(d.items())))
    if group_key and cand.get("moving_by_group"):
        L.append(f"  corrected cells per {group_key}: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(cand["moving_by_group"].items())))
        L.append(f"  (levels of {group_key} receiving none of it are not listed)")
    if lost and lost.get("labels"):
        L.append("  WHAT THIS JOINT CLUSTERING LOST elsewhere in the same run:")
        for r in lost["labels"]:
            into = ", ".join(f"{k} {v}" for k, v in list(r["absorbed_into"].items())[:3])
            L.append(f"    {r['n']:,} {r['label']} absorbed into {into}")
    return "\n".join(L)


def parse_verdict(text):
    """Free text in, a graded verdict out - or `unresolved`, kept verbatim and never coerced.

    The same shape as `agent.resolve_label`: the reviewer answers in its own words and the answer
    is RESOLVED afterwards. A reply that names no grade is not silently read as `undecided`; it
    is recorded as unresolved with its text, because a reviewer that did not answer and one that
    answered "undecided" are different findings.
    """
    import re

    raw = str(text or "").strip()
    grade, reason = None, ""
    m = re.search(r"\bGRADE\s*[:\-]\s*(\w+)", raw, re.I)
    if m and m.group(1).lower() in GRADES:
        grade = m.group(1).lower()
    else:
        for g in GRADES:
            if re.search(rf"\b{g}\b", raw, re.I):
                grade = g
                break
    m = re.search(r"\bREASON\s*[:\-]\s*(.+)", raw, re.I | re.S)
    reason = (m.group(1) if m else raw).strip().replace("\n", " ")[:600]
    return {"grade": grade or "undecided", "reason": reason,
            "tier": "graded" if grade else "unresolved", "raw": raw[:2000]}


def review(candidates, verdicts, tiers=None, provenance=None):
    """Grade each candidate, from a CLOSED vocabulary, with a reason that is required.

    THE MEASUREMENT IS THE TOOL'S AND THE JUDGEMENT IS NOT. `compare` names which samples lack a
    label and stops there, because deciding that a study's arms differ is not the tool's call and
    a gate that tried it was built here once and removed. But somebody still has to decide, and a
    decision taken in conversation and typed into a document is not reproducible, cannot be
    checked against the run it describes, and is gone with the session that made it.

    So the decision is recorded HERE, against the candidate it is about, with its reason, and the
    report renders it. It changes no label: the joint column is written by `reconcile` and a
    verdict is a reader's note on it.

    Refuses a grade for a cluster that is not a candidate, a grade outside the vocabulary, and an
    empty reason. Reports every candidate with NO verdict, because an ungraded candidate is not
    adopted by silence.
    """
    by_cluster = {str(r["cluster"]): r for r in candidates}
    graded, errors = {}, []
    for cl, (grade, reason) in verdicts.items():
        cl = str(cl)
        if cl not in by_cluster:
            errors.append(f"cluster {cl!r} is not a candidate; candidates are "
                          f"{', '.join(sorted(by_cluster)) or 'none'}")
            continue
        if grade not in GRADES:
            errors.append(f"cluster {cl!r}: grade {grade!r} is not one of {', '.join(GRADES)}")
            continue
        if not str(reason).strip():
            errors.append(f"cluster {cl!r}: a verdict needs a reason")
            continue
        r = by_cluster[cl]
        graded[cl] = {"cluster": cl, "grade": grade, "reason": str(reason).strip(),
                      "label_absent": r["label_absent"], "n_cells": r["n_cells"],
                      "pct_route_a_agrees": r.get("pct_route_a_agrees"),
                      "top_share_pct": r.get("top_share_pct"),
                      "tier": (tiers or {}).get(cl, "declared")}
    ungraded = sorted(set(by_cluster) - set(graded), key=lambda c: -by_cluster[c]["n_cells"])
    return {"schema": "scanno/joint-review@1", "grades": GRADES,
            "provenance": provenance or {"source": "declared"},
            "n_unresolved": sum(1 for v in graded.values() if v.get("tier") == "unresolved"),
            "n_candidates": len(by_cluster), "n_graded": len(graded),
            "verdicts": graded, "ungraded": ungraded, "errors": errors,
            "n_cells_by_grade": {g: sum(v["n_cells"] for v in graded.values()
                                        if v["grade"] == g) for g in GRADES},
            "n_cells_ungraded": sum(by_cluster[c]["n_cells"] for c in ungraded),
            "limit": "a verdict is a reader's note recorded against the run. It changes no "
                     "label: the joint column is what `reconcile` wrote, and an ungraded "
                     "candidate is NOT adopted by silence - it is applied in the column like "
                     "every other and simply has nobody's name against it."}


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
             f'({rec["pct_corrected"]}%)</b> &mdash; '
             f'<b>{rec.get("n_recovered", 0):,} recovered</b> onto a label the joint partition '
             f'separated, <b>{rec.get("n_absorbed", 0):,} absorbed</b> off a label it could not '
             f'separate at all &mdash; carry a different label in '
             f'<code>{_esc(payload["out_key"])}</code> than in '
             f'<code>{_esc(payload["forced_key"])}</code>, from '
             f'{rec["n_candidates_applied"]} candidate(s).</p>')
    rows = [{"cluster": m["cluster"], "to": m["to"], "cells": m["n"],
             "from": ", ".join(f"{k} {v}" for k, v in sorted(m["from"].items(),
                                                             key=lambda kv: -kv[1])[:3]),
             "route A agrees": f'{m["pct_route_a_agrees"]}%',
             # An unmeasured sweep is NAMED, not left blank. A candidate whose sweep mostly
             # disagreed and one measured without a sweep are opposite findings, and an empty
             # cell reads as the first.
             "sweep agrees": (f'{m["pct_sweep_agrees"]}%'
                              if m.get("pct_sweep_agrees") is not None else "not measured"),
             "most one sample": f'{m["top_share_pct"]}% {m["top_sample"]}',
             "samples lacking it": ", ".join(m["samples"])}
            for m in sorted(rec["moved"], key=lambda r: -r["n"])]
    P.append(_table(["cluster", "to", "cells", "from", "route A agrees", "sweep agrees",
                     "most one sample", "samples lacking it"], rows, numeric=("cells",)))
    P.append('<div class="cannot"><b>What this cannot show.</b> ' + _esc(rec["gating"]) + " "
             + _esc(rec["limit"]) + "</div>")

    P.append("<h2>What it cost, per label</h2>")
    P.append(_table(["label", "n_before", "n_after", "n_delta"],
                    [r for r in summ["per_label"] if r["n_delta"]],
                    numeric=("n_before", "n_after", "n_delta")))

    if rec.get("absorbed"):
        P.append("<h3>Absorbed &mdash; labels this partition could not separate</h3>")
        P.append(_table(["label", "cells", "moved onto", "from samples"],
                        [{"label": k, "cells": v["n"],
                          "moved onto": ", ".join(f"{a} {b}" for a, b in
                                                  sorted(v["into"].items(), key=lambda kv: -kv[1])),
                          "from samples": ", ".join(v["samples"])}
                         for k, v in sorted(rec["absorbed"].items(),
                                            key=lambda kv: -kv[1]["n"])],
                        numeric=("cells",)))
        P.append('<div class="cannot"><b>What this cannot show.</b> That absorbing them was '
                 'right. These cells carried a label the FIRST route resolved and this one '
                 'delivers nowhere; the column follows the joint partition because that is what '
                 'the column is, and the earlier annotation keeps them.</div>')

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

    imp = mc.get("impact") or {}
    if imp.get("group_key"):
        P.append(f'<h2>Across <code>{_esc(imp["group_key"])}</code></h2>')
        tot = {}
        for r in imp["labels"]:
            for g, n in (r.get("by_group") or {}).items():
                tot[g] = tot.get(g, 0) + n
        P.append(_table(["label", "gained", "lost", "net"] + sorted(tot),
                        [dict({"label": r["label"], "gained": r["n_gained"],
                               "lost": r["n_lost"], "net": r["n_delta"]},
                              **{g: (r.get("by_group") or {}).get(g, 0) for g in tot})
                         for r in imp["labels"] if r["n_gained"]],
                        numeric=("gained", "lost", "net") + tuple(sorted(tot))))
        P.append('<div class="cannot"><b>What this cannot show.</b> Whether that pattern '
                 'follows the study&rsquo;s design. This names the levels it was given and does '
                 'not know what they mean; a level receiving none of a correction, or all of '
                 'it, is a fact about the correction and the reader&rsquo;s to weigh. Deciding '
                 'that a study&rsquo;s arms differ is not this tool&rsquo;s call.</div>')

    rev = payload.get("review")
    if rev:
        P.append("<h2>Verdicts</h2>")
        P.append(f'<p>{rev["n_graded"]} of {rev["n_candidates"]} candidates carry a recorded '
                 f'verdict. ' + " &middot; ".join(
                     f'<b>{_esc(g)}</b> {rev["n_cells_by_grade"].get(g, 0):,} cells'
                     for g in rev["grades"]) +
                 f' &middot; ungraded {rev["n_cells_ungraded"]:,} cells.</p>')
        P.append(_table(["cluster", "grade", "label", "cells", "reason"],
                        [{"cluster": v["cluster"], "grade": v["grade"],
                          "label": v["label_absent"], "cells": v["n_cells"],
                          "reason": v["reason"]}
                         for v in sorted(rev["verdicts"].values(), key=lambda v: -v["n_cells"])],
                        numeric=("cells",)))
        if rev["ungraded"]:
            P.append(f'<div class="cannot"><b>Ungraded.</b> cluster(s) '
                     f'{_esc(", ".join(rev["ungraded"]))} carry no verdict. '
                     + _esc(rev["limit"]) + "</div>")
        else:
            P.append(f'<div class="cannot"><b>What this cannot show.</b> {_esc(rev["limit"])}'
                     "</div>")

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
