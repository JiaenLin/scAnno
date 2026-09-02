"""TARGETED RESCUE — a rare cell type missing from one unit, looked for in that unit alone.

THE OBSERVATION THIS EXISTS FOR

A cluster-based annotation delivers a rare cell type in some units and zero in others. Those two
outcomes look identical in a composition table and are not the same event: the population may be
genuinely absent, or it may have had too few cells to form a cluster in that unit, so it was
never offered to the classifier at all. No abundance claim across units is safe until they are
told apart, and nothing in a single annotation tells them apart.

THE ASYMMETRY OF TRUST, WHICH IS THE WHOLE DESIGN

Where the label WAS called, a cluster formed and was annotated, and that call is not in question
here. Only the ZEROS are. So this is not a re-annotation and not a second opinion: it is a
targeted search for a specific label in a specific unit, and everywhere else is left alone.

THE SEARCH

For each (label, unit) pair where the unit lacks a label that some other unit carries: cluster
that unit more finely, step by step, and annotate each step in the ordinary unbiased way. At each
step ask one question — did any cluster come back as the target label? On the first step where one
did, the cells of THAT CLUSTER take the label.

WHAT IS RENAMED, AND WHAT IS NOT

Only the located cluster's cells. Nothing else in the unit moves — not the labels that shifted at
the finer granularity, not the cells that stopped being resolvable there, not the clusters that
split. The finer clustering is used to LOCATE and is then discarded; it is never adopted. A
mechanism that adopted it would be re-annotating the whole unit to recover one population, which
is a different and much larger claim.

TWO KINDS OF "NOT FOUND", AND THEY MUST NOT BE PRINTED THE SAME WAY

A search that ends without the label has two possible meanings, and the difference is arithmetic
rather than opinion. If the finest clustering reached still produces clusters LARGER than the
population would be, the label could not have been isolated whether or not the cells are there,
and the pair is UNDECIDED. If the clustering reached a granularity that could have held it and
nothing was called, the zero stands. `reach()` computes that comparison from measured quantities
only - the unit's size, the clusters it produced, and the label's rate where it exists - and it
is REPORTED. It gates nothing.

NO THRESHOLD LIVES IN THIS MODULE. How small a resolution difference, or how small a rename, has
to be before a caller believes it is the caller's judgement; both numbers travel with every row.
"""
from __future__ import annotations

import numpy as np

KEPT = "kept"
RESCUED = "rescued"
SENTINELS = ("EXCLUDED", "UNRESOLVED")


def imbalanced(labels_by_unit, sentinels=SENTINELS):
    """The trigger set: labels some units carry and others do not. Returns {label: (with, without)}.

    A label present everywhere has nothing to explain and a label present nowhere is not a
    finding about any unit, so neither is searched. With `tree`, only LEAVES are eligible: a
    cell sitting on an internal node carries a compartment name, and a compartment is not a
    population that can be missing.

    NO RARITY THRESHOLD, and none is wanted. Imbalance is the criterion; rarity is its
    consequence, because an abundant population does not go missing from a unit.

    AND NO VOCABULARY CHECK, because `scanno annotate --scope` already made one impossible to
    need: the seals leave the walk structurally unable to emit a label outside the scope, and
    `--resolve` pushes every walked cell down to a leaf of that sealed tree. A delivered column
    produced that way contains the scope's vocabulary and the sentinels, and nothing else.

    A leaf filter used to sit here and it CAUSED the error it was meant to prevent: a node the
    scope SEALED is terminal by decision while still carrying children in the declared taxonomy,
    so testing against that taxonomy dropped a label the annotation actually delivers - and it
    was zero in two units, an imbalance silently never searched. The upstream mechanism is the
    one to trust; duplicating it here only added a way to disagree with it.
    """
    sent = set(sentinels)
    units = list(labels_by_unit)
    seen = {u: (set(np.asarray(labels_by_unit[u]).astype(str).tolist()) - sent) for u in units}
    out = {}
    for lab in sorted({x for u in units for x in seen[u]}):
        have = [u for u in units if lab in seen[u]]
        lack = [u for u in units if lab not in seen[u]]
        if have and lack:
            out[lab] = (have, lack)
    return out


def find(target, sweep_by_rung, clusters_by_rung, rungs):
    """The first rung at which ANY cluster is annotated `target`. Returns a dict, or None.

    `rungs` is in search order - coarsest first - and the caller supplies it, because the order
    of a search is part of the search. `sweep_by_rung` and `clusters_by_rung` map a rung to that
    unit's per-cell label and cluster arrays.

    The mask is the cells of the located cluster(s), which is exactly the set that will be
    renamed. Where more than one cluster comes back as the target at the same rung, all of them
    are taken: they are equally the answer to the question asked.
    """
    for r in rungs:
        lab = np.asarray(sweep_by_rung[r]).astype(str)
        m = lab == str(target)
        if not m.any():
            continue
        clu = np.asarray(clusters_by_rung[r]).astype(str)
        ids = sorted(set(clu[m].tolist()))
        return {"rung": r, "clusters": ids, "mask": np.isin(clu, ids) & m,
                "n_cells": int(m.sum())}
    return None


def reach(n_unit, n_clusters_finest, rate_pct):
    """Could the finest clustering reached have HELD this population? Reported, never a gate.

    `rate_pct` is the label's percentage where it exists. The comparison is between two measured
    quantities and has no free parameter: a population of `rate_pct` of `n_unit` cells against
    the mean cluster the finest rung produced. Below it, a search that found nothing has
    established nothing - the instrument does not reach - and saying so is the finding.
    """
    mean_cluster = float(n_unit) / max(1, int(n_clusters_finest))
    expected = float(rate_pct) / 100.0 * float(n_unit)
    return {"expected_cells": expected, "mean_cluster_finest": mean_cluster,
            "could_form": bool(expected >= mean_cluster)}


def rescue(labels_by_unit, sweep_by_unit_rung, clusters_by_unit_rung, rungs,
           sentinels=SENTINELS):
    """Run every targeted search and rename ONLY what each one located.

    Returns `(new_by_unit, origin_by_unit, record)`. `new_by_unit[u]` is that unit's label array
    with the located cells relabelled and every other cell untouched; `origin_by_unit[u]` is
    `kept` or `rescued` per cell, so reverting is a column drop and the two can never disagree
    about which cells moved.

    Searches are applied in order of the rung that found them, coarsest first, so the cheapest
    rescue wins any overlap - two targets located at different granularities can name the same
    cell, and a later, more expensive search does not get to take it back. Cells declined for
    that reason are counted and named in the record rather than silently dropped.
    """
    trig = imbalanced(labels_by_unit, sentinels=sentinels)
    units = list(labels_by_unit)
    sent = set(sentinels)
    # dtype=object, NOT the incoming fixed-width string dtype. A numpy `<U3` array holding
    # "Big" silently TRUNCATES "Rare" to "Rar" on assignment - no error, no warning, a corrupted
    # label in the delivered column. Caught on the first smoke test of this module and worth the
    # comment: every rescue writes a label that is, by construction, absent from the array and
    # therefore quite likely longer than anything in it.
    new = {u: np.asarray(labels_by_unit[u]).astype(str).astype(object).copy() for u in units}
    origin = {u: np.full(len(new[u]), KEPT, dtype=object) for u in units}

    rate = {}
    for lab, (have, _lack) in trig.items():
        v = [100.0 * float((np.asarray(labels_by_unit[u]).astype(str) == lab).mean())
             for u in have]
        rate[lab] = float(np.median(v)) if v else 0.0

    hits, misses = [], []
    for lab, (have, lack) in trig.items():
        for u in lack:
            f = find(lab, sweep_by_unit_rung[u], clusters_by_unit_rung[u], rungs)
            if f is None:
                n_fin = len(set(np.asarray(clusters_by_unit_rung[u][rungs[-1]]).astype(str)))
                misses.append({"label": lab, "unit": u, "searched": [str(r) for r in rungs],
                               **reach(len(new[u]), n_fin, rate[lab])})
            else:
                hits.append({"label": lab, "unit": u, **f})

    hits.sort(key=lambda h: (rungs.index(h["rung"]), str(h["label"]), str(h["unit"])))
    moved = []
    for h in hits:
        u, lab = h["unit"], h["label"]
        free = h["mask"] & (origin[u] == KEPT) & ~np.isin(new[u], list(sent) + [lab])
        taken = int(h["mask"].sum()) - int(free.sum())
        src = {}
        if free.any():
            k, c = np.unique(new[u][free], return_counts=True)
            src = {str(a): int(b) for a, b in zip(k, c)}
            new[u][free] = lab
            origin[u][free] = RESCUED
        moved.append({"label": lab, "unit": u, "rung": str(h["rung"]),
                      "clusters": h["clusters"], "n_renamed": int(free.sum()),
                      "n_declined": taken, "from": src})

    record = {
        "schema": "scanno/rescue@1",
        "rungs": [str(r) for r in rungs],
        "n_targets": sum(len(v[1]) for v in trig.values()),
        "n_found": len(moved), "n_not_found": len(misses),
        "n_renamed": int(sum(m["n_renamed"] for m in moved)),
        "trigger": {k: {"with": v[0], "without": v[1]} for k, v in trig.items()},
        "moved": moved, "not_found": misses,
        "origin_values": [KEPT, RESCUED],
        "rule": "for a label a unit lacks and another unit carries, the unit is clustered more "
                "finely step by step and annotated in the ordinary way; on the first step where "
                "a cluster comes back as that label, THAT CLUSTER'S CELLS take it. Nothing else "
                "is renamed and the finer clustering is discarded, not adopted.",
        "limit": "co-membership is not identity: a renamed cell was in a cluster the classifier "
                 "called that label, at a granularity the delivered annotation does not use. The "
                 "original column sits beside this one and reverting is a column drop.",
        "undecided": "a search that found nothing has established nothing where the finest "
                     "clustering still produces clusters larger than the population would be. "
                     "`could_form` says which of the two a miss is; it gates nothing.",
    }
    return ({u: new[u].astype(str) for u in units},
            {u: origin[u].astype(str) for u in units}, record)


def summarise(before_by_unit, after_by_unit, sentinels=SENTINELS):
    """Per-unit, per-label counts before and after, read off the two arrays themselves.

    Derived from the delivered arrays rather than from the record, so a mistake in the record
    cannot agree with itself. A caller that reports both has checked one against the other.
    """
    sent = set(sentinels)
    rows = []
    for u in before_by_unit:
        b = np.asarray(before_by_unit[u]).astype(str)
        a = np.asarray(after_by_unit[u]).astype(str)
        n = len(b)
        for lab in sorted(set(b.tolist()) | set(a.tolist())):
            nb, na = int((b == lab).sum()), int((a == lab).sum())
            if nb == na:
                continue
            rows.append({"unit": u, "label": lab, "n_before": nb, "n_after": na,
                         "n_delta": na - nb, "pct_before": round(100.0 * nb / n, 4),
                         "pct_after": round(100.0 * na / n, 4),
                         "pct_delta": round(100.0 * (na - nb) / n, 4),
                         "is_sentinel": lab in sent})
    return {"denominator": "every cell of that unit, sentinels included", "rows": rows}


def document(payload) -> str:
    """One self-contained page: what was searched, what was found, and what moved."""
    from .report import _CSS, _esc, _table

    rec, summ = payload["record"], payload["summary"]
    P = ['<div class="wrap"><h1>Targeted rescue of missing cell types</h1>',
         f'<div class="sub">{_esc(payload["label_key"])} &middot; '
         f'{rec["n_targets"]} targeted searches over rungs '
         f'{_esc(", ".join(rec["rungs"]))} &middot; scAnno '
         f'{_esc(payload.get("version", ""))} &middot; '
         f'generated {_esc(payload["generated"])}</div>']

    P.append("<h2>What was searched, and what was not</h2>")
    P.append(f'<p>A label carried by at least one unit and absent from at least one is searched '
             f'<b>in the units that lack it, and nowhere else</b>. Where the label was called, a '
             f'cluster formed and was annotated; that call is not in question here. '
             f'<b>{len(rec["trigger"])} labels</b> are imbalanced, giving '
             f'<b>{rec["n_targets"]} targeted searches</b>.</p>')
    P.append(_table(["label", "units carrying it", "units searched"],
                    [{"label": k, "units carrying it": len(v["with"]),
                      "units searched": ", ".join(v["without"])}
                     for k, v in sorted(rec["trigger"].items())],
                    numeric=("units carrying it",)))

    P.append("<h2>What was renamed</h2>")
    P.append(f'<p><b>{rec["n_renamed"]:,} cells</b> across {rec["n_found"]} of '
             f'{rec["n_targets"]} searches. Only the located cluster&rsquo;s cells move; the '
             f'finer clustering is discarded, never adopted.</p>')
    P.append(_table(["label", "unit", "found at", "clusters", "cells renamed", "taken from"],
                    [{"label": m["label"], "unit": m["unit"], "found at": m["rung"],
                      "clusters": ", ".join(m["clusters"]), "cells renamed": m["n_renamed"],
                      "taken from": ", ".join(f"{k} {v}" for k, v in
                                              sorted(m["from"].items(), key=lambda kv: -kv[1]))}
                     for m in rec["moved"]], numeric=("cells renamed",)))
    P.append('<div class="cannot"><b>What this cannot show.</b> ' + _esc(rec["limit"]) + "</div>")

    P.append("<h2>Where nothing was found</h2>")
    P.append('<p>' + _esc(rec["undecided"]) + '</p>')
    P.append(_table(["label", "unit", "population would be", "finest cluster reached",
                     "the search could have found it"],
                    [{"label": m["label"], "unit": m["unit"],
                      "population would be": round(m["expected_cells"]),
                      "finest cluster reached": round(m["mean_cluster_finest"]),
                      "the search could have found it": "yes" if m["could_form"] else
                      "no - UNDECIDED"}
                     for m in rec["not_found"]],
                    numeric=("population would be", "finest cluster reached")))

    P.append("<h2>Every count that changed, per unit</h2>")
    P.append(_table(["unit", "label", "n_before", "n_after", "n_delta", "pct_before",
                     "pct_after", "pct_delta"], summ["rows"],
                    numeric=("n_before", "n_after", "n_delta", "pct_before", "pct_after",
                             "pct_delta")))
    P.append(f'<p class="foot">{_esc(summ["denominator"])}. Read from the delivered arrays '
             f'themselves, not from the record above, so the two cannot agree by construction.'
             f'</p>')
    P.append("</div>")
    return f"<!doctype html><meta charset='utf-8'><style>{_CSS}</style>" + "".join(P)
