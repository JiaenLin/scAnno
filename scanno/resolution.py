"""Choosing a clustering resolution from the annotation, not from the geometry.

WHY THIS IS DECIDED ON LABELS

A resolution sweep is usually judged by silhouette, modularity or a clustree picture - all of
which describe the PARTITION. What a downstream analysis consumes is the LABEL, and the two are
not the same question: a partition can shift substantially while every cell keeps its identity,
and that is the case where granularity does not matter and any choice is safe.

So the criterion here is: at which resolution does the annotation agree with what the sweep as a
whole says, and stop changing when the granularity is nudged. Nothing in this module opens a
matrix. It reads label arrays, which makes it free to run and impossible to disagree with the
annotation it is describing.

WHY IT REPORTS A TIE INSTEAD OF ALWAYS PICKING

On real sweeps the top few resolutions are routinely separated by less than the sweep's own
step-to-step variation. Declaring a winner there is picking noise. This module measures that
variation and uses it as the tolerance - so "tied" is derived from the data rather than from a
threshold someone chose - then breaks the tie on what the candidates actually cost, and reports
every candidate so the choice can be argued with.

THE TIE-BREAK, IN ORDER, AND WHY EACH COMES WHERE IT DOES

  1. stability      the question that was asked
  2. completeness   among equally stable options, prefer the one that resolves more cells to a
                    LEAF of the tree. A resolution that is stable because it truncates
                    everything is stable and useless
  3. rare retention prefer the option whose smallest named population is largest. Finer
                    clustering absorbs rare types into their dominant neighbour, and no other
                    metric here notices - a population halving costs a fraction of a point of
                    stability while being the most damaging thing a resolution choice can do
  4. parsimony      among genuinely equivalent options, the coarsest

Rare retention sits above parsimony deliberately: losing a population is irreversible for every
analysis downstream, and a few extra clusters are not.
"""
from __future__ import annotations

import numpy as np

#: Floor on the derived tolerance, in percentage points. A sweep whose adjacent resolutions
#: agree perfectly would otherwise get a tolerance of zero and call a 0.001-point difference
#: decisive. Not a tuning knob - it is the resolution below which a percentage of cells is not
#: a meaningful difference.
MIN_TOL = 0.1


def _truncate(paths, depth):
    if depth is None:
        return np.asarray([str(p) for p in paths])
    return np.asarray(["/".join(str(p).split("/")[:depth]) for p in paths])


def _modal(stack):
    """Per-cell majority label down a (n_res, n_cells) array of strings."""
    n = stack.shape[1]
    out = np.empty(n, dtype=object)
    for i in range(n):
        vals, counts = np.unique(stack[:, i], return_counts=True)
        out[i] = vals[int(np.argmax(counts))]
    return out.astype(str)


def _leaves(tree):
    """Nodes with no children - the deepest a walk can legitimately stop."""
    if not tree:
        return None
    ch = tree.get("children", {})
    seen = {c for kids in ch.values() for c in kids} | set(ch)
    return {n for n in seen if not ch.get(n)}


def sweep_stability(labels_by_res, tree=None, groups=None, depth=None,
                    unresolved="UNRESOLVED", clusters_by_res=None):
    """Per-resolution annotation stability. Labels only; nothing is recomputed.

    `labels_by_res` maps a resolution (any orderable key) to an array of one label PATH per
    cell, all arrays the same length and in the same cell order. `depth` truncates the path
    before comparing, so level-1 and level-2 stability are two calls rather than two functions.

    `groups` is an optional per-cell array - sample, donor, batch - used only to report in how
    many groups the rarest label appears. A label seen in one group of ten is not a population
    a downstream comparison can use, and nothing else here would say so.

    Returns one dict per resolution, in resolution order:

      modal        % of cells whose label equals their majority label across the whole sweep
      neighbour    % unchanged at BOTH adjacent resolutions; None at the two ends
      unresolved   % the walk could not place at all
      truncated    % whose label ends at a node that HAS children - a partial identity, which
                   is not the same failure as UNRESOLVED and is counted apart from it
      complete     % ending at a leaf. Needs `tree`; None without one
      n_labels     distinct named labels
      smallest     cells in the smallest named label
      min_groups   groups the rarest label appears in; None without `groups`
    """
    res = list(labels_by_res)
    if len(res) < 2:
        raise ValueError("a sweep needs at least two resolutions to be stable across")
    L = {r: _truncate(labels_by_res[r], depth) for r in res}
    n = len(L[res[0]])
    if any(len(L[r]) != n for r in res):
        raise ValueError("every resolution must label the same cells, in the same order")

    modal = _modal(np.vstack([L[r] for r in res]))
    leaves = _leaves(tree)
    groups = None if groups is None else np.asarray(groups)

    out = []
    for k, r in enumerate(res):
        lab = L[r]
        is_unres = np.char.startswith(lab.astype(str), unresolved)
        named = [x for x in np.unique(lab) if not str(x).startswith(unresolved)]
        sizes = {x: int((lab == x).sum()) for x in named}
        row = {
            "resolution": r,
            "modal": float((lab == modal).mean() * 100),
            "neighbour": (float(((lab == L[res[k - 1]]) & (lab == L[res[k + 1]])).mean() * 100)
                          if 0 < k < len(res) - 1 else None),
            "unresolved": float(is_unres.mean() * 100),
            "n_labels": len(named),
            "smallest": min(sizes.values()) if sizes else 0,
            # The parsimony unit. CLUSTERS if the caller supplied them, otherwise distinct
            # labels - which is a different quantity and is named as one. Labels alone cannot
            # tell you how many clusters produced them, and pretending otherwise put "19
            # clusters" next to a sweep that had 116.
            "n_units": (int(len(np.unique(np.asarray(clusters_by_res[r]))))
                        if clusters_by_res is not None else len(named)),
            "units_are": "clusters" if clusters_by_res is not None else "labels",
        }
        if leaves is not None:
            term = np.asarray([str(p).split("/")[-1] for p in labels_by_res[r]])
            comp = np.isin(term, list(leaves)) & ~is_unres
            row["complete"] = float(comp.mean() * 100)
            row["truncated"] = float(((~comp) & (~is_unres)).mean() * 100)
        else:
            row["complete"] = row["truncated"] = None
        if groups is not None and named:
            row["min_groups"] = min(len(set(groups[lab == x])) for x in named)
            row["n_groups"] = len(set(groups))
        else:
            row["min_groups"] = row["n_groups"] = None
        out.append(row)
    return out


def sweep_agreement(labels_by_res, reference, depth=None):
    """How much of the sweep agrees with a label already chosen. One number per cell.

    `sweep_stability` answers "which resolution is stable" - per RESOLUTION. This answers "which
    CELLS are stable", which is the question a route correcting one annotation with another has
    to ask: a correction resting on cells the sweep agrees about is a different claim from one
    resting on cells whose identity changes whenever the granularity is nudged, and a
    per-resolution score has already averaged over the cells.

    `reference` is the label array the run actually delivered - one resolution's annotation, the
    thing being described. This does NOT vote a new label, and that is deliberate. A voted
    column is per CELL, while an annotation is per CLUSTER, and `joint.reconcile` reads "route B
    delivers L" off the label column on the assumption that the two are the same set. They are,
    for any single-resolution annotation, and they are not for a vote - which spared a label
    from absorption while leaving it unable to be recovered, because the two directions were
    then asking different questions. So the sweep reports and does not decide, which is also
    what `docs/PRINCIPLES.md` 3 asks of any statistic whose agreement with correctness has not
    been shown.

    Returns a float per cell in [0, 1]: the share of resolutions whose label equals the
    reference. 1.0 means every granularity agreed; 0.0 means the delivered call is unique to the
    resolution that made it, which is a real and reportable state.

    REFUSES a sweep of fewer than two resolutions: an agreement column that is 1.0 for every
    cell by construction carries no information and reads exactly like one that was measured.
    """
    res = list(labels_by_res)
    if len(res) < 2:
        raise ValueError("agreement needs at least two resolutions to agree or disagree")
    L = {r: _truncate(labels_by_res[r], depth) for r in res}
    ref = _truncate(reference, depth)
    n = len(ref)
    if any(len(L[r]) != n for r in res):
        raise ValueError("every resolution must label the same cells, in the same order")
    stack = np.vstack([L[r] for r in res])
    return (stack == ref).sum(axis=0) / float(len(res))


def derived_tolerance(values, floor=MIN_TOL):
    """How much a metric moves between ADJACENT resolutions, as the scale of its own noise.

    A declared tolerance would be one more number to justify per dataset. The sweep already
    contains the answer: the typical step-to-step change is what a difference has to exceed
    before it means anything. Median rather than mean - one granularity boundary in the sweep
    should not set the tolerance for the whole of it.
    """
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return floor
    return max(float(np.median(np.abs(np.diff(v)))), floor)


def pick_resolution(labels_by_res, tree=None, groups=None, depths=(1,),
                    unresolved="UNRESOLVED", clusters_by_res=None):
    """Choose one resolution, and say what it was chosen over.

    `depths` are the tree depths the choice has to serve - typically (1, 2). Stability is
    averaged over them, because a resolution good for level 1 and bad for level 2 is not a
    resolution this pipeline can fix.

    Returns {"pick", "tied", "reason", "tolerance", "per_depth", "table"}. `tied` is every
    candidate the evidence could not separate from the winner; when it holds more than one
    entry the pick rests on the tie-break, and `reason` names which one decided it.
    """
    per_depth = {d: sweep_stability(labels_by_res, tree, groups, d, unresolved,
                                    clusters_by_res)
                 for d in depths}
    res = [r["resolution"] for r in per_depth[depths[0]]]
    stab = {r["resolution"]: r for r in per_depth[depths[-1]]}

    mean_modal = {x: float(np.mean([per_depth[d][i]["modal"] for d in depths]))
                  for i, x in enumerate(res)}
    tol = derived_tolerance([mean_modal[x] for x in res])
    top = max(mean_modal.values())
    tied = [x for x in res if mean_modal[x] >= top - tol]
    reason = "stability" if len(tied) == 1 else None

    def narrow(cands, key, label, higher_is_better=True):
        """Keep the candidates that this metric cannot separate from the best of them.

        `reason` records the step that got the field down to ONE - not the first step that
        removed anything. Reporting the first was wrong on the calibration sweep: completeness
        narrowed three candidates to two and took the credit, while rare-population retention
        was what actually made the pick.
        """
        nonlocal reason
        vals = [stab[c].get(key) for c in cands]
        if len(cands) == 1 or any(v is None for v in vals):
            return cands
        t = derived_tolerance([stab[x].get(key) for x in res])
        if higher_is_better:
            best = max(vals)
            keep = [c for c in cands if stab[c][key] >= best - t]
        else:
            best = min(vals)
            keep = [c for c in cands if stab[c][key] <= best + t]
        keep = keep or cands
        if len(keep) == 1 and len(cands) > 1:
            reason = label
        return keep

    tied = narrow(tied, "complete", "completeness")
    tied = narrow(tied, "smallest", "rare-population retention")
    pick = tied[0] if len(tied) == 1 else min(tied, key=lambda x: stab[x]["n_units"])
    if reason is None:
        reason = "parsimony" if len(tied) > 1 else "stability"
    return {"pick": pick, "tied": tied, "reason": reason, "tolerance": tol,
            "per_depth": per_depth, "mean_modal": mean_modal}


def format_report(result, depths=(1,)) -> str:
    """The table a reader argues with. Every candidate, not just the winner."""
    lines = []
    for d in depths:
        rows = result["per_depth"][d]
        lines.append(f"level {d}")
        lines.append("  %-8s %8s %8s %8s %9s %9s %8s %9s %9s" %
                     ("res", rows[0]["units_are"], "modal%", "nbr%", "complete", "truncat.",
                      "unres.%", "labels", "smallest"))
        for r in rows:
            mark = " <-" if r["resolution"] == result["pick"] else (
                "  *" if r["resolution"] in result["tied"] else "   ")
            lines.append("  %-8s %8d %7.1f%% %8s %8s %9s %7.1f%% %9d %9s%s" % (
                r["resolution"], r["n_units"], r["modal"],
                "n/a" if r["neighbour"] is None else f"{r['neighbour']:.1f}%",
                "n/a" if r["complete"] is None else f"{r['complete']:.1f}%",
                "n/a" if r["truncated"] is None else f"{r['truncated']:.1f}%",
                r["unresolved"], r["n_labels"],
                f"{r['smallest']:,}" + (f" ({r['min_groups']}/{r['n_groups']})"
                                        if r["min_groups"] is not None else ""),
                mark))
        lines.append("")
    lines.append(f"tolerance {result['tolerance']:.2f} points, derived from the sweep's own "
                 f"step-to-step variation")
    lines.append(f"candidates it could not separate: {', '.join(map(str, result['tied']))}")
    lines.append(f"PICK {result['pick']}  - decided by {result['reason']}")
    if len(result["tied"]) > 1 and result["reason"] == "parsimony":
        lines.append("  no measurement separated these; the coarsest was taken. Read the table "
                     "before accepting it.")
    return "\n".join(lines)

