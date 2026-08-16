"""FORCE — nothing may TERMINATE on a node the cohort agreed to split.

WHY THIS EXISTS

`scanno scope` returns three verdicts per internal node. Two of them already had an
implementation: KEEP changes nothing, and SEAL deletes a node's entire child set so the walk
breaks there by construction. The third did not.

FORCE is the case where the cohort AGREED the split is admissible — every sample that reached
the node descended below it — and yet some cells still stopped ON the node, because their own
cluster's gap fell short where every other sample's cleared. Those cells end up carrying the
name of a COMPARTMENT: the identical string the L1 column uses for every cell beneath it. Read
side by side, the two delivered columns then disagree about what one word means — on the cohort
this was written for, one animal's 961 `Endothelial` against the compartment's 26,552.

A seal cannot repair that. Sealing a node the cohort agreed to split would throw away the split
for all ten samples in order to tidy up two, which is a removal paid for by every animal. So the
node keeps its children and the STRANDED CELLS move instead: each is pushed to the child it was
already measured to be most similar to.

NO NEW SCORING, AND NO CHANGE TO THE WALK

`classify()` records a `trace`, one entry per node it visited:

    trace.append({"at": node, "top": order[srt[0]], "gap": gap, "survival": ..., "cover": ...})

`top` is the ARGMAX CHILD at that node — the most similar one — and the final entry is written
at the very node where the walk truncated. So for a cluster that stopped at a FORCE node the
assignment is `trace[-1]["top"]` and its margin is `trace[-1]["gap"]`, both already computed by
the unchanged walk. This module reads them. It does not re-score, it does not re-walk, and
`scanno/classify.py` is not touched: the gap test, `GAP_CORPUS` and truncate-never-abstain are
exactly as they were.

WHAT IS RECORDED, AND WHY IT HAS TO BE

A forced call and a gap-cleared call are NOT the same evidential claim. The forced cell lands on
that child on a margin BELOW the bar, while other samples' cells reached the same child above
it. Pooling the two silently would trade a visible problem — a compartment name where a subtype
belongs — for an invisible one, which is the worse of the two, because nothing downstream would
ever ask.

So every cell carries HOW it was assigned, and the margin travels with it:

  `<prefix>_assignment`   `gap` | `forced` | `EXCLUDED`, one value per cell, no missing values.
  `<prefix>_gap`          UNCHANGED. For a forced row this already IS the margin at the FORCE
                          node — `classify()` writes `trace[-1]["gap"]` into `gap` whether the
                          walk descended or truncated — so the number is not copied into a
                          second column that could drift from it. What `assignment` adds is the
                          reading: where it is `forced`, `gap` is below `--gap-min` by
                          construction; where it is `gap`, it is at or above.
  uns provenance          per cluster: the FORCE node, the child chosen, the margin, survival,
                          cover and the cell count — so a sensitivity check can be run without
                          re-annotating anything.

The label the cell WOULD have carried is recoverable exactly and needs no column of its own: it
is the parent of the forced path, because a forced path is `<force node>/<child>` by
construction. `scanno_path` therefore still contains the whole story.

WHAT THIS MODULE REFUSES

Four shapes are refused up front, on the SCOPE and the TREE, before an object is read — a
refusal after the expensive part is a refusal the caller pays for twice:

  - the ROOT marked FORCE. Truncation at the root is `UNRESOLVED`, a different decision that
    `scanno scope` does not emit and that nobody has taken. Honouring it here would rewrite the
    sentinel quietly.
  - a FORCE node the tree does not give children to. There is nothing to be pushed onto, and
    inventing a destination is exactly what a classifier that truncates must never do.
  - a FORCE node whose own child is also FORCE. Forcing there would leave the cell carrying a
    bare FORCE-node name one level down, which is the problem this module exists to remove.
  - a name that sits at two positions in the tree. `trace["at"]` and `children` are both keyed
    by BARE name, so a duplicate would force the wrong lineage. `scanno scope` refuses the same
    shape for the same reason.

And one more after the walk: if any cluster still carries a bare FORCE-node label, the caller is
told and the run refuses. That is the post-condition, and `tests/test_force.py` asserts it.
"""
from __future__ import annotations

import math

SEP = "/"
ROOT = "root"

#: The scope verdicts this module acts on. Read from the scope file; never declared here.
FORCE = "FORCE"
SEAL = "SEAL"

#: How a cluster's label was arrived at — the values of the `<prefix>_assignment` column.
#: `EXCLUDED` is the third because a withheld nucleus was assigned by NOTHING, and calling that
#: `gap` would claim a decision that was never taken.
BY_GAP = "gap"
BY_FORCE = "forced"
EXCLUDED = "EXCLUDED"
ASSIGNMENTS = (BY_GAP, BY_FORCE, EXCLUDED)


def scope_verdicts(scope):
    """The per-node verdicts of a `scanno scope --out` payload. Refuses anything else.

    A dict that is not a scope result is the failure mode worth naming: a caller pointing
    `--scope` at the SEALED TREE instead of the vote would otherwise get an empty verdict set,
    no seals, no FORCE, and a run that looks like it honoured a scope and did not.
    """
    if not isinstance(scope, dict) or not isinstance(scope.get("nodes"), dict):
        raise ValueError(
            "not a `scanno scope --out` result: it has no 'nodes' mapping. The scope is the "
            "VOTE, not the sealed tree — pass the file `scanno scope --out` wrote.")
    return scope["nodes"]


def nodes_with(verdicts, verdict):
    """Every node carrying `verdict`, as sorted full paths from the root."""
    return sorted(n for n, v in verdicts.items()
                  if isinstance(v, dict) and v.get("verdict") == verdict)


def force_nodes(verdicts):
    """The nodes annotation must not let anything terminate on."""
    return nodes_with(verdicts, FORCE)


def sealed_nodes(verdicts):
    """The nodes whose entire child set the scope deletes."""
    return nodes_with(verdicts, SEAL)


def _children(tree, node, sep=SEP):
    """The declared children of a node given as a PATH. `children` is keyed by bare name."""
    return list((tree.get("children") or {}).get(node.split(sep)[-1], []))


def check_scope(scope, tree, sep=SEP):
    """Every reason this scope and this tree cannot be walked together. Empty means go.

    Returned as strings rather than raised one at a time, so a caller fixing a scope file sees
    all of its problems in one run instead of discovering them one refusal per hour.
    """
    from .scope import bare_names_unique, internal_nodes

    verdicts = scope_verdicts(scope)
    problems = []

    dup = bare_names_unique(tree)
    if dup:
        problems.append(
            f"these names appear at more than one position in --tree, so a seal and the walk's "
            f"own `trace['at']` cannot be matched unambiguously: {dup}")

    declared = internal_nodes(tree, sep=sep)
    forced = set(force_nodes(verdicts))

    if ROOT in forced:
        problems.append(
            "the scope marks the ROOT as FORCE. Truncation at the root is UNRESOLVED, which is "
            "a different decision from a cell stranded on a compartment; `scanno scope` never "
            "emits it and this command will not invent it.")

    for node in sorted(forced - {ROOT}):
        kids = _children(tree, node, sep=sep)
        if not kids:
            problems.append(
                f"{node} is FORCE but --tree gives it no children, so there is nothing to push "
                f"its stranded cells onto. Either the tree is already sealed there or the scope "
                f"describes a different taxonomy.")
            continue
        clash = [c for c in kids if f"{node}{sep}{c}" in forced]
        if clash:
            problems.append(
                f"{node} is FORCE and so is its child {', '.join(clash)}. Forcing would land "
                f"the cell on a bare FORCE-node name one level down, which is the very thing "
                f"FORCE exists to remove.")

    # Taxonomy drift. A scope voted on one tree and applied to another is silent: the seals land
    # on whatever happens to share a name and the rest is walked unscoped. A sealed node AND
    # EVERYTHING BENEATH IT is exempt, because a tree that has ALREADY been sealed legitimately
    # declares neither — applying a scope to its own output must be idempotent, not a refusal.
    # Exempting only the sealed node itself was not enough and made re-running refuse.
    sealed = set(sealed_nodes(verdicts))
    gone = tuple(s + sep for s in sealed)
    missing = sorted(n for n in verdicts
                     if n != ROOT and n not in declared and n not in sealed
                     and not n.startswith(gone))
    if missing:
        problems.append(
            f"the scope voted on {len(missing)} internal node(s) --tree does not declare and "
            f"does not seal: {', '.join(missing[:6])}"
            f"{' ...' if missing[6:] else ''}. The scope was voted on a different tree.")
    return problems


def apply_force(res, force_paths, counts=None, sep=SEP):
    """Reassign every cluster that TERMINATED on a FORCE node. Returns `(rows, record)`.

    POST-WALK AND NOTHING ELSE. `res` is `classify()`'s output; this returns new row dicts with
    `label`, `path`, `depth`, `survival` and `cover` moved on by one level and an `assignment`
    field added. `gap` is deliberately left alone — for a truncated cluster `classify()` already
    stored the failing step's gap there, which IS the margin of the forced call.

    `survival` and `cover` DO move, to the statistics of `trace[-1]`, because classify.py's own
    rule is that a call reports "the statistics of the step that produced the LABEL — the last
    accepted one". Forcing makes `trace[-1]` that step; leaving the previous level's numbers
    beside the new label would describe a decision that was not taken.

    `counts` is optional per-cluster cell counts, indexable by cluster id. With it the record
    carries how many CELLS each reassignment moved, which is the number a reader asks for first.
    """
    force = set(force_paths)
    out, assigned, unforceable, by_node = [], {}, {}, {}
    tally = {k: 0 for k in ASSIGNMENTS}
    n_cells_forced = 0

    for r in res:
        row = dict(r)
        cid = int(row.get("cluster", len(out)))
        n = _count(counts, cid)

        if row.get("excluded"):
            # Withheld upstream: never walked, so nothing assigned it and nothing may.
            row["assignment"] = EXCLUDED
            tally[EXCLUDED] += 1
            out.append(row)
            continue

        path = str(row.get("path", ""))
        if path not in force:
            row["assignment"] = BY_GAP
            tally[BY_GAP] += 1
            out.append(row)
            continue

        trace = row.get("trace") or []
        last = trace[-1] if trace else None
        bare = path.split(sep)[-1]
        if last is None or str(last.get("at")) != bare:
            # The walk broke BEFORE scoring this node's children — `node_weights` returned None,
            # so no argmax was ever recorded here. There is no measured most-similar child, and
            # inventing one is the one thing a classifier that truncates must not do.
            unforceable[str(cid)] = {
                "node": path,
                "stopped_at": (str(last.get("at")) if last else ""),
                "n_cells": n,
                "reason": ("the walk recorded no scores at this node, so no most-similar child "
                           "was measured; its children cannot be represented by the weights in "
                           "use"),
            }
            row["assignment"] = BY_GAP
            tally[BY_GAP] += 1
            out.append(row)
            continue

        child = str(last["top"])
        row["path"] = f"{path}{sep}{child}"
        row["label"] = child
        row["depth"] = int(row.get("depth", len(path.split(sep)))) + 1
        row["survival"] = _f(last.get("survival"))
        row["cover"] = _f(last.get("cover"))
        row["assignment"] = BY_FORCE
        tally[BY_FORCE] += 1
        n_cells_forced += n
        assigned[str(cid)] = {"node": path, "to": child, "path": row["path"],
                              "margin": _f(last.get("gap")), "survival": row["survival"],
                              "cover": row["cover"], "n_cells": n}
        by_node.setdefault(path, {})
        by_node[path][child] = by_node[path].get(child, 0) + n
        out.append(row)

    record = {
        "verdict": FORCE,
        "nodes": sorted(force),
        "assigned": assigned,
        "by_node": by_node,
        "unforceable": unforceable,
        "n_clusters": len(out),
        "n_forced": tally[BY_FORCE],
        "n_cells_forced": n_cells_forced,
        "clusters_by_assignment": tally,
        "margin_source": "the gap recorded at the FORCE node by the unchanged walk "
                         "(classify() trace[-1]['gap']), below --gap-min by construction",
    }
    return out, record


def bare_force(res, force_paths):
    """Clusters still terminating on a FORCE node. The post-condition; empty is the only pass.

    Kept separate from `apply_force` so it can be run on ANY `res` — including one the caller
    built some other way — and so the check reads the finished result rather than trusting the
    function that produced it.
    """
    force = set(force_paths)
    return [{"cluster": int(r.get("cluster", -1)), "node": str(r.get("path", "")),
             "assignment": str(r.get("assignment", ""))}
            for r in res if str(r.get("path", "")) in force]


def format_force(record, gap_min=None):
    """The reassignment as lines a human reads. What moved, where to, and on what margin."""
    L = [f"  FORCE  {len(record['nodes'])} node(s) the cohort agreed to split, on which nothing "
         f"may terminate: {', '.join(record['nodes']) or '(none)'}"]
    if not record["nodes"]:
        return L
    if record["n_forced"]:
        cells = record["n_cells_forced"]
        L.append(f"    {record['n_forced']} cluster(s)"
                 + (f" / {cells:,} cell(s)" if cells else "")
                 + " were stranded and have been pushed to the most similar child recorded by "
                   "the walk:")
        for cid, a in sorted(record["assigned"].items(), key=lambda kv: -kv[1]["n_cells"]):
            n = f"{a['n_cells']:,}" if a["n_cells"] else "?"
            L.append(f"      cluster {cid:>6}  {n:>9} cell(s)  {a['node']} -> {a['to']}"
                     f"   margin {a['margin']:.3f}")
    else:
        L.append("    no cluster terminated on one of them — nothing was reassigned")
    if gap_min is not None and record["n_forced"]:
        L.append(f"    A forced call is NOT a gap-cleared call: every margin above is below "
                 f"the bar ({gap_min}), while cells")
        L.append(f"    reaching the same child in other samples cleared it. "
                 f"`<prefix>_assignment` says which is which, per cell,")
        L.append(f"    and `<prefix>_gap` is that margin — filter on it before any "
                 f"cross-sample claim.")
    for cid, u in sorted(record["unforceable"].items()):
        L.append(f"    REFUSE  cluster {cid} stopped on {u['node']} and cannot be forced: "
                 f"{u['reason']}")
    return L


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _count(counts, cid):
    if counts is None:
        return 0
    try:
        return int(counts[cid])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0
