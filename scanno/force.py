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
this was written for, one animal's several hundred `Endothelial` against the tens of
thousands the compartment holds.

A seal cannot repair that. Sealing a node the cohort agreed to split would throw away the split
for all ten samples in order to tidy up two, which is a removal paid for by every animal. So the
node keeps its children and the STRANDED CELLS move instead: each is pushed to the child it was
already measured to be most similar to.

RECURSIVE — AND THE WALK IS STILL UNCHANGED

One push is not enough, and the cohort this was written for showed why: a stranded cluster in
one sample landed on a child that is itself an INTERNAL node, the parent of two real subtypes.
The cell finished carrying a compartment name one level down. That is the defect FORCE exists to
remove, moved rather than fixed.

So FORCE means: A CELL MAY NOT TERMINATE ON AN INTERNAL NODE. The push repeats until a LEAF of
the tree in force is reached, however many levels that takes. A SEALED node has had its entire
child set deleted and is therefore a leaf, so a seal stops the recursion by construction and
nothing special is written for it. No depth is ever assumed: the loop ends where the tree ends.

THE FIRST STEP IS FREE. `classify()` records a `trace`, one entry per node it visited:

    trace.append({"at": node, "top": order[srt[0]], "gap": gap, "survival": ..., "cover": ...})

`top` is the ARGMAX CHILD at that node — the most similar one — and the final entry is written
at the very node where the walk truncated. So for a cluster that stopped at a FORCE node the
first destination is `trace[-1]["top"]` and its margin is `trace[-1]["gap"]`, both already
computed by the unchanged walk.

EVERY STEP AFTER IT NEEDS A REAL SCORE. The walk stopped AT the FORCE node, so it never scored
that node's children and it certainly never scored the grandchildren — there is nothing in
`trace` to read, and a second reading of `trace[-1]` would return the first step again.
`scanno/step.py` scores them with the machinery `classify()` itself uses: the same weights over
that node's own child set and the same `Z[c] @ W` product, so a forced step is computed exactly
as a walked step would have been. Given no scorer, or no tree to say what a leaf is, this module
pushes once and stops — which is what it did before, unchanged.

`scanno/classify.py` is not touched. The gap test, `GAP_CORPUS` and truncate-never-abstain are
exactly as they were, and every line here runs after the walk has returned.

WHAT IS RECORDED, AND WHY IT HAS TO BE

A forced call and a gap-cleared call are NOT the same evidential claim: the forced cell lands on
that child on a margin BELOW the bar, while other samples' cells reached the same child above
it. And TWO forced steps are not the same claim as one — the label is the end of a chain of
decisions THE WALK DID NOT TAKE, and it is only as strong as the WEAKEST of them. Pooling any of
these silently would trade a visible problem — a compartment name where a subtype belongs — for
an invisible one, which is the worse of the two, because nothing downstream would ever ask.

A LATER STEP IS NOT AUTOMATICALLY A SUB-THRESHOLD ONE, and saying so would be the same error in
the other direction. Only the FIRST step is below the bar by construction — that is why the walk
stopped there. Each step after it is scored fresh at a node the walk never reached and its
margin may fall either side of the bar; measured on the cohort this was written for, a second
step cleared it at 0.540 where the first had failed at 0.190. So depth counts decisions, not
rejections, and it is never a substitute for reading the margins: every one of them is reported.

So every cell carries HOW it was assigned and HOW FAR it was pushed:

  `<prefix>_assignment`    `gap` | `forced` | `EXCLUDED`, one value per cell, no missing values.
                           The vocabulary is UNCHANGED, so a filter written against the older
                           column still selects exactly the forced cells.
  `<prefix>_force_depth`   how many forced steps stand behind that cell's label: 0 where the
                           walk produced it, 1 where it was pushed once, 2 where the push landed
                           on an internal node and had to be pushed again. Zero for a withheld
                           nucleus too — nothing was forced there — and `assignment` is what
                           tells the two zeroes apart. It says how many decisions were taken
                           outside the walk, NOT how weak they were; the margins say that.
  `<prefix>_gap`           UNCHANGED, and still the margin at the FORCE node: `classify()` wrote
                           `trace[-1]["gap"]` there whether the walk descended or truncated. It
                           is the FIRST step's margin, which is exactly why it stopped being
                           sufficient on its own.
  uns provenance          per cluster: the FORCE node, the leaf chosen, the path after each
                          step, the margin of each step in order, survival, cover and the cell
                          count — so a sensitivity check can be run without re-annotating.

Why the per-step margins are `uns` and not columns. `force_provenance` states the rule this
package already follows: the irreducible per-cell fact is a column, and everything else is not.
Depth is irreducible — it survives `anndata.concat`, which drops `uns`, and nothing else in obs
implies it. A variable-length list of margins is not: it has no cell-level scalar form, and a
second per-cell home for a number `<prefix>_gap` already holds is how two homes come to disagree.

The label the cell WOULD have carried is recoverable exactly and still needs no column of its
own: it is the prefix of the forced path, and `<prefix>_force_depth` says how many levels to
strip. `scanno_path` therefore still contains the whole story.

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

And three more after the walk, each read off the FINISHED result rather than trusted to the
function that produced it:

  - any cluster still carrying a bare FORCE-node label — `bare_force`;
  - any FORCED cluster terminating on an internal node — `internal_terminals`. The recursion is
    meant to make this impossible; the check is what establishes that it did;
  - a chain that could not reach a leaf, because the tree loops, because a node's child set is
    missing, or because those children cannot be scored. It is recorded as `unforceable` and the
    push is NOT half-applied: the row is left exactly as the walk returned it, so the first check
    above catches it and the caller refuses. A destination is never invented and a chain is
    never re-entered.

`tests/test_force.py` asserts each of them.
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


def push_to_leaf(cid, path, first, tree=None, scorer=None, sep=SEP):
    """Follow the forced chain from `path` down to a LEAF. Returns `(steps, refusal)`.

    `first` is the trace entry the WALK already wrote at `path` — its argmax is step one and
    costs nothing. Each step after it is scored by `scorer(cluster, node)`, which returns a trace
    entry for a node the walk never reached, or None when it cannot be scored.

    Stops, with `refusal` set and the partial chain returned unapplied, on any of the three ways
    a descent can fail to terminate: a name the chain has already passed through (a tree that
    loops), a node the tree gives no scorer, and a node whose children the weights cannot
    represent. None of them may be papered over — a classifier that truncates must not invent a
    destination — and none may be retried, or a cycle becomes an infinite one.

    With no `tree` there is no way to know what a leaf IS, so exactly one step is taken. That is
    the pre-recursion behaviour, kept whole for a caller that has only the walk's output.
    """
    steps, seen = [], {path.split(sep)[-1]}
    entry, here = first, path
    while True:
        child = str(entry["top"])
        if child in seen:
            return steps, (f"the forced path arrives back at {child}, which it has already "
                           f"passed through — a tree that loops cannot be descended")
        seen.add(child)
        here = f"{here}{sep}{child}"
        steps.append({"to": child, "path": here, "margin": _f(entry.get("gap")),
                      "survival": _f(entry.get("survival")), "cover": _f(entry.get("cover"))})
        if tree is None:
            return steps, ""                      # leafness is unknowable: one step, as before
        if not _children(tree, child, sep=sep):
            return steps, ""                      # a LEAF. The push is finished
        if scorer is None:
            return steps, (f"{child} is an internal node and no scorer was given, so the child "
                           f"it most resembles was never measured")
        entry = scorer(cid, child)
        if entry is None:
            return steps, (f"{child} is an internal node whose children cannot be scored — the "
                           f"weights in use cannot represent them, so no most-similar child was "
                           f"measured")


def apply_force(res, force_paths, counts=None, tree=None, scorer=None, sep=SEP):
    """Reassign every cluster that TERMINATED on a FORCE node. Returns `(rows, record)`.

    POST-WALK AND NOTHING ELSE. `res` is `classify()`'s output; this returns new row dicts whose
    `label`, `path`, `depth`, `survival` and `cover` have moved on by as many levels as it took
    to reach a leaf, with `assignment` and `force_depth` fields added. `gap` is deliberately left
    alone — for a truncated cluster `classify()` already stored the failing step's gap there,
    which IS the margin of the FIRST forced step; the rest of them are in the record.

    `survival` and `cover` DO move, to the statistics of the LAST step taken, because classify's
    own rule is that a call reports "the statistics of the step that produced the LABEL — the
    last accepted one". Forcing makes that the final push; leaving an earlier level's numbers
    beside the new label would describe a decision that was not taken.

    `tree` and `scorer` are what make the push RECURSIVE. With both, a cluster is pushed until it
    reaches a node the tree gives no children — including a node a seal made childless. With
    neither, one step is taken, exactly as before. A chain that cannot reach a leaf is recorded
    in `unforceable` and NOT half-applied: that row is returned as the walk produced it, so
    `bare_force` still reports it and the caller still refuses.

    `counts` is optional per-cluster cell counts, indexable by cluster id. With it the record
    carries how many CELLS each reassignment moved, which is the number a reader asks for first.
    """
    force = set(force_paths)
    out, assigned, unforceable, by_node = [], {}, {}, {}
    tally = {k: 0 for k in ASSIGNMENTS}
    by_depth = {}
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
                "reached": path,
                "n_steps": 0,
                "margins": [],
                "n_cells": n,
                "reason": ("the walk recorded no scores at this node, so no most-similar child "
                           "was measured; its children cannot be represented by the weights in "
                           "use"),
            }
            row["assignment"] = BY_GAP
            tally[BY_GAP] += 1
            out.append(row)
            continue

        steps, refusal = push_to_leaf(cid, path, last, tree=tree, scorer=scorer, sep=sep)
        if refusal:
            # NOT half-applied. A cell left on an intermediate internal node carries a
            # compartment name one level down, which is the very thing being removed — so the
            # partial chain is reported and the row is returned exactly as the walk produced it.
            unforceable[str(cid)] = {
                "node": path,
                "stopped_at": (steps[-1]["to"] if steps else bare),
                "reached": (steps[-1]["path"] if steps else path),
                "n_steps": len(steps),
                "margins": [s["margin"] for s in steps],
                "n_cells": n,
                "reason": refusal,
            }
            row["assignment"] = BY_GAP
            tally[BY_GAP] += 1
            out.append(row)
            continue

        end = steps[-1]
        row["path"] = end["path"]
        row["label"] = end["to"]
        row["depth"] = int(row.get("depth", len(path.split(sep)))) + len(steps)
        row["survival"] = end["survival"]
        row["cover"] = end["cover"]
        row["assignment"] = BY_FORCE
        row["force_depth"] = len(steps)
        tally[BY_FORCE] += 1
        by_depth[str(len(steps))] = by_depth.get(str(len(steps)), 0) + 1
        n_cells_forced += n
        assigned[str(cid)] = {"node": path, "to": end["to"], "path": end["path"],
                              "margin": steps[0]["margin"], "survival": row["survival"],
                              "cover": row["cover"], "n_cells": n,
                              "force_depth": len(steps),
                              "steps": [s["path"] for s in steps],
                              "margins": [s["margin"] for s in steps]}
        by_node.setdefault(path, {})
        by_node[path][end["to"]] = by_node[path].get(end["to"], 0) + n
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
        "clusters_by_force_depth": by_depth,
        "recursive": scorer is not None and tree is not None,
        "margin_source": "step one is the gap the unchanged walk recorded at the FORCE node "
                         "(classify() trace[-1]['gap']), below --gap-min by construction; every "
                         "later step is scored at a node the walk never reached, by the same "
                         "weights over that node's own children",
    }
    return out, record


#: Values of the `<prefix>_resolved_origin` column — HOW each cell got its resolved leaf.
#: A column that mixes confident calls with root-level guesses and does not say which is which
#: is worse than no column, because every consumer will read the guesses as calls.
FROM_WALK = "walk"            # the walk reached a leaf on its own; nothing was forced
FROM_INTERNAL = "forced"      # pushed down from an internal node the walk stopped on
FROM_ROOT = "root_forced"     # was UNRESOLVED: pushed from the root itself
UNRESOLVED = "UNRESOLVED"     # could not be pushed, and nothing was invented
RESOLVED_ORIGINS = (FROM_WALK, FROM_INTERNAL, FROM_ROOT, UNRESOLVED, EXCLUDED)


def resolve_to_leaf(res, tree=None, scorer=None, counts=None, sep=SEP):
    """Give every walked cluster a LEAF label, in columns of its own. Returns `(rows, record)`.

    WHAT THIS IS FOR. The walk truncates rather than guessing, so a cohort carries cells labelled
    `UNRESOLVED` — the root's children could not be told apart for them — and cells labelled with
    a compartment, where an internal node's children could not. That is the honest answer and it
    stays the answer. But a great deal of downstream work needs a column with no holes in it: a
    composition table, a colour-by in a viewer, a label handed to a semi-supervised model. Those
    consumers otherwise invent their own rule for the gaps, off the record and differently each
    time.

    So this writes a SECOND set of columns in which every walked cell sits on a leaf, and leaves
    the first set untouched. **It is additive and reversible**: nothing is overwritten, the
    principled label is still there beside it, and the origin column says for every cell whether
    its leaf was reached or assigned.

    NOTHING IS INVENTED. The descent is the same `push_to_leaf` the FORCE pass uses, over the
    same trace the walk already wrote — for an `UNRESOLVED` cluster the root's argmax is
    `trace[0]`, which was scored during the walk and cost nothing to keep. Where a chain cannot
    reach a leaf — a tree that loops, a node with no scorer, children the weights cannot
    represent — the cell stays `UNRESOLVED` in the resolved column too and the reason is
    recorded. A classifier that truncates must not acquire a destination it never measured.

    EXCLUDED cells are never resolved. They were withheld before the walk, so there is no trace
    to descend and no measurement to descend it with.

    The margin of a root-forced call is below the gap bar BY CONSTRUCTION — that is why it was
    unresolved — and it is already in `<prefix>_gap` for those rows. These are the least certain
    labels in the object and the origin column is how a reader finds them.
    """
    out, record = [], {"by_origin": {k: 0 for k in RESOLVED_ORIGINS},
                       "cells_by_origin": {k: 0 for k in RESOLVED_ORIGINS},
                       "unresolvable": {}, "pushed": {}}

    for r in res:
        row = dict(r)
        cid = int(row.get("cluster", len(out)))
        n = _count(counts, cid)
        path = str(row.get("path", ""))

        def settle(origin, label, full, depth):
            row["resolved_label"] = label
            row["resolved_path"] = full
            row["resolved_origin"] = origin
            row["resolved_depth"] = int(depth)
            record["by_origin"][origin] += 1
            record["cells_by_origin"][origin] += n
            out.append(row)

        if row.get("excluded") or path == EXCLUDED:
            settle(EXCLUDED, EXCLUDED, EXCLUDED, 0)
            continue

        trace = row.get("trace") or []
        unresolved = (path == UNRESOLVED or not path)

        if unresolved:
            # Start AT THE ROOT. `push_to_leaf` builds a path by joining, so it is given the
            # literal root name and the prefix is stripped afterwards — the alternative, an empty
            # starting path, yields a leading separator on every result.
            first = trace[0] if trace else None
            start, strip = ROOT, True
            origin = FROM_ROOT
        else:
            here = path.split(sep)[-1]
            if not _children(tree, here, sep=sep) if tree is not None else True:
                # Already a leaf, or leafness is unknowable without a tree. Either way the walk's
                # own answer stands and nothing is forced.
                settle(FROM_WALK, str(row.get("label", here)), path, int(row.get("depth", 0)))
                continue
            first = next((e for e in reversed(trace) if str(e.get("at")) == here), None)
            start, strip = path, False
            origin = FROM_INTERNAL

        if first is None:
            # The walk never scored this node's children, so there is no measured most-similar
            # child. Recorded, not guessed.
            record["unresolvable"][str(cid)] = {
                "from": path, "n_cells": n,
                "why": "the walk recorded no argmax here, so no most-similar child was measured"}
            settle(UNRESOLVED, UNRESOLVED, UNRESOLVED, 0)
            continue

        steps, refusal = push_to_leaf(cid, start, first, tree=tree, scorer=scorer, sep=sep)
        if refusal or not steps:
            record["unresolvable"][str(cid)] = {
                "from": path, "n_cells": n,
                "why": refusal or "the descent took no step"}
            settle(UNRESOLVED, UNRESOLVED, UNRESOLVED, 0)
            continue

        full = steps[-1]["path"]
        if strip:
            full = full[len(ROOT) + len(sep):] if full.startswith(ROOT + sep) else full
        record["pushed"][str(cid)] = {
            "from": path, "to": full, "n_steps": len(steps), "n_cells": n,
            "margins": [s["margin"] for s in steps]}
        settle(origin, steps[-1]["to"], full, full.count(sep) + 1)

    return out, record


def format_resolved(record, sep=SEP):
    """The resolved-label summary, as report lines. Says what was ASSIGNED, not merely what is."""
    if not record:
        return []
    by, cells = record["by_origin"], record["cells_by_origin"]
    L = [f"resolved labels: every walked cell carries a leaf in `<prefix>_resolved`, and "
         f"`<prefix>_resolved_origin` says how it got there."]
    for k in RESOLVED_ORIGINS:
        if by.get(k):
            L.append(f"    {k:<12} {by[k]:>4} cluster(s), {cells[k]:>8,} cell(s)")
    if by.get(FROM_ROOT):
        L.append(f"    {cells[FROM_ROOT]:,} cell(s) were UNRESOLVED and are now labelled. Their "
                 f"margin is below the gap bar BY CONSTRUCTION - that is why the walk stopped - "
                 f"so these are the least certain labels in the object.")
    if record.get("unresolvable"):
        L.append(f"    {len(record['unresolvable'])} cluster(s) could NOT be resolved and stay "
                 f"UNRESOLVED; nothing was invented for them:")
        for cid, d in sorted(record["unresolvable"].items())[:6]:
            L.append(f"      cluster {cid} from {d['from']!r}: {d['why']}")
    return L


def internal_terminals(res, tree, sep=SEP):
    """Clusters whose delivered path is a node the tree still gives CHILDREN to.

    The guarantee FORCE now stands for, read off the finished result: nothing may terminate on an
    internal node. A sealed node is not one — its children are gone from this tree, which is what
    a seal means — so this reports exactly the nodes that kept a split a cell then stopped short
    of. `assignment` rides along because the two cases have different remedies: a `forced` row
    here is a broken recursion, while a `gap` row is the walk truncating where the scope left the
    split standing, and only the caller knows which of those it is willing to deliver.
    """
    from .scope import internal_nodes

    inner = set(internal_nodes(tree, sep=sep)) - {ROOT}
    return [{"cluster": int(r.get("cluster", -1)), "node": str(r.get("path", "")),
             "assignment": str(r.get("assignment", "")),
             "children": ", ".join(_children(tree, str(r.get("path", "")), sep=sep))}
            for r in res if str(r.get("path", "")) in inner]


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


def format_force(record, gap_min=None, sep=SEP):
    """The reassignment as lines a human reads. What moved, where to, and on what margin.

    The WHOLE chain is printed, not the destination alone. A cell pushed twice was decided twice,
    and a line reading `A -> C` would let a reader believe a single measurement put it there.
    """
    L = [f"  FORCE  {len(record['nodes'])} node(s) the cohort agreed to split, on which nothing "
         f"may terminate: {', '.join(record['nodes']) or '(none)'}"]
    if not record["nodes"]:
        return L
    if record["n_forced"]:
        cells = record["n_cells_forced"]
        L.append(f"    {record['n_forced']} cluster(s)"
                 + (f" / {cells:,} cell(s)" if cells else "")
                 + " were stranded and have been pushed, step by step, to the most similar "
                   "child at each node until a leaf:")
        for cid, a in sorted(record["assigned"].items(), key=lambda kv: -kv[1]["n_cells"]):
            n = f"{a['n_cells']:,}" if a["n_cells"] else "?"
            chain = " -> ".join([a["node"]] + [p.split(sep)[-1] for p in a["steps"]])
            margins = ", ".join(f"{m:.3f}" for m in a["margins"])
            L.append(f"      cluster {cid:>6}  {n:>9} cell(s)  {chain}"
                     f"   margin(s) {margins}")
        deep = {cid: a for cid, a in record["assigned"].items() if a["steps"][1:]}
        if deep:
            L.append(f"    {len(deep)} of them needed MORE THAN ONE step. Such a label is the "
                     f"end of a chain of decisions the")
            L.append(f"    walk did not take, and is only as strong as the WEAKEST of them:")
            for cid, a in sorted(deep.items(), key=lambda kv: -kv[1]["n_cells"]):
                L.append(f"      cluster {cid:>6}  weakest step {min(a['margins']):.3f}"
                         f"   ({a['force_depth']} steps)")
            L.append(f"    `<prefix>_force_depth` carries the count per cell. It says how many "
                     f"decisions were taken")
            L.append(f"    outside the walk, NOT how weak they were — read the margins for "
                     f"that.")
    else:
        L.append("    no cluster terminated on one of them — nothing was reassigned")
    if gap_min is not None and record["n_forced"]:
        L.append(f"    A forced call is NOT a gap-cleared call: the FIRST margin of each chain "
                 f"above is below the bar")
        L.append(f"    ({gap_min}) by construction — that is why the walk stopped — while cells "
                 f"reaching the same child in")
        L.append(f"    other samples cleared it. A LATER step is scored fresh at a node the "
                 f"walk never reached and may")
        L.append(f"    fall either side of the bar; it does not inherit the first step's "
                 f"weakness and does not cancel it.")
        L.append(f"    `<prefix>_assignment` says which cells are which, `<prefix>_gap` is the "
                 f"FIRST step's margin, and")
        L.append(f"    uns holds the rest — filter on them before any cross-sample claim.")
    for cid, u in sorted(record["unforceable"].items()):
        L.append(f"    REFUSE  cluster {cid} stopped on {u['node']}, reached {u['reached']} and "
                 f"cannot be forced to a leaf: {u['reason']}")
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
