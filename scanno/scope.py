"""The common scope — which splits every sample agreed to make.

WHY THIS EXISTS

Ten samples walked independently produce ten scopes. On a lineage where the sibling contrast
sits near the bar, one animal descends and the next truncates, so the SAME cells get a subtype
in one library and its parent in another. Measured on a real cohort: two animals of the same
arm, same batch and same chemistry reported 11.25% `Fibroblast` / 0.00% `Matrifibrocyte` and
0.00% / 17.71%. Downstream that is indistinguishable from a compositional shift, and no analysis
can undo it — abundance is conserved and merely re-labelled.

The fix is to decide the scope ONCE, from what the samples agree on, and annotate everyone
against it. That is three steps, and this module is the middle one:

    PASS 1  SCOUT     ten independent walks, declared tree, unchanged bar
    VOTE              -> this module: one common scope, written as a pruned tree
    PASS 2  ANNOTATE  the UNCHANGED walk, per sample, against that tree

SEAL, NOT PRUNE — the distinction is the whole mechanism

  PRUNE removes SOME children of a node. The node stays open, `gap` is still computed over the
        survivors, and samples can still disagree. On a real cohort this was measured to change
        NOTHING: at 8 of 8 decisions with recorded score vectors every prunable child ranked 3rd
        or lower, so it never touched `s[srt[0]]` or `s[srt[1]]` and could not move the gap.

  SEAL  removes ALL children, so the node becomes a LEAF. `node_weights` returns None, the walk
        breaks, and every sample truncates there BY CONSTRUCTION rather than statistically.
        There is no bar to land either side of, so samples cannot diverge.

Sealing is therefore the only edit that guarantees equal depth, and it is the only edit this
module makes. Never-reached children at nodes that stay OPEN are reported, not removed: deleting
them would change `node_weights` at a node still being decided, which is an unmeasured
perturbation bought for no benefit.

WHAT IS REMOVED, AND WHAT IS NOT

A seal removes the possibility of a LABEL, not any observation. Every cell keeps its pass-1 path
in `obs`, so a sealed subtype call remains on disk and the decision is reversible by re-running
pass 2 against the declared tree. `sealed_labels()` returns the actual list — the project rule is
that a removal is stated as its members, not described as a category.
"""
from __future__ import annotations

import collections
import copy

SEP = "/"
ROOT = "root"
EXCLUDED = "EXCLUDED"
UNRESOLVED = "UNRESOLVED"
SENTINELS = (EXCLUDED, UNRESOLVED)

#: Seal any node where a sample that REACHED it declined to descend. Unanimity is the default
#: because a split one animal would not make is a split the cohort cannot be asked to compare
#: across; loosen it deliberately, and expect the residual disagreement it lets through.
MIN_SUPPORT = 1.0

#: A node reached by fewer samples than this is UNVOTABLE, not sealed. Sealing on one animal's
#: evidence is a removal with no quorum behind it, which is the failure this module exists to
#: prevent — so the node stays open and the shortfall is reported by name.
MIN_REACH = 2


def node_votes(paths_by_sample, sep=SEP, sentinels=SENTINELS, descend_rule="any",
               excluded=EXCLUDED, unresolved=UNRESOLVED):
    """Per node: which samples REACHED it, and which of those DESCENDED below it.

    THE CONDITIONING IS THE PART THAT GOES WRONG. A sample whose walk never reached the node
    casts NO vote — it is a missing observation, not a vote against. Counting absence as
    opposition seals every branch that is simply rare, and rare is not the same as unsupported.
    Measured case: one cohort's `Lymphoid` node was reached by 7 of 10 animals, three of which
    had too few lymphocytes to form a cluster at all.

    `descend_rule` decides what it means for one sample to have descended, and the two answers
    genuinely differ because a single sample can do BOTH — different clusters of one lineage
    truncating differently inside one animal was observed in 4 of 10:

      "any"       the sample descended if ANY of its cells went below. Strict: one cluster
                  descending is enough to count that animal as having made the split.
      "majority"  the sample descended if MORE THAN HALF the cells arriving at the node went
                  below. Tolerant of a single stray cluster.
    """
    if descend_rule not in ("any", "majority"):
        raise ValueError(f"descend_rule must be 'any' or 'majority', got {descend_rule!r}")

    arrive = collections.defaultdict(collections.Counter)
    deeper = collections.defaultdict(collections.Counter)
    for sample, paths in paths_by_sample.items():
        for p in paths:
            p = str(p)

            # THE TWO SENTINELS ARE NOT THE SAME EVENT, AND CONFLATING THEM BLINDS THE VOTE.
            #
            # EXCLUDED was withheld upstream and never walked, so it reached nothing and votes
            # on nothing. UNRESOLVED is a TRUNCATION AT THE ROOT — the walk ran, arrived, and
            # declined to descend. Skipping both, as the first version of this did, left the
            # root with zero reaching samples and the vote structurally unable to see root-level
            # failure. On the cohort this was written for that is 1,880 nuclei in 4 of 10
            # animals, 973 of one animal's and 613 of another's being cells the joint route
            # calls Pericyte — a whole population lost, invisible to a vote that ignores it.
            if p == excluded:
                continue
            arrive[ROOT][sample] += 1
            if p == unresolved:
                continue                       # arrived at the root, did not descend
            deeper[ROOT][sample] += 1

            parts = p.split(sep)
            for d in range(1, len(parts) + 1):
                node = sep.join(parts[:d])
                arrive[node][sample] += 1
                if d < len(parts):
                    deeper[node][sample] += 1

    out = {}
    for node, counts in arrive.items():
        reached = sorted(counts)
        below = deeper[node]
        if descend_rule == "majority":
            desc = [s for s in reached if below[s] * 2 > counts[s]]
        else:
            desc = [s for s in reached if below[s] > 0]
        out[node] = {
            "reached": reached,
            "descended": desc,
            "n_reached": len(reached),
            "n_descended": len(desc),
            "support": (len(desc) / len(reached)) if reached else float("nan"),
            "cells": dict(counts),
            "cells_below": {s: below[s] for s in reached},
        }
    return out


def _stranded(node, v):
    """Cells that ARRIVED at this node and went no further, per sample.

    Derived, never declared: it is the arrival count minus the below count, which the vote has
    already measured. A node with zero stranded cells needs no forcing, so the verdict stays
    KEEP and nothing about that branch changes.
    """
    return {s: v["cells"][s] - v["cells_below"].get(s, 0)
            for s in v.get("reached", [])
            if v["cells"][s] - v["cells_below"].get(s, 0) > 0}


def internal_nodes(tree, sep=SEP):
    """Every node the declared tree gives children to, as full paths from the root.

    Returned as PATHS even though `tree["children"]` is keyed by BARE name, because a vote is
    about a position in the taxonomy and two lineages may reuse a name. `bare_names_unique`
    reports whether that ever actually happens in this tree.
    """
    kids = tree.get("children", {})
    out, stack = {}, [("root", [])]
    while stack:
        name, prefix = stack.pop()
        path = sep.join(prefix) if prefix else ""
        if name in kids:
            out[path or "root"] = list(kids[name])
        for child in kids.get(name, []):
            stack.append((child, prefix + [child]))
    return out


#: Why a delivered label stops where it stops. Several different facts about a taxonomy that
#: produce the SAME string in the label column, and whose remedies have nothing in common.
TERMINAL_REASONS = ("leaf", "sealed", "unvotable", "stranded", "sentinel")


def why_terminal(path, tree, verdicts=None, sep=SEP, sentinels=SENTINELS):
    """Why the delivered annotation stops at `path`. One of TERMINAL_REASONS.

    THESE LOOK IDENTICAL IN THE LABEL COLUMN AND THEIR REMEDIES ARE UNRELATED.

      leaf       the DECLARED tree gives this node no children. The walk went as far as the
                 taxonomy goes and the call is complete. Nothing to remedy.
      sealed     the node has declared children and the COHORT removed them. The subtype is
                 recoverable — re-run pass 2 against the declared tree — and which labels were
                 lost is named in the scope's own removal table.
      unvotable  too few samples reached it to vote, so it was left OPEN and never sealed. A
                 cell terminating here did so on its own gap, not on a cohort decision.
      stranded   the node is OPEN, the cohort agreed the split is admissible, and this cell's
                 gap failed anyway. Nothing was removed; the evidence was thin for this cell.
      sentinel   EXCLUDED or UNRESOLVED — not a cell type at all.

    A reader who cannot tell `leaf` from `sealed` reads a sealed compartment as the finest
    resolution the tissue supports, which is the opposite of what it means. So this is computed
    from the DECLARED tree and the verdicts together and never inferred from the label's shape:
    a depth-1 label is a complete call when the tree gives it no children and an unfinished one
    when it does, and the string in the column is the same either way.

    Keyed on the BARE name for the tree, as `tree["children"]` is, and on the full PATH for the
    verdicts, as `vote()` returns them. `bare_names_unique` reports whether a bare name is
    ambiguous in a given tree.
    """
    p = str(path)
    if p in sentinels:
        return "sentinel"
    kids = (tree or {}).get("children", {}) or {}
    declared = list(kids.get(p.rsplit(sep, 1)[-1]) or [])
    if not declared:
        return "leaf"
    verdict = str(((verdicts or {}).get(p) or {}).get("verdict") or "")
    if verdict == "SEAL":
        return "sealed"
    if verdict == "UNVOTABLE":
        return "unvotable"
    return "stranded"


def bare_names_unique(tree):
    """Names that appear at more than one position. Empty means bare-name sealing is safe."""
    seen = collections.Counter()
    for name, children in tree.get("children", {}).items():
        for c in children:
            seen[c] += 1
    return {n: c for n, c in seen.items() if c > 1}


def vote(paths_by_sample, tree, min_support=MIN_SUPPORT, min_reach=MIN_REACH,
         descend_rule="any", sep=SEP, sentinels=SENTINELS):
    """The common scope. Returns one verdict per internal node of the declared tree.

    KEEP       every sample that reached it descended (or enough of them did) — the split stays
    SEAL       a sample that reached it declined to descend — the node becomes a leaf
    UNVOTABLE  fewer than `min_reach` samples reached it — NOT sealed, reported instead
    UNREACHED  no sample reached it at all — NOT sealed either, and this is deliberate: an empty
               branch removes nothing, so deleting it buys nothing and costs an unmeasured change
               to `node_weights` at whatever node is still open above it.
    """
    votes = node_votes(paths_by_sample, sep=sep, sentinels=sentinels, descend_rule=descend_rule)
    declared = internal_nodes(tree, sep=sep)

    verdicts = {}
    for node in declared:
        v = dict(votes.get(node, {"reached": [], "descended": [], "n_reached": 0,
                                  "n_descended": 0, "support": float("nan"),
                                  "cells": {}, "cells_below": {}}))
        v["children_declared"] = declared[node]
        v["stranded"] = _stranded(node, v)
        if node == ROOT:
            # SEALING THE ROOT IS NOT A SCOPE, IT IS AN EMPTY ANNOTATION. Its child set is the
            # level-1 compartments; delete them and every nucleus comes back UNRESOLVED. The
            # root's evidence is still computed and reported — root truncation is real and worth
            # seeing — but it is never actionable as a seal.
            v["verdict"] = "KEEP"
            v["note"] = ("root is never sealed; its shortfall is root-level truncation "
                         "(UNRESOLVED), which no seal can repair")
        elif v["n_reached"] == 0:
            v["verdict"] = "UNREACHED"
        elif v["n_reached"] < min_reach:
            v["verdict"] = "UNVOTABLE"
        elif v["support"] >= min_support:
            # AN OPEN NODE MAY NOT BE A TERMINAL LABEL. A node that keeps its children is a
            # branch, not a call: a cell left sitting on it carries the name of a COMPARTMENT,
            # which is the same string the L1 column uses for every cell beneath it. Read side
            # by side those two columns then disagree about what the word means -- one animal's
            # 961 `Endothelial` against the 26,552 `Endothelial` of the compartment.
            #
            # So the node is marked FORCE: annotation must push each stranded cell to its most
            # similar child rather than stopping. This is not a seal (the split is admissible,
            # the cohort agreed on it) and it is not a truncation (nothing terminates here).
            v["verdict"] = "FORCE" if v["stranded"] else "KEEP"
        else:
            v["verdict"] = "SEAL"
        verdicts[node] = v
    return verdicts


def sealed_labels(verdicts, paths_by_sample, sep=SEP, sentinels=SENTINELS):
    """The ACTUAL labels a seal removes, with their cell counts. Not a description of them.

    The project rule is that a removal is assessed by printing its members and reading them —
    `^Rp[sl]\\d` was "ribosomal genes" until it was printed, at which point it contained a
    kinase. A scope that says it seals two nodes is not assessable; one that says it removes
    `Matrifibrocyte` (7,187) and `Quiescent fibroblast` (2,845) is.
    """
    sealed = [n for n, v in verdicts.items() if v["verdict"] == "SEAL"]
    counts = collections.Counter()
    for paths in paths_by_sample.values():
        for p in paths:
            p = str(p)
            if p not in sentinels:
                counts[p] += 1
    out = {}
    for node in sealed:
        lost = {p: n for p, n in counts.items() if p.startswith(node + sep)}
        out[node] = dict(sorted(lost.items(), key=lambda kv: -kv[1]))
    return out


def seal_tree(tree, verdicts, sep=SEP):
    """A copy of `tree` with every SEALed node's ENTIRE child set removed.

    Returns `(sealed_tree, removed)`. `patterns` entries for nodes that are no longer reachable
    from the root are dropped too, so the corpus is not consulted for a label the tree can no
    longer emit — otherwise `node_weights` would keep scoring children that cannot be chosen.
    """
    out = copy.deepcopy(tree)
    kids = out.setdefault("children", {})
    removed = {}
    for node, v in sorted(verdicts.items()):
        if v["verdict"] != "SEAL" or node == ROOT:
            continue
        bare = node.split(sep)[-1]
        if bare in kids:
            removed[node] = list(kids.pop(bare))

    reachable = {"root"}
    stack = ["root"]
    while stack:
        for c in kids.get(stack.pop(), []):
            if c not in reachable:
                reachable.add(c)
                stack.append(c)
    if "patterns" in out:
        out["patterns"] = {k: v for k, v in out["patterns"].items() if k in reachable}
    if out.get("members"):
        out["members"] = {k: v for k, v in out["members"].items() if k in reachable}
    return out, removed


def apply_scope(path, verdicts, sep=SEP, sentinels=SENTINELS):
    """Truncate one pass-1 path to what the sealed tree could have produced.

    This is what makes the scope checkable WITHOUT re-running the walk: a sealed node's
    descendants collapse to the node itself, which is exactly what pass 2 must produce there.
    Where the tree stays open it returns the path unchanged, because sealing one node cannot
    alter a decision made at another — `node_weights` is computed over each node's own child set.
    """
    p = str(path)
    if p in sentinels:
        return p
    for node, v in verdicts.items():
        if v["verdict"] == "SEAL" and p.startswith(node + sep):
            return node
    return p


def format_report(verdicts, removed=None, sealed=None, n_samples=None):
    """The vote as lines a human reads before approving it.

    `n_samples` is the cohort size. It is a PARAMETER because the first version of this printed
    "/10" as a literal, which is the cohort this was written for baked into the tool: on seven
    samples it would have printed "7/10" and nobody reading it would have known.
    """
    order = {"SEAL": 0, "FORCE": 1, "UNVOTABLE": 2, "KEEP": 3, "UNREACHED": 4}
    if n_samples is None:
        n_samples = max((v["n_reached"] for v in verdicts.values()), default=0)
    rows = sorted(verdicts.items(), key=lambda kv: (order.get(kv[1]["verdict"], 9),
                                                    -kv[1]["n_reached"], kv[0]))
    out = [f"{'node':<34}{'reached':>8}{'descend':>8}{'support':>9}   verdict", "-" * 76]
    for node, v in rows:
        s = "  n/a" if v["n_reached"] == 0 else f"{v['support']:.3f}"
        reach = f"{v['n_reached']}/{n_samples}"
        out.append(f"{node:<34}{reach:>8}{v['n_descended']:>8}{s:>9}   {v['verdict']}")
    if sealed:
        out += ["", "WHAT EACH SEAL REMOVES — the labels themselves, not the category:"]
        for node, lost in sealed.items():
            if not lost:
                out.append(f"  {node}: nothing (no sample descended)")
            for p, n in lost.items():
                out.append(f"  {node}:  {n:>7,}  {p}")
    out += ["",
            "A seal removes the possibility of a LABEL, never an observation. Every cell keeps",
            "its pass-1 path, so this is reversible by re-running against the declared tree."]
    return out


def scoped_counts(verdicts, paths_by_sample, sep=SEP, sentinels=SENTINELS):
    """Per node of the SEALED tree: nuclei landing there, and how many samples have any.

    Counted through `apply_scope`, so a sealed node carries the nuclei its removed children used
    to hold — `Stromal/Fibroblast` is 17,961, not the 7,929 that truncated there in pass 1. That
    is the point: after the scope those are one population, and the tree must say so.
    """
    cells = collections.Counter()
    present = collections.defaultdict(set)
    for sample, paths in paths_by_sample.items():
        for p in paths:
            q = apply_scope(p, verdicts, sep=sep, sentinels=sentinels)
            if q in sentinels:
                continue
            cells[q] += 1
            present[q].add(sample)
    return cells, {k: len(v) for k, v in present.items()}


def format_tree(tree, verdicts, paths_by_sample, sep=SEP, sentinels=SENTINELS):
    """The scope drawn as the taxonomy pass 2 will actually walk.

    The node table says what the vote DECIDED; this says what you GET, which is the thing a
    reader checks against their expectation of the tissue. Only branches some sample reached are
    drawn — a declared branch nobody reached is reported underneath rather than drawn as though
    it were part of the scope, because it is not a seal and it holds nothing.
    """
    cells, present = scoped_counts(verdicts, paths_by_sample, sep=sep, sentinels=sentinels)
    n = len(paths_by_sample)
    kids = tree.get("children", {})

    reached = set()
    for node in list(cells):
        parts = node.split(sep)
        for d in range(1, len(parts) + 1):
            reached.add(sep.join(parts[:d]))

    out = ["root"]

    def walk(name, path, pad):
        children = [c for c in kids.get(name, [])
                    if (path + [c] and sep.join(path + [c]) in reached)]
        for i, c in enumerate(children):
            last = i == len(children) - 1
            cpath = sep.join(path + [c])
            stem = "└── " if last else "├── "
            grand = [g for g in kids.get(c, []) if sep.join(path + [c, g]) in reached]
            if grand:
                # AN OPEN INTERNAL NODE CAN STILL HOLD CELLS OF ITS OWN, and the first version
                # of this drew it with no count. Those nuclei are cells a sample stranded at the
                # parent because its gap failed where other samples' cleared -- exactly the
                # disagreement the scope exists to surface -- and drawing the node bare made
                # them invisible. Measured when it was found: 1,060 nuclei missing from a
                # drawing that looked complete, and a stated total 1,060 short of the truth.
                own = cells.get(cpath, 0)
                head = f"{pad}{stem}{c}"
                if own:
                    ns = len(verdicts.get(cpath, {}).get("stranded", {}) or {})
                    out.append(f"{head:<44}{'':>8}   "
                               f"[{own:,} from {ns} sample(s) -> most similar child]")
                else:
                    out.append(head)
            else:
                mark = "  \u25a0 SEALED" if verdicts.get(cpath, {}).get("verdict") == "SEAL" else ""
                head = f"{pad}{stem}{c}{mark}"
                # pad to a fixed column so the counts line up regardless of nesting depth
                out.append(f"{head:<44}{cells.get(cpath, 0):>8,}   {present.get(cpath, 0)}/{n}")
            walk(c, path + [c], pad + ("    " if last else "│   "))

    walk("root", [], "")

    unreached = sorted({f"{k}/{c}" for k, v in kids.items() for c in v} -
                       {p.split(sep)[-1] for p in reached} -
                       {f"{k}/{c}" for k, v in kids.items() for c in v
                        if any(p.split(sep)[-1] == c for p in reached)})
    if unreached:
        out += ["", "declared but reached by no sample — not sealed, not drawn, still in the tree:"]
        out += [f"  {u.split('/')[-1]}" for u in unreached]
    return out


def truncate_tree(tree, depth=1, sep=SEP):
    """The top `depth` levels of a taxonomy, as a tree in its own right.

    WHY AN INDEPENDENT L1 RUN RATHER THAN path[:1]

    Truncating the deep walk's path gives an L1 that INHERITS the deep walk's failures. A cell
    the walk sent to UNRESOLVED at the root has no path to truncate, so it has no L1 either --
    on the cohort this was written for that is 1,880 nuclei in 4 of 10 animals, and 973 of one
    animal's are a Pericyte population that simply vanishes from the L1 table.

    Walking a depth-1 tree instead makes L1 its own annotation, produced by the UNCHANGED walk
    against a tree with no depth to seal. Two consequences worth stating plainly:

      - No seal, at any depth, can move the L1 column. The scope and L1 are independent by
        construction rather than by convention.
      - The root decision itself is unchanged: root's child set is the same in both trees, so
        `node_weights` and therefore the gap at the root are identical. An independent L1 run
        does NOT rescue a nucleus the root already declined -- it isolates L1 from everything
        BELOW the root, which is a different and achievable guarantee. Claiming otherwise would
        be claiming the walk had changed, and it has not.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    out = copy.deepcopy(tree)
    kids = out.get("children", {})

    keep_children, frontier, level = {}, [ROOT], 0
    while frontier and level < depth:
        nxt = []
        for name in frontier:
            if name in kids:
                keep_children[name] = list(kids[name])
                nxt.extend(kids[name])
        frontier, level = nxt, level + 1
    out["children"] = keep_children

    reachable = {ROOT}
    stack = [ROOT]
    while stack:
        for c in keep_children.get(stack.pop(), []):
            if c not in reachable:
                reachable.add(c)
                stack.append(c)
    if "patterns" in out:
        out["patterns"] = {k: v for k, v in out["patterns"].items() if k in reachable}
    if out.get("members"):
        out["members"] = {k: v for k, v in out["members"].items() if k in reachable}
    return out


def tree_depth(tree):
    """How many levels BELOW the root this tree can emit. `truncate_tree(t, 1)` gives 1.

    This is the check that makes `--l1-tree` refuse a file that is not a depth-1 tree, and it
    reads the TREE rather than the walk's output. A result-based check would be data-dependent:
    a depth-2 tree whose gap happened to fail everywhere in sample 3 would pass there and refuse
    on sample 4, so a cohort would get half a column written one way and half the other. The
    declaration gives every sample the same verdict, which is what a per-sample pipeline needs.

    A name reachable by two routes is counted at the SHALLOWER one and not revisited — the guard
    that stops a malformed cyclic `children` spinning forever. `bare_names_unique` is what
    DETECTS that shape; this function only refuses to hang on it.
    """
    kids = (tree or {}).get("children", {}) or {}
    depth, frontier, seen = 0, [ROOT], {ROOT}
    while frontier:
        nxt = []
        for name in frontier:
            for c in kids.get(name, []):
                if c not in seen:
                    seen.add(c)
                    nxt.append(c)
        if not nxt:
            break
        depth, frontier = depth + 1, nxt
    return depth


def root_child_diff(tree, other):
    """`(only_in_tree, only_in_other)` for the two root child sets, as sorted lists.

    An independent L1 run is comparable with the deep walk's `path[:1]` only when both walks
    faced the SAME decision at the root — same children, therefore the same weights and the same
    gap. `truncate_tree` guarantees that; a hand-written depth-1 tree does not, and the
    difference is silent, giving two label columns that look like one taxonomy and are not.
    Reported, never refused: scAnno does not get to decide that a caller meant them to match.
    """
    a = set((tree or {}).get("children", {}).get(ROOT, []) or [])
    b = set((other or {}).get("children", {}).get(ROOT, []) or [])
    return sorted(a - b), sorted(b - a)
