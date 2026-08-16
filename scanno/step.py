"""ONE step of the walk, computed at a node the walk never reached.

WHY THIS EXISTS

`classify()` descends from the root and STOPS — at a leaf, at a node whose children cannot be
represented, or at the first gap below the bar. Everything it scored is in `trace`; everything
below where it stopped was never scored at all. So `trace[-1]["top"]` is the argmax of exactly
ONE node, and the result contains nothing whatsoever about which GRANDCHILD a cluster resembles.

`scanno/force.py` needs precisely that. FORCE pushes a stranded cluster off a node the cohort
agreed to split, and when the child it lands on is itself internal the push is not finished: the
cell still carries a compartment name, one level down, which is the whole defect FORCE exists to
remove. Finishing it needs a real score at a node the walk stopped short of. There is no way to
read one out of the walk's output, because the walk never computed it.

THE SAME MACHINERY, NOT A SECOND SCORER

A forced step has to be computed the way a walked step would have been, or a forced call and a
gap-cleared call on the same child are two different measurements wearing one name — and the
`<prefix>_gap` column would hold two different quantities depending on how the row got there.

So this calls what `classify()` calls: `corpus.node_weights` or `classify.profile_weights` over
that node's own child set, the same `Z[c] @ W`, the same descending sort for the argmax, and the
same normalisation of the gap by the spread. It returns a `trace` ENTRY — the same five keys, in
the same units — so the caller can put it exactly where a trace entry goes.

WHAT IT DOES NOT DO

It does not walk, does not descend, applies no threshold and decides nothing. Which node to
score, and what to do with the answer, belong to the caller. `scanno/classify.py` is NOT
modified and NOT imported wholesale: the walk owns the loop, the truncation rule and the bar,
and none of those appear here.

That leaves one real hazard — this file and the walk drifting apart — and it is answered by
measurement rather than by care. `tests/test_force.py` runs a REAL `classify()` and asserts this
function reproduces every entry of its trace, key for key, at every node it visited. A textual
copy can drift silently; an agreement that is re-measured on every test run cannot.

RETURNS None RATHER THAN A GUESS

`None` where the walk would have broken: a node with no children, or a child set the weights
cannot represent (`node_weights` returns no matrix when fewer than two children survive the
marker floor). The caller must treat that as "not measured", never as "no preference" — a
classifier that truncates must not invent a destination, and that is the one rule this whole
area of the package exists to keep.
"""
from __future__ import annotations

import numpy as np

from .classify import profile_weights
from .corpus import node_weights as corpus_node_weights


def node_scorer(Z, usable, tree, store=None, assertions=None):
    """Return `step(cluster, node)` — the trace entry `classify()` would have written there.

    The arguments are the ones `classify()` itself was given, and `use_corpus` is decided by the
    same expression, so a caller cannot accidentally score a node with weights the walk did not
    use. Bound once and reused, because the tree and the weights source do not change between
    steps and re-deriving them per call would invite them to.
    """
    use_corpus = store is None or assertions is not None
    children = tree.get("children", {})
    patterns = tree.get("patterns", {}) or {}

    def step(c, node):
        kids = list(children.get(node) or [])
        if not kids:
            return None
        if use_corpus:
            W, order, cover, _hits, surv = corpus_node_weights(
                assertions, {k: patterns[k] for k in kids if k in patterns},
                tree["genes"], usable)
        else:
            W, order, cover = profile_weights(store, tree["members"], kids, usable)
            surv = None
        if W is None:
            return None
        s = Z[c] @ W
        srt = np.argsort(-s)
        spread = float(np.abs(s).max()) or 1.0
        gap = float(s[srt[0]] - s[srt[1]]) / spread if len(order) > 1 else 1.0
        return {"at": node, "top": order[srt[0]], "gap": gap,
                "survival": (float(surv[srt[0]]) if surv is not None else float("nan")),
                "cover": (float(cover[srt[0]]) if cover is not None else float("nan"))}

    return step
