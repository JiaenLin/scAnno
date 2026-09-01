"""The rooted tree walk — descend, or truncate.

Truncation, not abstention, is the failure action. A flat classifier with a reject option
was implemented here once and scored 4/8 where its own argmax scored 7/8: when two
siblings are genuinely close the honest answer is their parent, not "no idea". The tree
must be ROOTED for that to be expressible - a forest of leaves has nowhere to truncate to.
"""
from __future__ import annotations

import numpy as np

from .corpus import node_weights as corpus_node_weights
from .store import safe_scale

#: Declared, not derived. See docs/CLASSIFIER.md for the sweep behind each value: the
#: untrained path needs more separation because its weights are unvalidated, and it pays
#: in label depth rather than in errors.
GAP_PROFILE = 0.15
GAP_CORPUS = 0.30


def node_profiles(store, members_of, nodes):
    """Pooled profile per node — support-weighted and TRIMMED.

    Max is the least robust aggregator available: one mislabelled corpus entry enters at
    full strength. With >=3 members the second-highest is taken instead, which keeps the
    anti-fragmentation property and drops the single-entry sensitivity.
    """
    idx = {c: i for i, c in enumerate(store.celltypes)}
    out = {}
    for node in nodes:
        mi = [idx[m] for m in members_of.get(node, []) if m in idx]
        mi = [i for i in mi if store.grade(i) != "C0"]
        if not mi:
            continue
        P = store.mean[mi]
        out[node] = np.sort(P, axis=0)[-2] if len(mi) >= 3 else P.max(axis=0)
    return out


def profile_weights(store, members_of, nodes, usable):
    """Contrast weights for one sibling set, normalised by FULL evidence mass."""
    nodeP = node_profiles(store, members_of, nodes)
    order = [n for n in nodes if n in nodeP]
    if len(order) < 2:
        return None, order, None
    P = np.vstack([nodeP[n] for n in order])
    scale = safe_scale(np.abs(P).std(axis=0))
    W = np.zeros((P.shape[1], len(order)))
    for j in range(len(order)):
        sib = np.delete(P, j, axis=0)
        comp = np.sort(sib, axis=0)[-2] if sib.shape[0] >= 4 else sib.max(axis=0)
        d = (P[j] - comp) / scale
        W[:, j] = np.clip(1.0 + d, 0.1, 4.0) * (P[j] - P.min())
    full = W.sum(axis=0, keepdims=True)
    W[~usable] = 0.0
    cover = W.sum(axis=0) / np.where(full[0] > 0, full[0], 1.0)
    return W / np.where(full > 0, full, 1.0), order, cover


def missing_nodes(store, members_of):
    """Declared nodes the store cannot represent at all."""
    have = set(store.celltypes)
    return {n: ms for n, ms in members_of.items() if not (set(ms) & have)}


def classify(Z, usable, tree, store=None, assertions=None, gap_min=None, exclude=None):
    """Walk the tree per cluster. Weights from profiles if a store is given, else corpus.

    `tree` is {"children": {node: [child, ...]}, "members": {...}, "patterns": {...}}.
    `members` maps a node to store cell-type names; `patterns` to corpus name substrings.

    `exclude` - a per-cluster boolean mask, or cluster indices - names clusters upstream QC
    flagged. They are NOT walked and receive the sentinel label `EXCLUDED`, which is not a cell
    type. Every cluster still appears in the output, in order, carrying `excluded: True`: a
    caller that drops rows loses the record of what was removed, and a caller that reindexes
    silently mislabels everything after the first gap. Pass the same mask to `standardise`, or
    an excluded cluster can still decide which genes are usable - see `scanno/exclude.py`.
    """
    from .exclude import EXCLUDED, as_mask

    use_corpus = store is None or assertions is not None
    if gap_min is None:
        gap_min = GAP_CORPUS if use_corpus else GAP_PROFILE
    children = tree["children"]
    drop = as_mask(exclude, Z.shape[0])
    out = []
    for c in range(Z.shape[0]):
        if drop[c]:
            # No walk, no gap, no survival. A number here would be a statistic about a decision
            # that was not taken, and would sort and average alongside ones that were.
            out.append({"cluster": c, "label": EXCLUDED, "path": EXCLUDED, "depth": 0,
                        "gap": float("nan"), "survival": float("nan"),
                        "cover": float("nan"), "excluded": True, "trace": []})
            continue
        node, path, trace = "root", [], []
        while children.get(node):
            kids = children[node]
            if use_corpus:
                W, order, cover, _hits, surv = corpus_node_weights(
                    assertions, {k: tree["patterns"][k] for k in kids
                                 if k in tree.get("patterns", {})},
                    tree["genes"], usable)
            else:
                W, order, cover = profile_weights(store, tree["members"], kids, usable)
                surv = None
            if W is None:
                break
            s = Z[c] @ W
            srt = np.argsort(-s)
            spread = float(np.abs(s).max()) or 1.0
            gap = float(s[srt[0]] - s[srt[1]]) / spread if len(order) > 1 else 1.0
            # `survival` travels with the call because the sibling contrast is depth-biased
            # and could not be de-biased without costing accuracy (scanno/corpus.py). A win on
            # a panel that lost 40% of its evidence to better-cited neighbours is a weaker
            # result than the same gap on an intact one, and nothing else in the output says so.
            # THE RUNNER-UP AND THE SCORES, because a call cannot be audited from its winner.
            # `gap` says the margin was 0.64; it does not say 0.64 over WHAT, and without that a
            # reader cannot tell a call that beat a near-tie from one that beat nothing close.
            # Measured on a real cohort: a cluster 72% of whose cells another route called
            # cardiomyocyte was labelled Neural, and the run recorded the label and the margin
            # and nothing about what lost - so the reason had to be reconstructed from a marker
            # table, which is inference and not a record.
            trace.append({"at": node, "top": order[srt[0]],
                          "second": (order[srt[1]] if len(order) > 1 else None),
                          "scores": {str(order[i]): round(float(s[i]), 4) for i in srt[:6]},
                          "gap": gap,
                          "survival": (float(surv[srt[0]]) if surv is not None
                                       else float("nan")),
                          "cover": (float(cover[srt[0]]) if cover is not None
                                    else float("nan"))})
            if gap < gap_min:
                break                                    # TRUNCATE, do not abstain
            node = order[srt[0]]
            path.append(node)
        # The statistics of the step that produced the LABEL - the last accepted one - not of
        # the step that truncated. Reporting the failed step's numbers beside a label the
        # previous step chose describes a decision that was not taken.
        acc = trace[len(path) - 1] if path else (trace[-1] if trace else None)
        out.append({"cluster": c,
                    "label": path[-1] if path else "UNRESOLVED",
                    "path": "/".join(path) or "UNRESOLVED",
                    "depth": len(path),
                    "gap": trace[-1]["gap"] if trace else 0.0,
                    "survival": acc["survival"] if acc else float("nan"),
                    "cover": acc["cover"] if acc else float("nan"),
                    "excluded": False,
                    "trace": trace})
    return out


def gate_auc(stat, correct):
    """Does a statistic separate correct calls from incorrect ones?

    Below ~0.6 it is noise and must not gate anything. Four statistics in this design's
    history were given veto power without this check and all four made results worse.
    """
    stat, correct = np.asarray(stat, float), np.asarray(correct, bool)
    pos, neg = stat[correct], stat[~correct]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
                 / (len(pos) * len(neg)))
