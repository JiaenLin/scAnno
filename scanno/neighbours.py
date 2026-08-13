"""Does the annotation respect the manifold? A diagnostic over the kNN graph.

WHY THIS IS A DIAGNOSTIC AND NOT A LABEL

The classifier scores a cluster's mean profile. It never asks whether a cell's NEIGHBOURS were
called the same thing, so three failures are currently invisible to it:

  * a mixed cluster - one partition holding two populations - which shows up as cells whose
    neighbourhoods disagree with their own label
  * a cell the markers cannot call but whose neighbourhood is unambiguous
  * an annotation that simply does not follow the graph, which nothing else here would report

The obvious next step - propagating labels over the graph - is deliberately NOT taken. A
propagated label depends on what else was sequenced beside the cell, and composition-independence
is the property `standardise()` exists to guarantee and that several removed components were
removed for breaking. So this measures and reports; it changes no call. If a smoothed label is
ever wanted it belongs in its own column, marked as composition-dependent, beside the one that
is not.

Read the numbers as a property of THIS object. Neighbour agreement is not transferable between
datasets and is not a quality score for a label - it is a statement about whether a partition and
an annotation tell the same story on the graph that was built.

COST

One sparse matrix product against a one-hot label indicator, the same shape as
`cluster_profile()`. The graph is already on the object; nothing is recomputed and no matrix of
expression is touched.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def _onehot(labels):
    cats = sorted(set(map(str, labels)))
    idx = {c: i for i, c in enumerate(cats)}
    y = np.array([idx[str(v)] for v in labels])
    n = len(y)
    return cats, y, sp.csr_matrix((np.ones(n), (np.arange(n), y)), shape=(n, len(cats)))


def label_flow(graph, labels, weighted=True):
    """(cats, per-cell own-label share, per-cell label mass). One sparse product.

    `graph` is a cell x cell adjacency - scanpy leaves one in `obsp['connectivities']`. With
    `weighted`, the connectivity weights are used; without, every edge counts once, which is the
    right choice when the weights encode a kernel whose scale differs between dense and sparse
    regions and you do not want that scale in the answer.

    The diagonal is removed. A cell is trivially its own neighbour and counting it inflates every
    agreement toward 1, most where a cell has fewest neighbours - which is exactly the rare
    population whose agreement you were trying to read.
    """
    G = sp.csr_matrix(graph)
    G.setdiag(0)
    G.eliminate_zeros()
    if not weighted:
        G = G.astype(bool).astype(np.float64)
    cats, y, Y = _onehot(labels)
    S = np.asarray((G @ Y).todense(), dtype=np.float64)      # cells x labels
    tot = S.sum(axis=1)
    own = S[np.arange(len(y)), y]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(tot > 0, own / np.maximum(tot, 1e-12), np.nan)
    return cats, share, S


def cluster_neighbourhood(graph, labels, clusters=None, weighted=True, min_isolated=0):
    """Per label - or per cluster - how much of each neighbourhood carries the same label.

    Returns one row per group with:

      agreement       mean over its cells of the share of neighbour mass carrying its own label
      foreign         fraction of its cells whose PLURALITY neighbour label is not their own.
                      This is the mixed-cluster signal: a cluster can average a respectable
                      agreement while a third of it sits inside somebody else's territory, and
                      the mean alone will not say so
      pieces          connected components of the group's OWN induced subgraph, and the share
                      in the largest. This is the second failure and `foreign` cannot see it:
                      a cluster holding two populations that each sit in a coherent
                      neighbourhood of their own label has foreign = 0 and agreement = 1.0,
                      and is still two things. Splitting is a property of the graph, not of
                      the labels, so it is measured without them
      isolated        cells with no neighbours at all, reported rather than dropped - an
                      unconnected cell has no agreement and averaging over it as if it did is
                      how a number becomes an opinion
      borders_on      the two labels its cells most often border, with their share of the
                      non-self neighbour mass

    The two signals catch different things and neither subsumes the other. `foreign` finds cells
    in the wrong place; `pieces` finds a partition that has merged two right places.
    """
    G_all = sp.csr_matrix(graph)
    G_all.setdiag(0)
    G_all.eliminate_zeros()
    grp = np.asarray(labels if clusters is None else clusters).astype(str)
    lab = np.asarray(labels).astype(str)
    cats, share, S = label_flow(graph, lab, weighted)
    ci = {c: i for i, c in enumerate(cats)}
    tot = S.sum(axis=1)
    plurality = np.array([cats[i] for i in S.argmax(axis=1)])
    plurality[tot <= 0] = ""

    rows = []
    for g in sorted(set(grp)):
        m = grp == g
        live = m & (tot > 0)
        own = lab[m][0] if len(set(lab[m])) == 1 else None
        # Neighbour mass by label, summed over the group, with the group's own label removed so
        # "what does this border on" is not answered by itself.
        mass = S[m].sum(axis=0).copy()
        if own is not None and own in ci:
            mass[ci[own]] = 0.0
        order = np.argsort(-mass)
        borders = [(cats[i], float(mass[i] / mass.sum())) for i in order[:2]
                   if mass.sum() > 0 and mass[i] > 0]
        # How many pieces is this group, on its own edges? Computed on the induced subgraph, so
        # it asks whether the group hangs together rather than whether its labels agree.
        idx = np.where(m)[0]
        sub = G_all[idx][:, idx]
        n_comp, comp = connected_components(sub, directed=False)
        big = float(np.bincount(comp).max() / len(idx)) if len(idx) else float("nan")
        rows.append({
            "group": g,
            "label": own,
            "n_cells": int(m.sum()),
            "isolated": int((m & (tot <= 0)).sum()),
            "agreement": float(np.nanmean(share[live])) if live.any() else float("nan"),
            "foreign": (float((plurality[live] != lab[live]).mean()) if live.any()
                        else float("nan")),
            "pieces": int(n_comp),
            "largest_piece": big,
            "borders_on": borders,
        })
    return rows


def format_report(rows, top=None) -> str:
    """The table, worst agreement first - which is the order anyone reads it in."""
    rs = sorted(rows, key=lambda r: (np.isnan(r["agreement"]), r["agreement"]))
    if top:
        rs = rs[:top]
    out = ["  %-30s %8s %10s %9s %8s %9s  %s" %
           ("group", "cells", "agreement", "foreign", "pieces", "isolated", "borders on")]
    for r in rs:
        b = ", ".join(f"{n} {100*s:.0f}%" for n, s in r["borders_on"]) or "-"
        pc = (f"{r['pieces']}" if r["pieces"] <= 1
              else f"{r['pieces']} ({100*r['largest_piece']:.0f}%)")
        out.append("  %-30s %8s %9s%% %8s%% %8s %9d  %s" % (
            str(r["group"])[:30], f"{r['n_cells']:,}",
            "  n/a" if r["agreement"] != r["agreement"] else f"{100*r['agreement']:5.1f}",
            "  n/a" if r["foreign"] != r["foreign"] else f"{100*r['foreign']:5.1f}",
            pc, r["isolated"], b))
    split = [r for r in rs if r["pieces"] > 1]
    if split:
        out.append(f"\n  {len(split)} group(s) are not one connected piece on their own edges - "
                   f"`foreign` cannot see this and it is the mixed-cluster signal")
    live = [r["agreement"] for r in rs if r["agreement"] == r["agreement"]]
    if live:
        out.append(f"\n  agreement {100*min(live):.1f}% to {100*max(live):.1f}%, "
                   f"median {100*float(np.median(live)):.1f}%")
    out.append("  A property of THIS object's graph, not a quality score for a label, and not "
               "comparable between datasets.")
    return "\n".join(out)
