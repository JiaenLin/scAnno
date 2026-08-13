"""The marker corpus — node weights for contexts with no atlas profile.

This is the path most users take. A curated corpus covers far more species x tissue
contexts than annotated atlases do, and a tool that only works once someone supplies an
atlas is not a general tool.

The corpus is NOT distributed with scAnno. `scanno build-markers` ingests a release the
user fetches themselves, the same way scQC ships a reference registry and not a genome.
"""
from __future__ import annotations

import sqlite3

import numpy as np

#: Tier 1 is strongest. Tier 5 is "a marker detector ran; no paper asserts it" and is the
#: bulk of any release - typically over 90% of a tissue slice. It sits at a
#: floor rather than at zero because zero can never be promoted by calibration.
TIER_W = {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0, 5: 0.25}


def load_assertions(db, species: str, tissue: str, min_tier: int = 4) -> dict:
    """{cell_name: {GENE: w_bib}} for one context.

        w_bib = TIER_W[tier] x (ln(1 + n_pmids) + 1)

    Publication counts are logged so 200 papers beats 6 by about 2x rather than 33x; tier
    dominates. Duplicate assertions for one (cell, gene) take the MAX, never the sum.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT cell_name, symbol_norm, evidence_tier, n_pmids FROM assertion "
        "WHERE species=? AND tissue_class=? AND evidence_tier<=? AND symbol_norm!=''",
        (species, tissue, min_tier)).fetchall()
    con.close()
    out: dict = {}
    for cell, sym, tier, npm in rows:
        w = TIER_W.get(tier, 0.0) * (float(np.log1p(max(npm or 0, 0))) + 1.0)
        if w > 0:
            d = out.setdefault(cell, {})
            d[sym] = max(d.get(sym, 0.0), w)
    return out


class GeneSpaceMismatch(SystemExit):
    """The corpus and the query do not share a gene naming scheme."""


def check_gene_space(assertions, genes, min_overlap=50):
    """Refuse when the corpus cannot address the query's genes at all.

    Nothing downstream notices this. Every marker panel comes out empty, every node is
    dropped for having too few markers, the walk finds fewer than two children at the root
    and returns UNRESOLVED - for every cluster, silently, at what looks like a successful
    run. It happened: an object keyed by Ensembl IDs against a symbol-keyed corpus produced
    0 of 34,290 overlapping genes and 100% UNRESOLVED with no error.

    The out-of-distribution check does not catch it either, because that compares the query
    against the STORE, and a store built from the same object shares its naming scheme. The
    mismatch is between both of them and the CORPUS.
    """
    corpus = set()
    for d in assertions.values():
        corpus |= set(d)
    have = {str(g).upper() for g in genes}
    n = len(have & corpus)
    if n < min_overlap:
        ex_c = ", ".join(sorted(corpus)[:4])
        ex_g = ", ".join(str(g) for g in list(genes)[:4])
        raise GeneSpaceMismatch("\n".join([
            f"scanno: REFUSE - the corpus and this object share only {n} gene symbols "
            f"(floor {min_overlap}).",
            f"        corpus looks like : {ex_c}",
            f"        object looks like : {ex_g}",
            "        Every marker panel would be empty and every cluster would come back",
            "        UNRESOLVED, which reads as a completed run. Map the object's var_names",
            "        to symbols first - many objects keep them in var['gene_symbol'].",
        ]))
    return n


def node_weights(assertions, node_patterns, genes, usable, min_markers=3,
                 sibling_contrast=True, relative_contrast=False):
    """genes x nodes, from the corpus alone.

    SIBLING CONTRAST — the untrained path's fix, and it needs no atlas.

    Reordering corpus panels against atlas data showed one dominant pattern: they are
    contaminated with markers of NEIGHBOURING LINEAGES. CD3D/CD3E/CD8A are claimed for B
    cells and NK cells; CD14/CD68/CD163/CSF1R are claimed for dendritic cells. The corpus
    aggregates papers that used different cell-type definitions, so panels accumulate their
    neighbours' genes.

    That contamination is visible in the corpus itself, so it can be removed with no
    expression data at all. Each gene is charged its best competing claim:

        w_disc(g, k | S) = max(0, w_bib(g, k) - max over siblings j of w_bib(g, j))

    Clipped at zero, NEVER negative: negative weights were measured and cost two errors on
    an independent dataset while being invisible on a self-test. This removes shared
    evidence; it never argues against a node.

    A node left with nothing keeps its raw panel rather than becoming unscoreable.

    THE CONTRAST IS DEPTH-BIASED, THIS IS MEASURED, AND FIVE FIXES ALL FAILED

    `w_bib` carries `ln(1 + n_pmids)`, so a well-studied cell type has a larger number for
    the SAME gene simply because more papers mention it, and an absolute subtraction charges
    a rare type its neighbour's fame. On mouse heart: panels of >=100 markers keep 91% of
    their mass through the contrast, panels of <20 keep 68%, corr(log10 panel size,
    survival) = +0.37. Lymphatic endothelial loses ALL SIX of its markers to Vascular
    endothelial; Monocyte keeps 41% against Macrophage; Endocardial 48%.

    That is a real bias and every attempt to remove it cost accuracy on PBMC, self-test and
    independent (`--relative-contrast` and three reweightings):

        weighting / contrast          bias r    3k self         68k independent
        tier x ln(pmids), absolute    +0.37     8 ex, 0 wrong   6 ex, 1 wrong   <- shipped
        tier x ln(pmids), RELATIVE    -0.46     7 ex, 0 wrong   6 ex, 2 wrong
        tier only,        absolute    +0.46     6 ex, 0 wrong   7 ex, 1 wrong
        tier x specificity            +0.37     8 ex, 0 wrong   6 ex, 2 wrong
        tier x pmids x specificity    +0.26     8 ex, 0 wrong   6 ex, 2 wrong

    Two things worth reading off that table. The relative form REVERSES the bias rather than
    removing it (+0.37 to -0.46) - small panels then win genes they share with large ones,
    and Macrophage drops from 95% survival to 77%. And dropping the publication term, the
    obvious suspect, makes the bias WORSE, not better: the depth advantage is not mainly
    coming from citation counts.

    So the shipped weighting is the least-bad of five measured, not a solution. The bias is
    a known limitation of a curated corpus - rare types genuinely have narrower panels - and
    it is reported per call through `cover` and `node_support` rather than corrected away.
    `relative_contrast=True` is kept so the alternative can be re-measured on another corpus,
    not because it is recommended here.
    """
    gi = {str(g): i for i, g in enumerate(genes)}
    names, raw, hits = [], [], []
    for node, pats in node_patterns.items():
        acc = {}
        for cell, d in assertions.items():
            low = cell.lower()
            if not any(p in low for p in pats):
                continue
            for sym, w in d.items():
                acc[sym] = max(acc.get(sym, 0.0), w)
        col = np.zeros(len(genes))
        n_hit = 0
        for sym, w in acc.items():
            i = gi.get(sym)
            if i is not None:
                col[i] = w
                n_hit += 1
        if n_hit < min_markers:
            continue
        names.append(node)
        raw.append(col)
        hits.append(n_hit)
    if len(names) < 2:
        return None, names, None, None, None

    R = np.vstack(raw)
    RAW = R.copy()
    if sibling_contrast:
        # Compare shares of each node's own panel, not raw citation weight. See the docstring:
        # the absolute form charged a rare type its neighbour's fame.
        C = (R / np.maximum(R.sum(axis=1, keepdims=True), 1e-12)) if relative_contrast else R
        D = np.empty_like(R)
        for j in range(len(names)):
            others = np.delete(C, j, axis=0)
            D[j] = np.clip(C[j] - others.max(axis=0), 0.0, None)
            if D[j].sum() <= 0:
                D[j] = R[j]
        R = D

    # How much of each node's own evidence still carries a claim after the contrast. This is
    # the depth bias made visible per call: it runs ~91% for a panel of 100+ markers and ~68%
    # for one under 20, so a call on a low-survival node beat its siblings on a panel that had
    # already been stripped by better-cited neighbours. The bias could not be removed without
    # costing accuracy (see the docstring's table), so it is reported instead.
    surv = np.array([float(RAW[j][R[j] > 0].sum() / RAW[j].sum()) if RAW[j].sum() > 0 else 0.0
                     for j in range(len(names))])

    cols, cover = [], []
    for j in range(len(names)):
        full = R[j].sum()
        masked = R[j].copy()
        masked[~usable] = 0.0
        # FULL mass, not surviving mass. Normalising by what survived the query's gene
        # coverage rescales a depleted node back to full weight - which is how a panel
        # that lost every real marker got scored on housekeeping leftovers.
        cols.append(masked / (full if full > 0 else 1.0))
        cover.append(masked.sum() / (full if full > 0 else 1.0))
    return np.vstack(cols).T, names, np.array(cover), hits, surv


def node_support(db, species, tissue, node_patterns, max_tier=2):
    """Curated assertions behind each node — the reliability signal the gap does not carry.

    A node's `gap` says how far it beat its siblings on THIS data. It says nothing about
    how much evidence the panel rests on, and a small, concentrated panel can beat a large
    diluted one with a perfectly healthy gap. Deep nodes are where that bites: a level-3
    node resting on a handful of curated assertions will look exactly as confident as a
    level-1 node resting on fifty.

    Reported beside every call so a reader can see the difference.
    """
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT cell_name, COUNT(*) FROM assertion WHERE species=? AND tissue_class=? "
        "AND evidence_tier<=? GROUP BY 1", (species, tissue, max_tier)).fetchall()
    con.close()
    out = {}
    for node, pats in node_patterns.items():
        out[node] = sum(n for cell, n in rows
                        if any(p in cell.lower() for p in pats))
    return out
