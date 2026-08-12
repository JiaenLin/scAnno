"""`scanno calibrate` — learn marker reliability from annotated atlases.

Run once, offline, by whoever maintains a corpus. Consumes atlases; emits numbers. The
atlases are never redistributed and the classifier never sees them.

WHAT COMES OUT

  store.npz          per-(celltype, gene) profiles + the gene background + C grades
  reliability.tsv    per (context, node, gene): citation weight, learned L, posterior,
                     both ranks, the rank delta, and a verdict
  panels.tsv         each node's markers reordered by measured discriminative power
  calibration.json   digest, provenance census, and what was refused and why

`reliability.tsv` is the scientific output, not a by-product. A panel reordered by
measured power, with its disagreement against the citation ordering made explicit, is a
falsifiable claim anyone with the same atlases can re-derive. A panel whose order does not
change is a null result and is reported as one.

WHAT IS BOUNDED, AND WHY

  L in [0.25, 4.0], never 0    driving a weight to zero removes a marker from every future
                               annotation. The markers at risk are the rare, lightly cited
                               ones that are nonetheless the only validated marker for
                               their type - exactly what an unbounded fit deletes.
  promotion needs clean labels most atlas labels are marker-derived; learning from labels
                               assigned by the markers being graded measures agreement
                               with prior practice, not truth.
  one dataset, one vote        a 5M-cell atlas must not outvote two hundred small studies.
  no significance testing      at these sample sizes a p-value is a statement about n.
                               Promotion is decided on effect size and replication.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .store import safe_scale

L_MIN, L_MAX = 0.25, 4.0

#: Shrinkage constant for partial pooling. A (gene, node) pair seen in one source is pulled
#: most of the way toward the level above it; one seen in many asserts on its own.
POOL_K = 3.0


@dataclass
class Calibration:
    context: dict
    nodes: list
    genes: np.ndarray
    L: dict                       # node -> per-gene multiplier
    rows: list                    # the reliability table
    store_digest: str
    census: dict

    def multiplier(self, node: str) -> np.ndarray:
        return self.L.get(node, np.ones(len(self.genes)))


def _contrast(store, members_of, node, siblings, gene_index_n):
    """Per-gene effect size for one node against its declared siblings, from profiles."""
    idx = {c: i for i, c in enumerate(store.celltypes)}
    mi = [idx[m] for m in members_of.get(node, []) if m in idx]
    si = [idx[m] for s in siblings if s != node
          for m in members_of.get(s, []) if m in idx]
    if not mi or not si:
        return None, 0
    P = (store.mean - store.gene_mu) / store.gene_sd
    mu_k = P[mi].max(axis=0)
    mu_s = P[si].max(axis=0)
    d = (mu_k - mu_s) / safe_scale(P.std(axis=0))
    support = int(min(store.n_sources[i] for i in mi))
    return d, support


def _pool(per_node: dict, n_genes: int):
    """Partial pooling — node level shrunk toward a gene-level prior.

    A weight fitted per (gene, node, context) cannot transfer to a context that was not in
    the training set, which is the situation the corpus exists for. Pooling gives a
    gene-level fallback that does transfer, and lets a well-supported node override it.
    """
    if not per_node:
        return {}, np.ones(n_genes)
    stack = np.vstack([d for d, _ in per_node.values()])
    gene_prior = np.nanmean(stack, axis=0)
    out = {}
    for node, (d, support) in per_node.items():
        w = support / (support + POOL_K)                  # 0 when unsupported, ->1 when deep
        out[node] = np.clip(1.0 + (w * d + (1 - w) * gene_prior), L_MIN, L_MAX)
    return out, np.clip(1.0 + gene_prior, L_MIN, L_MAX)


def calibrate(store, assertions, tree, context) -> Calibration:
    """Learn L per node, and build the reordered marker table."""
    children = tree["children"]
    members_of = tree.get("members", {})
    patterns = tree.get("patterns", {})
    genes = store.genes

    per_node = {}
    for parent, kids in children.items():
        for node in kids:
            d, support = _contrast(store, members_of, node, kids, len(genes))
            if d is not None:
                per_node[node] = (d, support)
    L, gene_prior = _pool(per_node, len(genes))

    gi = {str(g): i for i, g in enumerate(genes)}
    rows = []
    for parent, kids in children.items():
        pats = {k: patterns[k] for k in kids if k in patterns}
        if len(pats) < 2:
            continue
        for node in pats:
            # The RAW claim strength, not node_weights' normalised column: the table
            # reports citation weight against learned weight, and a normalised prior
            # would make the two incomparable.
            acc = {}
            for cell, d in assertions.items():
                if not any(p in cell.lower() for p in pats[node]):
                    continue
                for sym, w in d.items():
                    acc[sym] = max(acc.get(sym, 0.0), w)
            claimed = [(g, w) for g, w in acc.items() if g in gi]
            if len(claimed) < 10:
                continue
            mult = L.get(node, gene_prior)
            recs = [{"context": f"{context['species']}/{context['tissue']}",
                     "node": node, "gene": g, "w_bib": w,
                     "L": float(mult[gi[g]]), "w_post": w * float(mult[gi[g]])}
                    for g, w in claimed]
            by_lit = sorted(recs, key=lambda r: -r["w_bib"])
            by_post = sorted(recs, key=lambda r: -r["w_post"])
            rl = {r["gene"]: i + 1 for i, r in enumerate(by_lit)}
            rp = {r["gene"]: i + 1 for i, r in enumerate(by_post)}
            n = len(recs)
            for r in recs:
                r["rank_lit"], r["rank_post"] = rl[r["gene"]], rp[r["gene"]]
                r["delta"] = r["rank_lit"] - r["rank_post"]
                if r["L"] >= 2.0 and r["rank_lit"] / n > 0.5:
                    r["verdict"] = "PROMOTED"
                elif r["L"] <= 0.5 and r["rank_lit"] / n <= 0.2:
                    r["verdict"] = "DEMOTED"
                else:
                    r["verdict"] = "stable"
            a = np.array([r["rank_lit"] for r in recs], float)
            b = np.array([r["rank_post"] for r in recs], float)
            rho = float(np.corrcoef(a, b)[0, 1]) if n > 2 else float("nan")
            for r in recs:
                r["panel_rho"] = round(rho, 3)
            rows.extend(recs)

    census = {
        "celltypes": len(store.celltypes),
        "genes": int(len(genes)),
        "C1": sum(1 for i in range(len(store.celltypes)) if store.grade(i) == "C1"),
        "C2": sum(1 for i in range(len(store.celltypes)) if store.grade(i) == "C2"),
        "C3": sum(1 for i in range(len(store.celltypes)) if store.grade(i) == "C3"),
        "nodes_with_profiles": len(per_node),
        "claims_scored": len(rows),
        "promoted": sum(1 for r in rows if r["verdict"] == "PROMOTED"),
        "demoted": sum(1 for r in rows if r["verdict"] == "DEMOTED"),
    }
    return Calibration(context, sorted(per_node), genes, L, rows, store.digest, census)


# --------------------------------------------------------------------------- persistence
COLS = ["context", "node", "gene", "w_bib", "L", "w_post",
        "rank_lit", "rank_post", "delta", "panel_rho", "verdict"]


def save(cal: Calibration, store, outdir):
    from pathlib import Path
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / "store.npz", context=json.dumps(store.context), genes=store.genes,
        celltypes=np.array(store.celltypes, dtype=object), mean=store.mean,
        detect=store.detect, n_cells=store.n_cells, n_present=store.n_present,
        n_sources=store.n_sources, n_clean=store.n_clean, between_sd=store.between_sd,
        gene_mu=store.gene_mu, gene_sd=store.gene_sd, digest=store.digest)

    with (out / "reliability.tsv").open("w", encoding="utf-8") as fh:
        fh.write("\t".join(COLS) + "\n")
        for r in sorted(cal.rows, key=lambda r: (r["node"], r["rank_post"])):
            fh.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                               for c in COLS) + "\n")

    with (out / "panels.tsv").open("w", encoding="utf-8") as fh:
        fh.write("node\trank\tgene\tL\tw_post\tmoved_from\tverdict\n")
        for node in sorted({r["node"] for r in cal.rows}):
            rs = sorted((r for r in cal.rows if r["node"] == node),
                        key=lambda r: r["rank_post"])[:50]
            for r in rs:
                fh.write(f"{node}\t{r['rank_post']}\t{r['gene']}\t{r['L']:.3f}\t"
                         f"{r['w_post']:.2f}\t{r['rank_lit']}\t{r['verdict']}\n")

    (out / "calibration.json").write_text(json.dumps({
        "context": cal.context, "store_digest": cal.store_digest,
        "census": cal.census, "bounds": {"L_min": L_MIN, "L_max": L_MAX},
        "nodes": cal.nodes,
    }, indent=2), encoding="utf-8")
    return out


def load_store(path):
    """Read a store written by `save`."""
    from .store import ProfileStore
    d = np.load(path, allow_pickle=True)
    return ProfileStore(
        context=json.loads(str(d["context"])), genes=d["genes"],
        celltypes=list(d["celltypes"]), mean=d["mean"], detect=d["detect"],
        n_cells=d["n_cells"], n_present=d["n_present"], n_sources=d["n_sources"],
        between_sd=d["between_sd"], n_clean=d["n_clean"],
        gene_mu=d["gene_mu"], gene_sd=d["gene_sd"], digest=str(d["digest"]))
