"""The annotation report: what was called, how reliably, and what it cannot show.

One self-contained HTML file and one `report.json` carrying every number in it. No external
requests, no CDN, no fonts to fetch - openable from a filesystem in five years, which is longer
than any link survives.

WHAT THIS IS FOR, WHICH IS NOT A GALLERY

An annotation is a set of claims about what cells are. The document that delivers it has to carry
the evidence behind each claim and the limits on all of them, because a label on a UMAP is
persuasive whether or not it is right. So every section states what it CANNOT establish, in the
same place as its numbers, and the report audits itself: a section that should have carried a
limit and did not is counted as a defect on the front page rather than reading as though there
were no limits.

The reliability section is the one that earns the document. A label without its decision gap,
its curated support and its panel survival is a claim with the evidence removed - and depth is
where that bites, because a deep call looks exactly as confident as a shallow one while resting
on a fraction of the assertions.

WHAT IT ASSEMBLES RATHER THAN RECOMPUTES

Everything comes from the run that produced the annotation. A report that derives its own
numbers can disagree with the run it describes, and nothing on the page would say which was
right.

    from scanno.report import collect, build
    doc = collect(adata, res, cats, y, decision=..., ...)
    html, payload = build(doc)
"""
from __future__ import annotations

import base64
import html as _html
import io
import json
from datetime import datetime, timezone

SCHEMA = "scanno/report@1"

#: Matplotlib is optional. A run without it still writes the document, with each figure replaced
#: by a NAMED ABSENCE saying what would produce it - never a blank space, which reads as "there
#: was nothing to show".
try:                                                                      # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:                                                         # noqa: BLE001
    HAVE_MPL = False

EXCLUDED = "EXCLUDED"
UNRESOLVED = "UNRESOLVED"

# A colour-blind-safe qualitative set. Sentinels are grey on purpose: EXCLUDED and UNRESOLVED are
# not cell types and should not look like one more population in the legend.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
           "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
SENTINEL_COLOUR = {EXCLUDED: "#3f3f3f", UNRESOLVED: "#b0b0b0"}


def _fig_to_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _colour_for(labels):
    out, i = {}, 0
    for lab in labels:
        if lab in SENTINEL_COLOUR:
            out[lab] = SENTINEL_COLOUR[lab]
        else:
            out[lab] = PALETTE[i % len(PALETTE)]
            i += 1
    return out


# ------------------------------------------------------------------------------- collection

def collect(adata, res, cats, y, *, label_key, decision=None, support=None, store_digest="",
            tree_path="", db_path="", species="", tissue="", cluster_key="", sample_key=None,
            condition_key=None, gap_min=None, weights="", background="", stats=None,
            version="") -> dict:
    """Everything the document needs, read from the run that produced it."""
    import numpy as np

    obs = adata.obs
    lab = np.asarray(obs[label_key].astype(str))
    n = int(adata.n_obs)

    def level(path_series, depth):
        return np.array(["/".join(str(p).split("/")[:depth]) if str(p) not in
                         (EXCLUDED, UNRESOLVED) else str(p) for p in path_series], dtype=object)

    path_key = label_key.replace("_cell_type", "_path")
    paths = np.asarray(obs[path_key].astype(str)) if path_key in obs else lab

    def counts_of(arr):
        vals, cnt = np.unique(arr, return_counts=True)
        order = np.argsort(-cnt)
        return [{"label": str(vals[i]), "n": int(cnt[i]), "pct": round(100 * cnt[i] / n, 2)}
                for i in order]

    l1 = counts_of(level(paths, 1))
    l2 = counts_of(level(paths, 2))

    # reliability by tree depth - the section that carries the evidence behind each call
    depth_key = label_key.replace("_cell_type", "_depth")
    gap_key = label_key.replace("_cell_type", "_gap")
    sup_key = label_key.replace("_cell_type", "_support")
    reliability = []
    if depth_key in obs:
        d = np.asarray(obs[depth_key], dtype=float)
        g = np.asarray(obs[gap_key], dtype=float) if gap_key in obs else np.full(n, np.nan)
        s = np.asarray(obs[sup_key], dtype=float) if sup_key in obs else np.full(n, np.nan)
        for dv in sorted(set(int(x) for x in d[~np.isnan(d)])):
            m = (d == dv) & (lab != EXCLUDED)
            if not m.any():
                continue
            thin = np.nan
            if not np.all(np.isnan(s[m])):
                thin = round(100 * float(np.nanmean(s[m] < 10)), 1)
            reliability.append({
                "depth": int(dv), "n_calls": int(len({c for c in y[m]})),
                "n_cells": int(m.sum()), "pct_cells": round(100 * m.sum() / n, 2),
                "median_gap": (None if np.all(np.isnan(g[m]))
                               else round(float(np.nanmedian(g[m])), 3)),
                "pct_under_10_assertions": (None if thin != thin else thin),
            })

    # per-sample composition, when the object says which sample a cell came from
    per_sample = []
    if sample_key and sample_key in obs:
        sam = np.asarray(obs[sample_key].astype(str))
        for s_ in sorted(set(sam)):
            m = sam == s_
            row = {"sample": str(s_), "n": int(m.sum())}
            for entry in l1[:8]:
                row[entry["label"]] = round(100 * float((lab[m] == entry["label"]).mean()), 2)
            per_sample.append(row)

    excl = None
    if decision is not None and getattr(decision, "active", False):
        mask = np.asarray(decision.mask, dtype=bool)
        d = decision.declaration or {}
        excl = {
            "source": decision.source, "column": decision.column,
            "n": int(mask.sum()), "pct": round(100 * float(mask.mean()), 3),
            "declared_by": str(d.get("tool", "")) if d else "",
            "run_key": str(d.get("run_key", "")) if d else "",
            "digest": str(d.get("flag_digest", "")) if d else "",
            "meaning": str(d.get("flag_meaning", "")) if d else "",
            "per_sample": [],
        }
        if sample_key and sample_key in obs:
            sam = np.asarray(obs[sample_key].astype(str))
            for s_ in sorted(set(sam)):
                m = sam == s_
                excl["per_sample"].append({"sample": str(s_), "n": int(mask[m].sum()),
                                           "pct": round(100 * float(mask[m].mean()), 2)})
        if condition_key and condition_key in obs:
            cond = np.asarray(obs[condition_key].astype(str))
            rates = {str(c): round(100 * float(mask[cond == c].mean()), 3)
                     for c in sorted(set(cond))}
            excl["per_condition"] = rates
            vals = [v for v in rates.values() if v > 0]
            excl["condition_ratio"] = (round(max(rates.values()) / min(vals), 2)
                                       if vals and min(vals) > 0 else None)

    return {
        "schema": SCHEMA,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scanno_version": version,
        "run": {
            "cluster_key": cluster_key, "label_key": label_key,
            "species": species, "tissue": tissue,
            "n_cells": n, "n_genes": int(adata.n_vars), "n_clusters": len(cats),
            "gap_min": gap_min, "weights": weights, "background": background,
            "store_digest": store_digest, "tree": str(tree_path), "corpus": str(db_path),
            "sample_key": sample_key or "", "condition_key": condition_key or "",
        },
        "headline": {
            "n_cells": n, "n_clusters": len(cats),
            "n_labels": len([e for e in l1 if e["label"] not in (EXCLUDED, UNRESOLVED)]),
            "pct_placed": round(100 * float((~np.isin(lab, [EXCLUDED, UNRESOLVED])).mean()), 1),
            "pct_unresolved": round(100 * float((lab == UNRESOLVED).mean()), 2),
            "pct_excluded": round(100 * float((lab == EXCLUDED).mean()), 2),
        },
        "composition_l1": l1,
        "composition_l2": l2,
        "per_sample": per_sample,
        "reliability": reliability,
        "exclusion": excl,
        "genes": stats or {},
        "cluster_calls": [
            {"cluster": str(cats[r["cluster"]]), "label": r["label"], "path": r["path"],
             "depth": int(r["depth"]),
             "gap": (None if r["gap"] != r["gap"] else round(float(r["gap"]), 4)),
             "support": (None if not support else support.get(r["label"]))}
            for r in res],
    }


# ---------------------------------------------------------------------------------- figures

def marker_panels(assertions, patterns, nodes, top=5) -> dict:
    """The corpus markers behind each called node, best-cited first.

    Built from the SAME assertions the classifier scored on, so the dotplot shows the evidence
    that produced the label rather than a panel someone curated for the figure. A curated grid
    gives a reader no way to know it was curated.
    """
    out = {}
    for node in nodes:
        pats = [p.lower() for p in (patterns.get(node) or [])]
        if not pats:
            continue
        pool = {}
        for cell_name, genes in (assertions or {}).items():
            if any(p in str(cell_name).lower() for p in pats):
                for g, w in genes.items():
                    pool[g] = max(pool.get(g, 0.0), float(w))
        if pool:
            out[node] = [g for g, _ in sorted(pool.items(), key=lambda kv: -kv[1])[:top]]
    return out


def draw(doc, adata, label_key, X=None, genes=None, markers=None) -> dict:
    """Every figure, as a data URI. A figure that cannot be drawn is a NAMED ABSENCE."""
    out = {}

    def absent(fid, why):
        out[fid] = {"uri": None, "absent": why}

    if not HAVE_MPL:
        for fid, why in (("A1", "matplotlib is not installed"),
                         ("A2", "matplotlib is not installed"),
                         ("A3", "matplotlib is not installed"),
                         ("A4", "matplotlib is not installed"),
                         ("A5", "matplotlib is not installed")):
            absent(fid, why + " - `pip install 'scanno[report]'` draws it")
        return out

    import numpy as np

    obs = adata.obs
    lab = np.asarray(obs[label_key].astype(str))
    order = [e["label"] for e in doc["composition_l1"]]
    colour = _colour_for(order)

    # ---- A1 composition -------------------------------------------------------------------
    rows = doc["per_sample"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    if rows:
        names = [r["sample"] for r in rows]
        bottom = np.zeros(len(rows))
        for entry in doc["composition_l1"]:
            key = entry["label"]
            vals = np.array([r.get(key, 0.0) for r in rows], dtype=float)
            ax.bar(names, vals, bottom=bottom, color=colour[key], label=key, width=0.72)
            bottom += vals
        ax.set_ylabel("% of nuclei")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("Composition per sample, level 1")
    else:
        vals = [e["pct"] for e in doc["composition_l1"]]
        ax.barh(range(len(vals))[::-1], vals,
                color=[colour[e["label"]] for e in doc["composition_l1"]])
        ax.set_yticks(range(len(vals))[::-1])
        ax.set_yticklabels([e["label"] for e in doc["composition_l1"]])
        ax.set_xlabel("% of nuclei")
        ax.set_title("Composition, level 1 (no sample column - the cohort as one)")
    ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    out["A1"] = {"uri": _fig_to_uri(fig), "absent": None}

    # ---- A2 reliability by depth ----------------------------------------------------------
    rel = doc["reliability"]
    if rel:
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        d = [r["depth"] for r in rel]
        gaps = [r["median_gap"] if r["median_gap"] is not None else np.nan for r in rel]
        ax.bar([str(x) for x in d], gaps, color="#4C72B0", width=0.6)
        ax.set_xlabel("tree depth of the call")
        ax.set_ylabel("median decision gap")
        thin = [r["pct_under_10_assertions"] for r in rel]
        ax2 = ax.twinx()
        ax2.plot([str(x) for x in d], [t if t is not None else np.nan for t in thin],
                 "o-", color="#C44E52", lw=1.5, ms=5)
        ax2.set_ylabel("% of cells on <10 curated assertions", color="#C44E52")
        ax2.set_ylim(0, 100)
        ax.set_title("Deeper calls look no less confident and rest on less evidence")
        out["A2"] = {"uri": _fig_to_uri(fig), "absent": None}
    else:
        absent("A2", "no depth column on the object, so reliability cannot be broken down")

    # ---- A3 the labels on the embedding ---------------------------------------------------
    emb = next((k for k in (getattr(adata, "obsm", {}) or {})
                if getattr(adata.obsm[k], "ndim", 0) == 2
                and adata.obsm[k].shape[1] >= 2
                and any(h in k.lower() for h in ("umap", "tsne", "draw_graph", "pca"))), None)
    if emb:
        E = np.asarray(adata.obsm[emb])[:, :2]
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        for key in order:
            m = lab == key
            if not m.any():
                continue
            ax.scatter(E[m, 0], E[m, 1], s=3, linewidths=0, c=colour[key], label=key,
                       alpha=0.85 if key not in SENTINEL_COLOUR else 1.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(emb)
        ax.set_title("Every nucleus, coloured by its label")
        ax.legend(fontsize=6, markerscale=3, frameon=False, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0))
        out["A3"] = {"uri": _fig_to_uri(fig), "absent": None}
    else:
        absent("A3", "no 2-D embedding in obsm. scAnno does not compute one; run UMAP upstream")

    # ---- A4 marker evidence ---------------------------------------------------------------
    #
    # The evidence the label rests on, shown against the data it was applied to. Drawn from the
    # corpus panels the classifier actually scored on - not a hand-picked grid, which would give
    # a reader no way to tell that it had been picked.
    if not markers:
        absent("A4", "no corpus was supplied (--db), so there are no marker panels to show")
    elif X is None or genes is None:
        absent("A4", "the expression matrix was not passed to the report")
    else:
        import scipy.sparse as _sp

        gi = {str(g).upper(): i for i, g in enumerate(genes)}
        panel, panel_of = [], []
        for node, gs in markers.items():
            for g in gs:
                if str(g).upper() in gi and str(g).upper() not in [p for p in panel]:
                    panel.append(str(g).upper())
                    panel_of.append(node)
        shown = [lab_ for lab_ in order if lab_ not in SENTINEL_COLOUR]
        if not panel or not shown:
            absent("A4", "none of the corpus markers for the called nodes are in this object")
        else:
            cols = [gi[g] for g in panel]
            mean = np.zeros((len(shown), len(panel)))
            det = np.zeros((len(shown), len(panel)))
            for r, lab_ in enumerate(shown):
                m = lab == lab_
                sub = X[m][:, cols]
                sub = sub.toarray() if _sp.issparse(sub) else np.asarray(sub)
                mean[r] = sub.mean(axis=0)
                det[r] = (sub > 0).mean(axis=0)
            # Scaled per GENE so a highly expressed marker does not flatten every other column.
            rng_ = mean.max(axis=0) - mean.min(axis=0)
            scaled = (mean - mean.min(axis=0)) / np.where(rng_ == 0, 1, rng_)
            fig, ax = plt.subplots(figsize=(max(6.0, 0.34 * len(panel) + 2.4),
                                            0.42 * len(shown) + 2.0))
            for r in range(len(shown)):
                sc = ax.scatter(range(len(panel)), [r] * len(panel), s=det[r] * 170 + 4,
                                c=scaled[r], cmap="viridis", vmin=0, vmax=1,
                                edgecolors="none")
            ax.set_yticks(range(len(shown)))
            ax.set_yticklabels(shown, fontsize=8)
            ax.set_xticks(range(len(panel)))
            ax.set_xticklabels(panel, rotation=90, fontsize=7)
            ax.set_xlim(-0.7, len(panel) - 0.3)
            ax.set_ylim(-0.7, len(shown) - 0.3)
            ax.set_title("Corpus markers for the called nodes, in this object", fontsize=10)
            cb = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
            cb.set_label("mean expression, scaled per gene", fontsize=7)
            cb.ax.tick_params(labelsize=6)
            out["A4"] = {"uri": _fig_to_uri(fig), "absent": None}

    # ---- A5 what was withheld -------------------------------------------------------------
    ex = doc.get("exclusion")
    if not ex:
        absent("A5", "nothing was withheld in this run")
    elif emb:
        E = np.asarray(adata.obsm[emb])[:, :2]
        m = lab == EXCLUDED
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4),
                                 gridspec_kw={"width_ratios": [1, 1]})
        axes[0].scatter(E[~m, 0], E[~m, 1], s=3, linewidths=0, c="#d9d9d9")
        axes[0].scatter(E[m, 0], E[m, 1], s=5, linewidths=0, c=SENTINEL_COLOUR[EXCLUDED])
        axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[0].set_title(f"where the {ex['n']:,} withheld nuclei sit")
        ps = ex.get("per_sample") or []
        if ps:
            axes[1].bar([r["sample"] for r in ps], [r["pct"] for r in ps], color="#C44E52")
            axes[1].set_ylabel("% of that sample withheld")
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].set_title("and how unevenly, per sample")
        else:
            axes[1].axis("off")
            axes[1].text(0.5, 0.5, "no sample column,\nso evenness cannot be shown",
                         ha="center", va="center", fontsize=9, color="#666")
        out["A5"] = {"uri": _fig_to_uri(fig), "absent": None}
    else:
        absent("A5", "no embedding, so the withheld nuclei cannot be placed")
    return out


# ------------------------------------------------------------------------------------ build

CANNOT = {
    "composition": (
        "Composition is what the clustering and the corpus produced together, not a measured "
        "abundance. Dissociation, nuclei isolation and QC all change what reaches the object, "
        "and none of them is uniform across cell types - a fragile population is under-counted "
        "before anything here runs."),
    "reliability": (
        "These grade the EVIDENCE behind a call, not its truth. A node with a confident-looking "
        "panel that happens to be wrong scores well on all three."),
    "labels": (
        "No figure or number here shows that a label is CORRECT. That needs a truth set - "
        "sorted cells, genetic labels, an independent assay - and there is none in this object. "
        "Agreement with a published composition is reassurance, not validation."),
    "exclusion": (
        "This shows what was withheld and how unevenly, not whether withholding it was right. "
        "That question belongs to whoever produced the flag; scAnno records the decision and "
        "reports its cost."),
    "embedding": (
        "The embedding is the input object's. Distance in it is not a measurement, and a "
        "population that looks contiguous may not be."),
    "markers": (
        "A marker expressing where its label was called is circular: the label was assigned "
        "from those markers. This shows the evidence is present in the data, which is worth "
        "checking, and cannot show the call is right. A broadly-expressed marker carries less "
        "identity information than a specific one and looks the same here."),
}

_CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--accent:#4C72B0;--warn:#C44E52}
@media(prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e8e6e3;--mut:#9aa0a6;
--line:#2c2f34;--accent:#7aa2d6;--warn:#e07b7e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:38px 0 10px;
padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--mut);font-size:13px;margin-bottom:24px}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0 6px}
.tile{border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:120px}
.tile b{display:block;font-size:21px}.tile span{color:var(--mut);font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:16px 0}figure img{width:100%;border:1px solid var(--line);border-radius:8px}
figcaption{color:var(--mut);font-size:12px;margin-top:6px}
.cannot{border-left:3px solid var(--warn);padding:8px 12px;margin:14px 0;
background:rgba(196,78,82,.06);font-size:13px}
.cannot b{color:var(--warn)}
.absent{border:1px dashed var(--line);border-radius:8px;padding:18px;color:var(--mut);
font-size:13px;text-align:center}
code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:rgba(128,128,128,.12);
padding:1px 5px;border-radius:4px}
.scroll{overflow-x:auto}
"""


def _esc(v) -> str:
    return _html.escape("" if v is None else str(v))


def _table(headers, rows, numeric=()) -> str:
    th = "".join(f'<th class="{"n" if h in numeric else ""}">{_esc(h)}</th>' for h in headers)
    body = []
    for r in rows:
        tds = "".join(f'<td class="{"n" if h in numeric else ""}">'
                      f'{"" if r.get(h) is None else _esc(r.get(h))}</td>' for h in headers)
        body.append(f"<tr>{tds}</tr>")
    return (f'<div class="scroll"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _figure(figs, fid, caption) -> str:
    f = figs.get(fid)
    if not f or not f.get("uri"):
        why = (f or {}).get("absent", "not produced")
        return (f'<figure><div class="absent"><b>{fid} is not drawn.</b><br>{_esc(why)}</div>'
                f'<figcaption>{_esc(caption)}</figcaption></figure>')
    return (f'<figure><img alt="{fid}" src="{f["uri"]}">'
            f'<figcaption><b>{fid}.</b> {_esc(caption)}</figcaption></figure>')


def build(doc, figs=None) -> tuple[str, dict]:
    """The document, and the payload carrying every number in it."""
    figs = figs or {}
    run, head = doc["run"], doc["headline"]
    defects = []

    # The report audits itself. A section that should carry a limit and does not would otherwise
    # read as a section with no limits, which is the more confident claim.
    for name in ("composition", "reliability", "labels"):
        if not CANNOT.get(name):
            defects.append(f"section {name} states no limit")
    for fid, f in figs.items():
        if not f.get("uri") and not f.get("absent"):
            defects.append(f"figure {fid} is neither drawn nor explained")

    tiles = [
        ("nuclei", f"{head['n_cells']:,}"), ("clusters", f"{head['n_clusters']:,}"),
        ("level-1 labels", f"{head['n_labels']}"), ("placed", f"{head['pct_placed']}%"),
        ("UNRESOLVED", f"{head['pct_unresolved']}%"), ("EXCLUDED", f"{head['pct_excluded']}%"),
    ]
    tile_html = "".join(f'<div class="tile"><b>{_esc(v)}</b><span>{_esc(k)}</span></div>'
                        for k, v in tiles)

    parts = [f"""<div class="wrap">
<h1>Annotation report</h1>
<div class="sub">{_esc(run['species'])} / {_esc(run['tissue'])} &middot;
clustered on <code>{_esc(run['cluster_key'])}</code> &middot;
labels in <code>{_esc(run['label_key'])}</code> &middot;
weights from {_esc(run['weights'])} &middot; generated {_esc(doc['generated'])}</div>
<div class="tiles">{tile_html}</div>"""]

    if defects:
        parts.append('<div class="cannot"><b>Defects in this report:</b> '
                     + _esc("; ".join(defects)) + "</div>")

    # --- composition
    parts.append("<h2>Composition</h2>")
    parts.append(_figure(figs, "A1", "Level-1 composition. Sentinel labels are grey: EXCLUDED "
                                     "and UNRESOLVED are not cell types."))
    parts.append(_table(["label", "n", "pct"], doc["composition_l1"], numeric=("n", "pct")))
    if doc["composition_l2"]:
        parts.append("<h2>Composition &mdash; level 2</h2>")
        parts.append(_table(["label", "n", "pct"], doc["composition_l2"], numeric=("n", "pct")))
    parts.append(f'<div class="cannot"><b>What this cannot show.</b> {_esc(CANNOT["composition"])}'
                 f"</div>")

    # --- the picture
    parts.append("<h2>The labels on the embedding</h2>")
    parts.append(_figure(figs, "A3", "Every nucleus, coloured by its label."))
    parts.append(f'<div class="cannot"><b>What this cannot show.</b> {_esc(CANNOT["embedding"])}'
                 f"</div>")

    # --- reliability
    parts.append("<h2>Reliability</h2>")
    parts.append("<p>A label without its decision gap and its curated support is a claim with "
                 "the evidence removed. Depth is where that bites.</p>")
    parts.append(_figure(figs, "A2", "Median decision gap by depth, with the share of cells "
                                     "whose winning node rests on fewer than 10 curated "
                                     "assertions."))
    parts.append(_table(["depth", "n_calls", "n_cells", "pct_cells", "median_gap",
                         "pct_under_10_assertions"], doc["reliability"],
                        numeric=("depth", "n_calls", "n_cells", "pct_cells", "median_gap",
                                 "pct_under_10_assertions")))
    parts.append(f'<div class="cannot"><b>What this cannot show.</b> {_esc(CANNOT["reliability"])}'
                 f"</div>")

    # --- marker evidence
    parts.append("<h2>The markers behind the calls</h2>")
    parts.append("<p>The corpus panels the classifier scored on, shown against the object they "
                 "were applied to. Dot size is the share of nuclei detecting the gene; colour "
                 "is mean expression, scaled per gene so one loud marker does not flatten the "
                 "rest.</p>")
    parts.append(_figure(figs, "A4", "Corpus markers for each called node."))
    parts.append(f'<div class="cannot"><b>What this cannot show.</b> {_esc(CANNOT["markers"])}'
                 f"</div>")

    # --- exclusion
    ex = doc.get("exclusion")
    parts.append("<h2>What was withheld</h2>")
    if not ex:
        parts.append("<p>Nothing was withheld in this run: every nucleus was annotated.</p>")
    else:
        parts.append(
            f"<p><b>{ex['n']:,} nuclei ({ex['pct']}%)</b> were withheld and labelled "
            f"<code>EXCLUDED</code>, from <code>{_esc(ex['column'])}</code>"
            + (f", declared by {_esc(ex['declared_by'])}"
               f" (run <code>{_esc(ex['run_key'])}</code>, digest "
               f"<code>{_esc(ex['digest'])}</code>)" if ex.get("declared_by") else "")
            + ". Nothing was deleted: each keeps its place in the object.</p>")
        if ex.get("meaning"):
            parts.append(f"<p><i>{_esc(ex['meaning'])}</i></p>")
        parts.append(_figure(figs, "A5", "Where the withheld nuclei sit, and how unevenly they "
                                         "fall across samples."))
        if ex.get("per_sample"):
            parts.append(_table(["sample", "n", "pct"], ex["per_sample"], numeric=("n", "pct")))
        if ex.get("per_condition"):
            rows = [{"condition": k, "pct withheld": v} for k, v in ex["per_condition"].items()]
            parts.append(_table(["condition", "pct withheld"], rows, numeric=("pct withheld",)))
            if ex.get("condition_ratio"):
                parts.append(
                    f"<p>Widest ratio between conditions: <b>{ex['condition_ratio']}&times;</b>. "
                    f"An exclusion that falls harder on one arm of a design has moved a "
                    f"technical decision into the comparison.</p>")
    parts.append(f'<div class="cannot"><b>What this cannot show.</b> {_esc(CANNOT["exclusion"])}'
                 f"</div>")

    # --- per cluster
    parts.append("<h2>Every cluster call</h2>")
    parts.append(_table(["cluster", "path", "depth", "gap", "support"], doc["cluster_calls"],
                        numeric=("depth", "gap", "support")))

    # --- provenance
    parts.append("<h2>Provenance</h2>")
    prov = [{"field": k, "value": v} for k, v in [
        ("scanno version", doc.get("scanno_version") or "unrecorded"),
        ("cells x genes", f"{run['n_cells']:,} x {run['n_genes']:,}"),
        ("clusters", run["n_clusters"]), ("cluster key", run["cluster_key"]),
        ("label key", run["label_key"]), ("weights", run["weights"]),
        ("gene background", run["background"]), ("store digest", run["store_digest"] or "-"),
        ("taxonomy", run["tree"]), ("corpus", run["corpus"]),
        ("gap_min", run["gap_min"]), ("sample column", run["sample_key"] or "none"),
        ("condition column", run["condition_key"] or "none"),
    ]]
    parts.append(_table(["field", "value"], prov))

    parts.append("<h2>What this report cannot show</h2>")
    parts.append(f'<div class="cannot"><b>Labels are not validated here.</b> '
                 f'{_esc(CANNOT["labels"])}</div>')
    parts.append("</div>")

    doc = dict(doc)
    doc["defects"] = defects
    doc["figures"] = sorted(figs)
    html = (f"<!doctype html><meta charset=utf-8><title>scAnno &mdash; annotation report</title>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<style>{_CSS}</style>{''.join(parts)}")
    return html, doc


def write(path, doc, figs) -> tuple[str, str]:
    """Write `path` (HTML) and `path`.with_suffix('.json'). Returns both paths."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    html, payload = build(doc, figs)
    p.write_text(html, encoding="utf-8")
    j = p.with_suffix(".json")
    j.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return str(p), str(j)
