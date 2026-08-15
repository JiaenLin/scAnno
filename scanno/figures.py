"""Every figure the reports carry, as a pure function of the `Context`.

WHAT THIS IS A PORT OF

These figures were developed against a real study, reviewed by its PI, and rebuilt several times
on that review. The geometry is not decoration - a grid of every swept resolution with the call
count in each panel title answers "what did granularity cost me" at a glance, and the same
information as ten separate files answers nothing. An earlier attempt here was written from the
figure LEGENDS rather than from the source and produced pictures describable in the same words
and useless to look at. Legends say what a figure means; only the source says what it shows.

FOUR RULES THE ORIGINALS ESTABLISHED, EACH THE RESIDUE OF A FAILURE

1. **The points are the figure.** Any bar summarising groups of samples carries one point per
   sample over it. A group mean drawn from two samples is a statement about which samples landed
   in which arm, and a bar alone lets a reader forget that.
2. **Highlight against grey, in the embedding the subset was defined in.** Never re-embedded
   without it, never on its own axes - there "a compact island" and "dispersed everywhere" look
   identical, and telling those apart is the entire question.
3. **A figure that cannot be drawn says which input was missing.** `NotDrawable` is printed on
   the page as a named absence. A silently omitted panel reads as an absent finding.
4. **Nothing is curated out of a marker panel.** A gene lit across every column supports no call,
   and showing that is the point. Breadth is measured and reported, never acted on.

DEPTH IS NOT ASSUMED ANYWHERE

Every "level 2" figure in the original is emitted here once per level the taxonomy actually has.
`F103`, `F135`, `F136`, `F143` take a `depth`; `F101` and `F142` put one panel per level on one
page. Nothing tests `"/" in label` to detect a level - that silently matches level 3 as well -
and nothing hardcodes 1, 2 or 3.
"""
from __future__ import annotations

import numpy as np

from .context import MIN_FLAGGED_PER_ANIMAL, SENTINELS
from .primitives import (ACCENT, AMBER, COOL, GREY, INK, MUT, NotDrawable, continuous_scatter,
                         dotplot, dotplot_key, leaf, line_with_band, paired_barh, panel_grid,
                         points_over_bars, plt, quartile_boxes, ratio_bars, save,
                         scale_per_column, scatter_sized, signed_barh, stacked_rows,
                         swatch_legend, umap_scatter, unique_ticks)

BOX_TINTS = ["#e3d7f2", "#cfe3f2", "#d9ecd9", "#f7e3cf", "#e9e9e9"]

#: Rule one's refusal line: a removal whose rate differs more than this between arms of a design
#: factor converts a technical property into an apparent biological difference.
DIFFERENTIAL_LIMIT = 3.0

#: Group colours are keyed by index into the DECLARED group order, never into the groups that
#: happen to be present - otherwise an arm missing from a subset shifts every later arm's colour
#: and two runs are not comparable by colour at all.
def _group_colours(n):
    p = plt()
    return [p.cm.tab10(i % 10) for i in range(max(n, 1))]


def _panels_for_depth(ctx, cap=4):
    """Which levels get a panel when the tree is deeper than fits on a page.

    Beyond `cap` levels the default is the two shallowest and the two deepest, and the omission
    is NAMED in the caption rather than left as a page that quietly shows fewer panels than the
    taxonomy has.
    """
    if ctx.depth <= cap:
        return ctx.levels, []
    keep = [1, 2, ctx.depth - 1, ctx.depth]
    return keep, [d for d in ctx.levels if d not in keep]


# ======================================================================= per-sample figures

def F100(ctx, sample):
    """Every swept resolution on one page, coloured by the label each one produced.

    One grid rather than N files, because the question is what a finer clustering COST and that
    is a comparison. Colours are fixed across every panel and every file: a per-panel palette
    makes the same population look like a different one between resolutions, which is exactly
    the comparison this figure exists to support. Each legend carries the label's share, so a
    population disappearing at a finer resolution is a legend entry that stops appearing rather
    than something to infer from a table.
    """
    xy = ctx.embedding(sample)
    if xy is None:
        raise NotDrawable(f"{sample}: no UMAP in obsm")
    sweep = ctx.sweep_keys(sample)
    if not sweep:
        raise NotDrawable(f"{sample}: no per-resolution label columns ({ctx.path_key}_r*)")
    A = ctx._matrix(sample)
    fig, axes = panel_grid(len(sweep), ncols=min(4, len(sweep)), panel_size=(4.5, 4.5))
    order, colours = ctx.label_order(1), ctx.colours(1)
    for ax, (rv, col, _tag) in zip(axes, sweep):
        lab = ctx._trunc(A.obs[col].astype(str), 1)
        n_calls = len(set(A.obs[col].astype(str)))
        umap_scatter(ax, xy=xy, labels=lab, colours=colours, order=order,
                     denominator=len(lab), legend=True,
                     title=f"res {rv:g}   {n_calls} calls")
    return fig, (f"{sample} — the level-1 annotation across resolutions   "
                 f"(n = {len(xy):,} nuclei, one embedding, fixed colours)")


def F101(ctx, sample):
    """The same embedding at every level of the taxonomy, side by side.

    NOT one panel coloured by how deep the walk got - that is F133's question. This shows the
    partition getting finer with subtypes keeping their parent's hue, so a reader sees which
    level-1 territory each subtype occupies and what the extra level actually bought.
    """
    xy = ctx.embedding(sample)
    if xy is None:
        raise NotDrawable(f"{sample}: no UMAP in obsm")
    D = ctx.sample_depth(sample)
    levels = list(range(1, max(D, 1) + 1))
    if len(levels) > 4:
        levels = [1, 2, D - 1, D]
    fig, axes = panel_grid(len(levels), ncols=len(levels), panel_size=(6.4, 6.2))
    for ax, d in zip(axes, levels):
        lab = ctx.labels(sample, d)
        order = [l for l in ctx.label_order(d) if (lab == l).any()]
        umap_scatter(ax, xy=xy, labels=lab, colours=ctx.colours(d), order=order,
                     denominator=len(lab), legend=True,
                     legend_fontsize=6.5 if len(order) < 20 else 5.5,
                     title=f"level {d}" + ("  (full path)" if d == D else "")
                           + f"   {len(order)} labels")
    return fig, (f"{sample} — what each extra level buys   (n = {len(xy):,} nuclei; "
                 f"subtypes keep their parent's hue)")


def F104(ctx, sample):
    """The clusters the annotation was made from, beside the labels it produced.

    A cluster-level annotator is only as good as its partition, and the two pictures together
    are what show a label spread over three clusters or one cluster split between two labels.
    Neither picture alone shows it.
    """
    xy = ctx.embedding(sample)
    if xy is None:
        raise NotDrawable(f"{sample}: no UMAP in obsm")
    clu = ctx.clusters(sample)
    if clu is None:
        raise NotDrawable(f"{sample}: no cluster column in obs")
    p = plt()
    fig, axes = panel_grid(2, ncols=2, panel_size=(6.6, 6.2))
    uniq = sorted(set(clu), key=lambda c: (len(str(c)), str(c)))
    for i, c in enumerate(uniq):
        m = clu == c
        axes[0].scatter(xy[m, 0], xy[m, 1], s=1.2, color=[p.cm.tab20(i % 20)], linewidths=0,
                        rasterized=True)
        axes[0].text(float(np.median(xy[m, 0])), float(np.median(xy[m, 1])), str(c),
                     fontsize=7, ha="center", va="center", color=INK)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title(f"{len(uniq)} clusters at resolution {ctx.chosen_resolution or '?'}",
                      fontsize=11, loc="left")
    lab = ctx.labels(sample, 1)
    umap_scatter(axes[1], xy=xy, labels=lab, colours=ctx.colours(1),
                 order=[l for l in ctx.label_order(1) if (lab == l).any()],
                 denominator=len(lab), legend=True, title="the label each cluster received")
    return fig, f"{sample} — the partition, and what it was called"


def F105(ctx, sample):
    """This sample's QC measurements on its own embedding.

    QC as a histogram says how many; QC on the embedding says WHERE, and only the second
    distinguishes a uniformly shallow library from one population that is shallow.
    """
    xy = ctx.embedding(sample)
    if xy is None:
        raise NotDrawable(f"{sample}: no UMAP in obsm")
    rows = ctx.sample_rows(sample)
    keep, seen = [], set()
    for c, n, lg in (("total_counts", "UMI", True), ("n_genes", "genes", True),
                     ("n_genes_by_counts", "genes", True),
                     ("pct_counts_mt", "% mitochondrial", False),
                     ("doublet_score", "doublet score", False)):
        if c in rows and rows[c].notna().any() and n not in seen:
            seen.add(n)
            keep.append((c, n, lg))
    if not keep:
        raise NotDrawable(f"{sample}: no QC columns in obs")
    fig, axes = panel_grid(len(keep), ncols=len(keep), panel_size=(4.6, 4.5))
    for ax, (c, n, lg) in zip(axes, keep):
        continuous_scatter(ax, xy=xy, values=np.asarray(rows[c], dtype=float), log=lg,
                           size=1.2, title=f"{n}{' (log10)' if lg else ''}", fig=fig,
                           sort_order=False)
    return fig, (f"{sample} — QC on the embedding: WHERE the weak nuclei are, not how many")


# ======================================================================= composition

def _composition(ctx, depth, by):
    rows = ctx.composition_rows(depth, by=by)
    if not rows:
        raise NotDrawable(f"no '{by}' column to group composition by")
    order = ctx.label_order(depth)
    n_cols = len(order)
    # Every per-depth constant is a function of the COLUMN COUNT, not of the level: a 3% segment
    # is readable among nine columns and unreadable among twenty-five.
    floor = 3.0 if n_cols <= 10 else min(8.0, 3.0 + 0.3 * (n_cols - 10))
    width = min(22.0, 12.0 + 0.35 * max(0, n_cols - 9))
    fig, ax = plt().subplots(figsize=(width, 0.62 * len(rows) + 3.0))
    stacked_rows(ax, rows=rows, order=order, colours=ctx.colours(depth), label_floor=floor)
    swatch_legend(fig, order, ctx.colours(depth), ncol=8 if n_cols <= 10 else 6,
                  fontsize=8.5 if n_cols <= 10 else 7.5)
    ax.set_title(
        (f"Level {depth} composition, per {by}. Bars sum to 100%; percentages are within a row, "
         f"and segments below {floor:g}% are unlabelled.\n"
         + ("Computed per sample and averaged, never pooled before division — pooling lets the "
            "largest library set the group's composition." if by == "group"
            else "One row per sample: the detail the group figure averages over. Where two "
                 "samples of one group differ by more than the groups differ, the group "
                 "difference is not a group property.")),
        fontsize=9.5, loc="left")
    return fig, None


def F102(ctx, by="group"):
    """Level-1 composition, 100% stacked. The figure composition is read in."""
    return _composition(ctx, 1, by)


def F103(ctx, depth=2, by="group"):
    """The same at any deeper level.

    This is where a compositional change actually appears: level-1 shares can sit still while
    everything underneath rearranges, and level 1 has no way to show it. It is equally the level
    at which the calls are weakest — read it against the reliability table.
    """
    if depth > ctx.depth:
        raise NotDrawable(f"the taxonomy has {ctx.depth} level(s); no level {depth}")
    return _composition(ctx, depth, by)


def F141(ctx, depth=1, min_share=0.5):
    """Group means as bars with ONE POINT PER SAMPLE over them.

    Rule 1, and the figure the whole composition section rests on. The threshold is on the mean
    of the per-sample percentages — which is what it says, and is not the same statistic as a
    share of the pooled cohort whenever samples differ in depth.
    """
    if depth > ctx.depth:
        raise NotDrawable(f"the taxonomy has {ctx.depth} level(s); no level {depth}")
    pts, groups = ctx.per_animal_points(depth)
    if not pts:
        raise NotDrawable("no group column: per-sample points need a design to sit in")
    labels = [l for l in ctx.label_order(depth)
              if np.mean([np.mean(pts.get((g, l), [0.0])) for g in groups]) >= min_share]
    if not labels:
        raise NotDrawable(f"no level-{depth} label averages {min_share}% of a sample")
    per_label = max(0.6, 1.5 - 0.28 * (depth - 1))
    fig, ax = plt().subplots(figsize=(max(per_label * len(labels) + 3.5, 7.0), 4.9))
    points_over_bars(ax, categories=labels, groups=groups, point_values=pts,
                     group_colours=_group_colours(len(groups)),
                     ylabel="% of the sample's nuclei", ticks=unique_ticks(labels),
                     tick_rotation=35, tick_fontsize=9 if depth == 1 else 8.5,
                     point_size=22 if depth == 1 else 20)
    ax.set_title(f"Level {depth}. Bars are group means; each point is one sample "
                 f"(threshold: mean per-sample share ≥ {min_share:g}%).\n"
                 f"A bar difference smaller than the spread of points within a group is not a "
                 f"group difference.", fontsize=9.5, loc="left")
    return fig, None


def F143(ctx, depth=2, min_share=0.5):
    """F141 at a deeper level. The points matter MORE here: a subtype share rests on fewer
    nuclei, so the between-sample spread within a group is wider by construction."""
    return F141(ctx, depth=depth, min_share=min_share)


# ======================================================================= reliability

def F140(ctx):
    """The three reliability signals, distributed by the depth of the call.

    All three are shown because THEY DISAGREE. The decision gap does not fall much with depth
    while curated support collapses, so a deep call can look as confident as a shallow one while
    resting on a handful of assertions. A missing metric drops its PANEL and the grid is rebuilt
    — never left as a blank axis inside a fixed row.
    """
    c = ctx.calls_at_chosen()
    if c is None or not len(c):
        raise NotDrawable("no per-call statistics in obs "
                          f"({ctx.label_key.replace('_cell_type', '_gap')} and siblings)")
    have = [(k, n, lo) for k, n, lo in
            (("gap", "decision gap", None),
             ("support", "curated assertions behind the call", None),
             ("survival", "panel survival", 0.0))
            if k in c and c[k].notna().any()]
    if not have:
        raise NotDrawable("gap, support and survival are all absent from obs")
    depths = sorted(set(c["depth"]))
    fig, axes = panel_grid(len(have), ncols=len(have), panel_size=(4.0, 3.9))
    for ax, (k, name, lo) in zip(axes, have):
        quartile_boxes(ax, data=[c.loc[c["depth"] == d, k].dropna().values for d in depths],
                       tick_labels=[str(int(d)) for d in depths],
                       showfliers=True, ylim_bottom=lo)
        ax.set_xlabel("tree depth of the call", fontsize=9)
        ax.set_title(name, fontsize=10, loc="left")
    return fig, "The three reliability signals disagree, which is why all three are shown"


def F133(ctx):
    """Panel survival against decision gap, one point per call.

    Top-LEFT is the case to read: a confident call won on evidence that better-cited neighbours
    had already taken. Bottom-right is safe. Point area is `sqrt(n)*1.6`, so the plotted radius
    grows as the fourth root of the cluster size and a huge cluster does not swallow the panel.
    """
    c = ctx.calls_at_chosen()
    if c is None or "survival" not in c or c["survival"].isna().all():
        raise NotDrawable("no `survival` column — the annotation predates panel survival")
    c = c[~c["survival"].isna()]
    fig, ax = plt().subplots(figsize=(6.6, 4.8))
    scatter_sized(ax, x=c["survival"], y=c["gap"], sizes=np.sqrt(c["n_cells"]) * 1.6,
                  colour_values=c["depth"], vline=0.7, fig=fig, colorbar_label="tree depth")
    ax.set_xlabel("panel survival — share of the node's evidence left after the sibling contrast")
    ax.set_ylabel("decision gap")
    ax.set_title("A confident call on a stripped panel is the case to read\n"
                 "bottom-right is safe; top-LEFT won on evidence already taken",
                 fontsize=9, loc="left")
    return fig, None


def F142(ctx, floor=50):
    """Per-nucleus neighbourhood agreement by label, ONE PANEL PER LEVEL.

    A property of this object's graph, not a quality score: a cell type that genuinely sits
    between two others scores low without being wrong. Read the LOW boxes — and read whether
    splitting deeper costs agreement, which says whether the partition is coarser than the
    manifold or finer.

    The dashed line is the pooled cohort median; each panel title carries the median of that
    panel's per-label medians. They are different numbers by construction and both are labelled.
    """
    nn = ctx.neighbourhood()
    if nn is None:
        raise NotDrawable("no `nn_agreement` column in obs")
    levels, omitted = _panels_for_depth(ctx)
    per = [(d, *ctx.neighbourhood_by_label(d, floor)) for d in levels]
    per = [(d, k, v) for d, k, v in per if k]
    if not per:
        raise NotDrawable(f"no label reaches the {floor}-nucleus floor")
    med = 100 * float(np.nanmedian(nn))
    widths = [max(len(k), 1) for _, k, _ in per]
    p = plt()
    fig, axs = p.subplots(1, len(per), sharey=True, squeeze=False,
                          figsize=(0.42 * sum(widths) + 3.5, 5.0),
                          gridspec_kw={"width_ratios": widths})
    for i, (ax, (d, keys, data)) in enumerate(zip(axs.ravel(), per)):
        quartile_boxes(ax, data=data, tick_labels=unique_ticks(keys), showfliers=False,
                       facecolors=BOX_TINTS[i % len(BOX_TINTS)], tick_rotation=35,
                       tick_fontsize=8)
        ax.axhline(med, ls="--", lw=1, c=AMBER)
        ax.set_title(f"level {d}   {len(keys)} labels   median of per-label medians "
                     f"{np.median([np.median(x) for x in data if len(x)]):.0f}%",
                     fontsize=9.5, loc="left")
    axs.ravel()[0].set_ylabel("% of neighbourhood carrying the same label")
    note = (f"   Levels {', '.join(map(str, omitted))} are omitted for space."
            if omitted else "")
    return fig, (f"Dashed line is the pooled cohort median ({med:.0f}%). A property of this "
                 f"graph, not a score.\nRead the LOW boxes: those labels sit among neighbours "
                 f"carrying a different one.{note}")


def F160(ctx):
    """The evidence behind the chosen resolution.

    The shaded band is the tolerance DERIVED from the sweep's own step-to-step variation rather
    than chosen; candidates inside it are ones the evidence cannot separate. It is drawn only on
    the percentage panels — a percentage-point tolerance on a nucleus count is meaningless.
    Read the last panel: stability is usually flat, and the smallest named population is what
    a finer clustering destroys first.
    """
    sweep = ctx.resolution_sweep()
    if not sweep:
        raise NotDrawable("no resolution sweep — run `scanno resolution` first")
    lo_d, hi_d = min(sweep), max(sweep)
    spec = [(f"modal agreement, depth {lo_d}", sweep[lo_d], "modal", "%"),
            (f"modal agreement, depth {hi_d}", sweep[hi_d], "modal", "%"),
            ("complete (resolved to a leaf)", sweep[hi_d], "complete", "%"),
            ("smallest named population", sweep[hi_d], "smallest", "nuclei")]
    if lo_d == hi_d:
        spec.pop(1)
    spec = [s for s in spec if any(r.get(s[2]) is not None for r in s[1])]
    if not spec:
        raise NotDrawable("the sweep carries none of modal, complete or smallest")
    pick = ctx.sweep_pick if ctx.sweep_pick is not None else ctx.chosen_resolution
    pick_x = float(str(pick).replace("p", ".")) if pick is not None else None
    fig, axes = panel_grid(len(spec), ncols=len(spec), panel_size=(4.2, 4.3))
    for i, (ax, (title, rows, key, unit)) in enumerate(zip(axes, spec)):
        x = [float(str(r["resolution"]).replace("p", ".")) for r in rows]
        y = [np.nan if r.get(key) is None else r[key] for r in rows]
        band = None
        if unit == "%" and ctx.tolerance is not None and np.isfinite(np.nanmax(y)):
            top = float(np.nanmax(y))
            band = (top - float(ctx.tolerance), top)
        line_with_band(ax, x=x, y=y, band=band, mark_x=pick_x, title=title,
                       xlabel="clustering resolution", ylabel=unit,
                       annotate=(f"  chosen {pick_x:g}" if i == 0 and pick_x is not None
                                 else None))
    head = (f"Chosen resolution {pick_x:g}"
            + (f", decided by {ctx.sweep_reason}" if ctx.sweep_reason else "")
            + ". Shaded band = the tolerance derived from the sweep's own variation."
            if pick_x is not None else
            f"The resolution sweep — {len(spec[0][1])} candidates, {len(ctx.samples)} samples")
    return fig, head


# ======================================================================= the joint route

def _require_joint_embedding(ctx, what):
    """The joint embedding, or a refusal naming why it cannot carry `what`.

    A STITCHED embedding is refused rather than drawn. It is the per-sample embeddings moved
    apart, so every sample sits in its own territory because it was put there - and a figure
    asking whether the structure follows libraries then answers yes by construction. Drawing it
    anyway would publish a tautology in the shape of a finding, which is worse than drawing
    nothing, because nothing is obviously nothing.
    """
    xy = ctx.joint_embedding()
    if xy is None:
        raise NotDrawable(f"no joint object - {what} needs ONE embedding computed over all "
                          f"samples together. Build one with `scanno embed`.")
    st = ctx.embedding_is_stitched()
    if st and st["stitched"]:
        raise NotDrawable(
            f"the joint object's embedding is STITCHED, not joint: each sample's coordinates "
            f"are its own per-sample embedding shifted by a constant "
            f"({st['samples_rigid']}/{len(st['samples_tested'])} samples match exactly). Every "
            f"sample occupies its own territory because it was placed there, so {what} would "
            f"show that separation as a finding when it is an artefact of the assembly. Compute "
            f"a real joint embedding with `scanno embed` and pass it as --joint.")
    return xy



def F132(ctx, depth=1):
    """The same embedding coloured by label and by library, side by side.

    No identity decision is presented on metrics alone. IF THE RIGHT PANEL LOOKS LIKE THE LEFT,
    the structure is libraries, not cell types — and on an un-integrated cohort that is a live
    possibility rather than a formality.
    """
    xy = _require_joint_embedding(ctx, "the label-against-library comparison")
    fig, axes = panel_grid(2, ncols=2, panel_size=(6.7, 6.2))
    lab = ctx.joint_labels(depth)
    umap_scatter(axes[0], xy=xy, labels=lab, colours=ctx.colours(depth),
                 order=[l for l in ctx.label_order(depth) if (lab == l).any()],
                 denominator=len(lab), legend=True, title=f"cell type, level {depth}")
    sam = ctx.joint_samples()
    uniq = sorted(set(sam))
    p = plt()
    umap_scatter(axes[1], xy=xy, labels=sam,
                 colours={s: p.cm.tab20(i % 20) for i, s in enumerate(uniq)},
                 order=uniq, denominator=len(sam), legend=True, legend_fontsize=6.0,
                 title=f"library   {len(uniq)} samples")
    return fig, ("If the right panel looks like the left, the structure is libraries, "
                 "not cell types")


# ======================================================================= markers

def _panel_columns(ctx, depth):
    """(row labels, gene columns, column->owning node, group spans) for a dotplot at `depth`."""
    panels = ctx.panels(depth)
    if not panels:
        raise NotDrawable(f"no marker panel for level {depth} — pass --panels")
    # Sentinel rows are KEPT. What the withheld nuclei express against the same panel is a
    # finding about the exclusion - here it is what shows EXCLUDED carrying the cardiomyocyte
    # markers - and a row dropped for tidiness is that finding removed.
    rows = list(ctx.label_order(depth))
    genes, owner, spans = [], [], []
    for l in [x for x in rows if x not in SENTINELS]:
        start = len(genes)
        for g in (panels.get(l) or []):
            gu = str(g).upper()
            if ctx.has_gene(gu) and gu not in genes:
                genes.append(gu)
                owner.append(l)
        if len(genes) > start:
            spans.append((start, len(genes) - 1, l))
    if not genes:
        raise NotDrawable(f"none of the level-{depth} panel genes are in this object")
    return rows, genes, owner, spans


def _dotplot_figure(ctx, depth, title):
    rows, genes, _owner, spans = _panel_columns(ctx, depth)
    frac, mean = ctx.expression_by_label(genes, depth, rows)
    # Room ABOVE for the brackets and their vertical labels, and to the RIGHT for the two keys.
    longest = max((len(str(n)) for _a, _b, n in spans), default=8)
    top_pad = 0.10 * longest + 0.8
    height = max(0.34, 0.55 - 0.10 * (depth - 1)) * len(rows) + top_pad + 1.6
    width = max(9.0, 0.26 * len(genes) + 3.2) + 3.4
    fig, ax = plt().subplots(figsize=(width, height))
    spec = dotplot(ax, rows=rows, cols=genes, frac=frac, mean_scaled=scale_per_column(mean),
                   col_group_spans=spans)
    ax.set_yticklabels(unique_ticks(rows)[::-1], fontsize=9)
    dotplot_key(fig, ax, spec)
    _ = title            # the caption carries it; see below
    fig.subplots_adjust(left=0.16, right=0.72, top=1 - (top_pad / height), bottom=0.22)
    fig.scanno_no_tight = True
    # NO title on the panel. The brackets already occupy the space above the grid, and a title
    # there lands on top of their labels; the explanation belongs in the caption, where it can
    # be as long as it needs to be. Returned so the caller can use it.
    return fig, None


def F130(ctx):
    """Marker dotplot, level 1. READ EXCLUSIVITY, NOT INTENSITY.

    A marker lit across every column supports no call, and a label whose own block is dim is not
    supported by the evidence it was assigned on. Nothing is curated out — F134 gives each
    gene's measured breadth so a broad one can be identified rather than hidden.

    The bar under each block is that node's colour, so which genes were listed for which label
    is readable without the corpus table.
    """
    return _dotplot_figure(
        ctx, 1,
        "Corpus markers against the level-1 label, each block bracketed with the node whose "
        "panel it is.\nDot size is the fraction of cells detecting the gene, on an ABSOLUTE "
        "scale; colour is mean expression scaled per column. Read EXCLUSIVITY, not intensity.")


def F135(ctx, depth=2):
    """The dotplot one level deeper — the harder figure and the more informative one.

    Level-1 blocks separate on genes with hundreds of curated assertions behind them; sibling
    subtypes are distinguished by whatever the corpus has left after the shared parent markers
    are removed. Blocks that fail to separate here are subtypes this corpus cannot resolve —
    not necessarily subtypes that are absent.
    """
    if depth > ctx.depth:
        raise NotDrawable(f"the taxonomy has {ctx.depth} level(s); no level {depth}")
    return _dotplot_figure(
        ctx, depth,
        f"Level-{depth} panels against the level-{depth} label — each node's OWN markers, not "
        f"its parent's.\nBlocks that fail to separate are subtypes this corpus cannot resolve, "
        f"not subtypes that are absent.")


def _featureplot(ctx, depth, per_node):
    panels = ctx.panels(depth)
    if not panels:
        raise NotDrawable(f"no marker panel for level {depth}")
    xy = _require_joint_embedding(ctx, "feature plots")
    picks, missing = [], []
    for l in ctx.label_order(depth):
        got = [str(g).upper() for g in (panels.get(l) or []) if ctx.has_gene(str(g).upper())]
        if not got and l not in SENTINELS:
            missing.append(l)
        picks += [(g, l) for g in got[:per_node]]
    if not picks:
        raise NotDrawable(f"none of the level-{depth} panel genes are in this object")
    fig, axes = panel_grid(len(picks), ncols=4, panel_size=(3.7, 3.7))
    for ax, (g, l) in zip(axes, picks):
        continuous_scatter(ax, xy=xy, values=ctx.joint_expression(g), size=1.8, title=g,
                           fig=fig)
        ax.title.set_color(ctx.colour(l))
    note = (f"   {len(missing)} node(s) had no usable panel and contribute nothing: "
            f"{', '.join(leaf(m) for m in missing[:6])}" if missing else "")
    return fig, (f"Level-{depth} markers on the joint embedding — is expression WHERE the label "
                 f"is, or scattered through it?\nTitles are coloured by the label the gene was "
                 f"listed for. Each panel autoscales to its own gene, so panels are NOT "
                 f"comparable.{note}")


def F131(ctx):
    return _featureplot(ctx, 1, 2)


def F136(ctx, depth=2):
    if depth > ctx.depth:
        raise NotDrawable(f"the taxonomy has {ctx.depth} level(s); no level {depth}")
    return _featureplot(ctx, depth, 1)


def F134(ctx, depth=1):
    """Measured breadth: detection within a gene's own label against its best OTHER label.

    A gene whose two bars are the same height carries no information about identity, however
    specific the corpus says it is. NOTHING IS REMOVED ON THIS MEASUREMENT — two corpus-side
    filters were built for exactly that and both were wrong, one dropping the cleanest marker in
    a lineage and the other keeping the offending gene while excluding every canonical one.
    Breadth is a property of the data, and only the data has it.

    Breadth is a statement about a PARTITION: "the best other label" means the best other label
    AT THIS DEPTH, which is why the depth is named in the title rather than left implicit.
    """
    b = ctx.marker_breadth(depth)
    if not b:
        raise NotDrawable(f"no level-{depth} marker panel to measure breadth over")
    b = list(b)[::-1]
    fig, ax = plt().subplots(figsize=(7.6, 0.26 * len(b) + 2.2))
    ticks = [f"{r['gene']}  ({leaf(r['plotted_for'])[:14]})" for r in b]
    paired_barh(ax, labels=ticks,
                filled=[100 * r["own"] for r in b],
                hollow=[100 * r["best_other"] for r in b],
                xlabel="% of nuclei with at least one count")
    ax.set_title(f"Equal bars mean the gene says nothing about identity at level {depth},\n"
                 f"however specific the corpus calls it", fontsize=9, loc="left")
    return fig, None


# ======================================================================= the withheld nuclei

def F150(ctx):
    """Every withheld nucleus on its OWN sample's embedding, everything else grey.

    Read whether they form a compact island or are dispersed through a larger population. An
    island apart from everything is consistent with debris or doublets; a set scattered through
    the body of a population is part of that population. Neither reading is available from the
    QC numbers that produced the flag.

    This figure is deliberately NOT stratified by cell type and must never become so — grouping
    the withheld nuclei by population reports the population, not these nuclei.
    """
    if not ctx.has_flag:
        raise NotDrawable("no flag column — nothing was withheld")
    ss = [s for s in ctx.samples if ctx.embedding(s) is not None]
    if not ss:
        raise NotDrawable("no per-sample embeddings")
    fig, axes = panel_grid(len(ss), ncols=min(5, len(ss)), panel_size=(3.6, 3.7))
    for ax, s in zip(axes, ss):
        xy, w = ctx.embedding(s), ctx.flag(s)
        umap_scatter(ax, xy=xy, labels=None, colours={}, denominator=len(xy),
                     background_mask=w,
                     title=f"{s}\n{len(xy):,} nuclei · {int(w.sum()):,} withheld "
                           f"({100 * w.mean():.1f}%)")
    return fig, ("Withheld nuclei (red) against every other nucleus of the same sample (grey), "
                 "in the embedding they helped define")


def F151(ctx, top=15):
    """What the withheld nuclei express: EFFECT SIZE, NOT A P-VALUE.

    At these group sizes every gene is significant and a p-value ranks genes by how abundant
    they are rather than by how different they are. Beside each bar: the detection-rate
    difference in percentage points, and in brackets how many samples the difference has the
    same sign in — the only one of the three that separates a cohort signature from one
    library's.
    """
    sig = ctx.exclusion_signature()
    if not sig:
        raise NotDrawable("no flag column, or no withheld nuclei to compare")
    show = sig[:top] + sig[-top:][::-1]
    ann = []
    for r in show:
        t = f"  {r['d_detect']:+.0f} pp"
        if r.get("animals_compared"):
            t += f"  [{r['animals_agree']}/{r['animals_compared']}]"
        ann.append(t)
    fig, ax = plt().subplots(figsize=(10.0, 0.30 * len(show) + 2.4))
    signed_barh(ax, values=[r["d_mean"] for r in show], labels=[r["gene"] for r in show],
                annotations=ann,
                xlabel="mean log-normalised expression, withheld − kept")
    ax.set_title("Elevated in the withheld nuclei (red), depleted (blue)\n"
                 "beside each bar: detection-rate difference, and [samples agreeing]",
                 fontsize=10, loc="left")
    return fig, None


def F152(ctx, top=12):
    """The same genes per sample. THIS FIGURE EXISTS BECAUSE A POOLED SIGNATURE CAN BE ONE
    SAMPLE'S SIGNATURE.

    A gene elevated in the sample contributing most of the withheld nuclei and nowhere else
    would still lead the pooled ranking. A row where the withheld column is consistently darker
    or larger than its kept neighbour across samples is a signature; a row carried by one or two
    columns is that sample's property and must not be described as the withheld nuclei's.
    """
    out = ctx.signature_per_animal(top)
    if out is None:
        raise NotDrawable(f"fewer than two samples clear the {MIN_FLAGGED_PER_ANIMAL}-nucleus "
                          f"floor for a per-sample comparison")
    genes, cols, F, M = out
    fig, ax = plt().subplots(figsize=(0.46 * len(cols) + 6.5, 0.34 * len(genes) + 3.4))
    spec = dotplot(ax, rows=genes, cols=[f"{a}\n{w}" for a, w in cols], frac=F, mean_scaled=M,
                   cmap_range=(0.15, 1.0), col_fontsize=7, row_fontsize=8.5)
    dotplot_key(fig, ax, spec, colour_label="Mean expression\nscaled per gene")
    fig.subplots_adjust(left=0.22, right=0.74, bottom=0.30, top=0.86)
    fig.scanno_no_tight = True
    n_an = len({a for a, _ in cols})
    ax.set_title(f"Top signature genes, per sample ({n_an} of {len(ctx.samples)} clear the "
                 f"{MIN_FLAGGED_PER_ANIMAL}-nucleus floor).\nDot size = detection rate, colour "
                 f"= mean expression scaled per gene. A row carried by one or two columns is "
                 f"that sample's property.", fontsize=9.5, loc="left")
    return fig, None


def F153(ctx):
    """The QC measurements of the withheld nuclei, POOLED — not stratified by cell type.

    Stratifying by label reports the populations rather than these nuclei, and that framing is
    what this figure was rebuilt to remove. These are the properties the upstream gate fired on
    and they cannot say whether it should have: a doublet-rich population and a genuinely
    dirtier one are identical here, which is why F151 and F152 exist.
    """
    qc = ctx.exclusion_qc()
    if not qc:
        raise NotDrawable("no QC columns in obs, or nothing withheld")
    fig, axes = panel_grid(len(qc), ncols=len(qc), panel_size=(3.4, 4.5))
    for ax, (_col, name, f, k, logy) in zip(axes, qc):
        quartile_boxes(ax, data=[f, k],
                       tick_labels=[f"withheld\nn={f.size:,}", f"kept\nn={k.size:,}"],
                       showfliers=False, facecolors=[ACCENT, "#ffffff"], log_y=logy,
                       tick_fontsize=8.5)
        ax.set_title(name, fontsize=10, loc="left")
    return fig, ("QC features of the withheld nuclei, POOLED — not stratified by cell type, "
                 "which reports the populations rather than these nuclei")


def F154(ctx, refuse_at=DIFFERENTIAL_LIMIT):
    """Rule one question three, DRAWN rather than tabulated.

    Left: the percentage of each sample's nuclei withheld, one point per sample, grouped by arm
    with the arm mean behind — an arm mean drawn from two or three samples is a statement about
    which samples landed in which arm. Right: the ratio between the highest and lowest arm rate
    for each design factor, against the refusal line.

    The two panels use different denominators deliberately: the left bar is the unweighted mean
    of per-sample rates, the right is the pooled per-nucleus rate.
    """
    if not ctx.has_flag:
        raise NotDrawable("no flag column — nothing was withheld")
    per = ctx.flag_per_animal()
    if not per:
        raise NotDrawable("no per-sample withholding rates")
    facs = ctx.flag_by_factor()
    p = plt()
    if facs:
        fig, axs = p.subplots(1, 2, figsize=(14.0, 5.0), squeeze=False,
                              gridspec_kw={"width_ratios": [1.5, 1]})
    else:
        fig, axs = p.subplots(1, 1, figsize=(8.0, 5.0), squeeze=False)
    ax = axs.ravel()[0]
    # The DECLARED order, so a 2x2 reads young/aged x chow/HFD rather than alphabetically,
    # which interleaves the two factors and leaves no pair of adjacent bars comparable.
    order = ctx._levels("group")
    have = {r["arm"] for r in per}
    arms = [a for a in order if a in have] + sorted(have - set(order))
    # One bar per ARM, points over it. The arms are the x categories, not parallel series:
    # modelling them as series gives one bar per arm spanning the whole axis, which reads as a
    # rate that applies everywhere.
    for i, arm in enumerate(arms):
        v = np.array([r["rate"] for r in per if r["arm"] == arm], dtype=float)
        ax.bar(i, float(v.mean()) if v.size else 0.0, width=.6, color="#e9e6e1",
               edgecolor="#7a7a7a", zorder=1)
        # A lone point is CENTRED: `linspace(-s, s, 1)` returns `[-s]`, which offsets a
        # single-sample arm's only point to the left of its own bar and reads as systematic.
        jitter = np.zeros(v.size) if v.size < 2 else np.linspace(-.16, .16, v.size)
        ax.scatter(np.full(v.size, i) + jitter, v, s=52, color=ACCENT, zorder=3,
                   edgecolor="white", linewidths=.8)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a or "(ungrouped)" for a in arms], fontsize=9.5)
    ax.set_xlim(-.6, len(arms) - .4)
    ax.set_ylabel("% of the sample's nuclei withheld", fontsize=9.5)
    ax.set_title("per sample, arm mean behind — the points are the figure", fontsize=10,
                 loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if facs:
        ratio_bars(axs.ravel()[1],
                   names=[f["factor"] + ("  (auto-detected)" if f["auto"] else "")
                          for f in facs],
                   ratios=[f["ratio"] for f in facs], lo=[f["lo"] for f in facs],
                   hi=[f["hi"] for f in facs], limit=refuse_at,
                   note=f"rule one refuses beyond {refuse_at:g}x",
                   xlabel="highest arm rate / lowest arm rate")
        axs.ravel()[1].set_title("rule one Q3, per design factor", fontsize=10, loc="left")
    return fig, "How the withheld fraction falls across the design"


def F155(ctx, depth=1):
    """What the withheld nuclei WOULD have been called — identity, measured, not inferred.

    Without this a reader has to guess whether the exclusion fell on one population, and
    guessing is how a population-selective filter goes unnoticed. The dashed line is the cohort
    rate: a bar above it is a population the filter hit harder than average, which makes that
    population's share not interpretable across the design.
    """
    rows = ctx.flag_identity(depth)
    if not rows:
        raise NotDrawable("no flag column, or no label survives beside it")
    if all(str(r["label"]) in SENTINELS for r in rows):
        # The withheld nuclei carry only the sentinel here, so this object cannot say what they
        # WOULD have been - the identity was never computed for them. Saying "100% EXCLUDED" is
        # true and answers nothing; recovering it needs a generation annotated WITHOUT the
        # exclusion, joined on barcode.
        raise NotDrawable(
            "every withheld nucleus carries only the sentinel label in this object, so what "
            "they would have been called is not recoverable from it. Annotate a second "
            "generation with --no-exclude and pass it as --joint, or report identity from that "
            "run: inferring it from the populations nearby is not measurement.")
    cohort = 100.0 * float(np.mean(np.asarray(ctx.P["flag"])))
    fig, ax = plt().subplots(figsize=(max(1.1 * len(rows) + 4, 8), 4.8))
    x = np.arange(len(rows))
    ax.bar(x, [r["pct_of_label"] for r in rows], width=.66,
           color=[ctx.colour(r["label"]) for r in rows], edgecolor="none")
    for i, r in enumerate(rows):
        ax.text(i, r["pct_of_label"], f"{r['flagged']:,}\n{r['pct_of_flagged']:.0f}% of all",
                ha="center", va="bottom", fontsize=7.5, color=MUT)
    ax.axhline(cohort, ls="--", lw=1.2, color=INK)
    ax.text(len(rows) - .4, cohort, f" cohort {cohort:.1f}%", fontsize=8.5, va="bottom",
            ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(unique_ticks([r["label"] for r in rows]), rotation=30, ha="right",
                       fontsize=9)
    ax.set_ylabel("% of that label withheld", fontsize=9.5)
    ax.set_title("What the withheld nuclei would have been called.\n"
                 "A bar above the line is a population the filter hit harder than average — "
                 "which makes that population's share not interpretable across the design.",
                 fontsize=9.5, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return fig, None


# ======================================================================= the registry

#: id -> (function, scope, title). `scope` is "cohort" or "sample". Depth-varying figures are
#: expanded by the assembler over the levels the taxonomy has, so this table stays one row per
#: KIND of figure rather than one per level.
FIGURES = {
    "F102": (F102, "cohort", "composition"),
    "F103": (F103, "cohort", "composition, deeper"),
    "F141": (F141, "cohort", "composition per sample"),
    "F143": (F143, "cohort", "composition per sample, deeper"),
    "F140": (F140, "cohort", "reliability by depth"),
    "F133": (F133, "cohort", "survival against gap"),
    "F142": (F142, "cohort", "neighbourhood agreement"),
    "F160": (F160, "cohort", "the chosen resolution"),
    "F132": (F132, "cohort", "label and library"),
    "F130": (F130, "cohort", "marker dotplot"),
    "F135": (F135, "cohort", "marker dotplot, deeper"),
    "F131": (F131, "cohort", "feature plots"),
    "F136": (F136, "cohort", "feature plots, deeper"),
    "F134": (F134, "cohort", "marker breadth"),
    "F150": (F150, "cohort", "where the withheld nuclei sit"),
    "F151": (F151, "cohort", "what they express"),
    "F152": (F152, "cohort", "signature per sample"),
    "F153": (F153, "cohort", "their QC features"),
    "F154": (F154, "cohort", "across the design"),
    "F155": (F155, "cohort", "what they would have been called"),
    "F100": (F100, "sample", "the resolution sweep"),
    "F101": (F101, "sample", "the annotation at each level"),
    "F104": (F104, "sample", "clusters and their labels"),
    "F105": (F105, "sample", "QC on the embedding"),
}


def draw(fid, ctx, path, **kw):
    """Draw one figure and save it at its own geometry. Raises `NotDrawable` with the reason.

    The figure functions return `(fig, suptitle)`; the suptitle is applied here, before the
    layout pass, because one added afterwards overlaps the top row of panels.
    """
    fn = FIGURES[fid][0]
    fig, suptitle = fn(ctx, **kw)
    return save(fig, path, fid, suptitle=suptitle)
