"""The drawing primitives every figure is built from, and the geometry table that saves them.

WHY A PRIMITIVE LAYER AND NOT A FUNCTION PER FIGURE

Three figures here are dotplots, four are embeddings, three are boxplots. When each drew itself,
the same quantity got encoded three ways - one dotplot on viridis with a linear size map beside
another on Reds with an area map - and a reader comparing them was comparing the code, not the
data. One primitive per encoding means two figures of one quantity CANNOT disagree.

THE ARGUMENTS THAT HAVE NO DEFAULT, AND WHY EACH ONE KILLED A FIGURE

  `denominator`   percentages in the embedding legends are over EVERY nucleus, including the
                  withheld ones; percentages in the composition family are over the annotated
                  ones only. Both live in one report. A default would silently pick one.
  `showfliers`    the reliability boxes SHOW outliers and the agreement boxes suppress them.
                  That is the only structural difference between two otherwise identical
                  boxplots, and it is the easiest thing in the whole module to lose.
  `label_floor`   the segment-label threshold is interpolated into the title from the same
                  variable that thresholds the labels, so the number a reader is told and the
                  number the code applied cannot drift apart.

SAVING IS PART OF THE GEOMETRY

`dpi` varies between figures and `bbox_inches="tight"` is right for most of them and wrong for
two - applying it changes their dimensions. So the module saves; it never hands a Figure to a
caller that saves it in its own way. `GEOMETRY` is the whole table, asserted in the tests.
"""
from __future__ import annotations

import math

import numpy as np

GREY = "#dcdcdc"
INK = "#1a1a1a"
MUT = "#5b5b5b"
ACCENT = "#c0504d"
COOL = "#4472a8"
AMBER = "#b06d12"
PASS = "#59a14f"

#: id -> (dpi, bbox_inches). Two figures are saved WITHOUT a tight bounding box because applying
#: one changes their dimensions relative to the layout they were composed for.
GEOMETRY = {
    "F100": (130, "tight"), "F101": (130, "tight"), "F104": (130, "tight"),
    "F105": (130, "tight"),
    "F102": (130, "tight"), "F103": (130, "tight"),
    "F130": (140, "tight"), "F135": (140, "tight"),
    "F131": (130, "tight"), "F132": (130, "tight"), "F136": (130, "tight"),
    "F133": (140, None), "F134": (140, None),
    "F140": (140, "tight"), "F141": (140, "tight"), "F142": (140, "tight"),
    "F143": (140, "tight"),
    "F150": (140, "tight"), "F151": (140, "tight"), "F152": (140, "tight"),
    "F153": (140, "tight"), "F154": (140, "tight"), "F155": (140, "tight"),
    "F160": (140, "tight"),
}
DEFAULT_GEOMETRY = (140, "tight")


class NotDrawable(Exception):
    """This figure's input is absent, and the message says WHICH input.

    Raised rather than returning None so the report can print a NAMED ABSENCE. A figure that
    vanishes silently reads as a finding that there was nothing to show, and the two are not
    the same statement.
    """


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    return p


# ------------------------------------------------------------------------------ layout

def panel_grid(n_panels, *, ncols=None, panel_size=(4.5, 4.5), sharey=False,
               width_ratios=None):
    """A grid guaranteed to have at least `n_panels` axes, trailing ones already hidden.

    Returns a FLAT list of exactly `n_panels` axes. This is what kills the `zip(axes, series)`
    class of bug: a hardcoded 2x4 grid zipped against a twelve-resolution sweep drops four
    panels and raises nothing, and a literal 1x4 zipped against five panels drops the one that
    decides the choice.
    """
    if n_panels <= 0:
        raise NotDrawable("nothing to draw: zero panels requested")
    if ncols is None:
        ncols = min(4, max(1, int(math.ceil(math.sqrt(n_panels)))))
    ncols = max(1, min(ncols, n_panels))
    nrows = int(math.ceil(n_panels / ncols))
    kw = {}
    if width_ratios is not None:
        kw["gridspec_kw"] = {"width_ratios": list(width_ratios)}
    fig, axs = plt().subplots(nrows, ncols, squeeze=False, sharey=sharey,
                              figsize=(panel_size[0] * ncols, panel_size[1] * nrows), **kw)
    flat = list(axs.ravel())
    assert len(flat) >= n_panels, "panel_grid allocated fewer axes than requested"
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat[:n_panels]


def save(fig, path, fid=None, *, suptitle=None):
    """Set the suptitle, lay out, save at this figure's own geometry, close.

    The order matters: a suptitle added after `tight_layout` overlaps the top row.
    """
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, x=.01, ha="left")
    if not getattr(fig, "scanno_no_tight", False):
        fig.tight_layout()
    dpi, bbox = GEOMETRY.get(str(fid), DEFAULT_GEOMETRY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, **({"bbox_inches": bbox} if bbox else {}))
    plt().close(fig)
    return path


def leaf(label, sep="/"):
    return str(label).rsplit(sep, 1)[-1]


def unique_ticks(labels, sep="/"):
    """Leaf names, with a parent prefix wherever the leaf alone is ambiguous.

    At depth 2 a bare leaf is almost always unique; past it, collisions are likely, and two
    different populations collapsing into one tick is a mislabelling with no visible symptom.
    """
    leaves = [leaf(l, sep) for l in labels]
    dupes = {x for x in leaves if leaves.count(x) > 1}
    out = []
    for l, lf in zip(labels, leaves):
        parts = str(l).split(sep)
        out.append(f"{parts[-2]}/{lf}" if lf in dupes and len(parts) > 1 else lf)
    return out


def readable_on(hex_colour):
    """Black or white text, whichever a reader can actually see on that segment.

    Unconditional white was near-illegible on the pale end of a shaded lineage - and the paler
    a segment, the smaller the population, so the labels being lost were the ones worth reading.
    """
    from .palette import _to_hls
    return INK if _to_hls(hex_colour)[1] > 0.55 else "#ffffff"


# ------------------------------------------------------------------------------ embeddings

def umap_scatter(ax, *, xy, labels, colours, denominator, order=None, size=1.2,
                 legend=False, legend_fontsize=6.5, legend_ncol=1, title=None,
                 background_mask=None, background_colour=GREY, background_size=1.1,
                 highlight_colour=ACCENT, highlight_size=3.0, sep="/"):
    """A categorical embedding scatter. Also the highlight-against-grey form.

    `order` is BOTH draw order and legend order; the default is `sorted(set(labels))` on plain
    Python strings, never case-folded and never locale-collated, because that ordering is
    visible in the legend and stable between runs.

    `denominator` is required: the legend percentage is a share of something, and which thing
    differs between figure families that appear in the same report.

    With `background_mask` the grey layer is drawn FIRST and the highlighted layer on top, so a
    highlighted point is never hidden under the population it sits in.
    """
    if background_mask is not None:
        m = np.asarray(background_mask, dtype=bool)
        ax.scatter(xy[~m, 0], xy[~m, 1], s=background_size, c=[background_colour],
                   linewidths=0, rasterized=True)
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=highlight_size, c=[highlight_colour],
                       linewidths=0, rasterized=True)
    else:
        labels = np.asarray(labels, dtype=object)
        order = list(order) if order is not None else sorted(set(labels))
        handles = []
        import matplotlib.lines as mlines
        for name in order:
            m = labels == name
            if not m.any():
                continue
            c = colours.get(name, "#D9D9D9")
            ax.scatter(xy[m, 0], xy[m, 1], s=size, c=[c], linewidths=0, rasterized=True)
            if legend:
                pct = 100.0 * m.sum() / max(denominator, 1)
                # The marker size is set HERE, on a proxy handle, and `markerscale` is left at 1.
                # Scaling the handle instead - which is what you need when the handles come from
                # the scatter itself, whose points are ~1pt - multiplies a proxy's default 6pt
                # into a 48pt blob that swallows its own label and the panel title behind it.
                handles.append(mlines.Line2D([], [], marker="o", linestyle="", color=c,
                                             markersize=5,
                                             label=f"{leaf(name, sep)} ({pct:.0f}%)"))
        if legend and handles:
            ax.legend(handles=handles, loc="upper left", frameon=False,
                      fontsize=legend_fontsize, ncol=legend_ncol,
                      handletextpad=.3, borderpad=.2, labelspacing=.32)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11 if background_mask is None else 9, loc="left")
    return ax


def continuous_scatter(ax, *, xy, values, cmap="viridis", size=1.6, title=None,
                       sort_order=True, colorbar=True, fig=None, log=False):
    """One continuous quantity on an embedding, expressing cells drawn LAST.

    Without the sort a high-expressing cell painted early is buried under the zeros drawn over
    it, and the gene reads as absent from the population that expresses it most.
    """
    v = np.asarray(values, dtype=float)
    if log:
        v = np.log10(np.clip(v, 1, None))
    o = np.argsort(v) if sort_order else np.arange(len(v))
    sc = ax.scatter(xy[o, 0], xy[o, 1], s=size, c=v[o], cmap=cmap, linewidths=0,
                    rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9.5, loc="left")
    if colorbar and fig is not None:
        fig.colorbar(sc, ax=ax, fraction=.045, pad=.02)
    return sc


# ------------------------------------------------------------------------------ composition

def stacked_rows(ax, *, rows, order, colours, label_floor, height=0.68,
                 label_fontsize=8.5, xlabel="% of the row's nuclei"):
    """100%-stacked horizontal bars, one row per group or per sample.

    `rows` is [{"name", "n", "pct": {label: pct}}] with the percentages ALREADY summing to 100
    within a row: this primitive never chooses a denominator.

    A segment is labelled when its UNROUNDED value clears `label_floor` and the text is printed
    rounded, so the visible labels do not sum to 100 - which is correct and worth knowing.
    """
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        left = 0.0
        for l in order:
            v = float(r["pct"].get(l, 0.0))
            if v <= 0:
                continue
            c = colours.get(l, "#D9D9D9")
            ax.barh(yi, v, left=left, height=height, color=c, edgecolor="white", linewidth=1.0)
            if v >= label_floor:
                ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center",
                        fontsize=label_fontsize, color=readable_on(c))
            left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['name']}\n{r['n']:,}" for r in rows], fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(xlabel, fontsize=9)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def swatch_legend(fig, order, colours, *, ncol, fontsize=8, sep="/"):
    """A figure-level key below the axes, in a band reserved for it.

    Marks the figure `no_tight`, because `tight_layout` recomputes margins from the AXES alone
    and does not know a figure legend is there - it reclaims the reserved band and drops the
    legend on top of the x-axis label.
    """
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=colours.get(l, "#D9D9D9"), label=leaf(l, sep))
               for l in order]
    rows = int(math.ceil(len(handles) / max(ncol, 1)))
    band = 0.055 + 0.030 * rows
    fig.subplots_adjust(bottom=band + 0.10, top=0.86, left=0.10, right=0.985)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=fontsize, bbox_to_anchor=(0.5, 0.008))
    fig.scanno_no_tight = True


def points_over_bars(ax, *, categories, groups, point_values, group_colours,
                     ylabel, spread=0.0, point_size=22, tick_fontsize=9,
                     tick_rotation=30, ticks=None):
    """Group means as recessive bars with ONE POINT PER UNIT over them.

    The alpha contrast is the encoding: the means recede and the individual samples dominate,
    because a group mean drawn from two or three samples is a statement about which samples
    landed in which arm.

    `spread` lays a group's points out horizontally. A lone point is CENTRED - `linspace(-s, s, 1)`
    returns `[-s]`, which puts a single-sample arm's only point to the left of its own bar and
    reads as a systematic offset.
    """
    x = np.arange(len(categories))
    w = 0.8 / max(len(groups), 1)
    for gi, g in enumerate(groups):
        vals = [point_values.get((g, c), []) for c in categories]
        means = [float(np.mean(v)) if len(v) else 0.0 for v in vals]
        off = (gi - (len(groups) - 1) / 2) * w
        col = group_colours[gi % len(group_colours)]
        ax.bar(x + off, means, width=w * 0.92, label=str(g), alpha=.55, color=col,
               edgecolor="none")
        for i, v in enumerate(vals):
            if not len(v):
                continue
            jitter = (np.zeros(len(v)) if spread <= 0 or len(v) == 1
                      else np.linspace(-spread, spread, len(v)))
            ax.scatter(x[i] + off + jitter, v, s=point_size, color=col, edgecolor=INK,
                       linewidth=.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ticks if ticks is not None else list(categories),
                       rotation=tick_rotation, ha="right", fontsize=tick_fontsize)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ------------------------------------------------------------------------------ boxes

def quartile_boxes(ax, *, data, tick_labels, showfliers, facecolors="#cfe3f2", widths=0.6,
                   tick_rotation=0, tick_fontsize=None, log_y=False, ylim_bottom=None):
    """Quartile boxes. `showfliers` has NO DEFAULT - see the module docstring.

    Tick labels are always set with `set_xticks` then `set_xticklabels`, never passed to
    `boxplot`: matplotlib renamed that keyword, and a version-dependent kwarg is a figure that
    stops being produced silently.
    """
    data = [np.asarray(d, dtype=float) for d in data]
    data = [d[~np.isnan(d)] for d in data]
    bp = ax.boxplot(data, widths=widths, patch_artist=True, showfliers=showfliers,
                    medianprops=dict(color=INK))
    faces = facecolors if isinstance(facecolors, (list, tuple)) else [facecolors] * len(data)
    for b, fc in zip(bp["boxes"], faces):
        b.set(facecolor=fc, edgecolor=MUT)
    ax.set_xticks(np.arange(1, len(tick_labels) + 1))
    ax.set_xticklabels(tick_labels, rotation=tick_rotation,
                       ha="right" if tick_rotation else "center", fontsize=tick_fontsize)
    if log_y:
        ax.set_yscale("log")
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return bp


# ------------------------------------------------------------------------------ dotplots

#: scanpy's DotPlot defaults, pinned as names rather than inherited. Area is proportional to
#: fraction**1.5 and the largest dot is the largest fraction ACTUALLY PRESENT - so the size
#: scale is data-dependent and two dotplots are not comparable to each other by dot size.
LARGEST_DOT = 200.0
SIZE_EXPONENT = 1.5


def dotplot(ax, *, rows, cols, frac, mean_scaled, cmap="Reds", cmap_range=(0.0, 1.0),
            size_scale=None, size_exponent=SIZE_EXPONENT, row_fontsize=8,
            col_fontsize=6.5, col_rotation=90, col_group_spans=None, span_colours=None):
    """Dot size = fraction detecting, colour = mean scaled per gene.

    `mean_scaled` must already be min-max scaled per COLUMN by the caller - the equivalent of
    scanpy's `standard_scale="var"`. This primitive scales nothing, so two dotplots cannot end
    up scaled two different ways.
    """
    p = plt()
    cm = p.get_cmap(cmap)
    lo, hi = cmap_range
    frac = np.asarray(frac, dtype=float)
    fmax = float(np.nanmax(frac)) if frac.size else 1.0
    fmax = fmax if fmax > 0 else 1.0
    for i in range(len(rows)):
        for j in range(len(cols)):
            f = frac[i, j]
            if not np.isfinite(f):
                continue
            if size_scale is None:
                s = LARGEST_DOT * (max(f, 0.0) / fmax) ** size_exponent
            else:
                s = size_scale[0] + size_scale[1] * max(f, 0.0)
            m = mean_scaled[i, j]
            ax.scatter(j, len(rows) - 1 - i, s=s,
                       c=[cm(lo + (hi - lo) * float(np.nan_to_num(m)))],
                       edgecolor="black", linewidths=.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=col_rotation, fontsize=col_fontsize)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(rows)[::-1], fontsize=row_fontsize)
    ax.set_xlim(-.7, len(cols) - .3)
    ax.set_ylim(-.7, len(rows) - .3)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if col_group_spans:
        for (start, end, name) in col_group_spans:
            c = (span_colours or {}).get(name, MUT)
            ax.plot([start - .35, end + .35], [-.75, -.75], lw=2.6, color=c,
                    solid_capstyle="butt", clip_on=False)
    return {"cmap": cm, "cmap_range": cmap_range, "fmax": fmax,
            "size_scale": size_scale, "size_exponent": size_exponent}


def dotplot_key(fig, ax, spec, *, size_label="fraction detecting",
                colour_label="mean expression,\nscaled per column"):
    """The key without which a dotplot is undecodable.

    Both channels are explained: three reference dots for size, and a colour bar. The size
    legend states the LARGEST fraction actually present, because the dots are normalised to it -
    so a reader knows the scale is data-dependent and two dotplots are not comparable by size.
    """
    import matplotlib.lines as mlines
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fmax = spec["fmax"]
    fracs = [0.25 * fmax, 0.5 * fmax, fmax]
    handles = []
    for f in fracs:
        s = (LARGEST_DOT * (f / fmax) ** spec["size_exponent"] if spec["size_scale"] is None
             else spec["size_scale"][0] + spec["size_scale"][1] * f)
        handles.append(mlines.Line2D([], [], marker="o", linestyle="", markerfacecolor="#bbb",
                                     markeredgecolor="black", markeredgewidth=.5,
                                     markersize=(s ** 0.5), label=f"{100 * f:.0f}%"))
    lg = ax.legend(handles=handles, title=f"{size_label}\n(largest = {100 * fmax:.0f}%, the "
                                          f"largest present)",
                   loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                   fontsize=7.5, title_fontsize=7.5, labelspacing=1.1, borderpad=.6,
                   handletextpad=1.0)
    lg._legend_box.align = "left"
    lo, hi = spec["cmap_range"]
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=spec["cmap"])
    cax = ax.inset_axes([1.02, 0.06, 0.02, 0.30])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(colour_label, fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    return lg


def scale_per_column(M):
    """Min-max per column, 0-1. `standard_scale="var"`."""
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return M
    lo = np.nanmin(M, axis=0)
    rng = np.nanmax(M, axis=0) - lo
    return (M - lo) / np.where(rng == 0, 1, rng)


def scale_per_row(M):
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return M
    lo = np.nanmin(M, axis=1)[:, None]
    rng = np.nanmax(M, axis=1)[:, None] - lo
    return (M - lo) / np.where(rng == 0, 1, rng)


# ------------------------------------------------------------------------------ bars

def signed_barh(ax, *, values, labels, pos_colour=ACCENT, neg_colour=COOL, height=0.68,
                annotations=None, ann_offset=0.012, ann_fontsize=8, xpad_frac=0.40,
                ytick_fontsize=8.5, xlabel=None):
    """Bars coloured by SIGN, not by scale. Row 0 is drawn at the top.

    `annotations` are placed just outside each bar's tip with the alignment flipped on the sign.
    An absent annotation renders as nothing at all, never as the string "nan".
    """
    vals = [float(v) for v in values]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=height, color=[pos_colour if v > 0 else neg_colour for v in vals])
    if annotations:
        for yy, v, txt in zip(y, vals, annotations):
            if not txt:
                continue
            ax.text(v + (ann_offset if v > 0 else -ann_offset), yy, txt, va="center",
                    ha="left" if v > 0 else "right", fontsize=ann_fontsize)
    ax.set_yticks(y)
    ax.set_yticklabels(list(labels), fontsize=ytick_fontsize)
    ax.axvline(0, color=INK, lw=1)
    if vals:
        pad = xpad_frac * max(abs(v) for v in vals)
        ax.set_xlim(min(vals) - pad, max(vals) + pad)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def paired_barh(ax, *, labels, filled, hollow, offset=0.19, height=0.36,
                filled_colour="#3d6f9e", hollow_edgecolor=AMBER, hollow_linewidth=1.1,
                legend_labels=("its own label", "best OTHER label"), ytick_fontsize=7.5,
                xlabel=None):
    """Two bars per row: filled above the pair centre, hollow below.

    The filled-versus-open contrast rather than hue is the encoding, and EQUAL BARS ARE THE
    MESSAGE. A NaN draws nothing rather than a zero, so a hand-made panel shows a visible gap
    instead of a silent absence of expression.
    """
    y = np.arange(len(labels))
    ax.barh(y + offset, filled, height=height, color=filled_colour, label=legend_labels[0])
    ax.barh(y - offset, hollow, height=height, facecolor="none", edgecolor=hollow_edgecolor,
            linewidth=hollow_linewidth, label=legend_labels[1])
    ax.set_yticks(y)
    ax.set_yticklabels(list(labels), fontsize=ytick_fontsize)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.margins(y=.01)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def ratio_bars(ax, *, names, ratios, lo, hi, limit=3.0, clip=12.0, height=0.55,
               ann_fontsize=8.5, note=None, xlabel=None):
    """Bar length is `min(ratio, clip)`; the printed label carries the TRUE ratio.

    A 206x differential draws a bar of length `clip` labelled "206.88x". Clipping the bar keeps
    the other factors readable; clipping the NUMBER would hide the finding.
    """
    yy = np.arange(len(names))
    ax.barh(yy, [min(r, clip) for r in ratios], height=height,
            color=[ACCENT if r > limit else PASS for r in ratios])
    ax.axvline(limit, color=INK, ls="--", lw=1.3)
    if note:
        # Anchored to the LINE in x and to the axes floor in y. Data coordinates in y put it
        # either under the x-axis label (past the last bar) or over the panel title (before the
        # first), because the y axis is inverted below and the bar count varies.
        from matplotlib.transforms import blended_transform_factory
        ax.text(limit + clip * .012, 0.015, note, fontsize=ann_fontsize, va="bottom",
                transform=blended_transform_factory(ax.transData, ax.transAxes))
    for i, (r, a, b) in enumerate(zip(ratios, lo, hi)):
        ax.text(min(r, clip) + clip * .012, i, f"{r:.2f}x   ({a:.2f}% – {b:.2f}%)",
                va="center", fontsize=ann_fontsize)
    ax.set_yticks(yy)
    ax.set_yticklabels(list(names), fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, clip * 1.35)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def line_with_band(ax, *, x, y, band=None, mark_x=None, colour=COOL, mark_colour=ACCENT,
                   title=None, xlabel=None, ylabel=None, annotate=None):
    """A sweep line, optionally with the tolerance band and the chosen point marked.

    `band` is an explicit `(lo, hi)` or None and the CALLER decides. A percentage-point
    tolerance drawn on a panel whose unit is a nucleus count is meaningless, and drawing it
    anyway makes a count look like it has a tolerance.
    """
    ax.plot(x, y, "-o", color=colour, ms=5, lw=1.6)
    if band is not None:
        ax.axhspan(band[0], band[1], color=mark_colour, alpha=.10, lw=0)
    if mark_x is not None:
        ax.axvline(mark_x, color=mark_colour, lw=1.6, ls="--")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if annotate and mark_x is not None:
        ax.text(mark_x, ax.get_ylim()[0], annotate, color=mark_colour, fontsize=9, va="bottom")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def scatter_sized(ax, *, x, y, sizes, colour_values, cmap="viridis", alpha=0.75,
                  vline=None, fig=None, colorbar_label=None):
    """`sizes` is matplotlib's `s`, i.e. AREA in points squared.

    The caller passes `sqrt(n)*1.6`, so the plotted RADIUS grows as the fourth root of n: a
    cluster of 10,000 draws ten times the radius of one of 100, not a hundred times.

    The colour bar is built from the returned collection, never from `ax.collections[0]` - any
    artist drawn earlier silently retargets it.
    """
    sc = ax.scatter(x, y, s=sizes, c=colour_values, cmap=cmap, alpha=alpha, edgecolor="none")
    if vline is not None:
        ax.axvline(vline, ls="--", lw=1, c=AMBER)
    if fig is not None:
        fig.colorbar(sc, ax=ax, label=colorbar_label)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return sc
