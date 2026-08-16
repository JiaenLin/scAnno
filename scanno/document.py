"""The two documents scAnno delivers: one cohort report, and one comprehensive report per sample.

WHY TWO AND NOT SEVEN

An earlier version wrote five overlapping documents - a per-sample gallery, a marker report, a
summary, an exclusion report and a resolution report - plus a second "filtered" copy of several
of them. A reader opening the directory had to know which of five to read first, three of them
carried the same composition figure, and no rebuild order left all of them fresh at once.

What a reader actually wants is the COHORT: composition across the design, whether the calls are
supported, whether anything was withheld and from where. Per-sample detail matters only once a
cohort number looks wrong, and then they want everything about that one sample in one place.

So: `reports/cohort.html` is the document, and `reports/samples/<name>.html` is one comprehensive
page per sample, linked from it. Nothing appears in both except by deliberate reference.

WHAT IT ASSEMBLES RATHER THAN RECOMPUTES

Every number comes from the `Context`, which derived it once. A report that computes its own
numbers can disagree with the figures beside it, and nothing on the page says which is right.

THREE THINGS THIS DOCUMENT DOES THAT A GALLERY DOES NOT

  - **A figure that could not be drawn is NAMED**, with the missing input, in the place it would
    have been. A silently absent panel reads as a finding that there was nothing to show.
  - **Every table says which file it came from**, so a number in the text can be opened.
  - **Figures are referenced with relative `src="..."` paths, in double quotes.** Not base64: a
    self-contained page has no inputs a freshness checker can see, so it can never be found
    stale - which is the failure that kind of check exists to prevent.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from .context import MIN_FLAGGED_PER_ANIMAL
from .figures import FIGURES, draw
from .primitives import NotDrawable, leaf

CSS = """
:root{--bg:#fff;--fg:#191919;--mut:#5b5b5b;--line:#e6e4e0;--card:#faf9f7;--ok:#eef6ee;
--okl:#3f7d3f;--warn:#fff8ec;--warnl:#b06d12;--bad:#fdeeed;--badl:#a8403c;--acc:#2b5f9e}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#16181c;--fg:#e8e6e3;
--mut:#a3a09b;--line:#2d3138;--card:#1d2025;--ok:#16241a;--okl:#6fbf73;--warn:#2a2115;
--warnl:#e0a44a;--bad:#2a1717;--badl:#e07b76;--acc:#7aa9e0}}
:root[data-theme=dark]{--bg:#16181c;--fg:#e8e6e3;--mut:#a3a09b;--line:#2d3138;--card:#1d2025;
--ok:#16241a;--okl:#6fbf73;--warn:#2a2115;--warnl:#e0a44a;--bad:#2a1717;--badl:#e07b76;
--acc:#7aa9e0}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2.5rem 1.5rem 5rem;
font:16px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto}
a{color:var(--acc)}
h1{font-size:1.9rem;margin:0 0 .35rem;letter-spacing:-.02em}
h2{font-size:1.22rem;margin:2.9rem 0 .8rem;padding-bottom:.35rem;border-bottom:1px solid var(--line)}
h3{font-size:.88rem;margin:1.6rem 0 .6rem;color:var(--mut);font-weight:600;
text-transform:uppercase;letter-spacing:.05em}
.sub{color:var(--mut);font-size:.9rem}
.lede{font-size:1.03rem;margin:1.1rem 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:.9rem;
margin:1.5rem 0}
.cell{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.9rem 1rem}
.cell .lbl{margin:0;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}
.cell .big{margin:.2rem 0 0;font-size:1.5rem;font-weight:600;font-variant-numeric:tabular-nums}
.cell .s{margin:.15rem 0 0;font-size:.78rem;color:var(--mut)}
.ok,.warn,.bad{padding:1rem 1.2rem;margin:1.4rem 0;border-radius:0 5px 5px 0;font-size:.93rem}
.ok{background:var(--ok);border-left:3px solid var(--okl)}
.warn{background:var(--warn);border-left:3px solid var(--warnl)}
.bad{background:var(--bad);border-left:3px solid var(--badl)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.87rem;
font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:.5rem .6rem;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{background:var(--card);font-size:.72rem;text-transform:uppercase;color:var(--mut)}
figure{margin:1.9rem 0;padding:1rem;background:var(--card);border:1px solid var(--line);
border-radius:8px}
img{max-width:100%;height:auto;display:block;border-radius:4px;background:#fff}
figcaption{color:var(--mut);font-size:.86rem;margin-top:.75rem}
code,.mono{font-family:ui-monospace,Consolas,monospace;font-size:.84em}
.src{font-size:.78rem;color:var(--mut);margin:.2rem 0 1.2rem}
.absent{border:1px dashed var(--line);border-radius:8px;padding:.9rem 1.1rem;margin:1.6rem 0;
color:var(--mut);font-size:.9rem;background:transparent}
.samples{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:.7rem;
margin:1.2rem 0}
.samples a{display:block;background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:.7rem .9rem;text-decoration:none}
.samples .n{display:block;font-size:.76rem;color:var(--mut)}
.sw{display:inline-block;width:.72rem;height:.72rem;border-radius:2px;vertical-align:-1px;
margin-right:.4rem}
ul.rule{font-size:.93rem;padding-left:1.2rem;margin:.8rem 0}
ul.rule li{margin:.45rem 0}
pre.tree{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:.9rem 1.1rem;margin:1rem 0;overflow-x:auto;font-family:ui-monospace,Consolas,monospace;
font-size:.8rem;line-height:1.5}
"""

#: Manuscript legends, keyed by figure id. They live here rather than in the drawing code so the
#: page a reader sees and the caption that was written for it are the same string, and so a
#: caption can interpolate a number the run computed instead of restating one.
LEGENDS = {
    "F100": "The same embedding coloured by the level-1 label at each swept clustering "
            "resolution, one panel per resolution, with the number of calls in each panel "
            "title and each label's share in its legend. <b>Colours are fixed across every "
            "panel and every file</b> — a per-panel palette makes the same population look "
            "like a different one between resolutions, which is the comparison this figure "
            "exists to support. A population that disappears at a finer resolution is a legend "
            "entry that stops appearing.",
    "F101": "The same embedding at each level of the taxonomy. Subtypes keep their parent's "
            "hue at a different lightness, so this reads as the level-1 picture subdivided "
            "rather than as a different partition of the same nuclei. <b>Read what the extra "
            "level bought</b>: a level whose panel looks like the one before it is a level the "
            "evidence did not support splitting.",
    "F104": "The clusters the annotation was made from, beside the labels it produced. A "
            "cluster-level annotator is only as good as its partition; these two pictures "
            "together are what show <b>one label spread over three clusters, or one cluster "
            "split between two labels</b>. Neither picture alone shows it.",
    "F105": "The per-nucleus QC measurements on this sample's own embedding. A histogram says "
            "how many; this says <b>where</b> — and only the second distinguishes a uniformly "
            "shallow library from one population that is shallow.",
    "F102": "Level-1 cell-type proportions, as a percentage of each row's nuclei. Bars sum to "
            "100%. Group rows are computed per sample and averaged, never pooled before "
            "division — pooling lets the largest library set the group's composition. <b>A "
            "difference between two rows may be biology, annotation behaviour or clustering "
            "granularity, and this figure cannot separate them.</b>",
    "F103": "The composition one level deeper. <b>This is the level at which a compositional "
            "change would actually appear</b> — level-1 shares can sit still while everything "
            "underneath rearranges — and equally the level at which the calls are weakest. "
            "Each subtype is shaded from its parent's colour and ordered under it. Read it "
            "against the reliability table.",
    "F141": "Group means as bars with <b>one point per sample</b> over them. The points are the "
            "figure: a group mean drawn from a handful of samples spanning a wide range is a "
            "statement about which samples landed in which arm, not about the arm.",
    "F143": "The same one level deeper. The points matter more here than at level 1 — a subtype "
            "share rests on fewer nuclei, so the between-sample spread within a group is wider "
            "by construction, and <b>a bar difference smaller than that spread is not a group "
            "difference</b>.",
    "F140": "Three reliability signals against the depth of the call. <b>They disagree, which "
            "is the point of showing all three</b>: the decision gap does not fall much with "
            "depth, but curated support collapses, so a deep call can look as confident as a "
            "shallow one while resting on a handful of assertions. Panel survival adds the "
            "third axis — how much of a node's evidence was left after better-cited siblings "
            "took their share.",
    "F133": "One point per call, sized by nuclei and coloured by tree depth. Panel survival is "
            "how much of a node's own evidence still carried a claim after the sibling "
            "contrast. <b>Top-left is a confident call won on evidence that had already been "
            "stripped by better-cited neighbours</b>; bottom-right is safe.",
    "F142": "For each nucleus, the share of its own neighbourhood carrying its label, split by "
            "label, one panel per level. <b>A property of this object's graph and not a quality "
            "score</b> — a cell type that genuinely sits between two others scores low without "
            "being wrong. The low boxes are the figure. Whether splitting deeper costs "
            "agreement says whether the partition is coarser than the manifold or finer.",
    "F160": "The evidence behind the chosen clustering resolution, over every candidate swept. "
            "The shaded band is the tolerance <b>derived from the sweep's own step-to-step "
            "variation</b> rather than chosen; candidates inside it are ones the evidence "
            "cannot separate, and it is drawn only on the percentage panels. <b>Read the last "
            "panel</b>: stability is often flat across a sweep, and the smallest named "
            "population is what a finer clustering destroys first.",
    "F132": "The same embedding coloured by cell type and by library. <b>If the right panel "
            "looks like the left, the structure is libraries, not cell types.</b> No identity "
            "decision should be presented on mixing metrics alone; a number can say a "
            "population was mixed, and only the picture distinguishes <i>aligned with its "
            "counterparts</i> from <i>dispersed everywhere</i>.",
    "F130": "Corpus markers against the level-1 label. Dot size is the fraction of nuclei "
            "expressing the gene, colour the mean among them scaled per gene so columns are "
            "comparable; the largest dot is the largest fraction actually present, so <b>two "
            "dotplots are not comparable to each other by dot size</b>. The bar under each "
            "block is that label's colour. <b>Read exclusivity, not intensity</b> — a marker "
            "lit across every column supports no call, and a label whose own block is dim is "
            "not supported by the evidence it was assigned on. Nothing has been curated out.",
    "F135": "The dotplot one level deeper, each node against its <b>own</b> markers rather than "
            "its parent's. This is the harder figure and the more informative one: level-1 "
            "blocks separate on genes with hundreds of curated assertions behind them, while "
            "sibling subtypes are distinguished by whatever the corpus has left after the "
            "shared parent markers are removed. <b>Blocks that fail to separate are subtypes "
            "this corpus cannot resolve, not necessarily subtypes that are absent.</b>",
    "F131": "The top markers per label on the joint embedding. Asks whether expression is where "
            "the label is rather than scattered through the manifold. Each panel autoscales to "
            "its own gene, so <b>panels are not comparable to each other</b>. Read it beside "
            "the label-and-library figure — the embedding is not integrated.",
    "F136": "The same question one level deeper: does a subtype's marker land on a sub-region "
            "of its parent's territory, or across all of it? <b>A marker covering its parent's "
            "whole territory is not distinguishing the subtype.</b>",
    "F134": "For every gene in the dotplot: the fraction of nuclei detecting it within the "
            "label it was plotted for (filled) against the highest fraction in any other label "
            "(open). <b>A gene whose two bars are the same height carries no information about "
            "identity</b>, however specific the corpus says it is. <b>Nothing is removed on "
            "this measurement</b> — two corpus-side filters were built for exactly that and "
            "both were wrong, one dropping the cleanest marker in a lineage and the other "
            "keeping the offending gene while excluding every canonical one. Breadth is a "
            "property of the data, and only the data has it.",
    "F150": "Every withheld nucleus on the embedding of the sample it belongs to, over all the "
            "others in grey, in the embedding it helped define rather than one recomputed "
            "without it. <b>Read whether they form a compact island or are dispersed through a "
            "larger population.</b> An island apart from everything is consistent with debris "
            "or doublets; a set scattered through the body of a population is part of that "
            "population. Neither reading is available from the QC numbers that produced the "
            "flag.",
    "F151": "Genes ranked by the difference in mean log-normalised expression between the "
            "withheld nuclei and the kept. <b>Effect size, not a p-value, deliberately</b>: at "
            "these group sizes every gene is significant and a p-value would rank genes by "
            "abundance. Beside each bar, the detection-rate difference in percentage points and "
            "in brackets the number of samples in which the difference has the same sign — the "
            "only one of the three that separates a cohort signature from one library's.",
    "F152": "The same genes resolved per sample, over the samples holding enough withheld "
            "nuclei to compare. <b>This figure exists because a pooled signature can be one "
            "sample's signature.</b> A gene elevated in the sample contributing most of the "
            "withheld nuclei and nowhere else would still lead the pooled ranking. A row "
            "carried by one or two columns is that sample's property.",
    "F153": "The per-nucleus QC measurements, withheld against kept, <b>pooled — not stratified "
            "by cell type</b>, which reports the populations rather than these nuclei. These "
            "are the properties the upstream gate fired on and they cannot say whether it "
            "should have: a doublet-rich population and a genuinely dirtier one are identical "
            "here, which is why the signature figures exist.",
    "F154": "Left: the percentage of each sample's nuclei withheld, one point per sample, "
            "grouped by arm with the arm mean behind. Right: the ratio between the highest and "
            "lowest arm rate for each design factor. <b>A filter falling several times harder "
            "on one arm converts a technical property into an apparent biological difference, "
            "and nothing downstream can undo it.</b> The bar is clipped where the ratio is "
            "extreme; the printed number is not. The two panels use different denominators "
            "deliberately — the left bar is the unweighted mean of per-sample rates, the right "
            "is the pooled per-nucleus rate.",
    "F155": "What the withheld nuclei would have been called, joined from the labels that "
            "survive beside the flag — <b>identity measured, not inferred</b>. Without it a "
            "reader has to guess whether the exclusion fell on one population, and guessing is "
            "how a population-selective filter goes unnoticed. The dashed line is the cohort "
            "rate: a bar above it is a population the filter hit harder than average, <b>which "
            "makes that population's share not interpretable across the design</b>.",
}

CANNOT_SHOW = (
    "<b>Nothing here can show that a label is CORRECT.</b> These figures and numbers grade the "
    "EVIDENCE behind a call, not its truth, and a node with a confident-looking panel that "
    "happens to be wrong will score well on all of them. The genes plotted are the genes the "
    "classifier scored on, so agreement between them is close to circular — read it as a check "
    "that the pipeline did what it claims, not as validation of the biology. It cannot detect a "
    "cell type the taxonomy has no node for, and it cannot separate a real population from one "
    "the corpus merely has a confident panel for. Composition matching the literature is "
    "reassurance, not validation. Unless the embedding was integrated, proximity on it may be "
    "library rather than identity.")


def _esc(v):
    return html.escape("" if v is None else str(v), quote=True)


def _num(v, nd=0):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return f"{v:,}"


def _table(headers, rows, source=None):
    if not rows:
        return "<p class='sub'>nothing to show</p>"
    h = "".join(f"<th>{_esc(x)}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    src = f"<p class='src'>source: <code>{_esc(source)}</code></p>" if source else ""
    return f"<div class='scroll'><table><tr>{h}</tr>{b}</table></div>{src}"


def _absent_section(what, why):
    """A section whose input is missing says so IN PLACE.

    A section that simply does not render is indistinguishable from one whose answer was
    "nothing to report", and the two are opposite statements. Naming the missing input also
    tells a reader how to get the section.
    """
    return (f"<div class='absent'><b>This section needs {_esc(what)}, which this run did not "
            f"have.</b> {_esc(why)}</div>")


def _kpi(cells):
    out = "".join(
        f"<div class='cell'><p class='lbl'>{_esc(l)}</p><p class='big'>{v}</p>"
        f"<p class='s'>{s}</p></div>" for l, v, s in cells)
    return f"<div class='grid'>{out}</div>"


def _page(title, body):
    return (f"<!doctype html><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_esc(title)}</title><style>{CSS}</style><main>{body}</main>")


class Assembler:
    """Draws each figure, records what could not be drawn, and emits the HTML block for it."""

    def __init__(self, ctx, out_dir, *, rel, subdir):
        self.ctx = ctx
        self.out = Path(out_dir)
        self.rel = rel                 # path prefix from the HTML file to the figure directory
        self.subdir = subdir           # figure directory under out/figures
        self.absent = []   # the DATA lacked an input
        self.failed = []   # scAnno broke. Never the same thing.
        self.made = []

    def fig(self, fid, *, name=None, caption=None, **kw):
        """Draw `fid` and return its HTML block, or a NAMED ABSENCE if it cannot be drawn."""
        stem = name or fid
        path = self.out / "figures" / self.subdir / f"{stem}.png"
        try:
            draw(fid, self.ctx, path, **kw)
        except NotDrawable as e:
            self.absent.append({"figure": fid, "name": stem, "reason": str(e)})
            return (f"<div class='absent'><b>{_esc(fid)} was not drawn.</b> "
                    f"{_esc(str(e))}</div>")
        except Exception as e:                                            # noqa: BLE001
            # A DEFECT, kept visually distinct from an absence. An absence says something about
            # the data; a defect says something about scAnno, and the two must never render the
            # same. A lookup failure once reported "no QC columns in obs" for ten samples whose
            # objects carried all four - a bug wearing a finding's clothes.
            self.failed.append({"figure": fid, "name": stem,
                                "reason": f"{type(e).__name__}: {e}"})
            return (f"<div class='bad'><b>{_esc(fid)} FAILED TO DRAW — this is a defect in "
                    f"scAnno, not a gap in your data.</b> "
                    f"<code>{_esc(type(e).__name__)}: {_esc(e)}</code><br>"
                    f"Please report it; do not read this as a finding about the cohort.</div>")
        self.made.append({"figure": fid, "name": stem,
                          "path": str(path.relative_to(self.out))})
        title = FIGURES[fid][2]
        cap = caption or LEGENDS.get(fid, "")
        return (f"<figure><h3>{_esc(fid)} · {_esc(title)}</h3>"
                f'<img src="{self.rel}/{stem}.png" alt="{_esc(fid)}">'
                f"<figcaption>{cap}</figcaption></figure>")


def _write_table(out_dir, name, headers, rows):
    """Every table on the page is also a CSV a reader can open. Returns the relative path."""
    import csv
    p = Path(out_dir) / "tables" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)
    return f"tables/{name}"


# =============================================================== the common scope

#: What a seal can and cannot be read as. Written in the same voice as CANNOT_SHOW because it is
#: the same kind of statement: a bound on what the section supports, placed where the section is,
#: rather than a caveat somewhere a reader has to go looking for.
SCOPE_CANNOT_SHOW = (
    "<b>A seal removes the possibility of a LABEL, never an observation.</b> Nothing was "
    "filtered, dropped or excluded here. Every nucleus is still annotated, still carries its "
    "pass-1 path in <code>obs</code>, and is merely called by its parent instead of a subtype — "
    "so the decision is reversible by re-running against the declared tree, and the removed "
    "labels are named individually below rather than described as a category. "
    "<b>It cannot distinguish a split the DATA cannot make from one the CORPUS cannot.</b> A "
    "node is sealed because samples disagreed about descending, and that disagreement is "
    "produced by the gap between sibling scores — which depends on both the biology and on which "
    "markers the corpus happens to carry for those siblings. A subtype the tissue does not "
    "contain and a subtype the corpus has no separating markers for produce the same vote, and "
    "nothing in this section separates them. "
    "<b>The vote counts SAMPLES, not nuclei.</b> A node can therefore be kept unanimously while "
    "a large share of one sample's nuclei still stop at it; the <i>stopped here</i> column is "
    "where that shows, and a seal does not repair it. "
    "<b>And it says nothing about whether any label is correct</b> — agreement between samples "
    "is agreement, not truth.")

#: The two `descend_rule` values, in words. Kept beside the section rather than in the CLI help
#: because the report must state the rule it was actually run under, and a reader of the HTML has
#: no access to the command line.
_DESCEND_WORDS = {
    "any": ("a sample counts as having descended if <b>any</b> of its nuclei went below the "
            "node — one cluster descending is enough to count that sample as having made the "
            "split"),
    "majority": ("a sample counts as having descended only if <b>more than half</b> the nuclei "
                 "arriving at the node went below it, so a single stray cluster does not carry "
                 "the sample"),
}

_VERDICT_WORDS = {
    "KEEP": "the split stays; every sample that reached it agreed to make it",
    "SEAL": "the node becomes a LEAF; its children are removed from the tree",
    "UNVOTABLE": "too few samples reached it to vote — reported, NOT sealed",
    "UNREACHED": "no sample reached it at all — left in the tree, holds nothing",
}

_VERDICT_ORDER = {"SEAL": 0, "UNVOTABLE": 1, "KEEP": 2, "UNREACHED": 3}


def _scope_stopped(v):
    """Nuclei that ARRIVED at a node and did not go below it, and in how many samples.

    Computed from the vote's own per-sample `cells` / `cells_below`, which is the only place the
    residual is visible: `support` is a count of SAMPLES, so a node where nine samples descended
    with every nucleus and the tenth descended with one reads as unanimous.
    """
    cells = v.get("cells") or {}
    below = v.get("cells_below") or {}
    per = {s: int(cells[s]) - int(below.get(s, 0)) for s in cells}
    return sum(per.values()), sum(1 for n in per.values() if n > 0)


def _scope_support(v):
    sup = v.get("support")
    try:
        sup = float(sup)
    except (TypeError, ValueError):
        return "—"
    if not v.get("n_reached") or sup != sup:          # nan
        return "n/a"
    return f"{sup:.3f}"


def scope_section(scope, out_dir=None):
    """The common scope, as the block of HTML the cohort document carries.

    Fed ENTIRELY by `scanno scope --out` (scope.json). Nothing here is recomputed from the
    objects: the report must show the scope the run was actually annotated against, and a
    section that re-derived it from the annotated objects would be describing pass 2's output
    rather than the decision pass 2 was given.

    Returns a list of HTML chunks. Never returns an empty list — a scope that is missing renders
    as a NAMED ABSENCE, because a section that simply does not appear is indistinguishable from
    one whose answer was "nothing was sealed", and those are opposite statements.
    """
    body = ["<h2>The common scope</h2>"]
    if not scope:
        body.append(_absent_section(
            "the common scope from `scanno scope --out scope.json`",
            "Without it this document cannot say which splits the cohort agreed to make, so it "
            "cannot tell you whether a label is absent from a sample because that sample has "
            "none, or because the scope removed the label everywhere. Run `scanno scope` over "
            "the pass-1 objects and pass the result with --scope."))
        return body

    nodes = scope.get("nodes") or {}
    rule = scope.get("rule") or {}
    sealed_children = scope.get("sealed") or {}
    removed = scope.get("removed_labels") or {}
    lines = scope.get("tree_lines") or []
    samples = scope.get("samples") or []
    n_samples = scope.get("n_samples") or len(samples) or None

    seals = [n for n, v in nodes.items() if v.get("verdict") == "SEAL"]
    n_lost = sum(sum(d.values()) for d in removed.values())
    lost_labels = sorted({p for d in removed.values() for p in d})

    body.append(
        f"<p class='lede'>Ten independent walks produce ten scopes: on a lineage where the "
        f"sibling contrast sits near the bar, one sample descends and the next truncates, so the "
        f"same cells get a subtype in one library and its parent in another — which downstream "
        f"is indistinguishable from a compositional shift. The scope decides the depth "
        f"<b>once</b>, from what the samples agree on, and every sample is annotated against it. "
        f"This section is that decision: the rule it was made under, the tree it produced, and "
        f"the labels it costs.</p>")

    body.append(_kpi([
        ("samples voting", f"{n_samples:,}" if n_samples else "—",
         ", ".join(_esc(s) for s in samples[:6]) + (" …" if len(samples) > 6 else "")
         or "no sample list in scope.json"),
        ("nodes sealed", f"{len(seals)}",
         f"of {len(nodes)} internal node(s) voted on"),
        ("labels removed", f"{len(lost_labels)}",
         "the possibility of a label, never a nucleus"),
        ("nuclei re-labelled to a parent", f"{n_lost:,}",
         "still annotated, still on disk, called one level higher"),
    ]))
    body.append(f"<div class='warn'>{SCOPE_CANNOT_SHOW}</div>")

    # ---- the rule, in words ---------------------------------------------------------------
    ms, mr = rule.get("min_support"), rule.get("min_reach")
    dr = str(rule.get("descend_rule", ""))
    words = ["<h3>the rule this scope was made under</h3>", "<ul class='rule'>"]
    if ms is not None:
        pct = f"{float(ms) * 100:.0f}%"
        words.append(
            f"<li><b>min-support {ms}</b> — a node is <b>SEALED</b> unless at least {pct} of the "
            f"samples that <i>reached</i> it descended below it."
            + (" At 1.0 that is unanimity among the samples that reached it: a split one sample "
               "would not make is a split the cohort cannot be asked to compare across."
               if float(ms) >= 1.0 else
               " Below 1.0 this deliberately admits residual disagreement; expect it, and read "
               "the <i>stopped here</i> column for how much got through.") + "</li>")
    if dr:
        words.append(f"<li><b>descend-rule <code>{_esc(dr)}</code></b> — "
                     + _DESCEND_WORDS.get(dr, "an unrecognised rule; see `scanno scope --help`")
                     + ".</li>")
    if mr is not None:
        words.append(
            f"<li><b>min-reach {mr}</b> — a node reached by fewer than {mr} sample(s) is "
            f"<b>UNVOTABLE</b>: it is reported by name and left OPEN, never sealed. Sealing on "
            f"one sample's evidence is a removal with no quorum behind it.</li>")
    words.append(
        "<li><b>A sample whose walk never reached a node casts NO vote there.</b> It is a "
        "missing observation, not a vote against — counting absence as opposition would seal "
        "every branch that is merely rare, and rare is not unsupported. The denominator of "
        "<i>support</i> is therefore the number of samples that <b>reached</b> the node, never "
        "the cohort size: a support of 4/7 and one of 4/10 are different statements, and the "
        "<i>reached</i> column below shows which one you are reading.</li>")
    words.append(
        "<li><b>The root is never sealed.</b> Its children are the level-1 compartments; "
        "removing them would return every nucleus as UNRESOLVED. Its evidence is still voted and "
        "still shown, because root-level truncation is real — but it is not actionable as a "
        "seal.</li>")
    if rule.get("path_key"):
        words.append(f"<li>Voted on the pass-1 column "
                     f"<code>obs[{_esc(repr(rule['path_key']))}]</code>"
                     + (f", declared tree <code>{_esc(scope['tree'])}</code>"
                        if scope.get("tree") else "") + ".</li>")
    words.append("</ul>")
    body.append("".join(words))

    # ---- the tree it produced --------------------------------------------------------------
    # The drawing is rendered VERBATIM from the JSON rather than redrawn here: redrawing would
    # need every pass-1 object just to make a picture, and would drift from what the vote
    # actually printed the day it ran. What this section adds is the CROSS-CHECK below, computed
    # from the same JSON, so a drawing that is short says so on the page.
    body.append("<h3>the scope, drawn — the taxonomy pass 2 walks</h3>")
    if lines:
        body.append("<pre class='tree'>" + _esc("\n".join(str(x) for x in lines)) + "</pre>")
        stranded, stranded_nodes, root_stopped = 0, 0, None
        for node, v in nodes.items():
            n, k = _scope_stopped(v)
            if node == "root":
                root_stopped = n            # truncation AT the root: these are UNRESOLVED
                continue
            if v.get("verdict") != "SEAL" and n:
                stranded += n
                stranded_nodes += 1
        cap = ["<p class='sub'>Counts are nuclei landing at each node of the SEALED tree; "
               "<code>n/N</code> is how many samples have any. A node marked "
               "<code>&lt;- stranded</code> is an <b>open</b> node holding nuclei of its own — a "
               "sample whose gap failed there where the other samples' cleared. Those are real "
               "values in the label column, so the scope has more terminal labels than it has "
               "leaves. Source: <code>tree_lines</code> in the scope JSON.</p>"]
        if stranded:
            cap.append(
                f"<p class='sub'><b>Cross-check, computed from the vote rather than the "
                f"drawing:</b> {stranded:,} nuclei are stranded at {stranded_nodes} open "
                f"internal node(s). If the tree above carries no <code>stranded</code> marking, "
                f"this scope JSON was written before that was drawn and <b>the picture is short "
                f"by those {stranded:,} nuclei</b> — regenerate it with `scanno scope --out`.</p>")
        if root_stopped:
            cap.append(
                f"<p class='sub'>A further <b>{root_stopped:,}</b> nuclei truncated at the "
                f"<b>root</b> and are UNRESOLVED. They appear in no branch of this drawing, and "
                f"no seal can repair them: the root is never sealed.</p>")
        body.append("".join(cap))
    else:
        body.append(_absent_section(
            "`tree_lines` in the scope JSON",
            "This scope.json predates the drawn tree travelling with the vote. Re-run "
            "`scanno scope --out` to get it; the node table below is unaffected."))

    # ---- the vote, per node ------------------------------------------------------------------
    rows, csv_rows = [], []
    for node, v in sorted(nodes.items(),
                          key=lambda kv: (_VERDICT_ORDER.get(kv[1].get("verdict"), 9),
                                          -int(kv[1].get("n_reached") or 0), kv[0])):
        verdict = str(v.get("verdict", "—"))
        stopped, stopped_n = _scope_stopped(v)
        nr = int(v.get("n_reached") or 0)
        badge = (f"<b style='color:var(--badl)'>{_esc(verdict)}</b>"
                 if verdict in ("SEAL", "UNVOTABLE") else _esc(verdict))
        rows.append([f"<span class='mono'>{_esc(node)}</span>", badge,
                     f"{nr}/{n_samples}" if n_samples else f"{nr}",
                     f"{int(v.get('n_descended') or 0)}", _scope_support(v),
                     f"{stopped:,}" + (f" <span class='sub'>({stopped_n} sample(s))</span>"
                                       if stopped else "")])
        csv_rows.append([node, verdict, nr, n_samples or "", int(v.get("n_descended") or 0),
                         v.get("support"), stopped, stopped_n,
                         ";".join(v.get("children_declared") or [])])
    src = (_write_table(out_dir, "scope_nodes.csv",
                        ["node", "verdict", "n_reached", "n_samples", "n_descended", "support",
                         "nuclei_stopped_here_pass1", "samples_with_any_stopped",
                         "children_declared"], csv_rows)
           if out_dir is not None else "the scope JSON, key `nodes`")
    body.append("<h3>the vote, node by node</h3>")
    body.append(_table(["node", "verdict", "reached", "descended", "support",
                        "stopped here (pass 1)"], rows, source=src))
    body.append("<p class='sub'>"
                + " · ".join(f"<b>{k}</b> {v}" for k, v in _VERDICT_WORDS.items())
                + ". <i>stopped here</i> counts nuclei that arrived at the node in <b>pass 1</b> "
                "and did not go below it, summed over samples — at a SEALED node every arriving "
                "nucleus stops there in pass 2 by construction, so that column describes what "
                "the vote SAW, not what pass 2 produces.</p>")

    # ---- what each seal removes, by label ----------------------------------------------------
    body.append("<h3>what each seal removes — the labels themselves, not the category</h3>")
    if not seals:
        body.append("<p class='sub'>Nothing was sealed: every node the cohort voted on was kept, "
                    "so pass 2 walks the declared tree unchanged and no label was removed.</p>")
    else:
        lrows, lcsv = [], []
        for node in sorted(seals, key=lambda n: -sum((removed.get(n) or {}).values())):
            lost = removed.get(node) or {}
            for p, n in sorted(lost.items(), key=lambda kv: -kv[1]):
                lrows.append([f"<span class='mono'>{_esc(node)}</span>",
                              f"<span class='mono'>{_esc(p)}</span>",
                              f"<b>{_esc(leaf(p))}</b>", f"{int(n):,}"])
                lcsv.append([node, p, leaf(p), int(n)])
            for c in (sealed_children.get(node) or []):
                cp = f"{node}/{c}"
                if not any(p == cp or p.startswith(cp + "/") for p in lost):
                    lrows.append([f"<span class='mono'>{_esc(node)}</span>",
                                  f"<span class='mono'>{_esc(cp)}</span>",
                                  f"<b>{_esc(c)}</b>", "0"])
                    lcsv.append([node, cp, c, 0])
            if not lost and not (sealed_children.get(node) or []):
                lrows.append([f"<span class='mono'>{_esc(node)}</span>", "—",
                              "<i>nothing — no sample descended</i>", "0"])
        lsrc = (_write_table(out_dir, "scope_removed_labels.csv",
                             ["sealed_node", "removed_path", "label", "nuclei"], lcsv)
                if out_dir is not None else "the scope JSON, keys `removed_labels` and `sealed`")
        body.append(_table(["sealed node", "removed path", "label", "nuclei in pass 1"],
                           lrows, source=lsrc))
        body.append(
            f"<p class='sub'>Those {n_lost:,} nuclei are <b>not gone</b>: in pass 2 each is "
            f"called by the sealed node itself. A removal is stated as its members and read, "
            f"never described as a category — a list called \"the fibroblast subtypes\" is not "
            f"assessable, and one that names each label and its count is. Rows at 0 are children "
            f"the seal removed from the tree that no sample had reached.</p>")

    return body


# =============================================================== the cohort document

def write_cohort(ctx, out_dir, *, title="Annotation", version="", sample_links=None):
    out = Path(out_dir)
    A = Assembler(ctx, out, rel="../figures/cohort", subdir="cohort")
    now = datetime.now(timezone.utc).astimezone()
    body = [f"<h1>{_esc(title)} — the cohort</h1>",
            f"<p class='sub'>{ctx.n:,} nuclei · {len(ctx.samples)} samples · "
            f"taxonomy depth {ctx.depth}"
            + (f" · resolution {_esc(ctx.chosen_resolution)}" if ctx.chosen_resolution else "")
            + (f" · scAnno {_esc(version)}" if version else "")
            + f" · generated {now:%Y-%m-%d %H:%M %Z}</p>"]

    # ---- the front strip -------------------------------------------------------------
    l1 = [l for l in ctx.label_order(1) if l not in ("UNRESOLVED", "EXCLUDED")]
    unres = ctx.count("UNRESOLVED", 1)
    kpi = [("nuclei annotated", f"{ctx.n:,}",
            f"{100 * (1 - unres / max(ctx.n, 1)):.1f}% placed"),
           ("level-1 labels", f"{len(l1)}",
            f"{ctx.depth} level(s) deep · {len(ctx.label_order(ctx.depth))} at the deepest"),
           ("samples", f"{len(ctx.samples)}",
            f"{len(ctx._levels('group'))} group(s)" if "group" in ctx.P else "no group column")]
    ja = ctx.joint_agreement(1)
    if ja and ja.get("error"):
        body.append(f"<div class='warn'><b>The two-route agreement was not computed.</b> "
                    f"{_esc(ja['error'])}</div>")
        ja = None
    if ja:
        d2 = ctx.joint_agreement(min(2, ctx.depth))
        d2 = d2 if (d2 and not d2.get("error")) else None
        kpi.append(("agreement, two routes", f"{ja['pct']:.1f}%",
                    "level 1" + (f" · <b>{d2['pct']:.1f}%</b> at level {min(2, ctx.depth)}"
                                 if d2 else "")
                    + f"<br>over the {ja['n_scored']:,} nuclei both routes annotated"))
    nn = ctx.neighbourhood()
    if nn is not None:
        import numpy as np
        kpi.append(("neighbourhood agreement", f"{100 * float(np.nanmean(nn)):.1f}%",
                    "mean of the per-nucleus share"))
    if ctx.has_flag:
        nf = int(ctx.P["flag"].sum())
        kpi.append(("withheld upstream", f"{nf:,}",
                    f"{100 * nf / max(ctx.n, 1):.2f}% · from "
                    f"{ctx.flag_column or 'a flag column'}"))
    body.append(_kpi(kpi))
    body.append(f"<div class='warn'>{CANNOT_SHOW}</div>")

    # ---- the common scope ----------------------------------------------------------------
    # BEFORE composition, deliberately. Composition is a table of labels, and the scope decides
    # which labels exist to be counted at all - a reader who meets the composition table first
    # has no way to tell a subtype that is absent from this tissue from one the scope removed
    # everywhere.
    body += scope_section(getattr(ctx, "scope", None), out_dir=out)

    # ---- composition, at every level ---------------------------------------------------
    body.append("<h2>Composition</h2>")
    body.append("<p class='lede'>Read this section downward. Level 1 is what a summary quotes; "
                "the deeper levels are where a compositional change actually appears, because "
                "level-1 shares can sit still while everything underneath rearranges.</p>")
    for d in ctx.levels:
        rows, csv_rows = [], []
        jl = ctx.joint_labels(d) if ctx.joint is not None else None
        for l in ctx.label_order(d):
            c = ctx.count(l, d)
            if not c:
                continue
            j = int((jl == l).sum()) if jl is not None else None
            sw = f"<span class='sw' style='background:{ctx.colour(l)}'></span>"
            rows.append([sw + _esc(leaf(l)),
                         _esc(ctx.parent_of(l)) if d > 1 else "—",
                         f"{c:,}", f"{ctx.share(l, d):.1f}%",
                         f"{j:,}" if j is not None else "—",
                         f"{ctx.animals_with(l, d)}/{len(ctx.samples)}"])
            csv_rows.append([l, ctx.parent_of(l) if d > 1 else "",
                             c, round(ctx.share(l, d), 3),
                             j if j is not None else "", ctx.animals_with(l, d)])
        src = _write_table(out, f"composition_level{d}.csv",
                           ["label", "parent", "nuclei", "share_pct", "joint_route", "samples"],
                           csv_rows)
        body.append(f"<h3>level {d}</h3>")
        body.append(_table(["label", "parent", "nuclei", "share", "joint route", "samples"],
                           rows, source=src))
        fid = "F102" if d == 1 else "F103"
        kw = {} if d == 1 else {"depth": d}
        body.append(A.fig(fid, name=f"{fid}_composition_level{d}_by_group",
                          by="group", **kw))
        body.append(A.fig(fid, name=f"{fid}_composition_level{d}_by_sample",
                          by="sample", **kw))
        pf = "F141" if d == 1 else "F143"
        body.append(A.fig(pf, name=f"{pf}_per_sample_level{d}", depth=d))

    # ---- reliability ---------------------------------------------------------------------
    rel = ctx.reliability_rows()
    body.append("<h2>Reliability</h2>")
    if not rel:
        body.append(_absent_section(
            "per-call statistics",
            f"obs carries none of {ctx.label_key.replace('_cell_type', '_depth')}, "
            f"{ctx.label_key.replace('_cell_type', '_gap')}, "
            f"{ctx.label_key.replace('_cell_type', '_support')} or "
            f"{ctx.label_key.replace('_cell_type', '_survival')}. Name the annotation with "
            f"--label-key so they are found, or re-run the annotation to write them."))
    if rel:
        rows = [[f"depth {r['depth']}", f"{r['calls']:,}",
                 f"{r['nuclei']:,} ({r['pct']:.1f}%)", _num(r["median_gap"], 2),
                 _num(r["median_support"]), _num(r["median_survival"], 2),
                 f"{r['pct_thin']:.0f}%" if r["pct_thin"] is not None else "—"]
                for r in rel]
        src = _write_table(out, "reliability_by_depth.csv",
                           ["depth", "calls", "nuclei", "pct", "median_gap", "median_support",
                            "median_survival", "pct_on_under_10_assertions"],
                           [[r["depth"], r["calls"], r["nuclei"], r["pct"], r["median_gap"],
                             r["median_support"], r["median_survival"], r["pct_thin"]]
                            for r in rel])
        body.append(_table(["depth", "calls", "nuclei", "median gap", "median support",
                            "median survival", "on <10 assertions"], rows, source=src))
        body.append("<p class='sub'><b>A deeper call is not a worse call, but it rests on less."
                    "</b> Support falls with depth while the gap does not, so a deep call can "
                    "look as confident as a shallow one on a fraction of the evidence. Every "
                    "object carries the full path, so a consumer can truncate to the depth it "
                    "can defend.</p>")
        body.append(A.fig("F140"))
        body.append(A.fig("F133"))
        worst = ctx.worst_evidence()
        if worst:
            body.append("<h3>calls won on the most depleted panels</h3>")
            body.append(_table(
                ["sample", "path", "gap", "support", "survival"],
                [[_esc(w["sample"]), f"<span class='mono'>{_esc(w['path'])}</span>",
                  _num(w["gap"], 2), _num(w["support"]), f"<b>{_num(w['survival'], 2)}</b>"]
                 for w in worst],
                source=_write_table(out, "worst_evidence.csv",
                                    ["sample", "path", "gap", "support", "survival"],
                                    [[w["sample"], w["path"], w["gap"], w["support"],
                                      w["survival"]] for w in worst])))
            body.append("<p class='sub'>Panel survival is the share of a node's own evidence "
                        "still carrying a claim after the sibling contrast. <b>A high gap on a "
                        "low survival is a confident call on evidence that better-cited "
                        "neighbours had already taken.</b></p>")
    if ctx.neighbourhood() is None:
        body.append(_absent_section(
            "neighbourhood agreement",
            "no `nn_agreement` column in obs. It is written by `scanno annotate --neighbours`; "
            "without it there is no measure of whether a label sits among neighbours carrying "
            "the same one."))
    if ctx.neighbourhood() is not None:
        body.append(A.fig("F142"))
        man = ctx.manifold_rows()
        if man:
            body.append("<h3>where the annotation leaves the manifold</h3>")
            body.append(_table(
                ["sample", "label", "nuclei", "agreement"],
                [[_esc(m["cluster"]), f"<span class='mono'>{_esc(m['label'])}</span>",
                  f"{m['nuclei']:,}", f"{m['agreement']:.1f}%"] for m in man],
                source=_write_table(out, "manifold_outliers.csv",
                                    ["sample", "label", "nuclei", "agreement"],
                                    [[m["cluster"], m["label"], m["nuclei"], m["agreement"]]
                                     for m in man])))

    # ---- the chosen resolution ------------------------------------------------------------
    body.append("<h2>The chosen clustering resolution</h2>")
    if not ctx.resolution_sweep():
        body.append(_absent_section(
            "the resolution sweep",
            "no sweep was given, or the file passed to --sweep carried no per-depth rows. "
            "`scanno resolution --out sweep.json` writes one; without it the chosen resolution "
            "is a value someone typed rather than one the evidence supports."))
    if ctx.resolution_sweep():
        body.append("<p class='lede'>The criterion is the <b>annotation</b>, not the geometry. "
                    "A partition can shift substantially while every nucleus keeps its "
                    "identity, and that is exactly when granularity does not matter. What "
                    "downstream analysis consumes is the label, so the label is what the "
                    "choice is made on.</p>")
        if ctx.sweep_pick is not None:
            body.append(f"<div class='ok'><b>Resolution {_esc(ctx.sweep_pick)}</b>"
                        + (f", decided by <b>{_esc(ctx.sweep_reason)}</b>"
                           if ctx.sweep_reason else "")
                        + (f". The tolerance the candidates were compared within is "
                           f"<b>{ctx.tolerance:.2f} points</b>, derived from the sweep's own "
                           f"step-to-step variation rather than chosen."
                           if ctx.tolerance is not None else "") + "</div>")
        for d, rws in sorted(ctx.resolution_sweep().items()):
            body.append(f"<h3>depth {d}</h3>")
            body.append(_table(
                ["resolution", "units", "modal %", "complete %", "truncated %",
                 "unresolved %", "labels", "smallest", "samples"],
                [[("<b>" + _esc(r["resolution"]) + "</b>"
                   if str(r["resolution"]) == str(ctx.sweep_pick) else _esc(r["resolution"])),
                  _num(r.get("n_units")), _num(r.get("modal"), 1), _num(r.get("complete"), 1),
                  _num(r.get("truncated"), 1), _num(r.get("unresolved"), 1),
                  _num(r.get("n_labels")), _num(r.get("smallest")),
                  _num(r.get("min_groups"))] for r in rws]))
        body.append(A.fig("F160"))
        body.append("<p class='sub'><b>modal</b> — nuclei whose label equals their majority "
                    "across the sweep. <b>complete</b> — resolved to a leaf of the taxonomy. "
                    "<b>truncated</b> — stopped at an internal node, which is a partial "
                    "identity and not the same failure as unresolved. <b>smallest</b> — nuclei "
                    "in the smallest named label. <b>samples</b> — how many samples the rarest "
                    "label appears in; a label seen in one sample of ten is not a population a "
                    "downstream comparison can use.</p>")

    # ---- markers ----------------------------------------------------------------------------
    body.append("<h2>Marker expression</h2>")
    if not ctx.panel_depths():
        body.append(_absent_section(
            "marker panels",
            "none were given. Pass --panels auto with --db and --tree to build them from the "
            "same corpus the classifier scored on, or --panels FILE. Without them nothing here "
            "checks a label against the evidence it was assigned on."))
    if ctx.panel_depths():
        body.append("<p class='lede'>The markers are taken <b>from the corpus</b> — the same "
                    "evidence the classifier scored on. A hand-picked panel would show whether "
                    "the labels match the genes someone chose to plot, which is a question "
                    "about that person. Taking them from the corpus makes the figure a check "
                    "on the call.</p>")
        if ctx.joint is not None:
            body.append(A.fig("F132"))
        for d in ctx.panel_depths():
            body.append(f"<h3>level {d}</h3>")
            body.append(A.fig("F130" if d == 1 else "F135",
                              name=f"{'F130' if d == 1 else 'F135'}_dotplot_level{d}",
                              **({} if d == 1 else {"depth": d})))
            body.append(A.fig("F131" if d == 1 else "F136",
                              name=f"{'F131' if d == 1 else 'F136'}_featureplot_level{d}",
                              **({} if d == 1 else {"depth": d})))
        for d in ({1, max(ctx.panel_depths())} if ctx.panel_depths() else {1}):
            b = ctx.marker_breadth(d)
            if not b:
                continue
            body.append(A.fig("F134", name=f"F134_breadth_level{d}", depth=d))
            src = _write_table(out, f"marker_breadth_level{d}.csv",
                               ["gene", "plotted_for", "own", "best_other", "overall",
                                "labels_over_25pct"],
                               [[r["gene"], r["plotted_for"], round(r["own"], 4),
                                 round(r["best_other"], 4), round(r["overall"], 4),
                                 r["labels_over_25"]] for r in b])
            broad = [r for r in b if r["labels_over_25"] >= 4]
            if broad:
                body.append(f"<h3>genes that carry no identity information at level {d}</h3>")
                body.append(_table(
                    ["gene", "plotted for", "own label", "best other", "overall", "labels ≥25%"],
                    [[f"<span class='mono'>{_esc(r['gene'])}</span>", _esc(leaf(r["plotted_for"])),
                      f"{100 * r['own']:.1f}%", f"{100 * r['best_other']:.1f}%",
                      f"{100 * r['overall']:.1f}%", r["labels_over_25"]] for r in broad],
                    source=src))
                body.append("<p class='sub'><b>Nothing has been removed from the panel.</b> "
                            "These genes are detected in at least 25% of nuclei in four or more "
                            "labels; discount their columns in the dotplot. They have not been "
                            "taken out of it, because a curated grid gives a reader no way to "
                            "know it was curated.</p>")

    # ---- the withheld nuclei -----------------------------------------------------------------
    body.append("<h2>The nuclei withheld before annotation</h2>")
    if not ctx.has_flag:
        body.append(_absent_section(
            "an upstream flag",
            "no flag column was found, so nothing was withheld before annotation - or the "
            "column was not named. Pass --flag-key, or annotate objects carrying an upstream "
            "provenance declaration."))
    if ctx.has_flag:
        nf = int(ctx.P["flag"].sum())
        body.append(f"<p class='lede'>This section is about <b>{nf:,} nuclei and nothing "
                    f"else</b>: where they sit, what they express, and how they differ from the "
                    f"{ctx.n - nf:,} that were kept. scAnno is an annotation tool, not QC — "
                    f"<b>every withheld nucleus comes from the upstream flag "
                    f"<code>{_esc(ctx.flag_column)}</code></b>, and this tool has no code that "
                    f"can widen one.</p>")
        body.append(A.fig("F150"))
        body.append(A.fig("F155"))
        body.append(A.fig("F151"))
        sig = ctx.exclusion_signature()
        if sig:
            hdr = ["gene", "Δ mean", "Δ detection", "detected, withheld", "detected, kept",
                   "samples agreeing"]
            def sig_rows(rs):
                return [[f"<span class='mono'>{_esc(r['gene'])}</span>",
                         f"{r['d_mean']:+.3f}", f"{r['d_detect']:+.1f} pp",
                         f"{100 * r['det_flagged']:.1f}%", f"{100 * r['det_kept']:.1f}%",
                         (f"{r['animals_agree']}/{r['animals_compared']}"
                          if r.get("animals_compared") else "—")] for r in rs]
            src = _write_table(out, "exclusion_signature.csv",
                               ["gene", "d_mean", "d_detect_pp", "det_withheld", "det_kept",
                                "samples_agree", "samples_compared"],
                               [[r["gene"], round(r["d_mean"], 5), round(r["d_detect"], 3),
                                 round(r["det_flagged"], 5), round(r["det_kept"], 5),
                                 r["animals_agree"], r.get("animals_compared", "")]
                                for r in sig])
            body.append("<h3>elevated in the withheld nuclei</h3>")
            body.append(_table(hdr, sig_rows(sig[:15]), source=src))
            body.append("<h3>depleted</h3>")
            body.append(_table(hdr, sig_rows(sig[-15:][::-1]), source=src))
        body.append(A.fig("F152"))
        cov = ctx.flag_per_animal()
        n_used = ctx.n_compare_animals
        if cov:
            if n_used < len(cov):
                body.append(
                    f"<div class='warn'><b>{n_used} of {len(cov)} samples</b> hold at least "
                    f"{MIN_FLAGGED_PER_ANIMAL} withheld nuclei, the floor below which a "
                    f"per-sample flagged-against-kept comparison is noise. "
                    f"The samples this omits <b>are not necessarily evenly spread across the "
                    f"design</b>, so the cross-sample check is stronger for some arms than "
                    f"others — read it as consistency among the samples that could be checked, "
                    f"never as consistency across the cohort.</div>")
            body.append(_table(
                ["sample", "arm", "nuclei", "withheld", "rate", "in the per-sample comparison"],
                [[_esc(r["animal"]), _esc(r["arm"]) or "—", f"{r['nuclei']:,}",
                  f"{r['flagged']:,}", f"{r['rate']:.2f}%",
                  "yes" if r["in_comparison"] else "<b>no</b>"] for r in cov],
                source=_write_table(out, "exclusion_per_sample.csv",
                                    ["sample", "arm", "nuclei", "withheld", "rate_pct",
                                     "in_comparison"],
                                    [[r["animal"], r["arm"], r["nuclei"], r["flagged"],
                                      r["rate"], r["in_comparison"]] for r in cov])))
        body.append(A.fig("F153"))
        body.append(A.fig("F154"))
        facs = ctx.flag_by_factor()
        if facs:
            body.append(_table(
                ["factor", "rate per level", "ratio"],
                [[_esc(f["factor"]) + (" <span class='sub'>(auto-detected)</span>"
                                       if f["auto"] else ""),
                  " · ".join(f"{k} {v:.2f}%" for k, v in f["rates"].items()),
                  (f"<b style='color:var(--badl)'>{f['ratio']:.2f}×</b>"
                   if f["ratio"] > 3 else f"<b>{f['ratio']:.2f}×</b>")] for f in facs],
                source=_write_table(out, "exclusion_by_factor.csv",
                                    ["factor", "levels", "lo_pct", "hi_pct", "ratio",
                                     "auto_detected"],
                                    [[f["factor"], json.dumps(f["rates"]), f["lo"], f["hi"],
                                      f["ratio"], f["auto"]] for f in facs])))
            body.append("<p class='sub'>A removal whose rate differs more than <b>3×</b> "
                        "between arms of a design factor has converted a technical property "
                        "into an apparent biological difference, and no downstream analysis can "
                        "undo it. Factors marked auto-detected were inferred from low-"
                        "cardinality obs columns, not declared.</p>")

    # ---- per-sample index ---------------------------------------------------------------------
    if sample_links:
        body.append("<h2>The samples</h2>")
        body.append("<p class='sub'>One comprehensive page per sample. Open one when a cohort "
                    "number looks wrong — everything about that sample is on its page.</p>")
        body.append("<div class='samples'>" + "".join(
            f'<a href="{_esc(href)}">{_esc(s)}<span class="n">'
            f'{int((ctx.P["sample"] == s).sum()):,} nuclei</span></a>'
            for s, href in sample_links) + "</div>")

    body.append(f"<h2>What this cannot show</h2><div class='warn'>{CANNOT_SHOW}</div>")

    # ---- provenance ------------------------------------------------------------------------
    prov = [["objects", f"<span class='mono'>{len(ctx.samples)} annotated object(s)</span>"],
            ["label column", f"<span class='mono'>{_esc(ctx.path_key)}</span>"],
            ["taxonomy depth", f"{ctx.depth}"]]
    if ctx.tree_path:
        prov.append(["taxonomy", f"<span class='mono'>{_esc(ctx.tree_path)}</span>"])
    if ctx.corpus_path:
        prov.append(["corpus", f"<span class='mono'>{_esc(ctx.corpus_path)}</span>"])
    if ctx.flag_column:
        prov.append(["the flag", f"<span class='mono'>obs[{_esc(ctx.flag_column)!r}]</span>"])
    if ctx.declaration:
        prov.append(["upstream declaration",
                     f"<span class='mono'>{_esc(json.dumps(ctx.declaration)[:200])}</span>"])
    if version:
        prov.append(["scAnno", f"<span class='mono'>{_esc(version)}</span>"])
    body.append("<h2>Provenance</h2>" + _table(["item", "value"], prov))

    clash = ctx.palette.collisions(ctx.label_order(ctx.depth) + ctx.label_order(1))
    if clash:
        body.append("<div class='bad'><b>Two labels share a colour.</b> "
                    + "; ".join(f"{_esc(a)} and {_esc(b)} are both {_esc(c)}"
                                for a, b, c in clash[:6])
                    + ". Pin them with <code>--palette</code>.</div>")
    body.append("<h3>colours</h3>" + _table(
        ["label", "colour"],
        [[f"<span class='sw' style='background:{v}'></span>{_esc(k)}",
          f"<span class='mono'>{_esc(v)}</span>"]
         for k, v in sorted(ctx.palette.as_dict().items())]))

    if A.failed:
        body.append("<h2>Figures that FAILED — defects in scAnno</h2>")
        body.append("<div class='bad'><b>These are not findings about your data.</b> Each is an "
                    "error inside scAnno that prevented a figure being drawn. Nothing here "
                    "should be read as an absence of signal.</div>")
        body.append(_table(["figure", "error"],
                           [[_esc(a["name"]), _esc(a["reason"])] for a in A.failed]))
    if A.absent:
        body.append("<h2>Figures that could not be drawn</h2>")
        body.append(_table(["figure", "the input it needed"],
                           [[_esc(a["name"]), _esc(a["reason"])] for a in A.absent]))
        body.append("<p class='sub'>Named rather than omitted: a figure that vanishes silently "
                    "reads as a finding that there was nothing to show. These are gaps in the "
                    "INPUT; anything that broke is listed separately above.</p>")

    path = out / "reports" / "cohort.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_page(f"{title} — cohort", "".join(body)), encoding="utf-8")
    return path, A


# =============================================================== the per-sample document

def write_sample(ctx, sample, out_dir, *, title="Annotation", version="", cohort_href=None):
    out = Path(out_dir)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(sample))
    A = Assembler(ctx, out, rel=f"../../figures/samples/{safe}", subdir=f"samples/{safe}")
    rows = ctx.sample_rows(sample)
    n = len(rows)
    D = ctx.sample_depth(sample)
    now = datetime.now(timezone.utc).astimezone()
    grp = str(rows["group"].iloc[0]) if "group" in rows and n else ""

    body = [f"<h1>{_esc(sample)}</h1>",
            f"<p class='sub'>{n:,} nuclei"
            + (f" · {_esc(grp)}" if grp else "")
            + f" · reaches level {D} of {ctx.depth}"
            + (f" · resolution {_esc(ctx.chosen_resolution)}" if ctx.chosen_resolution else "")
            + f" · generated {now:%Y-%m-%d %H:%M %Z}</p>"]
    if cohort_href:
        body.append(f"<p class='sub'>← <a href=\"{_esc(cohort_href)}\">back to the cohort "
                    f"report</a>, which is the document; this page is the detail behind one "
                    f"of its rows.</p>")

    kpi = [("nuclei", f"{n:,}", f"{100 * n / max(ctx.n, 1):.1f}% of the cohort")]
    lab1 = rows["L1"]
    kpi.append(("level-1 labels", f"{lab1.nunique()}",
                f"deepest level reached: {D}"))
    if "flag" in rows:
        nf = int(rows["flag"].sum())
        kpi.append(("withheld", f"{nf:,}", f"{100 * nf / max(n, 1):.2f}% of this sample"))
    if "nn_agreement" in rows and rows["nn_agreement"].notna().any():
        import numpy as np
        kpi.append(("neighbourhood agreement",
                    f"{100 * float(np.nanmean(rows['nn_agreement'])):.1f}%",
                    "mean of the per-nucleus share"))
    body.append(_kpi(kpi))

    body.append("<h2>The annotation</h2>")
    body.append(A.fig("F101", name=f"F101_{safe}_by_level", sample=sample))
    body.append(A.fig("F104", name=f"F104_{safe}_clusters", sample=sample))
    body.append("<h2>What granularity cost</h2>")
    body.append(A.fig("F100", name=f"F100_{safe}_by_resolution", sample=sample))
    body.append("<h2>Quality</h2>")
    body.append(A.fig("F105", name=f"F105_{safe}_qc", sample=sample))

    body.append("<h2>Composition, this sample against the cohort</h2>")
    for d in range(1, D + 1):
        col = rows[f"L{d}"]
        tab = []
        for l in ctx.label_order(d):
            c = int((col == l).sum())
            if not c:
                continue
            here, coh = 100.0 * c / max(n, 1), ctx.share(l, d)
            tab.append([f"<span class='sw' style='background:{ctx.colour(l)}'></span>"
                        + _esc(leaf(l)), f"{c:,}", f"{here:.1f}%", f"{coh:.1f}%",
                        f"{here - coh:+.1f} pp"])
        body.append(f"<h3>level {d}</h3>")
        body.append(_table(["label", "nuclei", "this sample", "cohort", "difference"], tab))
    body.append("<p class='sub'>The last column is this sample minus the cohort. A large "
                "difference is not by itself a finding — read it against the per-sample points "
                "in the cohort report's composition figures, which show the spread the rest of "
                "the samples occupy.</p>")

    calls = ctx.calls_at_chosen()
    if calls is not None and len(calls):
        mine = calls[calls["sample"] == sample]
        if len(mine):
            mine = mine.sort_values("n_cells", ascending=False)
            body.append("<h2>The calls</h2>")
            body.append(_table(
                ["path", "nuclei", "depth", "gap", "support", "survival"],
                [[f"<span class='mono'>{_esc(r['path'])}</span>", f"{int(r['n_cells']):,}",
                  int(r["depth"]), _num(r["gap"], 2), _num(r["support"]),
                  _num(r["survival"], 2)] for _, r in mine.iterrows()],
                source=_write_table(out, f"calls_{safe}.csv",
                                    ["path", "n_cells", "depth", "gap", "support", "survival"],
                                    [[r["path"], r["n_cells"], r["depth"], r["gap"],
                                      r["support"], r["survival"]]
                                     for _, r in mine.iterrows()])))

    body.append(f"<h2>What this cannot show</h2><div class='warn'>{CANNOT_SHOW}</div>")
    if A.absent:
        body.append("<h2>Figures that could not be drawn</h2>")
        body.append(_table(["figure", "why"],
                           [[_esc(a["name"]), _esc(a["reason"])] for a in A.absent]))

    path = out / "reports" / "samples" / f"{safe}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_page(f"{sample} — {title}", "".join(body)), encoding="utf-8")
    return path, A


# =============================================================== the whole delivery

def write_all(ctx, out_dir, *, title="Annotation", version="", per_sample=True):
    """Both documents, the figures and the tables, under one output directory.

    Returns the payload that is also written as `report.json` - every number the pages carry, so
    a consumer never has to scrape HTML to get at them.
    """
    out = Path(out_dir)
    links, absent, failed = [], [], []
    if per_sample:
        for s in ctx.samples:
            p, A = write_sample(ctx, s, out, title=title, version=version,
                                cohort_href="../cohort.html")
            links.append((s, f"samples/{p.name}"))
            absent += A.absent
            failed += A.failed
    cohort, CA = write_cohort(ctx, out, title=title, version=version, sample_links=links)
    absent += CA.absent
    failed += CA.failed

    payload = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "scanno_version": version,
        "n_nuclei": int(ctx.n), "n_samples": len(ctx.samples),
        "taxonomy_depth": int(ctx.depth),
        "label_key": ctx.path_key, "flag_column": ctx.flag_column,
        "chosen_resolution": ctx.chosen_resolution,
        "composition": {d: [{"label": l, "nuclei": ctx.count(l, d),
                             "share_pct": round(ctx.share(l, d), 4),
                             "samples": ctx.animals_with(l, d)}
                            for l in ctx.label_order(d) if ctx.count(l, d)]
                        for d in ctx.levels},
        "reliability": ctx.reliability_rows(),
        "worst_evidence": ctx.worst_evidence(),
        "joint_agreement": {d: ctx.joint_agreement(d) for d in ctx.levels}
                           if ctx.joint is not None else None,
        "withheld": ({"n": int(ctx.P["flag"].sum()),
                      "per_sample": ctx.flag_per_animal(),
                      "by_factor": ctx.flag_by_factor(),
                      "identity": ctx.flag_identity(1)} if ctx.has_flag else None),
        "resolution_sweep": ctx.resolution_sweep() or None,
        # The scope travels into report.json UNCHANGED. A consumer asking "which splits was this
        # cohort annotated against" must get the decision itself, not this document's rendering
        # of it, and must be able to diff it against the scope.json the run was given.
        "scope": getattr(ctx, "scope", None),
        "palette": ctx.palette.as_dict(),
        "figures_not_drawn": absent,
        "figures_failed": failed,
        "reports": {"cohort": str(cohort.relative_to(out)),
                    "samples": [h for _, h in links]},
    }
    (out / "report.json").write_text(json.dumps(payload, indent=1, default=str),
                                     encoding="utf-8")
    return cohort, payload
