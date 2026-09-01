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

from .context import MIN_FLAGGED_PER_ANIMAL, SENTINELS
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
tr.band td{background:var(--card);font-weight:600;border-top:2px solid var(--line)}
tr.guide td{color:var(--mut)}
.tw{display:inline-block;min-width:1.1em}
.why{font-size:.7rem;text-transform:uppercase;letter-spacing:.04em;padding:.1rem .4rem;
border-radius:3px;border:1px solid var(--line);color:var(--mut);white-space:nowrap}
.why.sealed,.why.stranded{color:var(--badl);border-color:var(--badl)}
.why.leaf{color:var(--okl);border-color:var(--okl)}
.why.unvotable{color:var(--warnl);border-color:var(--warnl)}
.bar{display:inline-block;height:.42rem;border-radius:2px;vertical-align:middle;
margin-left:.45rem;min-width:1px}
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
        # THE TITLE NAMES WHAT WAS DRAWN. The registry holds the figure's KIND - "composition",
        # "composition per sample" - and the caller supplies the subject, because one kind is
        # drawn over several different columns. Baking the subject into the registry put
        # "composition, forced" over a figure of the JOINT ROUTE's column: the picture and the
        # words beside it named different annotations, and nothing on the page said so.
        title = FIGURES[fid][2]
        if kw.get("what"):
            title = f"{title} — {kw['what']}"
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
    "<b>A FORCE is not a seal and removes nothing.</b> It reassigns nuclei that stopped on an "
    "open node to the child they already scored highest, so it changes which label those nuclei "
    "carry and never which labels exist. The cells it moves are the ones whose evidence was "
    "thinnest, so they are the least certain calls in the deliverable, not the most. "
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
    "SEAL": "the node becomes a LEAF; its ENTIRE child set is removed from the tree",
    "FORCE": "the split STAYS, but nothing may terminate here — each nucleus left on the node "
             "is pushed to its most similar child",
    "UNVOTABLE": "too few samples reached it to vote — reported, NOT sealed",
    "KEEP": "the split stays; every sample that reached it agreed to make it",
    "UNREACHED": "no sample reached it at all — left in the tree, holds nothing",
}

#: SEAL and FORCE first: they are the two verdicts that CHANGE something, and a reader scanning
#: the table for what the vote did should not have to find them among the ones that did nothing.
_VERDICT_ORDER = {"SEAL": 0, "FORCE": 1, "UNVOTABLE": 2, "KEEP": 3, "UNREACHED": 4}

#: The verdicts a reader must not skim past. Rendered in the alert colour.
_VERDICT_LOUD = ("SEAL", "FORCE", "UNVOTABLE")


def _scope_forced(v):
    """Nuclei a FORCE node must reassign, and in how many samples. From the vote's own `stranded`.

    Read from `stranded` rather than recomputed from `cells - cells_below`, because `stranded` is
    what the VERDICT was decided on: a node is FORCE precisely when that dict is non-empty, so
    reporting a different number here could contradict the verdict beside it.
    """
    st = v.get("stranded") or {}
    try:
        return sum(int(x) for x in st.values()), len(st)
    except (TypeError, ValueError):
        return 0, 0


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
    forced = [n for n, v in nodes.items() if v.get("verdict") == "FORCE"]
    n_lost = sum(sum(d.values()) for d in removed.values())
    lost_labels = sorted({p for d in removed.values() for p in d})
    n_forced = sum(_scope_forced(nodes[n])[0] for n in forced)

    # "Each sample walks once" rather than a number: the cohort size is `n_samples`, read from the
    # JSON. A sentence that says "ten" is correct for one cohort and silently wrong for the next,
    # and it is the same defect as the literal `/10` that shipped in `format_report`.
    body.append(
        f"<p class='lede'>Each sample walks independently, so a cohort produces as many scopes as "
        f"it has samples: on a lineage where the sibling contrast sits near the bar, one sample "
        f"descends and the next truncates, so the same cells get a subtype in one library and its "
        f"parent in another — which downstream is indistinguishable from a compositional shift. "
        f"The scope decides the depth <b>once</b>, from what the samples agree on, and every "
        f"sample is annotated against it. This section is that decision: the rule it was made "
        f"under, the tree it produced, and the labels it costs.</p>")

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
        ("nodes forced", f"{len(forced)}",
         "split kept, but nothing may terminate there"),
        ("nuclei a FORCE reassigns", f"{n_forced:,}",
         "pushed to their most similar child; nothing removed"),
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
        "the cohort size: the same numerator over a partial denominator and over the whole "
        "cohort are different statements, and the <i>reached</i> column below shows which one "
        "you are reading.</li>")
    words.append(
        "<li><b>An open node may not be a terminal label.</b> A node that keeps its children is "
        "a branch, not a call — a nucleus left sitting on it would carry the name of a "
        "<i>compartment</i>, which is the same string the independent L1 column uses for every "
        "nucleus beneath it, so the two delivered columns would disagree about what that word "
        "means. Such a node is marked <b>FORCE</b>: the split stands, and each stranded nucleus "
        "is pushed to the child it already scored highest. This is <b>not</b> a seal — nothing "
        "is removed — and <b>not</b> a truncation, because nothing terminates there.</li>")
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
                 if verdict in _VERDICT_LOUD else _esc(verdict))
        fn, fs = _scope_forced(v)
        rows.append([f"<span class='mono'>{_esc(node)}</span>", badge,
                     f"{nr}/{n_samples}" if n_samples else f"{nr}",
                     f"{int(v.get('n_descended') or 0)}", _scope_support(v),
                     f"{stopped:,}" + (f" <span class='sub'>({stopped_n} sample(s))</span>"
                                       if stopped else ""),
                     (f"<b>{fn:,}</b>" if verdict == "FORCE" and fn else "—")])
        csv_rows.append([node, verdict, nr, n_samples or "", int(v.get("n_descended") or 0),
                         v.get("support"), stopped, stopped_n, fn, fs,
                         ";".join(v.get("children_declared") or [])])
    src = (_write_table(out_dir, "scope_nodes.csv",
                        ["node", "verdict", "n_reached", "n_samples", "n_descended", "support",
                         "nuclei_stopped_here_pass1", "samples_with_any_stopped",
                         "nuclei_forced_to_a_child", "samples_with_any_forced",
                         "children_declared"], csv_rows)
           if out_dir is not None else "the scope JSON, key `nodes`")
    body.append("<h3>the vote, node by node</h3>")
    body.append(_table(["node", "verdict", "reached", "descended", "support",
                        "stopped here (pass 1)", "forced to a child"], rows, source=src))
    body.append("<p class='sub'>"
                + " · ".join(f"<b>{k}</b> {v}" for k, v in _VERDICT_WORDS.items())
                + ". <i>stopped here</i> counts nuclei that arrived at the node in <b>pass 1</b> "
                "and did not go below it, summed over samples — at a SEALED node every arriving "
                "nucleus stops there in pass 2 by construction, so that column describes what "
                "the vote SAW, not what pass 2 produces. <i>forced to a child</i> is the same "
                "residual at a node that stayed OPEN, which is the one case where pass 2 moves "
                "a nucleus rather than truncating it.</p>")

    # ---- what each FORCE reassigns ------------------------------------------------------------
    # Its own subsection rather than a column note, because a FORCE and a SEAL are opposite
    # operations that both read as "the vote did something here", and a reader who conflates them
    # concludes that a label was removed when a label was in fact re-assigned.
    body.append("<h3>what each FORCE reassigns — nuclei moved, never labels removed</h3>")
    if not forced:
        body.append(
            "<p class='sub'>No node was forced: every node that kept its children had every "
            "arriving nucleus descend below it, so no nucleus was left carrying a compartment "
            "name as its final label and pass 2 moved nothing.</p>")
    else:
        frows, fcsv = [], []
        for node in sorted(forced, key=lambda n: -_scope_forced(nodes[n])[0]):
            v = nodes[node]
            fn, fs = _scope_forced(v)
            st = v.get("stranded") or {}
            worst = sorted(st.items(), key=lambda kv: -int(kv[1]))
            frows.append([
                f"<span class='mono'>{_esc(node)}</span>",
                f"<b>{fn:,}</b>",
                f"{fs}/{n_samples}" if n_samples else f"{fs}",
                ", ".join(f"{_esc(s)} <span class='sub'>({int(c):,})</span>"
                          for s, c in worst[:4]) + (" …" if len(worst) > 4 else ""),
                ", ".join(_esc(c) for c in (v.get("children_declared") or [])) or "—"])
            fcsv.append([node, fn, fs, ";".join(f"{s}={int(c)}" for s, c in worst),
                         ";".join(v.get("children_declared") or [])])
        fsrc = (_write_table(out_dir, "scope_forced_nodes.csv",
                             ["node", "nuclei_forced", "n_samples_with_any",
                              "per_sample", "children_declared"], fcsv)
                if out_dir is not None else "the scope JSON, key `nodes`, sub-key `stranded`")
        body.append(_table(["forced node", "nuclei reassigned", "samples", "which samples",
                            "candidate children"], frows, source=fsrc))
        body.append(
            f"<p class='sub'>Each of those {n_forced:,} nuclei is assigned the child it already "
            f"scored highest at that node, so the reassignment needs no new scoring and no change "
            f"to the walk — it reads the choice the walk had already made and declined to act on "
            f"because the margin was thin. <b>The margin is why these are the least certain calls "
            f"in the deliverable</b>: every one of them is a cell whose gap failed where the rest "
            f"of the cohort's cleared. They are concentrated rather than spread — the "
            f"<i>samples</i> column shows how few libraries produced them — so a subtype's count "
            f"can be moved appreciably in one sample and not at all in another, which is a "
            f"per-sample effect and not a compositional one.</p>")

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
            f"never described as a category — \"the subtypes under that node\" is not "
            f"assessable, and a list naming each label with its count is. Rows at 0 are children "
            f"the seal removed from the tree that no sample had reached.</p>")

    return body


# ================================================= the delivered tree, with L1 integrated

#: The bound on the section below, in the same voice as the others.
TAXONOMY_CANNOT_SHOW = (
    "<b>This is the annotation that was DELIVERED, not evidence that it is right.</b> No label "
    "here is established as correct: there is no truth set for this tissue, and the tool reports "
    "what the corpus supports, not what is true. "
    "<b>The two columns are two walks, and their agreement is a measurement.</b> The compartment "
    "column is an independent walk against a depth-1 tree; the taxonomy column is the deep walk "
    "against the scope. Nothing forces them to agree, so a concordance of 100% is a result about "
    "this cohort and not a property of the format — and where they disagree, BOTH are shown "
    "rather than one being silently preferred. "
    "<b>It cannot tell you why a branch stopped, only which of the reasons applies.</b> "
    "<i>leaf</i> points at the declared tree, <i>sealed</i> at the cohort vote, <i>stranded</i> "
    "at one nucleus's own gap, and the remedy differs completely: extend the taxonomy, re-run "
    "against the declared tree, or improve the evidence. A reader who reads them all as \"this "
    "is as deep as the biology goes\" is wrong in three different ways. "
    "<b>Counts are nuclei, not cells and not animals.</b> A label carried by one sample and one "
    "spread across the cohort read identically in the nuclei column; the <i>samples</i> column is "
    "the only place that shows, and a label in few samples cannot support a compositional claim "
    "however many nuclei it has.")

#: How each terminal reason is phrased on the page, and what a reader should do about it.
_WHY_WORDS = {
    "leaf": "the declared taxonomy has nothing below this — the call is complete",
    "sealed": "the cohort removed this node's children; the subtype is recoverable by re-running "
              "against the declared tree",
    "stranded": "the node stayed OPEN and this nucleus's gap failed anyway — thin evidence, not "
                "a removal",
    "unvotable": "too few samples reached it to vote, so it was never sealed; this nucleus "
                 "stopped on its own gap",
    "sentinel": "not a cell type — withheld upstream, or never resolved",
}


def taxonomy_section(ctx, out_dir=None):
    """The delivered cell-type tree with the independent L1 compartment integrated.

    WHAT THIS ANSWERS THAT THE COMPOSITION TABLES DO NOT

    The deliverable is TWO label columns: an independent level-1 compartment, and the
    scope-based label from the deep walk. A reader who wants to know "what is this cell at L1,
    and how far down did the scope let us go" currently has to read one table of paths, hold the
    taxonomy in their head, and cross-reference a second table for the compartment. This section
    is that question answered in one picture.

    WHY A BANDED, INDENTED TABLE RATHER THAN A DRAWN TREE

    A `pre` drawing (which the scope section uses, correctly, because there it is reproducing the
    vote's own output verbatim) puts the numbers inside the picture, where they cannot be aligned,
    sorted or read down a column, and it wraps once the taxonomy is wide. The delivered tree has
    to carry four numbers per row and stay readable when a lineage is deep, so:

      - the INDEPENDENT L1 compartment is the BAND, not a column. It is the coarsest partition
        and the one a biologist navigates by, so it becomes the thing you scan for; and because
        a band is defined by the independent column while its rows are defined by the deep one,
        a disagreement between the two shows up as a row whose path does not begin with its own
        band's name — visible without a separate table.
      - the taxonomy is an INDENT inside the band, computed from the path's own depth, so the
        layout has no idea how many levels exist and gains none when a fourth is added.
      - intermediate nodes nothing terminates at are kept as dimmed GUIDE rows. Dropping them
        would flatten a deep branch into a list of unrelated names, which is exactly the reading
        error this section exists to prevent.
      - siblings are ordered by size within their parent, so the eye meets the populations that
        matter first at every level, and the ordering rule is the same one the composition
        figures use.

    Returns a list of HTML chunks; never empty. Without the independent column the section is a
    NAMED ABSENCE, because a silently missing section reads as a cohort that had no L1.
    """
    body = ["<h2>The delivered annotation — the cell-type tree, with L1 integrated</h2>"]
    if not getattr(ctx, "has_l1", False):
        body.append(_absent_section(
            "the INDEPENDENT level-1 column, named with `--l1-key`",
            "The deliverable is two label columns and this section shows them together. Without "
            "the independent one there is only the deep walk's own path, whose level-1 prefix "
            "agrees with it by construction — showing that beside it would be showing one column "
            "twice and calling it agreement. Annotate with `scanno annotate --l1-tree` and name "
            "the resulting column with `--l1-key`."))
        return body

    bands = ctx.l1_bands()
    con = ctx.l1_concordance()
    scope = getattr(ctx, "scope", None) or {}
    verdicts = scope.get("nodes") or {}
    tree = getattr(ctx, "tree", None) or {}

    from .scope import why_terminal

    sent = set(SENTINELS)
    real = [b for b in bands if b["l1"] not in sent]
    tail = [b for b in bands if b["l1"] in sent]
    n_terminal = sum(1 for b in real for r in b["rows"]
                     if r["n_here"] and r["path"] not in sent)

    body.append(
        "<p class='lede'>Two columns were delivered for every nucleus: an <b>independent</b> "
        "level-1 compartment, walked against a depth-1 tree of its own, and the <b>scope-based</b> "
        "label from the deep walk. They are shown together here because they answer different "
        "halves of one question — <i>what is this cell</i>, and <i>how far down did the cohort "
        "let us go</i> — and reading either alone gives a confident answer to half of it. The "
        "compartment is the band; the taxonomy beneath it is what the scope permitted.</p>")

    kpi = [("compartments (independent L1)", f"{len(real)}",
            f"from <code>obs[{_esc(repr(ctx.l1_key))}]</code>"),
           ("terminal labels delivered", f"{n_terminal}",
            f"across {ctx.depth} level(s) of taxonomy"),
           ("deepest level reached", f"{ctx.depth}",
            "read from the labels, never assumed")]
    if con:
        kpi.append(("the two columns agree", f"{con['pct']:.2f}%",
                    f"on {con['n_scored']:,} nuclei · "
                    + (f"<b>{con['n_disagree']:,}</b> disagree"
                       if con["n_disagree"] else "none disagree")))
    body.append(_kpi(kpi))
    body.append(f"<div class='warn'>{TAXONOMY_CANNOT_SHOW}</div>")

    # ---- do the two delivered columns agree? -------------------------------------------------
    body.append("<h3>do the two delivered columns agree about the compartment?</h3>")
    if con and not con["n_disagree"]:
        body.append(
            f"<p class='sub'>The independent walk and the deep walk's own root return the "
            f"<b>same compartment for all {con['n_scored']:,} nuclei</b> scored. That is what an "
            f"unchanged root decision predicts, and it is <b>measured here rather than assumed</b> "
            f"— the two columns are separate walks and nothing in the pipeline constrains them to "
            f"match. Because they do, every band below contains exactly one lineage and the "
            f"compartment column can be read as a strict parent of the taxonomy column. Had they "
            f"differed, the bands would not partition the tree and this paragraph would say so.</p>")
    elif con:
        drows = [[f"<span class='mono'>{_esc(a)}</span>", f"<span class='mono'>{_esc(b)}</span>",
                  f"{n:,}", f"{100.0 * n / max(con['n_scored'], 1):.3f}%"]
                 for (a, b), n in con["pairs"].items()]
        dsrc = (_write_table(out_dir, "l1_concordance.csv",
                             ["deep_walk_root", "independent_l1", "nuclei", "pct_of_scored"],
                             [[a, b, n, round(100.0 * n / max(con["n_scored"], 1), 4)]
                              for (a, b), n in con["pairs"].items()])
                if out_dir is not None else "the two label columns in obs")
        body.append(
            f"<p class='sub'>The two columns disagree about the compartment for "
            f"<b>{con['n_disagree']:,}</b> of {con['n_scored']:,} nuclei "
            f"({100.0 - con['pct']:.3f}%). <b>Neither is corrected to match the other.</b> Those "
            f"nuclei appear in the band their INDEPENDENT column names, carrying the full "
            f"scope-based path they were actually given — so a row whose path does not begin with "
            f"its band's name is one of these, and is visible in the tree below without needing "
            f"this table.</p>")
        body.append(_table(["deep walk's root", "independent L1", "nuclei", "share of scored"],
                           drows, source=dsrc))

    # ---- the tree ----------------------------------------------------------------------------
    body.append("<h3>the tree as delivered — compartment, then everything the scope allowed</h3>")
    rows, csv_rows = [], []
    total = max(int(ctx.n), 1)
    for b in real + tail:
        is_sent = b["l1"] in sent
        sw = f"<span class='sw' style='background:{ctx.colour(b['l1'])}'></span>"
        rows.append([
            f"<tr class='band'><td>{sw}<b>{_esc(b['l1'])}</b>"
            + ("  <span class='why sentinel'>not a cell type</span>" if is_sent else "")
            + "</td>",
            "<td>—</td>",
            "<td>—</td>",
            f"<td><b>{b['n']:,}</b>"
            f"<span class='bar' style='width:{60.0 * b['n'] / total:.2f}rem;"
            f"background:{ctx.colour(b['l1'])}'></span></td>",
            f"<td>{100.0 * b['n'] / total:.2f}%</td>",
            f"<td>{b['samples']}/{len(ctx.samples)}</td></tr>"])
        csv_rows.append([b["l1"], "", "", 0, b["n"], b["n"], round(100.0 * b["n"] / total, 4),
                         b["samples"], "compartment"])
        for r in b["rows"]:
            term = bool(r["n_here"])
            why = why_terminal(r["path"], tree, verdicts) if term else ""
            # The path is shown in full whenever it does not simply extend its band, so a
            # disagreement between the two delivered columns is legible in place.
            shows_path = not str(r["path"]).startswith(b["l1"])
            name = (f"<span class='mono'>{_esc(r['path'])}</span>" if shows_path
                    else _esc(r["label"]))
            glyph = "└ " if r["depth"] > 1 else ""
            cls = "" if term else " class='guide'"
            rows.append([
                f"<tr{cls}><td><span style='padding-left:{1.15 * (r['depth'] - 1):.2f}rem'></span>"
                f"<span class='tw'>{glyph}</span>{name}</td>",
                (f"<td><span class='why {why}'>{_esc(why)}</span></td>" if term
                 else "<td><span class='sub'>branch</span></td>"),
                (f"<td>{r['n_here']:,}</td>" if term else "<td>—</td>"),
                f"<td>{r['n_below']:,}"
                f"<span class='bar' style='width:{60.0 * r['n_below'] / total:.2f}rem;"
                f"background:{ctx.colour(r['path'])}'></span></td>",
                f"<td>{100.0 * r['n_below'] / total:.2f}%</td>",
                f"<td>{r['samples']}/{len(ctx.samples)}</td></tr>" if term
                else "<td>—</td></tr>"])
            csv_rows.append([b["l1"], r["path"], r["label"], r["depth"], r["n_here"],
                             r["n_below"], round(100.0 * r["n_below"] / total, 4),
                             r["samples"] if term else "", why or "branch"])

    src = (_write_table(out_dir, "delivered_tree_with_l1.csv",
                        ["independent_l1", "scope_path", "label", "depth", "nuclei_terminal",
                         "nuclei_at_or_below", "pct_of_cohort", "samples", "stops_because"],
                        csv_rows)
           if out_dir is not None else "the two label columns in obs")
    head = "".join(f"<th>{_esc(x)}</th>" for x in
                   ["compartment / cell type", "stops here because", "nuclei here",
                    "at or below", "share", "samples"])
    body.append(f"<div class='scroll'><table><tr>{head}</tr>"
                + "".join("".join(r) for r in rows)
                + f"</table></div><p class='src'>source: <code>{_esc(src)}</code></p>")

    body.append(
        "<p class='sub'><b>Bold rows are compartments</b> — the independent L1 column. Indented "
        "rows beneath them are the scope-based taxonomy, one level of indent per level of path. "
        "<b><i>nuclei here</i> is the terminal count</b>: nuclei whose delivered label is exactly "
        "that path and goes no deeper. <b><i>at or below</i> is the subtotal</b> including every "
        "descendant, so the two are equal only at a row with no children. A greyed row is a "
        "branch nothing terminates at — it carries no label of its own and exists to show the "
        "lineage's shape. Rows shown as a full monospaced path are nuclei whose independent "
        "compartment differs from the path they were given; where the concordance above is total, "
        "there are none.</p>")
    body.append("<p class='sub'>"
                + " · ".join(f"<b>{k}</b> {v}" for k, v in _WHY_WORDS.items()) + ".</p>")
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

    # ---- the delivered tree, with L1 integrated ------------------------------------------
    # AFTER the scope and BEFORE composition. The scope says what the labels below were allowed
    # to be; this says what they turned out to be; composition then counts them across the
    # design. Put before the scope it would show a taxonomy with no account of why it stops
    # where it does, which is the reading error the `stops here because` column exists to stop.
    body += taxonomy_section(ctx, out_dir=out)

    # ---- composition: the TWO annotations, and only those --------------------------------
    body.append("<h2>Composition</h2>")
    body.append("<p class='lede'>Each block below states the <b>obs column it was drawn from</b>. "
                "The delivered annotation comes first, then the same annotation with nothing left "
                "UNRESOLVED, then the joint route's correction of that where one was given, and "
                "the independent L1 walk last &mdash; L1 is a CHECK on the delivered call rather "
                "than a coarser view of it.</p>")
    body.append("<p class='lede'>The two are independent evidence, not two resolutions of one "
                "call: the <b>L1 annotation</b> is an independent depth-1 walk over the complete declared "
                "compartment set against the full corpus, and the <b>scope annotation</b> is the "
                "same unchanged walk against the labels the cohort's own vote left standing. They "
                "describe the same nuclei — so they <i>can</i> disagree, and a disagreement is a finding rather "
                "than an error. There is no level-2 or level-3 annotation: nothing was annotated "
                "at those depths, and a share quoted at a depth no walk terminated on is a number "
                "with no call behind it.</p>")

    # THE TWO ANNOTATIONS, EACH FROM ITS OWN COLUMN. scAnno delivers exactly two label columns:
    # the INDEPENDENT depth-1 walk and the SCOPE annotation. They are independent evidence about
    # the same nuclei, not a coarse and a fine view of one call.
    #
    # Neither is read from a level index. `L1` is the scope path TRUNCATED, which inherits every
    # edit the scope made - rendering it as "the L1 annotation" would manufacture perfect
    # agreement between the two columns for exactly the objects that measured none. And the
    # deepest `L{n}` coincides with the delivered terminals only when n happens to be the
    # declared tree's maximum depth, so a table built on it is right by accident.
    ctx_l1 = ctx.l1_rows()
    ctx_fl1 = ctx.forced_l1_rows()
    ctx_fsc = ctx.forced_scope_rows()
    # FOUR BLOCKS, IN PAIRS. Each annotation is shown as the walk delivered it and again with
    # every walked nucleus pushed to a leaf. The forced pair is NOT a better version of the
    # honest one - it is the same annotation with the calls the walk declined made on the
    # reader's behalf, and it is placed next to its own original so the difference is the thing
    # on the page rather than a footnote.
    ctx_jr = ctx.joint_route_rows()
    # THE DELIVERED ANNOTATION LEADS. It used to be L1 first, on the reasoning that a reader
    # meets the compartments before the subtypes - but L1 is CONTEXT for the delivered call and
    # is not what anything downstream consumes, so leading with it puts the answer third on the
    # page. The order is now: the annotation as the walk delivered it, the same with nothing
    # UNRESOLVED, the joint route's correction of that, and then L1 as the independent check.
    #
    # Every block NAMES THE OBS COLUMN IT WAS DRAWN FROM. Four blocks described as "the scope
    # annotation" and "FORCED" read identically whichever object they were built over, so a
    # report of a JOINT route's own columns and a report of a per-sample route's are
    # indistinguishable on the page - which is exactly how a reader comes to believe a figure
    # shows one annotation while it shows another.
    for _title, _rows, _what, _key in (
            ("the scope annotation", ctx.scope_rows(),
             "the labels the cohort's own vote left available, each at the depth it terminates "
             "at - so the set is mixed across levels by construction", ctx.path_key),
            ("the scope annotation, FORCED", ctx_fsc,
             "the same walk with nothing left UNRESOLVED: each such nucleus descends from where "
             "it stopped to a leaf, by the argmax already recorded at every step",
             ctx.forced_key),
            ("the JOINT ROUTE", ctx_jr,
             "the forced annotation with a second, JOINT clustering's corrections applied - a "
             "third reading of the same nuclei and not a better one, because the joint "
             "partition is the coarser of the two and absorbs populations as well as "
             "recovering them", ctx.joint_route_key),
            ("the L1 annotation", ctx_l1,
             "an INDEPENDENT depth-1 walk - one decision at the root, and no seal at any depth "
             "can move it. Shown after the delivered annotation because it is a CHECK on it, "
             "not a coarser view of it", ctx.l1_key),
            ("the L1 annotation, FORCED", ctx_fl1,
             "the same independent walk with every UNRESOLVED nucleus pushed onto the root's "
             "argmax - a child the walk already scored and then declined, because the margin "
             "was below the bar", ctx.forced_l1_key)):
        if _rows is None and _title == "the JOINT ROUTE":
            continue          # no joint column was named; the block is not an absence to explain
        body.append(f"<h3>{_title}</h3>")
        if _key:
            body.append(f"<p class='sub'>drawn from <code>{_esc(_key)}</code></p>")
        if _rows is None and "FORCED" in _title:
            body.append(_absent_section(
                "a forced label column (`scanno annotate --resolve`, reported with "
                "`--forced-key` / `--forced-l1-key`)",
                "Without it this document cannot show what the annotation looks like with "
                "nothing left UNRESOLVED. It is NOT substituted by redistributing the "
                "unresolved share here: which leaf each nucleus would take is a measurement "
                "the walk makes, not an assumption this document may make for it."))
            continue
        if _rows is None:
            body.append(_absent_section(
                "the independent L1 column (`scanno annotate --l1-tree`, reported with "
                "`--l1-key`)",
                "Without it this document cannot show the L1 annotation at all. It is NOT "
                "substituted with the scope path truncated to depth 1: that column inherits "
                "every seal the scope made, so the two would agree by construction and the "
                "agreement would mean nothing."))
            continue
        body.append(f"<p class='lede'>{_what}.</p>")
        _csv = [[r["label"], r["depth"], r["nuclei"], round(r["share"], 3), r["samples"]]
                for r in _rows]
        _stem = ("l1" if "L1" in _title else "scope") + ("_forced" if "FORCED" in _title else "")
        _src = _write_table(out, f"annotation_{_stem}.csv",
                            ["label", "depth", "nuclei", "share_pct", "samples"], _csv)
        body.append(_table(
            ["label", "depth", "nuclei", "share", "samples"],
            [[_esc(r["label"]), f"L{r['depth']}" if r["depth"] else "—", f"{r['nuclei']:,}",
              f"{r['share']:.1f}%", f"{r['samples']}/{len(ctx.samples)}"] for r in _rows],
            source=_src))
        # The composition figures are keyed by DEPTH, so each annotation is drawn at the depth its
        # labels live at: 1 for the L1 walk, the deepest for the scope, where a full path is
        # itself. Their established F-names and filenames are unchanged — renaming them would
        # orphan the files `figures.py` writes.
        if "FORCED" in _title:
            # WHAT MOVED, which is the only thing this block adds over the one above it.
            _mv = ctx.forced_moved("forced_l1" if "L1" in _title else "forced",
                                   "l1" if "L1" in _title else "path")
            if _mv["n"]:
                body.append(_table(
                    ["from -> to", "nuclei"],
                    [[_esc(k), f"{v:,}"] for k, v in list(_mv["moves"].items())[:15]]))
                body.append(f"<p class='sub'><b>{_mv['n']:,} nuclei moved</b> "
                            f"({100.0 * _mv['n'] / max(ctx.n, 1):.2f}% of the cohort). Every one "
                            f"of them is a call the walk declined to make; their margin was below "
                            f"the gap bar BY CONSTRUCTION, which is why it stopped. These are the "
                            f"least certain labels in the column.</p>")
            else:
                body.append("<p class='sub'>No nucleus moved: the walk resolved every one it "
                            "annotated, so this column equals the one above it.</p>")
            # Drawn from the FORCED column itself, never the unforced figure reused under a
            # forced heading. Same palette as the pair above, so a cell type keeps its colour and
            # the two are comparable by eye.
            _c = "forced_l1" if "L1" in _title else "forced"
            _w = f"the L1 annotation ({_key})" if "L1" in _title else f"the scope annotation ({_key})"
            _s = "l1" if "L1" in _title else "scope"
            body.append(A.fig("F106", name=f"F106_composition_forced_{_s}_by_group",
                              col=_c, by="group", what=_w))
            body.append(A.fig("F106", name=f"F106_composition_forced_{_s}_by_sample",
                              col=_c, by="sample", what=_w))
            body.append(A.fig("F107", name=f"F107_per_sample_forced_{_s}", col=_c, what=_w))
            continue
        if _rows is ctx_jr:
            # Drawn from the JOINT ROUTE's own column, never the forced figure reused under a
            # joint heading - which is the defect this whole block exists to make impossible.
            body.append(A.fig("F106", name="F106_composition_joint_route_by_group",
                              col="joint_route", by="group",
                              what=f"the joint route ({_key})"))
            body.append(A.fig("F106", name="F106_composition_joint_route_by_sample",
                              col="joint_route", by="sample",
                              what=f"the joint route ({_key})"))
            body.append(A.fig("F107", name="F107_per_sample_joint_route", col="joint_route",
                              what=f"the joint route ({_key})"))
            continue
        if _rows is ctx_l1:
            body.append(A.fig("F102", name="F102_composition_level1_by_group", by="group"))
            body.append(A.fig("F102", name="F102_composition_level1_by_sample", by="sample"))
            body.append(A.fig("F141", name="F141_per_sample_level1", depth=1))
        else:
            _d = list(ctx.levels)[-1]
            body.append(A.fig("F103", name=f"F103_composition_level{_d}_by_group",
                              by="group", depth=_d))
            body.append(A.fig("F103", name=f"F103_composition_level{_d}_by_sample",
                              by="sample", depth=_d))
            body.append(A.fig("F143", name=f"F143_per_sample_level{_d}", depth=_d))

    # TWO ANNOTATIONS, AND ONLY TWO. scAnno delivers the INDEPENDENT L1 walk and the SCOPE
    # annotation; there is no "level 2 annotation" and no "level 3 annotation". The intermediate
    # depths are TRUNCATIONS of the scope path, and presenting them as sections of their own
    # invites a reader to quote a level nothing was ever annotated at — and to read a sealed
    # compartment appearing at "level 2" as a level-2 call rather than as the scope's own
    # terminal. So the first level and the delivered scope are shown, and nothing between them.
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
        # THE TWO ANNOTATIONS, not every depth. The deepest panel depth is where the scope's
        # labels live; the depths between are truncations nothing was annotated at.
        # FOUR BLOCKS, matching the composition section, and NOT one per taxonomy level.
        #
        # The L1 block keeps F130/F131: L1 is a single level, its panels are the depth-1 panels,
        # and the level figure is exactly right for it.
        #
        # The SCOPE block does NOT use the deepest level dotplot any more. That figure read the
        # panels of ONE depth and rows truncated to it, so every scope label that terminates
        # SHORT of the deepest level - which is what a seal produces - had no panel there and
        # contributed no gene columns. The figure rendered, looked complete, and omitted the
        # evidence for precisely the labels the cohort's vote created. F108 reads the delivered
        # column and takes each label's panel from the depth that label terminates at.
        body.append("<h3>the L1 annotation</h3>")
        body.append(A.fig("F130", name="F130_dotplot_level1"))
        body.append(A.fig("F131", name="F131_featureplot_level1"))

        _marker_blocks = [("the L1 annotation, FORCED", "forced_l1", ctx.has_forced_l1),
                          ("the scope annotation", ctx.path_key, True),
                          ("the scope annotation, FORCED", "forced", ctx.has_forced)]
        for _t2, _c2, _have in _marker_blocks:
            body.append(f"<h3>{_t2}</h3>")
            if not _have:
                body.append(_absent_section(
                    "a forced label column (`scanno annotate --resolve`)",
                    "Without it the forced marker figure cannot be drawn. It is NOT substituted "
                    "with the unforced dotplot, which would show the unforced rows under a "
                    "forced heading."))
                continue
            _key = "forced" if _c2 == "forced" else ("forced_l1" if _c2 == "forced_l1" else "path")
            _stem = {"forced": "scope_forced", "forced_l1": "l1_forced"}.get(_c2, "scope")
            body.append(A.fig("F108", name=f"F108_dotplot_{_stem}", col=_key, what=_t2))
            if "FORCED" in _t2:
                # The FEATURE plot is expression on the embedding, coloured by gene, and does not
                # read the label column at all - a forced version is byte-identical. Saying so
                # beats drawing the same picture twice under a heading that implies a difference.
                body.append("<p class='sub'>The feature plots are not repeated here: they show "
                            "gene expression on the embedding and do not read the label column, "
                            "so the forced version is the same figure as the one above.</p>")
            else:
                body.append(A.fig("F136", name="F136_featureplot_scope",
                                  depth=max(ctx.panel_depths()) if ctx.panel_depths() else 1))
        # SORTED, and each figure under its own heading. This was a set literal, so the two
        # F134 panels came out in arbitrary order with one shared heading between them - which
        # reads as a stray figure rather than as one per annotation.
        for d in (sorted({1, max(ctx.panel_depths())}) if ctx.panel_depths() else [1]):
            b = ctx.marker_breadth(d)
            if not b:
                continue
            body.append(f"<h3>marker breadth — "
                        f"{'the L1 annotation' if d <= 1 else 'the scope annotation'}</h3>")
            body.append(A.fig("F134", name=f"F134_breadth_level{d}", depth=d))
            src = _write_table(out, f"marker_breadth_level{d}.csv",
                               ["gene", "plotted_for", "own", "best_other", "overall",
                                "labels_over_25pct"],
                               [[r["gene"], r["plotted_for"], round(r["own"], 4),
                                 round(r["best_other"], 4), round(r["overall"], 4),
                                 r["labels_over_25"]] for r in b])
            broad = [r for r in b if r["labels_over_25"] >= 4]
            if broad:
                _an = ("the L1 annotation" if d <= 1 else "the scope annotation")
                body.append(f"<h3>genes that carry no identity information for {_an}</h3>")
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
        # `ctx.sample_rows()`, NOT `ctx.P["sample"] == s`. `sample_links` is keyed by the OBJECT
        # name — `<sample>.filtered` from `<sample>.filtered_annotated.h5ad` — while obs holds
        # `<sample>`, so the direct comparison matches nothing and every card read "0 nuclei".
        # Context carries a resolver for exactly this mismatch and the sample PAGES already use
        # it; this call site was written without it.
        #
        # 0 is the worst value this could have failed to: it renders as a FACT — this sample has
        # no nuclei — rather than as a lookup that found nothing. An unresolvable name is now an
        # em dash, which is visibly not a count.
        cards = []
        for s, href in sample_links:
            rows = ctx.sample_rows(s)
            n = None if rows is None else len(rows)
            cards.append(f'<a href="{_esc(href)}">{_esc(s)}<span class="n">'
                         + (f'{n:,} nuclei' if n else '— nuclei not resolved')
                         + '</span></a>')
        body.append("<div class='samples'>" + "".join(cards) + "</div>")

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
    # L1 and the scope, per sample — the same two annotations the cohort page carries. The
    # intermediate depths are truncations and are not shown here either.
    # The two DELIVERED annotations, plus their forced twins where the run wrote them - the same
    # four blocks the cohort page shows, so a reader moving between the two pages is not asked to
    # match different structures.
    #
    # The scope block reads the DELIVERED column, not `L{D}`. The deepest level index coincides
    # with the delivered set only when the tree's maximum depth happens to equal it, so a table
    # built on the index is right by accident and silently wrong on a deeper tree.
    _blocks = [("the L1 annotation", "l1" if ctx.has_l1 else "L1", 1)]
    if ctx.has_forced_l1:
        _blocks.append(("the L1 annotation, FORCED", "forced_l1", 1))
    _blocks.append(("the scope annotation", "path", D))
    if ctx.has_forced:
        _blocks.append(("the scope annotation, FORCED", "forced", D))
    for _title2, _col2, d in _blocks:
        body.append(f"<h3>{_title2}</h3>")
        if _col2 not in rows:
            body.append(_absent_section(f"the {_col2!r} column on this object",
                                        "so this block cannot be shown for this sample."))
            continue
        col = rows[_col2]
        order = (ctx.label_order_for(_col2) if _col2 in ("forced", "forced_l1", "path", "l1")
                 else ctx.label_order(d))
        tab = []
        for l in order:
            c = int((col == l).sum())
            if not c:
                continue
            here = 100.0 * c / max(n, 1)
            coh = 100.0 * int((ctx.P[_col2] == l).sum()) / max(ctx.n, 1)
            tab.append([f"<span class='sw' style='background:{ctx.colour(l)}'></span>"
                        + _esc(leaf(l)), f"{c:,}", f"{here:.1f}%", f"{coh:.1f}%",
                        f"{here - coh:+.1f} pp"])
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
        # THE TWO ANNOTATIONS, machine-readable. `composition` above is a per-depth breakdown and
        # stays because a consumer may want it; these are the two things that were ANNOTATED, and
        # a consumer should not have to pick a level index to find them.
        "annotation_l1": ctx.l1_rows(),
        "annotation_scope": ctx.scope_rows(),
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
        # Both delivered columns, machine-readable. A consumer asking "what is this cell at L1 and
        # how deep did it go" must get it without scraping the HTML the section renders.
        "l1": ({"column": ctx.l1_key,
                "concordance": (lambda c: c and {k: v for k, v in c.items() if k != "pairs"}
                                | {"pairs": [{"deep_walk_root": a, "independent_l1": b,
                                              "nuclei": n} for (a, b), n in c["pairs"].items()]}
                                )(ctx.l1_concordance()),
                "bands": ctx.l1_bands()} if getattr(ctx, "has_l1", False) else None),
        "palette": ctx.palette.as_dict(),
        "figures_not_drawn": absent,
        "figures_failed": failed,
        "reports": {"cohort": str(cohort.relative_to(out)),
                    "samples": [h for _, h in links]},
    }
    (out / "report.json").write_text(json.dumps(payload, indent=1, default=str),
                                     encoding="utf-8")
    return cohort, payload
