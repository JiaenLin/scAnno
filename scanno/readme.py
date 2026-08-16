"""Write the README that says WHICH FILE TO USE.

A directory of ten `*_annotated.h5ad` beside ten `*_clustered.h5ad`, three similarly-named joint
objects and an `.npz` is not self-describing, and the cost of that falls on whoever arrives next.
The question they have is never "what is in this directory" - they can list it - but **which of
these is the answer, and which must I not use**.

So this module INSPECTS what is actually on disk and writes what it finds. It does not template
a description of what scAnno usually produces: a README that describes the intended output of a
run that half-failed is worse than none, because it reads exactly like a correct one.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

#: What each thing in a scAnno output tree IS, and whether a consumer should read it.
#: (glob, role, use-it?, one line)
KNOWN = [
    ("annotated/*_annotated.h5ad", "the annotation", True,
     "**THE DELIVERABLE.** One object per sample, carrying every label column. This is what "
     "downstream stages read."),
    ("clustered/*_clustered.h5ad", "intermediate", False,
     "Superseded by `annotated/`. Same cells, clustering only, no labels. Kept because the "
     "annotation was built from it; **do not read it for labels** - it has none."),
    ("joint/cohort_joint_embedding.h5ad", "the joint embedding", True,
     "ONE embedding computed over all samples together, by `scanno embed`. Use this for any "
     "figure that needs a single manifold. **Not integrated** - no batch correction was applied."),
    ("joint/cohort_annotated.h5ad", "the second route", False,
     "An independent joint clustering, annotated the same way, kept for route comparison. It is "
     "the WEAKER route where clusters are library-dominated; not the deliverable."),
    ("joint/cohort_clustered.h5ad", "intermediate", False,
     "The joint clustering before annotation. Superseded by `cohort_annotated.h5ad`."),
    ("store.npz", "the background", False,
     "Marker-score background built once over the pooled libraries so a cluster's score does "
     "not depend on which sample it sat in. An input to annotation, not a result."),
    ("report/reports/cohort.html", "the document", True,
     "**START HERE.** The cohort report: composition, reliability, markers, and whatever was "
     "withheld upstream."),
    ("report/reports/samples/*.html", "per sample", True,
     "One comprehensive page per sample. Open one when a cohort number looks wrong."),
    ("report/report.json", "machine-readable", True,
     "Every number the pages show, so nothing has to be scraped from HTML."),
    ("report/tables/*.csv", "the numbers", True, "Each table the report displays, openable."),
    ("report/figures/**/*.png", "the figures", True, "Referenced by the reports."),
]


def _label_columns(obs_names, path_key=None):
    """Group the obs columns a consumer has to choose between, and name the one that is the answer."""
    import re
    swept, plain = {}, []
    for c in obs_names:
        m = re.match(r"^(scanno|scAnno)_([a-zA-Z0-9]+?)_r([0-9p.]+)$", str(c))
        if m:
            swept.setdefault(m.group(3), []).append(str(c))
        elif str(c).startswith(("scanno_", "scAnno_")):
            plain.append(str(c))
    return swept, sorted(plain)


def build(out_dir, *, chosen_resolution=None, path_key=None, obs_columns=(), n_cells=None,
          n_samples=None, taxonomy_depth=None, version="", command=None, inputs=None,
          withheld=None, limits=()):
    """Inspect `out_dir` and return the README text describing what is actually there."""
    out = Path(out_dir)
    now = datetime.now(timezone.utc).astimezone()

    found, missing = [], []
    for pat, role, use, why in KNOWN:
        hits = sorted(out.glob(pat))
        (found if hits else missing).append((pat, role, use, why, len(hits)))

    use_rows = "".join(
        f"| `{pat}` | {n} | {'**YES**' if use else 'no'} | {why} |\n"
        for pat, role, use, why, n in found)

    swept, plain = _label_columns(obs_columns, path_key)
    res_note = ""
    if swept:
        tags = sorted(swept)
        cols_at = swept.get(str(chosen_resolution).replace(".", "p"), [])
        res_note = f"""
## Which label column is the answer

`obs` carries **{len(obs_columns)} columns**, because the annotation was swept over
**{len(tags)} clustering resolutions** ({', '.join(tags)}) and each one wrote its own set.

**Use the columns for the chosen resolution: `{chosen_resolution}`.**
{'They are: ' + ', '.join(f'`{c}`' for c in sorted(cols_at)) + '.' if cols_at else ''}

{'`' + str(path_key) + '` is the full label path — `Immune/Myeloid/Macrophage` — and is the column downstream analysis should read.' if path_key else ''}
Truncate it to work at the depth you can defend: the first component is the level-1 label, the
first two are level 2, and so on. **The other resolutions are kept so the choice can be argued
with, not because they are alternatives to use.**
"""

    lim = "".join(f"- {l}\n" for l in limits) or "- (none recorded by the run)\n"
    miss_note = ""
    if missing:
        miss_note = ("\n## Not present in this directory\n\n"
                     + "".join(f"- `{p}` — {w}\n" for p, r, u, w, n in missing)
                     + "\nAbsent because the command that writes it was not run, or it failed. "
                       "Listed so a reader does not go looking.\n")

    return f"""# scAnno output

Written by scAnno {version} on {now:%Y-%m-%d %H:%M %Z}. **This file was generated by inspecting
this directory**, not from a template, so it describes what is actually here.

{f'- **{n_cells:,} cells**' if n_cells else ''}{f' across **{n_samples} samples**' if n_samples else ''}
{f'- taxonomy depth **{taxonomy_depth}**' if taxonomy_depth else ''}
{f'- withheld before annotation: **{withheld:,}**' if withheld is not None else ''}

## Which file do I use?

| path | files | use it? | what it is |
|---|---|---|---|
{use_rows}
{res_note}
## What this cannot tell you

{lim}
{miss_note}
## How it was produced

{f'```{chr(10)}{command}{chr(10)}```' if command else '_(command not recorded by this run)_'}

{f'**Input**: `{inputs}`' if inputs else ''}

scAnno does not ship a marker corpus or a taxonomy — both are supplied per study, the same way a
QC tool ships no genome. The labels here are only as good as those two inputs, and **no label is
established as correct by this tool**: it reports what the corpus supports, not what is true.
"""


def write(out_dir, **kw):
    p = Path(out_dir) / "README.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build(out_dir, **kw), encoding="utf-8")
    return p
