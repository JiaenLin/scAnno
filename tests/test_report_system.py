"""The report system: palette, primitives, depth generalisation, and the two documents.

Every test here is a defect that reached a rendered figure or a published page. They are written
as assertions on ARTISTS and DATA, not on pixels: a pixel diff tells you something moved and not
what, and it fails on a matplotlib upgrade for reasons that are nobody's fault.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanno.palette import BASE_HUES, Palette, SENTINELS, _to_hls, depth_of  # noqa: E402
from scanno.primitives import (GEOMETRY, NotDrawable, panel_grid, readable_on,  # noqa: E402
                               unique_ticks)

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# ------------------------------------------------------------------ palette
def test_palette():
    print("palette")
    paths = ["Cardiomyocyte/Working/Atrial", "Cardiomyocyte/Working/Ventricular",
             "Endothelial/Vascular", "Endothelial/Lymphatic",
             "Immune/Myeloid/Macrophage", "Immune/Lymphoid/T cell",
             "Stromal/Fibroblast", "UNRESOLVED", "EXCLUDED"]
    p = Palette(paths)
    check("depth is read from the paths", p.depth == 3, f"got {p.depth}")

    # rule 3: never None, for anything
    check("every lookup has a default",
          all(isinstance(p.of(x), str) and p.of(x).startswith("#")
              for x in ["", "Nope", None, "a/b/c/d/e/f"]))

    # rule 2: two different greys, neither shaded
    check("sentinels are two different greys",
          p.of("EXCLUDED") == SENTINELS["EXCLUDED"]
          and p.of("UNRESOLVED") == SENTINELS["UNRESOLVED"]
          and p.of("EXCLUDED") != p.of("UNRESOLVED"))
    check("a sentinel below the root is still a sentinel",
          Palette(["Immune/EXCLUDED"]).of("Immune/EXCLUDED") == SENTINELS["EXCLUDED"])

    # rule 1: hue is the ancestor's
    for kid in ("Immune/Myeloid", "Immune/Myeloid/Macrophage", "Immune/Lymphoid/T cell"):
        check(f"{kid} keeps its root's hue",
              abs(_to_hls(p.of(kid))[0] - _to_hls(p.of("Immune"))[0]) < 0.01,
              "tolerance is for 8-bit rounding (~0.001 of hue), not for drift")

    # rule 4: no collision across MIXED depths
    nodes = ["Cardiomyocyte", "Cardiomyocyte/Working", "Cardiomyocyte/Working/Atrial",
             "Cardiomyocyte/Working/Ventricular", "Immune", "Immune/Myeloid",
             "Immune/Myeloid/Macrophage", "Immune/Lymphoid", "Immune/Lymphoid/T cell",
             "Endothelial", "Endothelial/Vascular", "Endothelial/Lymphatic",
             "Stromal", "Stromal/Fibroblast", "EXCLUDED", "UNRESOLVED"]
    check("no two labels that appear together share a colour", not p.collisions(nodes),
          str(p.collisions(nodes)[:2]))

    # the two recorded failures of the earlier shading rules
    ls = [_to_hls(p.of(n))[1] for n in nodes if n not in SENTINELS]
    check("nothing is driven to black or white", min(ls) >= 0.29 and max(ls) <= 0.83,
          f"lightness {min(ls):.2f}-{max(ls):.2f}")
    import itertools
    deep = ["/".join(("R",) + c) for c in itertools.product(*[["a", "b"]] * 5)]
    q = Palette(deep)
    allnodes = sorted({"/".join(l.split("/")[:i + 1])
                       for l in deep for i in range(len(l.split("/")))})
    check("depth 6 does not collide", not q.collisions(allnodes),
          f"{len(allnodes)} nodes, {len(q.collisions(allnodes))} clashes")

    # base hues are distinguishable from one another
    worst = min((abs(_to_hls(a)[0] - _to_hls(b)[0]), a, b)
                for a, b in itertools.combinations(BASE_HUES[:8], 2))
    check("the first eight base hues are separated", worst[0] > 0.02,
          f"closest pair {worst[1]} / {worst[2]}")

    # pinning
    pin = Palette(paths, pinned={"Immune": "#7B3FA0"})
    check("a pin wins", pin.of("Immune") == "#7B3FA0")
    check("pinning a lineage recolours its subtree",
          abs(_to_hls(pin.of("Immune/Myeloid/Macrophage"))[0]
              - _to_hls("#7B3FA0")[0]) < 0.01)
    check("a pinned leaf is exactly the pin",
          Palette(paths, pinned={"Immune/Myeloid/Macrophage": "#abcdef"})
          .of("Immune/Myeloid/Macrophage") == "#abcdef")
    check("Palette.load rejects a non-colour",
          _raises(lambda: Palette.load(_tmp_json({"A": "red"}))))


def _raises(fn):
    try:
        fn()
    except Exception:                                                     # noqa: BLE001
        return True
    return False


def _tmp_json(obj):
    import tempfile
    p = Path(tempfile.mkstemp(suffix=".json")[1])
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


# ------------------------------------------------------------------ primitives
def test_primitives():
    print("primitives")
    # the zip()-truncation class of bug: a grid must never be smaller than the series
    for n in (1, 2, 5, 8, 12, 13):
        fig, axes = panel_grid(n)
        check(f"panel_grid({n}) returns exactly {n} axes", len(axes) == n, f"got {len(axes)}")
        import matplotlib.pyplot as plt
        plt.close(fig)

    check("panel_grid refuses zero panels", _raises(lambda: panel_grid(0)))

    # segment labels must be legible on their own segment
    check("label colour flips on a pale segment", readable_on("#f0e0d0") == "#1a1a1a")
    check("label colour flips on a dark segment", readable_on("#20303f") == "#ffffff")

    # a leaf name that is not unique must not collapse two populations into one tick
    t = unique_ticks(["Immune/Myeloid/Mac", "Stromal/Other/Mac", "Immune/Lymphoid/T"])
    check("ambiguous leaves are disambiguated", t[0] != t[1] and t[2] == "T", str(t))

    # saving geometry is part of the figure, and two figures deliberately have no tight box
    check("F133 and F134 save without a tight bounding box",
          GEOMETRY["F133"][1] is None and GEOMETRY["F134"][1] is None)
    check("the sweep and level figures save at 130 dpi",
          GEOMETRY["F100"][0] == 130 and GEOMETRY["F101"][0] == 130)

    # showfliers has no default: it is the only difference between three identical boxplots
    import inspect
    from scanno.primitives import quartile_boxes
    sig = inspect.signature(quartile_boxes)
    check("showfliers is required", sig.parameters["showfliers"].default is inspect._empty)
    from scanno.primitives import umap_scatter, stacked_rows
    check("denominator is required",
          inspect.signature(umap_scatter).parameters["denominator"].default is inspect._empty)
    check("label_floor is required",
          inspect.signature(stacked_rows).parameters["label_floor"].default is inspect._empty)


# ------------------------------------------------------------------ one colour authority
def test_one_colour_rule():
    print("one colour authority")
    src = {p.name: p.read_text(encoding="utf-8")
           for p in (Path(__file__).resolve().parents[1] / "scanno").glob("*.py")}
    import re
    offenders = []
    for name, text in src.items():
        if name in ("palette.py",):
            continue
        for m in re.finditer(r'^[A-Z_]*(PALETTE|HUES|COLOURS|COLORS)\s*=\s*[\[{]', text, re.M):
            offenders.append(f"{name}:{m.group(0)[:30]}")
        if re.search(r'^def shade\(', text, re.M):
            offenders.append(f"{name}: defines its own shade()")
    check("no module outside palette.py defines a hue list or a shade rule",
          not offenders, "; ".join(offenders[:3]))


# ------------------------------------------------------------------ depth generalisation
def test_depth_general():
    print("depth generalisation")
    src = (Path(__file__).resolve().parents[1] / "scanno")
    import re
    bad = []
    for f in ("figures.py", "context.py", "document.py"):
        text = (src / f).read_text(encoding="utf-8")
        # `"/" in label` silently matches level 3 as well as level 2
        # Code, not prose: these module docstrings NAME the patterns in order to forbid them,
        # so a lint that reads the docstring reports the rule as its own violation.
        code = re.sub(r'"""..*?"""', "", text, flags=re.S)
        for m in re.finditer(r'\b(if|elif|and|or|not)\s+[^\n]*"/"\s+in\s+\w+', code):
            bad.append(f"{f}: {m.group(0)[:40]}")
        for m in re.finditer(r'\.split\("/"\)\[0\]', code):
            bad.append(f"{f}: split('/')[0]")
    check("no module detects a level with a separator test", not bad, "; ".join(bad[:3]))
    check("depth_of ignores sentinels",
          depth_of(["A/B/C", "EXCLUDED", "UNRESOLVED"]) == 3)

    from scanno.figures import FIGURES
    for fid in ("F103", "F135", "F136", "F143", "F134"):
        import inspect
        check(f"{fid} takes a depth",
              "depth" in inspect.signature(FIGURES[fid][0]).parameters)


# ------------------------------------------------------------------ per-sample lookup
def test_sample_rows():
    """An object named for its FILE must still find its own rows.

    `S1.filtered.h5ad` gives an object named `S1.filtered` whose obs says `S1`.
    Matching per-sample rows on obs alone returned an empty frame, and every per-sample figure
    then reported a named absence — "no QC columns in obs" — for objects that carried them.
    A lookup failure that renders as a finding is worse than a crash.
    """
    print("per-sample lookup")
    import numpy as np, pandas as pd
    from scanno.context import Context

    class Fake:
        def __init__(self, name, n=40):
            self.n_obs = n
            self.obs = pd.DataFrame({
                "sample": ["S1"] * n,                       # NOT the object name
                "scanno_path": ["Immune/Myeloid"] * n,
                "total_counts": np.arange(n, dtype=float),
                "pct_counts_mt": np.ones(n),
            }, index=[f"{name}_c{i}" for i in range(n)])
            self.var = pd.DataFrame(index=["G1", "G2"])
            self.obsm, self.layers, self.X = {}, {}, None
            self.var_names = self.var.index

    ctx = Context(assay="snrna", objects=[("S1.filtered", Fake("S1.filtered"))], path_key="scanno_path")
    r = ctx.sample_rows("S1.filtered")
    check("rows found by OBJECT name", len(r) == 40, f"got {len(r)}")
    check("its QC columns are visible", "total_counts" in r and r["total_counts"].notna().any())
    check("obs-name lookup still works too", len(ctx.sample_rows("S1")) == 40)

    # THE INVARIANT. Every object this context holds must be able to find its own rows. If it
    # cannot, that is a defect and it must RAISE - never return empty, because empty becomes a
    # named absence and a named absence reads as a statement about the data.
    for smp in ctx.samples:
        check(f"invariant: {smp} finds its rows", len(ctx.sample_rows(smp)) > 0)
    try:
        ctx.P.loc[:, "_obj"] = "renamed"
        ctx.P.loc[:, "sample"] = "renamed"
        ctx.sample_rows("S1.filtered")
        raised = False
    except LookupError as e:
        raised = "bug in scAnno" in str(e)
    check("a lookup failure RAISES rather than returning empty", raised)


# ------------------------------------------------------------------ named absence
def test_named_absence():
    """An absence is NAMED and a defect is DISTINCT - run on the assembler, not read off its source
    (single-cell-harness B1-034)."""
    print("named absence")
    import tempfile
    from scanno import document as D
    from scanno import figures as F
    check("NotDrawable exists and carries a reason",
          issubclass(F.NotDrawable, Exception))
    real = D.draw
    try:
        with tempfile.TemporaryDirectory() as td:
            A = D.Assembler(ctx=None, out_dir=td, rel="figures/x", subdir="x")
            def absent(*_a, **_k):
                raise F.NotDrawable("no group column")
            D.draw = absent
            h_abs = A.fig("F154")
            def broken(*_a, **_k):
                raise RuntimeError("a bug")
            D.draw = broken
            h_bad = A.fig("F155")
    finally:
        D.draw = real
    check("an absent FIGURE renders as a named absence with its reason",
          "class='absent'" in h_abs and "no group column" in h_abs, h_abs[:120])
    # A DEFECT and an ABSENCE must never render the same. One is about scAnno, the other about
    # the data, and the page that conflates them reports a bug as a finding.
    check("a defect renders as a DEFECT, not an absence",
          "FAILED TO DRAW" in h_bad and "defect in" in h_bad and "class='absent'" not in h_bad,
          h_bad[:120])
    check("defects are collected separately from absences",
          [x["figure"] for x in A.absent] == ["F154"] and [x["figure"] for x in A.failed] == ["F155"])
    check("an absent SECTION is named, not omitted",
          "This section needs" in D._absent_section("a flag column", "nothing was withheld"))
    # Words beside every figure: its registered title, which the assembler prints over it, and a
    # legend where one is registered (F106-F108's captions come from the call that draws them).
    untitled = sorted(f for f, v in F.FIGURES.items() if not str(v[2]).strip())
    check("every figure id has a title the page prints over it", not untitled, str(untitled))
    check("and the legends registered are for figures that exist", set(D.LEGENDS) <= set(F.FIGURES),
          str(sorted(set(D.LEGENDS) - set(F.FIGURES))))


# ------------------------------------------------------------------ a removed gate draws no line
def test_the_design_figure_draws_no_gate():
    """F154 drew a dashed line at 3x labelled "rule one refuses beyond 3x" for a design-differential
    gate that was REMOVED (KNOWN_ISSUES; single-cell-harness B1-038). A line on a figure reads as a
    gate. Drawn here from a stub context: the ratio panel has bars and their true ratios, no line,
    no refusal wording, and is titled for what it shows."""
    print("design figure")
    from scanno.assay import words as assay_words
    from scanno.figures import F154

    class Ctx:
        words = assay_words("scrna")
        has_flag = True

        def flag_per_animal(self):
            return [{"arm": "arm1", "rate": 2.0}, {"arm": "arm1", "rate": 3.0},
                    {"arm": "arm2", "rate": 9.0}, {"arm": "arm2", "rate": 11.0}]

        def flag_by_factor(self):
            return [{"factor": "factor_a", "auto": False, "ratio": 4.0, "lo": 2.5, "hi": 10.0}]

        def _levels(self, key):
            return ["arm1", "arm2"]

    fig, caption = F154(Ctx())
    right = fig.axes[1]
    texts = [t.get_text() for t in right.texts] + [right.get_title(loc="left")]
    check("the ratio panel draws no line across the bars", not right.lines,
          f"{len(right.lines)} line(s)")
    check("and no refusal wording", not [t for t in texts if "refuse" in t.lower()], str(texts))
    check("it still prints the true ratio", any("4.00x" in t for t in texts), str(texts))
    check("titled for what it shows", right.get_title(loc="left") == "withheld fraction by design factor",
          right.get_title(loc="left"))


# ------------------------------------------------------------------ viewer fitness
def test_viewer_audit():
    print("viewer audit")
    from scanno.emit import audit_file, SCRATCH_UNS_PREFIXES
    check("audit_file is importable", callable(audit_file))
    check("scratch uns prefixes are declared", "_tmp" in SCRATCH_UNS_PREFIXES)
    # That `scanno lab` exists and offers --fix is run in tests/test_report_cli.py §4; it was a
    # search of cli.py's text for `sub.add_parser("lab"` (single-cell-harness B1-034).


if __name__ == "__main__":
    for t in (test_palette, test_primitives, test_one_colour_rule, test_depth_general,
              test_sample_rows,
              test_named_absence, test_the_design_figure_draws_no_gate, test_viewer_audit):
        t()
    print("")
    if FAIL:
        print(f"{len(FAIL)} FAILED: {', '.join(FAIL)}")
        raise SystemExit(1)
    print("all report-system checks passed")
